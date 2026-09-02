"""
Database Write Layer for AI Revenue Recovery Agent.

Provides robust SQLAlchemy session management, atomic transaction persistence,
deduplication on razorpay_payment_id, recovery action logging, and status state transitions.
"""

import os
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List, Generator

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from agent.models import Base, Transaction, RetryAttempt, RecoveryAction
from api_integration.schemas import NormalizedEvent

logger = logging.getLogger(__name__)

# Default SQLite path if not specified in environment
DEFAULT_DB_URL = "sqlite:///./data/recovery_agent.db"

_engine = None
_SessionFactory = None


def get_database_url() -> str:
    """Retrieves configured DATABASE_URL with fallback."""
    return os.getenv("DATABASE_URL", DEFAULT_DB_URL)


def _create_engine_instance(url: str):
    """Internal helper to instantiate an engine with appropriate connection parameters."""
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if "///" in url and not url.startswith("sqlite:///:memory:"):
            db_path = url.split("///", 1)[1].split("?")[0]
            dir_name = os.path.dirname(db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

    engine = create_engine(
        url,
        connect_args=connect_args,
        echo=os.getenv("SQL_ECHO", "False").lower() in ("true", "1"),
        pool_pre_ping=True
    )

    if url.startswith("sqlite") and not url.startswith("sqlite:///:memory:"):
        try:
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))
        except Exception as e:
            logger.debug(f"Could not set SQLite PRAGMA journal_mode: {e}")

    return engine


def get_db_engine(database_url: Optional[str] = None):
    """
    Creates or retrieves the SQLAlchemy engine singleton.
    Strictly connects to the database configured in DATABASE_URL or passed as argument.
    Raises explicit SQLAlchemyError if the database connection fails.
    """
    global _engine, _SessionFactory
    if _engine is not None and database_url is None:
        return _engine

    url = database_url or get_database_url()

    try:
        engine = _create_engine_instance(url)
        Base.metadata.create_all(engine)
    except Exception as exc:
        logger.error(f"Failed to connect to database at '{url}': {exc}")
        raise RuntimeError(
            f"Database connection failed for '{url}'. "
            f"Please verify that the database server is running and credentials in .env are correct. "
            f"Original error: {exc}"
        ) from exc

    _engine = engine
    _SessionFactory = None  # Reset session factory to bind to new engine
    return engine


def get_session_factory(engine=None):
    """Retrieves or creates a thread-safe scoped session factory."""
    global _SessionFactory
    if _SessionFactory is not None and engine is None:
        return _SessionFactory

    eng = engine or get_db_engine()
    factory = scoped_session(sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=eng))
    if engine is None:
        _SessionFactory = factory
    return factory


def init_db(database_url: Optional[str] = None):
    """Initializes tables in database and binds singleton engine/session."""
    global _engine, _SessionFactory
    _engine = None
    _SessionFactory = None
    engine = get_db_engine(database_url)
    Base.metadata.create_all(engine)
    logger.info(f"Database schema initialized against {engine.url}")
    return engine


@contextmanager
def get_db_session(session: Optional[Session] = None) -> Generator[Session, None, None]:
    """
    Context manager providing a transactional SQLAlchemy session.
    Automatically commits on normal exit, rolls back on exceptions, and closes session.
    If an external session is passed, it manages transaction rollback on error without closing.
    """
    if session is not None:
        try:
            yield session
        except Exception as exc:
            session.rollback()
            logger.error(f"Database transaction rolled back due to error: {exc}", exc_info=True)
            raise
        return

    session_factory = get_session_factory()
    db_session: Session = session_factory()
    try:
        yield db_session
        db_session.commit()
    except Exception as exc:
        db_session.rollback()
        logger.error(f"Database transaction rolled back due to error: {exc}", exc_info=True)
        raise
    finally:
        db_session.close()


def save_transaction(
    event: NormalizedEvent,
    session: Optional[Session] = None
) -> Tuple[Transaction, bool]:
    """
    Persists a transaction from a NormalizedEvent into the database.
    Deduplicates on `razorpay_payment_id`.

    If a transaction with the same `razorpay_payment_id` already exists:
    - Returns the existing Transaction instance.
    - Updates status/failure details if the event is more recent or has new status.
    - Flag `is_created` is False.

    If it does not exist:
    - Creates and persists a new Transaction.
    - Flag `is_created` is True.

    Args:
        event: NormalizedEvent instance from normalizer.
        session: Optional external SQLAlchemy Session.

    Returns:
        Tuple of (Transaction, is_created: bool)
    """
    target_payment_id = event.payment_id or event.entity_id
    if not target_payment_id:
        target_payment_id = f"pay_syn_{event.event_id}"

    with get_db_session(session) as s:
        # Check for existing record by razorpay_payment_id (deduplication)
        existing_tx = s.query(Transaction).filter_by(razorpay_payment_id=target_payment_id).first()

        if existing_tx:
            logger.info(f"Transaction with payment_id='{target_payment_id}' already exists (id='{existing_tx.id}'). Deduplicated.")
            
            # Optionally update status and failure information if changed
            if event.status and existing_tx.status != event.status:
                existing_tx.status = event.status
            if event.error_code and not existing_tx.failure_code:
                existing_tx.failure_code = event.error_code
            if event.error_description and not existing_tx.failure_reason:
                existing_tx.failure_reason = event.error_description

            s.flush()
            return existing_tx, False

        # Build new Transaction ORM instance
        tx_id = f"tx_{event.entity_id}" if not event.entity_id.startswith("tx_") else event.entity_id
        
        # Ensure primary key uniqueness in rare cases where entity_id duplicates
        existing_by_pk = s.query(Transaction).filter_by(id=tx_id).first()
        if existing_by_pk:
            tx_id = f"tx_{target_payment_id}_{int(datetime.now(timezone.utc).timestamp())}"

        failure_reason = (
            event.error_description
            or event.error_reason
            or f"Event {event.event_type} received"
        )
        failure_code = event.error_code or event.error_reason or "UNKNOWN_ERROR"

        new_tx = Transaction(
            id=tx_id,
            razorpay_payment_id=target_payment_id,
            merchant_id=event.merchant_id or "unknown_merchant",
            amount=float(event.amount),
            currency=event.currency or "INR",
            status=event.status or "FAILED",
            failure_reason=failure_reason,
            failure_code=failure_code,
            created_at=event.created_at or datetime.utcnow()
        )

        try:
            s.add(new_tx)
            s.flush()
            logger.info(f"Saved new Transaction: id='{new_tx.id}', payment_id='{new_tx.razorpay_payment_id}', status='{new_tx.status}'")
            return new_tx, True
        except IntegrityError:
            s.rollback()
            existing_tx = s.query(Transaction).filter_by(razorpay_payment_id=target_payment_id).first()
            if existing_tx:
                logger.info(f"Concurrent insert detected for payment_id='{target_payment_id}'. Deduplicated via UNIQUE constraint.")
                return existing_tx, False
            raise


def save_recovery_action(
    transaction_id: str,
    action_type: str,
    action_payload: Any,
    status: str = "PENDING",
    session: Optional[Session] = None
) -> RecoveryAction:
    """
    Saves a recovery action taken for a specific transaction into recovery_actions table.

    Args:
        transaction_id: The primary key (id) of the transaction in transactions table.
        action_type: Action category (e.g. 'RETRY', 'PAYMENT_LINK', 'INVOICE_CHASER', 'NOTIFICATION').
        action_payload: Dictionary, JSON string, or text payload sent or scheduled.
        status: Action status ('PENDING', 'EXECUTED', 'FAILED').
        session: Optional external SQLAlchemy Session.

    Returns:
        The created RecoveryAction instance.
    """
    # Serialize payload to JSON string if it is dict or list
    if isinstance(action_payload, (dict, list)):
        payload_str = json.dumps(action_payload, default=str)
    else:
        payload_str = str(action_payload) if action_payload is not None else "{}"

    with get_db_session(session) as s:
        # Validate that transaction exists
        tx = s.query(Transaction).filter_by(id=transaction_id).first()
        if not tx:
            # Check by razorpay_payment_id in case payment_id was provided
            tx = s.query(Transaction).filter_by(razorpay_payment_id=transaction_id).first()
            if tx:
                transaction_id = tx.id
            else:
                logger.warning(f"Creating recovery action for unconfirmed transaction_id='{transaction_id}'")

        action = RecoveryAction(
            transaction_id=transaction_id,
            action_type=action_type,
            action_payload=payload_str,
            status=status,
            created_at=datetime.utcnow()
        )
        s.add(action)
        s.flush()
        logger.info(f"Saved RecoveryAction: id={action.id}, tx='{transaction_id}', type='{action_type}', status='{status}'")
        return action


def update_transaction_status(
    razorpay_payment_id: str,
    new_status: str,
    failure_reason: Optional[str] = None,
    session: Optional[Session] = None
) -> Optional[Transaction]:
    """
    Updates the status (and optional failure_reason) of a transaction given its razorpay_payment_id or id.

    Args:
        razorpay_payment_id: Razorpay payment ID (e.g. 'pay_xxx') or internal tx id ('tx_xxx').
        new_status: New status string (e.g. 'RECOVERED', 'RETRY_PENDING', 'EXHAUSTED', 'FAILED').
        failure_reason: Optional updated failure reason string.
        session: Optional external SQLAlchemy Session.

    Returns:
        Updated Transaction instance, or None if transaction was not found.
    """
    with get_db_session(session) as s:
        tx = (
            s.query(Transaction)
            .filter(
                (Transaction.razorpay_payment_id == razorpay_payment_id) |
                (Transaction.id == razorpay_payment_id)
            )
            .first()
        )

        if not tx:
            logger.warning(f"Cannot update status: No transaction found for identifier='{razorpay_payment_id}'")
            return None

        tx.status = new_status
        if failure_reason is not None:
            tx.failure_reason = failure_reason

        s.flush()
        logger.info(f"Updated Transaction status: payment_id='{tx.razorpay_payment_id}', new_status='{new_status}'")
        return tx


def save_retry_attempt(
    transaction_id: str,
    attempt_number: int,
    result: str,
    next_retry_at: Optional[datetime] = None,
    session: Optional[Session] = None
) -> RetryAttempt:
    """
    Records a retry attempt for a transaction in retry_attempts table.

    Args:
        transaction_id: Transaction primary key.
        attempt_number: Current retry index (1 or 2).
        result: Outcome ('SUCCESS', 'FAILED', 'TIMEOUT').
        next_retry_at: Optional datetime for subsequent scheduled attempt.
        session: Optional external SQLAlchemy Session.

    Returns:
        Created RetryAttempt instance.
    """
    with get_db_session(session) as s:
        # Validate tx exists
        tx = s.query(Transaction).filter_by(id=transaction_id).first()
        if not tx:
            tx_by_pay = s.query(Transaction).filter_by(razorpay_payment_id=transaction_id).first()
            if tx_by_pay:
                transaction_id = tx_by_pay.id

        retry = RetryAttempt(
            transaction_id=transaction_id,
            attempt_number=attempt_number,
            attempted_at=datetime.utcnow(),
            result=result,
            next_retry_at=next_retry_at
        )
        s.add(retry)
        s.flush()
        logger.info(f"Saved RetryAttempt: id={retry.id}, tx='{transaction_id}', attempt={attempt_number}, result='{result}'")
        return retry


def get_transaction_by_payment_id(
    razorpay_payment_id: str,
    session: Optional[Session] = None
) -> Optional[Transaction]:
    """Retrieves a single transaction by razorpay_payment_id or primary key id."""
    with get_db_session(session) as s:
        return (
            s.query(Transaction)
            .filter(
                (Transaction.razorpay_payment_id == razorpay_payment_id) |
                (Transaction.id == razorpay_payment_id)
            )
            .first()
        )


def get_all_transactions(
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None,
    session: Optional[Session] = None
) -> List[Transaction]:
    """Retrieves a paginated list of transactions ordered by created_at desc."""
    with get_db_session(session) as s:
        query = s.query(Transaction)
        if status_filter:
            query = query.filter(Transaction.status == status_filter)
        return query.order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()
