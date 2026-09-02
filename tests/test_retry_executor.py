# tests/test_retry_executor.py
"""Unit tests for Retry Executor (agent/retry_executor.py) and pipeline pending retries."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
import pytest

from agent.db_writer import init_db, save_transaction, get_db_session
from agent.models import Transaction, RetryAttempt, RecoveryAction
from agent.retry_scheduler import schedule_retry
from agent.retry_executor import execute_retry
from agent.pipeline import run_pending_retries
from api_integration.schemas import NormalizedEvent, FailureCategory


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """Temporary SQLite database for isolated test execution."""
    db_file = tmp_path / "test_executor.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    yield engine


@pytest.fixture
def sample_transaction():
    """Create a sample failed transaction in DB with unique payment_id."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    event = NormalizedEvent(
        event_id=f"evt_test_exec_{uid}",
        event_type="payment.failed",
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=f"pay_EXEC_{uid}",
        merchant_id="merchant_001",
        amount=2499.0,
        currency="INR",
        status="FAILED",
        error_code="GATEWAY_ERROR",
        error_reason="gateway_issue",
        error_description="Bank gateway timeout",
    )
    tx, _ = save_transaction(event)
    return tx


@patch("agent.retry_executor._get_razorpay_client")
def test_execute_retry_success_flow(mock_get_client, sample_transaction):
    """When Razorpay reports payment captured/authorized, status updates to 'recovered'."""
    mock_client = MagicMock()
    mock_client.payment.fetch.return_value = {
        "id": sample_transaction.razorpay_payment_id,
        "status": "captured",
        "amount": 249900,
        "currency": "INR",
    }
    mock_get_client.return_value = mock_client

    # Schedule attempt 1
    retry_attempt = schedule_retry(sample_transaction, retry_after_seconds=300)
    assert retry_attempt.result == "SCHEDULED"

    # Execute retry
    res = execute_retry(retry_attempt)

    assert res["status"] == "recovered"
    assert res["payment_id"] == sample_transaction.razorpay_payment_id
    assert res["amount"] == 2499.0

    # Verify DB state
    with get_db_session() as session:
        tx = session.query(Transaction).filter_by(id=sample_transaction.id).first()
        assert tx.status == "recovered"

        att = session.query(RetryAttempt).filter_by(id=retry_attempt.id).first()
        assert att.result == "SUCCESS"


@patch("agent.retry_executor._get_razorpay_client")
def test_execute_retry_failure_cascading(mock_get_client, sample_transaction):
    """When retry fails and attempts remaining (<3), schedules next attempt."""
    mock_client = MagicMock()
    mock_client.payment.fetch.return_value = {
        "id": "pay_EXEC_001",
        "status": "failed",
        "error_description": "Card blocked",
    }
    mock_get_client.return_value = mock_client

    # Schedule attempt 1
    r1 = schedule_retry(sample_transaction, retry_after_seconds=300)
    
    # Execute attempt 1 -> fails -> should schedule attempt 2
    res = execute_retry(r1)

    assert res["status"] == "failed"
    assert "Card blocked" in res["reason"]
    assert res["next_retry_at"] is not None

    with get_db_session() as session:
        attempts = session.query(RetryAttempt).filter_by(transaction_id=sample_transaction.id).order_by(RetryAttempt.attempt_number.asc()).all()
        assert len(attempts) == 2
        assert attempts[0].result == "FAILED"
        assert attempts[1].result == "SCHEDULED"


@patch("agent.retry_executor._get_razorpay_client")
def test_execute_retry_exhaustion_notification(mock_get_client, sample_transaction):
    """When 3rd attempt fails, should escalate to customer notification."""
    mock_client = MagicMock()
    mock_client.payment.fetch.return_value = {
        "id": "pay_EXEC_001",
        "status": "failed",
        "error_description": "Do not honor",
    }
    mock_get_client.return_value = mock_client

    # Schedule 1st, 2nd, and 3rd attempts
    r1 = schedule_retry(sample_transaction, retry_after_seconds=100)
    with get_db_session() as s:
        s.query(RetryAttempt).filter_by(id=r1.id).first().result = "FAILED"
    
    r2 = schedule_retry(sample_transaction, retry_after_seconds=100)
    with get_db_session() as s:
        s.query(RetryAttempt).filter_by(id=r2.id).first().result = "FAILED"
        
    r3 = schedule_retry(sample_transaction, retry_after_seconds=100)

    # Execute 3rd attempt
    res = execute_retry(r3)

    assert res["status"] == "failed"
    assert res["next_retry_at"] is None

    with get_db_session() as session:
        attempts = session.query(RetryAttempt).filter_by(transaction_id=sample_transaction.id).all()
        assert len(attempts) == 3
        
        tx = session.query(Transaction).filter_by(id=sample_transaction.id).first()
        assert tx.status == "customer_notified"
        
        notif = session.query(RecoveryAction).filter_by(
            transaction_id=sample_transaction.id,
            action_type="customer_notified",
        ).first()
        assert notif is not None


@patch("agent.retry_executor._get_razorpay_client")
def test_run_pending_retries_processing(mock_get_client, sample_transaction):
    """run_pending_retries() processes all due SCHEDULED records."""
    mock_client = MagicMock()
    mock_client.payment.fetch.return_value = {
        "id": "pay_EXEC_001",
        "status": "captured",
        "amount": 249900,
        "currency": "INR",
    }
    mock_get_client.return_value = mock_client

    # Schedule a retry with next_retry_at in the past
    past_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    r1 = schedule_retry(sample_transaction, retry_after_seconds=10)
    with get_db_session() as session:
        db_att = session.query(RetryAttempt).filter_by(id=r1.id).first()
        db_att.next_retry_at = past_time

    summary = run_pending_retries()

    assert summary["processed"] == 1
    assert summary["recovered"] == 1
    assert summary["failed"] == 0
