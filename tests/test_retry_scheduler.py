# tests/test_retry_scheduler.py
"""Unit tests for Intelligent Retry Scheduler (agent/retry_scheduler.py).

Tests:
    - schedule_retry() attempt 1 delay (1x)
    - schedule_retry() attempt 2 delay (2x backoff)
    - schedule_retry() attempt 3 delay (4x backoff)
    - schedule_retry() attempt 4 blocked and customer notification called
    - get_retry_status() accurate returns
"""

from datetime import datetime, timezone
import pytest

from agent.db_writer import init_db, save_transaction, get_db_session
from agent.models import Transaction, RetryAttempt, RecoveryAction
from agent.retry_scheduler import schedule_retry, get_retry_status, MAX_RETRIES
from api_integration.schemas import NormalizedEvent, FailureCategory


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """Temporary SQLite database for isolated test execution."""
    db_file = tmp_path / "test_scheduler.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    yield engine


@pytest.fixture
def sample_transaction():
    """Create a sample failed transaction in DB with unique payment_id."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    event = NormalizedEvent(
        event_id=f"evt_test_sched_{uid}",
        event_type="payment.failed",
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=f"pay_SCHED_{uid}",
        merchant_id="merchant_001",
        amount=2499.0,
        currency="INR",
        status="FAILED",
        error_code="BAD_REQUEST_ERROR",
        error_reason="network_timeout",
        error_description="Bank network timed out",
    )
    tx, _ = save_transaction(event)
    return tx


def test_schedule_retry_attempt_1(sample_transaction):
    """Attempt 1: delay should be retry_after_seconds * 1 (as-is)."""
    base_delay = 300
    t_before = datetime.now(timezone.utc)
    
    retry = schedule_retry(sample_transaction, retry_after_seconds=base_delay)
    
    assert retry is not None
    assert retry.attempt_number == 1
    assert retry.result == "SCHEDULED"
    assert retry.next_retry_at is not None
    
    # Check calculated delay is approx base_delay (300s)
    delay = (retry.next_retry_at.replace(tzinfo=timezone.utc) - t_before).total_seconds()
    assert 295 <= delay <= 305


def test_schedule_retry_exponential_backoff_attempts_1_to_3(sample_transaction):
    """Attempts 1, 2, 3 should scale delays by 1x, 2x, 4x."""
    base_delay = 100
    
    # Attempt 1 -> 100s
    r1 = schedule_retry(sample_transaction, retry_after_seconds=base_delay)
    assert r1.attempt_number == 1
    
    # Attempt 2 -> 200s (2x)
    t2 = datetime.now(timezone.utc)
    r2 = schedule_retry(sample_transaction, retry_after_seconds=base_delay)
    assert r2.attempt_number == 2
    delay2 = (r2.next_retry_at.replace(tzinfo=timezone.utc) - t2).total_seconds()
    assert 195 <= delay2 <= 205
    
    # Attempt 3 -> 400s (4x)
    t3 = datetime.now(timezone.utc)
    r3 = schedule_retry(sample_transaction, retry_after_seconds=base_delay)
    assert r3.attempt_number == 3
    delay3 = (r3.next_retry_at.replace(tzinfo=timezone.utc) - t3).total_seconds()
    assert 395 <= delay3 <= 405


def test_schedule_retry_max_retries_enforced(sample_transaction):
    """4th attempt must be blocked and customer notification triggered."""
    base_delay = 60
    
    # Exhaust 3 attempts
    for _ in range(3):
        schedule_retry(sample_transaction, retry_after_seconds=base_delay)
        
    with get_db_session() as session:
        count = session.query(RetryAttempt).filter_by(transaction_id=sample_transaction.id).count()
        assert count == 3
        
    # 4th attempt call
    r4 = schedule_retry(sample_transaction, retry_after_seconds=base_delay)
    assert r4 is None  # Blocked
    
    with get_db_session() as session:
        count = session.query(RetryAttempt).filter_by(transaction_id=sample_transaction.id).count()
        assert count == 3  # Still 3, not 4
        
        # Verify customer notification was created
        notif = session.query(RecoveryAction).filter_by(
            transaction_id=sample_transaction.id,
            action_type="customer_notified"
        ).first()
        assert notif is not None
        assert notif.status == "EXECUTED"


def test_get_retry_status(sample_transaction):
    """Test get_retry_status schema and remaining attempts calculations."""
    status_initial = get_retry_status(sample_transaction.id)
    assert status_initial["transaction_id"] == sample_transaction.id
    assert status_initial["attempts_made"] == 0
    assert status_initial["attempts_remaining"] == 3
    assert status_initial["next_retry_at"] is None
    
    # Schedule attempt 1
    schedule_retry(sample_transaction, retry_after_seconds=300)
    status_1 = get_retry_status(sample_transaction.id)
    assert status_1["attempts_made"] == 1
    assert status_1["attempts_remaining"] == 2
    assert status_1["next_retry_at"] is not None
    assert status_1["status"] == "retry_scheduled"
    
    # Schedule attempt 2
    schedule_retry(sample_transaction, retry_after_seconds=300)
    status_2 = get_retry_status(sample_transaction.id)
    assert status_2["attempts_made"] == 2
    assert status_2["attempts_remaining"] == 1
