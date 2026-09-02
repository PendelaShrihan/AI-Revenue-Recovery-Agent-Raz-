# tests/test_retry_loop_e2e.py
"""End-to-End Sandbox Retry Loop Tests.

Simulates the complete real-world payment failure and recovery lifecycle:
1. Failed payment event ingestion & DB persistence
2. Retry scheduling with exponential backoff & delay calculation
3. Pending retries polling (run_pending_retries)
4. Success recovery outcome vs Failure exhaustion outcome with customer notification
"""

from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
import pytest

from agent.db_writer import init_db, save_transaction, get_db_session
from agent.models import Transaction, RetryAttempt, RecoveryAction
from agent.retry_scheduler import schedule_retry, get_retry_status, MAX_RETRIES
from agent.retry_executor import execute_retry
from agent.pipeline import run_recovery_pipeline, run_pending_retries
from api_integration.schemas import NormalizedEvent, FailureCategory, EventType


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """Temporary SQLite database for clean E2E loop isolation."""
    db_file = tmp_path / "test_e2e_loop.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    yield engine


@patch("agent.retry_executor._get_razorpay_client")
def test_retry_loop_e2e_success_recovery(mock_get_client):
    """E2E Loop: Failed Payment -> Scheduled Retry (15m) -> Time elapsed -> Retry Execution -> Recovered."""
    import uuid
    payment_id = f"pay_E2E_SUCC_{uuid.uuid4().hex[:8]}"

    # Mock real Razorpay API returning captured on retry attempt
    mock_client = MagicMock()
    mock_client.payment.fetch.return_value = {
        "id": payment_id,
        "status": "captured",
        "amount": 199900,
        "currency": "INR",
    }
    mock_get_client.return_value = mock_client

    # 1. Ingest Failed Payment
    event = NormalizedEvent(
        event_id=f"evt_{payment_id}",
        event_type=EventType.PAYMENT_FAILED.value,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=payment_id,
        merchant_id="merchant_e2e_01",
        amount=1999.0,
        currency="INR",
        status="FAILED",
        payment_id=payment_id,
        error_code="GATEWAY_ERROR",
        error_reason="network_timeout",
        error_description="Bank gateway timeout",
    )
    tx, is_created = save_transaction(event)
    assert is_created is True
    assert tx.status == "FAILED"

    # 2. Schedule Initial Retry (Attempt 1: 900s)
    r1 = schedule_retry(tx, retry_after_seconds=900)
    assert r1 is not None
    assert r1.attempt_number == 1
    assert r1.result == "SCHEDULED"

    status_info = get_retry_status(tx.id)
    assert status_info["attempts_made"] == 1
    assert status_info["attempts_remaining"] == 2
    assert status_info["status"] == "retry_scheduled"

    # 3. Simulate time progression (making next_retry_at due in the past)
    with get_db_session() as session:
        db_r1 = session.query(RetryAttempt).filter_by(id=r1.id).first()
        db_r1.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=10)

    # 4. Run Pending Retries Engine
    summary = run_pending_retries()
    assert summary["processed"] == 1
    assert summary["recovered"] == 1
    assert summary["failed"] == 0

    # 5. Verify Database Records
    with get_db_session() as session:
        final_tx = session.query(Transaction).filter_by(id=tx.id).first()
        assert final_tx.status == "recovered"
        
        final_attempt = session.query(RetryAttempt).filter_by(id=r1.id).first()
        assert final_attempt.result == "SUCCESS"


@patch("agent.retry_executor._get_razorpay_client")
def test_retry_loop_e2e_exhaustion_to_notification(mock_get_client):
    """E2E Loop: Failed Payment -> 3 Retries Fail -> State transitions to customer_notified."""
    import uuid
    payment_id = f"pay_E2E_EXH_{uuid.uuid4().hex[:8]}"

    # Mock Razorpay API continually returning failed status
    mock_client = MagicMock()
    mock_client.payment.fetch.return_value = {
        "id": payment_id,
        "status": "failed",
        "error_description": "Bank network unavailable",
    }
    mock_get_client.return_value = mock_client

    # 1. Ingest Failed Payment
    event = NormalizedEvent(
        event_id=f"evt_{payment_id}",
        event_type=EventType.PAYMENT_FAILED.value,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=payment_id,
        merchant_id="merchant_e2e_02",
        amount=3499.0,
        currency="INR",
        status="FAILED",
        payment_id=payment_id,
        error_code="GATEWAY_ERROR",
        error_reason="network_timeout",
    )
    tx, _ = save_transaction(event)

    # 2. Schedule & Execute Attempt 1 (1x delay)
    r1 = schedule_retry(tx, retry_after_seconds=100)
    res1 = execute_retry(r1)
    assert res1["status"] == "failed"

    # Verify attempt 2 was automatically scheduled by retry_executor
    status_2 = get_retry_status(tx.id)
    assert status_2["attempts_made"] == 2
    assert status_2["attempts_remaining"] == 1

    # 3. Execute Attempt 2 (2x delay)
    with get_db_session() as session:
        r2 = session.query(RetryAttempt).filter_by(transaction_id=tx.id, attempt_number=2).first()
    res2 = execute_retry(r2)
    assert res2["status"] == "failed"

    # Verify attempt 3 was automatically scheduled
    status_3 = get_retry_status(tx.id)
    assert status_3["attempts_made"] == 3
    assert status_3["attempts_remaining"] == 0

    # 4. Execute Attempt 3 (4x delay - final attempt)
    with get_db_session() as session:
        r3 = session.query(RetryAttempt).filter_by(transaction_id=tx.id, attempt_number=3).first()
    res3 = execute_retry(r3)
    assert res3["status"] == "failed"
    assert res3["next_retry_at"] is None  # No more retries scheduled

    # 5. Verify Final State & Customer Notification Dispatch
    with get_db_session() as session:
        final_tx = session.query(Transaction).filter_by(id=tx.id).first()
        assert final_tx.status == "customer_notified"

        notif_actions = session.query(RecoveryAction).filter_by(
            transaction_id=tx.id,
            action_type="customer_notified"
        ).all()
        assert len(notif_actions) >= 1
