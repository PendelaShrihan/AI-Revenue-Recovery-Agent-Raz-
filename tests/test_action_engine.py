# tests/test_action_engine.py
"""Unit tests for Action Engine (agent/action_engine.py).

Tests:
    - execute_auto_retry()
    - execute_alternate_suggestion()
    - execute_customer_notification()
    - dispatch_recovery_action()
"""

import json
from datetime import datetime, timezone
import pytest

from agent.action_engine import (
    execute_auto_retry,
    execute_alternate_suggestion,
    execute_customer_notification,
    dispatch_recovery_action,
)
from agent.db_writer import init_db, save_transaction
from agent.models import Transaction, RetryAttempt, RecoveryAction
from api_integration.schemas import NormalizedEvent, FailureCategory


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """Use temporary SQLite DB for fast unit tests."""
    db_file = tmp_path / "test_action_engine.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    yield engine


@pytest.fixture
def sample_transaction(sqlite_db):
    """Create a test transaction in the DB."""
    event = NormalizedEvent(
        event_id="evt_test_ae_001",
        event_type="payment.failed",
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id="pay_AE_TEST_001",
        merchant_id="merchant_ae_01",
        amount=1499.0,
        currency="INR",
        status="FAILED",
        error_code="BAD_REQUEST_ERROR",
        error_reason="insufficient_funds",
        error_description="Insufficient balance",
    )
    tx, _ = save_transaction(event)
    return tx


def test_execute_auto_retry(sample_transaction):
    retry = execute_auto_retry(sample_transaction, retry_after_seconds=900)
    assert isinstance(retry, RetryAttempt)
    assert retry.attempt_number >= 1
    assert retry.result == "SCHEDULED"
    assert retry.next_retry_at is not None


def test_execute_alternate_suggestion(sample_transaction):
    action = execute_alternate_suggestion(sample_transaction, alternate_method="upi")
    assert isinstance(action, RecoveryAction)
    assert action.action_type == "alternate_method_suggested"
    assert action.status == "EXECUTED"
    payload = json.loads(action.action_payload)
    assert payload["alternate_method"] == "upi"


def test_execute_customer_notification(sample_transaction):
    action = execute_customer_notification(
        sample_transaction,
        message="Please check your account balance.",
        channel="email",
    )
    assert isinstance(action, RecoveryAction)
    assert action.action_type == "customer_notified"
    assert action.status == "EXECUTED"
    payload = json.loads(action.action_payload)
    assert payload["channel"] == "email"
    assert "balance" in payload["message"]


def test_dispatch_recovery_action_auto_retry(sample_transaction):
    decision = {
        "action": "auto_retry",
        "retry_after": 600,
        "alternate_method": "none",
        "message": "Retrying shortly.",
        "priority": "medium",
        "reasoning": "Transient issue.",
    }
    result = dispatch_recovery_action(decision, sample_transaction)
    assert result["action_taken"] == "auto_retry"
    assert result["retry_after"] == 600
    assert result["status"] == "dispatched"
    assert result["db_record_id"] is not None


def test_dispatch_recovery_action_suggest_alternate(sample_transaction):
    decision = {
        "action": "suggest_alternate_method",
        "retry_after": 0,
        "alternate_method": "netbanking",
        "message": "Use netbanking.",
        "priority": "high",
        "reasoning": "Card blocked.",
    }
    result = dispatch_recovery_action(decision, sample_transaction)
    assert result["action_taken"] == "suggest_alternate_method"
    assert result["alternate_method"] == "netbanking"
    assert result["status"] == "dispatched"
