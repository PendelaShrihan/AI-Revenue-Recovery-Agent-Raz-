# tests/test_notification_engine.py
"""Unit tests for Customer Notification & Communication Engine (agent/notification_engine.py)."""

from datetime import datetime, timezone
import pytest

from agent.db_writer import init_db, save_transaction, get_db_session
from agent.models import Transaction, RecoveryAction
from agent.notification_engine import (
    draft_whatsapp_message,
    draft_sms_message,
    draft_email_message,
    generate_personalized_notification,
    dispatch_customer_notification,
    FAILURE_GUIDANCE,
)
from api_integration.schemas import NormalizedEvent, FailureCategory


@pytest.fixture(autouse=True)
def sqlite_db(tmp_path):
    """Temporary SQLite DB for testing."""
    db_file = tmp_path / "test_notifications.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    yield engine


@pytest.fixture
def sample_transaction():
    """Create sample failed transaction in DB."""
    import uuid
    uid = uuid.uuid4().hex[:8]
    event = NormalizedEvent(
        event_id=f"evt_notif_{uid}",
        event_type="payment.failed",
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=f"pay_NOTIF_{uid}",
        merchant_id="Croma Electronics",
        amount=4999.0,
        currency="INR",
        status="FAILED",
        error_code="BAD_REQUEST_ERROR",
        error_reason="insufficient_funds",
        error_description="Customer account balance is low",
    )
    tx, _ = save_transaction(event)
    return tx


def test_draft_whatsapp_message():
    msg = draft_whatsapp_message(
        merchant_name="Acme Corp",
        amount_str="₹1,499.00",
        payment_link="https://rzp.io/i/testlink",
        failure_category="insufficient_funds",
        alternate_method="UPI / Google Pay",
        customer_name="Rahul",
    )
    assert "Hi Rahul," in msg
    assert "₹1,499.00" in msg
    assert "Acme Corp" in msg
    assert "https://rzp.io/i/testlink" in msg
    assert "UPI / Google Pay" in msg


def test_draft_sms_message():
    msg = draft_sms_message(
        merchant_name="Zomato",
        amount_str="₹540.00",
        payment_link="https://rzp.io/i/zmt",
        failure_category="card_blocked",
    )
    assert "₹540.00" in msg
    assert "Zomato" in msg
    assert "https://rzp.io/i/zmt" in msg
    assert "blocked" in msg


def test_draft_email_message():
    email = draft_email_message(
        merchant_name="SaaS Enterprise",
        amount_str="₹12,000.00",
        payment_link="https://rzp.io/i/sub123",
        failure_category="expired_card",
        customer_name="Priya",
    )
    assert "Action Required: Complete your payment of ₹12,000.00 to SaaS Enterprise" in email["subject"]
    assert "Dear Priya," in email["body"]
    assert "https://rzp.io/i/sub123" in email["body"]
    assert "expired" in email["body"]


def test_generate_personalized_notification(sample_transaction):
    notif = generate_personalized_notification(
        transaction=sample_transaction,
        failure_category="insufficient_funds",
        channel="whatsapp",
        merchant_name="Croma Electronics",
        payment_link="https://rzp.io/i/croma_pay",
        alternate_method="UPI",
        customer_name="Aditi",
        use_llm=False,
    )
    assert notif["transaction_id"] == sample_transaction.id
    assert notif["merchant_name"] == "Croma Electronics"
    assert notif["amount_formatted"] == "₹4,999.00"
    assert notif["payment_link"] == "https://rzp.io/i/croma_pay"
    assert notif["channels"]["whatsapp"] is not None
    assert notif["channels"]["sms"] is not None
    assert notif["channels"]["email"] is not None
    assert "₹4,999.00" in notif["channels"]["whatsapp"]


def test_dispatch_customer_notification(sample_transaction):
    action = dispatch_customer_notification(
        transaction=sample_transaction,
        failure_category="authentication_failed",
        channel="whatsapp",
        merchant_name="Flipkart",
        payment_link="https://rzp.io/i/fk123",
        alternate_method="UPI 1-Click",
    )
    assert isinstance(action, RecoveryAction)
    assert action.action_type == "customer_notified"
    assert action.status == "EXECUTED"

    with get_db_session() as session:
        tx = session.query(Transaction).filter_by(id=sample_transaction.id).first()
        assert tx.status == "customer_notified"
        
        saved_action = session.query(RecoveryAction).filter_by(id=action.id).first()
        assert saved_action is not None
