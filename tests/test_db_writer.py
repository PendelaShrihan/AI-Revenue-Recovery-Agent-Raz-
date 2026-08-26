"""
Unit & Integration Tests for Database Write Layer (agent/db_writer.py).
Tests transaction persistence, deduplication on razorpay_payment_id, recovery actions,
status updates, and commit/rollback handling.
"""

import os
import unittest
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.models import Base, Transaction, RecoveryAction, RetryAttempt
from agent.db_writer import (
    save_transaction,
    save_recovery_action,
    update_transaction_status,
    save_retry_attempt,
    get_transaction_by_payment_id,
    get_all_transactions,
    get_db_session,
    init_db
)
from api_integration.schemas import NormalizedEvent, FailureCategory


class TestDBWriter(unittest.TestCase):

    def setUp(self):
        self.test_db_url = "sqlite:///:memory:"
        self.engine = create_engine(self.test_db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.session = self.SessionLocal()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def _create_sample_event(self, payment_id="pay_test_123", amount=1499.00, error_code="INSUFFICIENT_FUNDS"):
        return NormalizedEvent(
            event_id="evt_test_001",
            event_type="payment.failed",
            failure_category=FailureCategory.CHECKOUT_FAILURE,
            entity_type="payment",
            entity_id=payment_id,
            merchant_id="acc_test_merchant",
            amount=amount,
            currency="INR",
            status="FAILED",
            payment_id=payment_id,
            error_code=error_code,
            error_reason="insufficient_funds",
            error_description="Customer account balance is low",
            created_at=datetime.now(timezone.utc)
        )

    def test_save_transaction_new_record(self):
        """Test persisting a brand new transaction."""
        event = self._create_sample_event(payment_id="pay_new_001", amount=2999.00)
        tx, is_created = save_transaction(event, session=self.session)
        self.session.commit()

        self.assertTrue(is_created)
        self.assertIsNotNone(tx)
        self.assertEqual(tx.razorpay_payment_id, "pay_new_001")
        self.assertEqual(tx.amount, 2999.00)
        self.assertEqual(tx.status, "FAILED")
        self.assertEqual(tx.failure_code, "INSUFFICIENT_FUNDS")

        # Verify querying back
        fetched = self.session.query(Transaction).filter_by(razorpay_payment_id="pay_new_001").first()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, tx.id)

    def test_save_transaction_deduplication(self):
        """Test that saving an event with the same razorpay_payment_id deduplicates cleanly."""
        event1 = self._create_sample_event(payment_id="pay_dup_001", amount=1500.00)
        tx1, is_created1 = save_transaction(event1, session=self.session)
        self.session.commit()
        self.assertTrue(is_created1)

        # Attempt to save the exact same payment_id again (simulate webhook redelivery or duplicate batch)
        event2 = self._create_sample_event(payment_id="pay_dup_001", amount=1500.00)
        tx2, is_created2 = save_transaction(event2, session=self.session)
        self.session.commit()

        self.assertFalse(is_created2)
        self.assertEqual(tx1.id, tx2.id)
        self.assertEqual(tx1.razorpay_payment_id, tx2.razorpay_payment_id)

        # Ensure only 1 record exists in DB
        count = self.session.query(Transaction).filter_by(razorpay_payment_id="pay_dup_001").count()
        self.assertEqual(count, 1)

    def test_save_recovery_action(self):
        """Test creating and persisting recovery actions linked to a transaction."""
        event = self._create_sample_event(payment_id="pay_rec_001")
        tx, _ = save_transaction(event, session=self.session)
        self.session.commit()

        payload = {
            "action": "PAYMENT_LINK",
            "channel": "whatsapp",
            "short_url": "https://rzp.io/i/testlink",
            "template": "hinglish_recovery_v1"
        }
        action = save_recovery_action(
            transaction_id=tx.id,
            action_type="PAYMENT_LINK",
            action_payload=payload,
            status="PENDING",
            session=self.session
        )
        self.session.commit()

        self.assertIsNotNone(action.id)
        self.assertEqual(action.transaction_id, tx.id)
        self.assertEqual(action.action_type, "PAYMENT_LINK")
        self.assertEqual(action.status, "PENDING")

        parsed_payload = json.loads(action.action_payload)
        self.assertEqual(parsed_payload["channel"], "whatsapp")
        self.assertEqual(parsed_payload["short_url"], "https://rzp.io/i/testlink")

        # Test relationship
        reloaded_tx = self.session.query(Transaction).filter_by(id=tx.id).first()
        self.assertEqual(len(reloaded_tx.recovery_actions), 1)
        self.assertEqual(reloaded_tx.recovery_actions[0].id, action.id)

    def test_update_transaction_status(self):
        """Test updating status and failure reason of a transaction."""
        event = self._create_sample_event(payment_id="pay_status_001")
        tx, _ = save_transaction(event, session=self.session)
        self.session.commit()

        updated_tx = update_transaction_status(
            razorpay_payment_id="pay_status_001",
            new_status="RECOVERED",
            failure_reason="Customer re-attempted via UPI link successfully",
            session=self.session
        )
        self.session.commit()

        self.assertIsNotNone(updated_tx)
        self.assertEqual(updated_tx.status, "RECOVERED")
        self.assertEqual(updated_tx.failure_reason, "Customer re-attempted via UPI link successfully")

        # Test querying non-existent payment ID
        none_tx = update_transaction_status("pay_nonexistent_999", "RECOVERED", session=self.session)
        self.assertIsNone(none_tx)

    def test_save_retry_attempt(self):
        """Test persisting retry attempts with next_retry_at."""
        event = self._create_sample_event(payment_id="pay_retry_001")
        tx, _ = save_transaction(event, session=self.session)
        self.session.commit()

        retry = save_retry_attempt(
            transaction_id=tx.id,
            attempt_number=1,
            result="FAILED",
            next_retry_at=datetime.utcnow(),
            session=self.session
        )
        self.session.commit()

        self.assertIsNotNone(retry.id)
        self.assertEqual(retry.transaction_id, tx.id)
        self.assertEqual(retry.attempt_number, 1)
        self.assertEqual(retry.result, "FAILED")

    def test_get_all_transactions_pagination(self):
        """Test pagination and filtering in get_all_transactions."""
        for i in range(5):
            ev = self._create_sample_event(payment_id=f"pay_page_{i}", amount=100.0 * (i + 1))
            save_transaction(ev, session=self.session)
        self.session.commit()

        txs = get_all_transactions(limit=3, offset=0, session=self.session)
        self.assertEqual(len(txs), 3)

        txs_offset = get_all_transactions(limit=3, offset=3, session=self.session)
        self.assertEqual(len(txs_offset), 2)

    def test_rollback_on_error(self):
        """Test that get_db_session rolls back on unhandled exceptions."""
        try:
            with get_db_session(self.session) as s:
                tx = Transaction(
                    id="tx_rollback_01",
                    razorpay_payment_id="pay_rollback_01",
                    merchant_id="acc_test",
                    amount=500.0,
                    status="FAILED"
                )
                s.add(tx)
                s.flush()
                # Deliberately raise exception
                raise ValueError("Simulated unexpected processing error")
        except ValueError:
            pass

        # Verify record was rolled back
        found = self.session.query(Transaction).filter_by(id="tx_rollback_01").first()
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
