"""
Tests for database schema creation (scripts/init_db.sql) and SQLAlchemy models (agent/models.py).
Supports both pytest and standard unittest runner.
"""

import os
import unittest
import sqlite3
from datetime import datetime

try:
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import sessionmaker
    from agent.models import Base, Transaction, RetryAttempt, RecoveryAction
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


class TestDatabaseSchema(unittest.TestCase):

    def test_init_db_sql(self):
        """Verify that scripts/init_db.sql executes cleanly and creates all 3 tables."""
        db_path = "data/test_sql_schema.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        with open("scripts/init_db.sql", "r", encoding="utf-8") as f:
            sql_script = f.read()

        conn = sqlite3.connect(db_path)
        conn.executescript(sql_script)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        self.assertIn("transactions", tables)
        self.assertIn("retry_attempts", tables)
        self.assertIn("recovery_actions", tables)

        if os.path.exists(db_path):
            os.remove(db_path)

    @unittest.skipUnless(HAS_SQLALCHEMY, "SQLAlchemy not installed in current python environment")
    def test_sqlalchemy_models(self):
        """Verify that agent/models.py generates tables and handles CRUD operations."""
        db_url = "sqlite:///data/test_orm_models.db"
        db_file = "data/test_orm_models.db"
        if os.path.exists(db_file):
            os.remove(db_file)

        engine = create_engine(db_url)
        Base.metadata.create_all(engine)

        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        self.assertIn("transactions", table_names)
        self.assertIn("retry_attempts", table_names)
        self.assertIn("recovery_actions", table_names)

        # Test inserting records
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        tx = Transaction(
            id="tx_test_001",
            razorpay_payment_id="pay_K12345678",
            merchant_id="mer_ABC123",
            amount=2499.00,
            currency="INR",
            status="FAILED",
            failure_reason="Customer bank OTP page froze during transaction",
            failure_code="GATEWAY_ERROR"
        )
        session.add(tx)
        session.commit()

        retry = RetryAttempt(
            transaction_id="tx_test_001",
            attempt_number=1,
            result="FAILED",
            next_retry_at=datetime.utcnow()
        )
        session.add(retry)

        action = RecoveryAction(
            transaction_id="tx_test_001",
            action_type="PAYMENT_LINK",
            action_payload='{"link": "https://rzp.io/i/test", "channel": "whatsapp"}',
            status="PENDING"
        )
        session.add(action)
        session.commit()

        # Query back
        fetched_tx = session.query(Transaction).filter_by(id="tx_test_001").first()
        self.assertIsNotNone(fetched_tx)
        self.assertEqual(fetched_tx.amount, 2499.00)
        self.assertEqual(len(fetched_tx.retry_attempts), 1)
        self.assertEqual(len(fetched_tx.recovery_actions), 1)
        self.assertEqual(fetched_tx.to_dict()["retry_count"], 1)

        session.close()
        engine.dispose()
        if os.path.exists(db_file):
            os.remove(db_file)


if __name__ == "__main__":
    unittest.main()
