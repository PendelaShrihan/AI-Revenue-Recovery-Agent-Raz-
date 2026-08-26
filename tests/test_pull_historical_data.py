"""
Integration Tests for Real Razorpay Historical Data Ingestion (scripts/pull_historical_data.py).

Tests real Razorpay API connectivity, credential validation,
PostgreSQL database writing, and strict error reporting without mocks or synthetic data.
"""

import os
import unittest
from dotenv import load_dotenv

load_dotenv()

from scripts.pull_historical_data import (
    fetch_from_razorpay_api,
    pull_and_ingest_historical_data
)
from agent.db_writer import get_all_transactions, init_db, get_database_url


class TestRealHistoricalDataPuller(unittest.TestCase):

    def setUp(self):
        self.db_url = get_database_url()
        init_db(self.db_url)

    def test_real_razorpay_api_fetch(self):
        """
        Verify live connection to Razorpay API (api.razorpay.com) using credentials from .env.
        Must succeed and return a list (empty if sandbox has no failed payments, or populated).
        """
        key_id = os.getenv("RAZORPAY_KEY_ID")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET")
        self.assertTrue(bool(key_id), "RAZORPAY_KEY_ID must be set in .env")
        self.assertTrue(bool(key_secret), "RAZORPAY_KEY_SECRET must be set in .env")

        # Fetch payments from live Razorpay API
        payments = fetch_from_razorpay_api(days=90, count=10)
        self.assertIsInstance(payments, list)

    def test_real_pull_and_ingest_pipeline(self):
        """
        Runs the real historical pull pipeline against PostgreSQL and api.razorpay.com.
        """
        summary = pull_and_ingest_historical_data(days=90, count=50)

        self.assertEqual(summary["days_window"], 90)
        self.assertEqual(summary["source"], "REAL_RAZORPAY_API (api.razorpay.com)")
        self.assertIn("total_fetched", summary)
        self.assertIn("total_saved", summary)
        self.assertIn("duplicates_skipped", summary)
        self.assertEqual(summary["total_fetched"], summary["total_saved"] + summary["duplicates_skipped"])

    def test_missing_credentials_raises_error(self):
        """
        Verifies that missing credentials immediately raise an explicit error
        rather than silently falling back to mock data.
        """
        original_key = os.environ.get("RAZORPAY_KEY_ID")
        try:
            os.environ["RAZORPAY_KEY_ID"] = ""
            with self.assertRaises(RuntimeError) as ctx:
                fetch_from_razorpay_api(days=90, count=10)
            self.assertIn("Missing Razorpay API credentials", str(ctx.exception))
        finally:
            if original_key:
                os.environ["RAZORPAY_KEY_ID"] = original_key


if __name__ == "__main__":
    unittest.main()
