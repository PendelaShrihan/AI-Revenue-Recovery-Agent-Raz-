"""
Unit tests for Razorpay Error Code Mapper & Classifier (ml/error_codes.py).
Validates all 8 failure categories, 20+ error codes/reasons, edge cases, and helper functions.
"""

import unittest
from ml.error_codes import (
    classify_failure,
    is_transient_failure,
    get_category_description,
    FailureCategory,
    CATEGORIES,
    ERROR_CODE_MAP,
    ERROR_REASON_MAP,
)


class TestErrorCodeMapper(unittest.TestCase):

    def test_all_8_categories_exist(self):
        """Verifies that all 8 required failure categories exist in CATEGORIES."""
        expected_categories = {
            "insufficient_funds",
            "card_blocked",
            "network_timeout",
            "gateway_issue",
            "expired_card",
            "authentication_failed",
            "limit_exceeded",
            "unknown"
        }
        self.assertEqual(set(CATEGORIES), expected_categories)
        self.assertEqual(len(CATEGORIES), 8)

    def test_at_least_20_error_codes_and_reasons_covered(self):
        """Verifies mapping tables contain at least 20 distinct codes and reasons."""
        self.assertGreaterEqual(len(ERROR_CODE_MAP), 20)
        self.assertGreaterEqual(len(ERROR_REASON_MAP), 20)

    def test_classify_insufficient_funds(self):
        """Test variations of insufficient funds errors."""
        test_cases = [
            ("BAD_REQUEST_ERROR", "insufficient_funds"),
            ("INSUFFICIENT_FUNDS", None),
            ("BAD_REQUEST_INSUFFICIENT_FUNDS", "low_balance"),
            (None, "not_enough_balance"),
            ("", "balance_insufficient"),
            ("PAYMENT_INSUFFICIENT_BALANCE", "Customer short of balance"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.INSUFFICIENT_FUNDS.value)

    def test_classify_card_blocked(self):
        """Test variations of card blocked, closed, blacklisted, or declined."""
        test_cases = [
            ("CARD_INACTIVE_OR_CLOSED", None),
            ("CARD_BLOCKED", "card_blocked"),
            ("ACCOUNT_BLOCKED", "account_frozen"),
            ("CARD_BLACKLISTED", None),
            ("PAYMENT_DECLINED", "declined_by_bank"),
            ("DO_NOT_HONOUR", "do_not_honour"),
            (None, "stolen_card"),
            ("PICK_UP_CARD", "lost_card"),
            ("BAD_REQUEST_ERROR", "card_inactive"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.CARD_BLOCKED.value)

    def test_classify_network_timeout(self):
        """Test variations of network timeouts and connection drops."""
        test_cases = [
            ("TRANSACTION_TIMED_OUT", None),
            ("NETWORK_ERROR", "network_timeout"),
            ("TIMEDOUT", "connection_timeout"),
            ("GATEWAY_TIMED_OUT", None),
            ("REQUEST_TIMEOUT", "timed_out"),
            ("UPI_COLLECT_EXPIRED", "upi_collect_expired"),
            ("BAD_REQUEST_ERROR", "socket_timeout"),
            (None, "otp_timeout"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.NETWORK_TIMEOUT.value)

    def test_classify_gateway_issue(self):
        """Test variations of gateway downtime, bank switch errors, and 5xx failures."""
        test_cases = [
            ("GATEWAY_ERROR", "gateway_error"),
            ("SERVER_ERROR", None),
            ("INTERNAL_SERVER_ERROR", "server_error"),
            ("ISSUER_DOWN", "issuer_down"),
            ("ISSUING_BANK_DOWN", "bank_down"),
            ("PAYMENT_METHOD_TEMPORARILY_UNAVAILABLE", None),
            ("SWITCH_DOWN", "switch_down"),
            ("BAD_REQUEST_ERROR", "service_unavailable"),
            (None, "bank_error"),
            (None, "acquirer_down"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.GATEWAY_ISSUE.value)

    def test_classify_expired_card(self):
        """Test variations of expired card errors."""
        test_cases = [
            ("CARD_EXPIRED", None),
            ("EXPIRED_CARD", "expired_card"),
            ("CARD_VALIDITY_EXPIRED", "card_expired"),
            ("INVALID_EXPIRY_DATE", None),
            ("BAD_REQUEST_ERROR", "card_validity_expired"),
            (None, "invalid_expiry"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.EXPIRED_CARD.value)

    def test_classify_authentication_failed(self):
        """Test variations of 3DS, OTP, PIN, and credential failures."""
        test_cases = [
            ("AUTHENTICATION_FAILED", "payment_authentication_failed"),
            ("OTP_EXPIRED", None),
            ("OTP_INCORRECT", "incorrect_otp"),
            ("INVALID_PIN", "pin_incorrect"),
            ("3DS_AUTHENTICATION_FAILED", "3ds_failed"),
            ("MPIN_INCORRECT", None),
            ("BAD_REQUEST_ERROR", "invalid_otp"),
            (None, "invalid_cvv"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.AUTHENTICATION_FAILED.value)

    def test_classify_limit_exceeded(self):
        """Test variations of transaction limit, daily velocity, and credit limits."""
        test_cases = [
            ("LIMIT_EXCEEDED", "limit_exceeded"),
            ("EXCEEDED_MAX_AMOUNT_PER_TRANSACTION", "transaction_limit_exceeded"),
            ("EXCEEDED_DAILY_AMOUNT_LIMIT", "daily_limit_exceeded"),
            ("MAX_AMOUNT_LIMIT_EXCEEDED", None),
            ("VELOCITY_EXCEEDED", "velocity_limit_exceeded"),
            ("CARD_LIMIT_EXCEEDED", "credit_limit_exceeded"),
            ("BAD_REQUEST_ERROR", "maximum_amount_exceeded"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.LIMIT_EXCEEDED.value)

    def test_classify_unknown_fallback(self):
        """Test unmapped or empty inputs gracefully falling back to 'unknown'."""
        test_cases = [
            (None, None),
            ("", ""),
            ("SOME_RANDOM_CUSTOM_CODE_123", "unrecognized_internal_reason_xyz"),
            ("UNKNOWN_ERROR", "general_failure"),
        ]
        for code, reason in test_cases:
            with self.subTest(code=code, reason=reason):
                self.assertEqual(classify_failure(code, reason), FailureCategory.UNKNOWN.value)

    def test_classify_failure_graceful_none_and_edge_types(self):
        """
        Guarantee classify_failure() handles None, empty string, numeric,
        and missing inputs gracefully without raising any exceptions.
        """
        # Both None
        self.assertEqual(classify_failure(None, None), "unknown")
        # No arguments passed
        self.assertEqual(classify_failure(), "unknown")
        # None as first arg with valid reason
        self.assertEqual(classify_failure(None, "insufficient_funds"), "insufficient_funds")
        # Valid code with None reason
        self.assertEqual(classify_failure("GATEWAY_ERROR", None), "gateway_issue")
        # Non-string types (integers, floats, dicts) handled gracefully
        self.assertEqual(classify_failure(500, None), "unknown")
        self.assertEqual(classify_failure(None, 404), "unknown")
        self.assertEqual(classify_failure([], {}), "unknown")
        # Whitespace-only strings
        self.assertEqual(classify_failure("   ", "   "), "unknown")

    def test_transient_failure_check(self):
        """Verify is_transient_failure correctly distinguishes retryable vs non-retryable errors."""
        self.assertTrue(is_transient_failure("network_timeout"))
        self.assertTrue(is_transient_failure("gateway_issue"))
        self.assertFalse(is_transient_failure("insufficient_funds"))
        self.assertFalse(is_transient_failure("card_blocked"))
        self.assertFalse(is_transient_failure("expired_card"))
        self.assertFalse(is_transient_failure("authentication_failed"))
        self.assertFalse(is_transient_failure("limit_exceeded"))
        self.assertFalse(is_transient_failure("unknown"))

    def test_category_descriptions(self):
        """Verify human readable descriptions exist for all categories."""
        for cat in CATEGORIES:
            desc = get_category_description(cat)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 10)


if __name__ == "__main__":
    unittest.main()
