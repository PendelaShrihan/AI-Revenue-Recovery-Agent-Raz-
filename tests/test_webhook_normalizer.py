"""
Unit and Integration Tests for Razorpay Webhook Ingestion & Payload Normalizer.
Tests:
1. Normalizer across all 3 event types (payment.failed, subscription.halted, invoice.overdue)
2. Edge cases (missing fields, zero amounts, malformed JSON, malformed envelope)
3. Signature verification (HMAC-SHA256, tampered body, missing headers)
4. FastAPI POST /webhooks/razorpay endpoint integration
"""

import json
import os
import unittest
from datetime import datetime

from fastapi.testclient import TestClient
from main import app

from api_integration.schemas import (
    EventType,
    FailureCategory,
    NormalizedEvent,
)
from api_integration.verifier import (
    verify_webhook_signature,
    compute_webhook_signature,
)
from api_integration.normalizer import (
    normalize_webhook_payload,
    _to_rupees,
    _parse_timestamp,
)


class TestWebhookNormalizer(unittest.TestCase):
    """Unit tests for payload normalization across all failure streams."""

    def test_payment_failed_normalization(self):
        """Verify normalization of a standard one-time checkout card/UPI failure."""
        raw_payload = {
            "entity": "event",
            "account_id": "acc_mer_12345",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_K987654321",
                        "entity": "payment",
                        "amount": 249900,  # 2499.00 INR in paise
                        "currency": "INR",
                        "status": "failed",
                        "order_id": "order_H87654321",
                        "invoice_id": None,
                        "international": False,
                        "method": "card",
                        "amount_refunded": 0,
                        "refund_status": None,
                        "captured": False,
                        "description": "Payment for order #87654",
                        "card_id": "card_XYZ987",
                        "bank": "HDFC",
                        "email": "customer@example.com",
                        "contact": "+919876543210",
                        "customer_id": "cust_C98765",
                        "notes": {
                            "customer_name": "Aditi Sharma",
                            "friction_note": "Customer bank OTP page froze during checkout",
                            "merchant_category": "electronics"
                        },
                        "error_code": "GATEWAY_ERROR",
                        "error_description": "Payment failed due to bank OTP verification timeout",
                        "error_source": "bank",
                        "error_step": "payment_authentication",
                        "error_reason": "network_timeout",
                        "created_at": 1756200000
                    }
                }
            },
            "created_at": 1756200000
        }

        normalized: NormalizedEvent = normalize_webhook_payload(raw_payload)

        # Assert core identification
        self.assertEqual(normalized.event_type, "payment.failed")
        self.assertEqual(normalized.failure_category, FailureCategory.CHECKOUT_FAILURE)
        self.assertEqual(normalized.entity_type, "payment")
        self.assertEqual(normalized.entity_id, "pay_K987654321")
        self.assertEqual(normalized.merchant_id, "acc_mer_12345")

        # Assert monetary normalization (paise -> INR)
        self.assertEqual(normalized.amount, 2499.00)
        self.assertEqual(normalized.currency, "INR")
        self.assertEqual(normalized.status, "FAILED")

        # Assert customer details
        self.assertEqual(normalized.customer_id, "cust_C98765")
        self.assertEqual(normalized.customer_name, "Aditi Sharma")
        self.assertEqual(normalized.customer_email, "customer@example.com")
        self.assertEqual(normalized.customer_phone, "+919876543210")

        # Assert error diagnostics
        self.assertEqual(normalized.error_code, "GATEWAY_ERROR")
        self.assertEqual(normalized.error_description, "Payment failed due to bank OTP verification timeout")
        self.assertEqual(normalized.error_source, "bank")
        self.assertEqual(normalized.error_step, "payment_authentication")
        self.assertEqual(normalized.error_reason, "network_timeout")

        # Assert contextual notes preserved
        self.assertIn("friction_note", normalized.notes)
        self.assertEqual(normalized.notes["friction_note"], "Customer bank OTP page froze during checkout")

        # Assert ORM model dictionary conversion
        tx_dict = normalized.to_transaction_dict()
        self.assertEqual(tx_dict["id"], "tx_pay_K987654321")
        self.assertEqual(tx_dict["razorpay_payment_id"], "pay_K987654321")
        self.assertEqual(tx_dict["amount"], 2499.00)
        self.assertEqual(tx_dict["status"], "FAILED")
        self.assertEqual(tx_dict["failure_code"], "GATEWAY_ERROR")

    def test_subscription_halted_normalization(self):
        """Verify normalization of a recurring billing / mandate halt event."""
        raw_payload = {
            "entity": "event",
            "account_id": "acc_mer_99887",
            "event": "subscription.halted",
            "contains": ["subscription", "payment"],
            "payload": {
                "subscription": {
                    "entity": {
                        "id": "sub_S12345678",
                        "entity": "subscription",
                        "plan_id": "plan_P998877",
                        "customer_id": "cust_C112233",
                        "status": "halted",
                        "current_start": 1756100000,
                        "current_end": 1758700000,
                        "auth_attempts": 3,
                        "total_count": 12,
                        "paid_count": 4,
                        "notes": {
                            "plan_name": "SaaS Pro Enterprise Monthly",
                            "department": "Engineering"
                        },
                        "short_url": "https://rzp.io/i/sub_reauth_123",
                        "payment_method": "emandate",
                        "charge_at": 1756200000
                    }
                },
                "payment": {
                    "entity": {
                        "id": "pay_sub_fail_9988",
                        "entity": "payment",
                        "amount": 1499900,  # 14,999.00 INR in paise
                        "currency": "INR",
                        "status": "failed",
                        "method": "emandate",
                        "email": "finance@clientcorp.com",
                        "contact": "+919811223344",
                        "customer_id": "cust_C112233",
                        "error_code": "GATEWAY_ERROR",
                        "error_description": "Recurring auto-debit mandate expired or rejected by issuing bank",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "mandate_inactive"
                    }
                }
            },
            "created_at": 1756200000
        }

        normalized: NormalizedEvent = normalize_webhook_payload(raw_payload)

        self.assertEqual(normalized.event_type, "subscription.halted")
        self.assertEqual(normalized.failure_category, FailureCategory.MANDATE_FAILURE)
        self.assertEqual(normalized.entity_type, "subscription")
        self.assertEqual(normalized.entity_id, "sub_S12345678")
        self.assertEqual(normalized.subscription_id, "sub_S12345678")
        self.assertEqual(normalized.payment_id, "pay_sub_fail_9988")
        self.assertEqual(normalized.amount, 14999.00)
        self.assertEqual(normalized.status, "HALTED")
        self.assertEqual(normalized.customer_email, "finance@clientcorp.com")
        self.assertEqual(normalized.customer_phone, "+919811223344")
        self.assertEqual(normalized.error_code, "GATEWAY_ERROR")
        self.assertEqual(normalized.error_reason, "mandate_inactive")
        self.assertEqual(normalized.notes.get("short_url"), "https://rzp.io/i/sub_reauth_123")
        self.assertEqual(normalized.notes.get("plan_name"), "SaaS Pro Enterprise Monthly")

    def test_subscription_halted_without_embedded_payment(self):
        """Verify subscription halt handles absence of embedded payment entity gracefully."""
        raw_payload = {
            "entity": "event",
            "account_id": "acc_mer_99887",
            "event": "subscription.halted",
            "contains": ["subscription"],
            "payload": {
                "subscription": {
                    "entity": {
                        "id": "sub_S99999",
                        "customer_id": "cust_C999",
                        "status": "halted",
                        "plan": {
                            "item": {
                                "amount": 49900  # 499.00 INR
                            }
                        },
                        "notes": {
                            "plan_name": "Basic Monthly"
                        }
                    }
                }
            }
        }
        normalized = normalize_webhook_payload(raw_payload)
        self.assertEqual(normalized.entity_id, "sub_S99999")
        self.assertEqual(normalized.amount, 499.00)
        self.assertEqual(normalized.failure_category, FailureCategory.MANDATE_FAILURE)
        self.assertEqual(normalized.error_code, "MANDATE_HALTED")

    def test_invoice_overdue_normalization(self):
        """Verify normalization of B2B commercial overdue invoice."""
        raw_payload = {
            "entity": "event",
            "account_id": "acc_mer_b2b_01",
            "event": "invoice.overdue",
            "contains": ["invoice"],
            "payload": {
                "invoice": {
                    "entity": {
                        "id": "inv_B2B_987654",
                        "entity": "invoice",
                        "customer_id": "cust_enterprise_01",
                        "customer_details": {
                            "name": "Tata Consultancy Subcontracting",
                            "email": "vendor.billing@tata.com",
                            "contact": "+919822334455"
                        },
                        "order_id": "order_inv_77665",
                        "status": "overdue",
                        "amount": 7500000,       # 75,000.00 INR in paise
                        "amount_paid": 0,
                        "amount_due": 7500000,   # 75,000.00 INR in paise
                        "currency": "INR",
                        "short_url": "https://rzp.io/i/inv_pay_987654",
                        "notes": {
                            "po_number": "PO-2026-AUG-9912",
                            "client_tier": "Enterprise A+",
                            "relationship_notes": "Awaiting VP Procurement sign-off on Net-30 invoice"
                        },
                        "created_at": 1755000000
                    }
                }
            },
            "created_at": 1756200000
        }

        normalized: NormalizedEvent = normalize_webhook_payload(raw_payload)

        self.assertEqual(normalized.event_type, "invoice.overdue")
        self.assertEqual(normalized.failure_category, FailureCategory.INVOICE_OVERDUE)
        self.assertEqual(normalized.entity_type, "invoice")
        self.assertEqual(normalized.entity_id, "inv_B2B_987654")
        self.assertEqual(normalized.invoice_id, "inv_B2B_987654")
        self.assertEqual(normalized.amount, 75000.00)
        self.assertEqual(normalized.currency, "INR")
        self.assertEqual(normalized.status, "OVERDUE")
        self.assertEqual(normalized.customer_name, "Tata Consultancy Subcontracting")
        self.assertEqual(normalized.customer_email, "vendor.billing@tata.com")
        self.assertEqual(normalized.customer_phone, "+919822334455")
        self.assertEqual(normalized.error_code, "INVOICE_OVERDUE")
        self.assertIn("Awaiting VP Procurement", normalized.notes.get("relationship_notes", ""))
        self.assertEqual(normalized.notes.get("invoice_url"), "https://rzp.io/i/inv_pay_987654")

    def test_invoice_expired_normalization(self):
        """Verify normalization of invoice.expired as an overdue receivable."""
        raw_payload = {
            "entity": "event",
            "account_id": "acc_mer_b2b_02",
            "event": "invoice.expired",
            "payload": {
                "invoice": {
                    "entity": {
                        "id": "inv_EXPIRED_001",
                        "amount_due": 3500000,
                        "status": "expired",
                        "customer_details": {
                            "name": "Infra Global",
                            "email": "ap@infraglobal.com"
                        }
                    }
                }
            }
        }
        normalized = normalize_webhook_payload(raw_payload)
        self.assertEqual(normalized.event_type, "invoice.expired")
        self.assertEqual(normalized.failure_category, FailureCategory.INVOICE_OVERDUE)
        self.assertEqual(normalized.amount, 35000.00)
        self.assertEqual(normalized.status, "OVERDUE")
        self.assertEqual(normalized.customer_name, "Infra Global")

    def test_informational_event_normalization(self):
        """Verify non-failure events like order.paid normalize as INFORMATIONAL."""
        raw_payload = {
            "entity": "event",
            "account_id": "acc_mer_123",
            "event": "order.paid",
            "payload": {
                "order": {
                    "entity": {
                        "id": "order_ORD12345",
                        "amount": 50000,
                        "currency": "INR",
                        "status": "paid"
                    }
                }
            }
        }
        normalized = normalize_webhook_payload(raw_payload)
        self.assertEqual(normalized.event_type, "order.paid")
        self.assertEqual(normalized.failure_category, FailureCategory.INFORMATIONAL)
        self.assertEqual(normalized.entity_id, "order_ORD12345")
        self.assertEqual(normalized.amount, 500.00)
        self.assertEqual(normalized.status, "PAID")

    def test_malformed_payload_raises_value_error(self):
        """Verify parser rejects corrupt or invalid envelopes with clear exceptions."""
        with self.assertRaises(ValueError):
            normalize_webhook_payload("not a json dict")

        with self.assertRaises(ValueError):
            normalize_webhook_payload({"missing_event": True})

        with self.assertRaises(ValueError):
            normalize_webhook_payload({"event": "payment.failed", "missing_payload_block": True})

        with self.assertRaises(ValueError):
            normalize_webhook_payload({
                "event": "payment.failed",
                "payload": {"payment": {}}  # missing entity
            })

    def test_utility_functions(self):
        """Verify helper math and timestamp parsing functions."""
        self.assertEqual(_to_rupees(100), 1.0)
        self.assertEqual(_to_rupees(249900), 2499.0)
        self.assertEqual(_to_rupees(0), 0.0)
        self.assertEqual(_to_rupees(None), 0.0)
        self.assertEqual(_to_rupees("invalid"), 0.0)

        # Timestamps
        now_dt = _parse_timestamp(1756200000)
        self.assertEqual(now_dt.year, 2025)  # 1756200000 epoch is in 2025


class TestSignatureVerification(unittest.TestCase):
    """Unit tests for Razorpay HMAC-SHA256 signature verification."""

    def setUp(self):
        self.secret = "test_webhook_secret_key_12345"
        self.payload = b'{"event":"payment.failed","payload":{"payment":{"entity":{"id":"pay_1"}}}}'

    def test_valid_signature_succeeds(self):
        sig = compute_webhook_signature(self.payload, self.secret)
        self.assertTrue(verify_webhook_signature(self.payload, sig, self.secret))

    def test_tampered_payload_fails(self):
        sig = compute_webhook_signature(self.payload, self.secret)
        tampered = b'{"event":"payment.failed","payload":{"payment":{"entity":{"id":"pay_TAMPERED"}}}}'
        self.assertFalse(verify_webhook_signature(tampered, sig, self.secret))

    def test_wrong_secret_fails(self):
        sig = compute_webhook_signature(self.payload, self.secret)
        self.assertFalse(verify_webhook_signature(self.payload, sig, "wrong_secret"))

    def test_empty_signature_fails(self):
        self.assertFalse(verify_webhook_signature(self.payload, "", self.secret))
        self.assertFalse(verify_webhook_signature(self.payload, None, self.secret))

    def test_empty_secret_fails(self):
        sig = compute_webhook_signature(self.payload, self.secret)
        self.assertFalse(verify_webhook_signature(self.payload, sig, ""))
        self.assertFalse(verify_webhook_signature(self.payload, sig, None))


class TestWebhookEndpointIntegration(unittest.TestCase):
    """Integration tests for POST /webhooks/razorpay FastAPI endpoint."""

    def setUp(self):
        self.client = TestClient(app)
        self.secret = "rzp_webhook_secret_for_tests"
        os.environ["RAZORPAY_WEBHOOK_SECRET"] = self.secret
        os.environ["SIMULATION_MODE"] = "false"

    def tearDown(self):
        if "RAZORPAY_WEBHOOK_SECRET" in os.environ:
            del os.environ["RAZORPAY_WEBHOOK_SECRET"]

    def test_post_payment_failed_with_valid_signature(self):
        body_dict = {
            "entity": "event",
            "account_id": "acc_001",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_TEST_FAILED_101",
                        "amount": 499900,
                        "currency": "INR",
                        "status": "failed",
                        "method": "card",
                        "error_code": "BAD_REQUEST_ERROR",
                        "error_description": "Card blocked by issuing bank"
                    }
                }
            }
        }
        body_bytes = json.dumps(body_dict).encode("utf-8")
        sig = compute_webhook_signature(body_bytes, self.secret)

        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["event_type"], "payment.failed")
        self.assertEqual(data["action_taken"], "payment_failure_routed")
        self.assertEqual(data["normalized_event"]["amount"], 4999.00)
        self.assertEqual(data["normalized_event"]["entity_id"], "pay_TEST_FAILED_101")

    def test_post_subscription_halted_with_valid_signature(self):
        body_dict = {
            "entity": "event",
            "account_id": "acc_002",
            "event": "subscription.halted",
            "payload": {
                "subscription": {
                    "entity": {
                        "id": "sub_TEST_HALTED_202",
                        "status": "halted",
                        "notes": {"plan_name": "Premium Tier"}
                    }
                },
                "payment": {
                    "entity": {
                        "id": "pay_sub_fail_202",
                        "amount": 299900,
                        "error_code": "GATEWAY_ERROR"
                    }
                }
            }
        }
        body_bytes = json.dumps(body_dict).encode("utf-8")
        sig = compute_webhook_signature(body_bytes, self.secret)

        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["event_type"], "subscription.halted")
        self.assertEqual(data["action_taken"], "subscription_halted_routed")
        self.assertEqual(data["normalized_event"]["subscription_id"], "sub_TEST_HALTED_202")

    def test_post_invoice_overdue_with_valid_signature(self):
        body_dict = {
            "entity": "event",
            "account_id": "acc_003",
            "event": "invoice.overdue",
            "payload": {
                "invoice": {
                    "entity": {
                        "id": "inv_TEST_OVERDUE_303",
                        "amount_due": 12000000,  # 120,000 INR
                        "status": "overdue",
                        "customer_details": {
                            "name": "Global Corp Ltd",
                            "email": "billing@globalcorp.com"
                        }
                    }
                }
            }
        }
        body_bytes = json.dumps(body_dict).encode("utf-8")
        sig = compute_webhook_signature(body_bytes, self.secret)

        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["event_type"], "invoice.overdue")
        self.assertEqual(data["action_taken"], "invoice_overdue_routed")
        self.assertEqual(data["normalized_event"]["amount"], 120000.00)

    def test_post_with_invalid_signature_rejected(self):
        body_bytes = b'{"event":"payment.failed"}'
        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": "invalid_hex_signature"
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid or missing X-Razorpay-Signature", response.json()["detail"])

    def test_post_with_missing_signature_rejected(self):
        body_bytes = b'{"event":"payment.failed"}'
        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 400)

    def test_post_with_malformed_json_rejected(self):
        body_bytes = b'invalid json body {'
        sig = compute_webhook_signature(body_bytes, self.secret)
        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig
            }
        )
    def test_post_in_simulation_mode_bypasses_signature(self):
        """Verify that developer SIMULATION_MODE allows testing without calculating HMAC header."""
        os.environ["SIMULATION_MODE"] = "true"
        body_dict = {
            "entity": "event",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_sim_001",
                        "amount": 19900,
                        "status": "failed",
                        "error_code": "GATEWAY_ERROR"
                    }
                }
            }
        }
        body_bytes = json.dumps(body_dict).encode("utf-8")
        response = self.client.post(
            "/webhooks/razorpay",
            content=body_bytes,
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["normalized_event"]["entity_id"], "pay_sim_001")


if __name__ == "__main__":
    unittest.main()
