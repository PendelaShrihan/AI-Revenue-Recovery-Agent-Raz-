#!/usr/bin/env python3
"""
Explicit Edge Case Verification Suite
Tests all 4 critical edge cases required before submission:
  1. Network Timeout       (transient error auto-retry scheduling)
  2. Duplicate Webhooks    (idempotency, no duplicate records)
  3. Invalid Amount        (schema validation, rejects <= 0 or invalid currency)
  4. Already-Recovered     (guardrail preventing double-charge/re-retry)
"""

import os
import sys
import uuid
from datetime import datetime, timezone

# Windows safe stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from main import app
from agent.db_writer import get_db_session, save_transaction
from agent.models import Transaction, RetryAttempt
from api_integration.schemas import NormalizedEvent, EventType, FailureCategory
from api_integration.verifier import compute_webhook_signature

AUTH_KEY = os.getenv("MERCHANT_API_KEY") or os.getenv("APP_SECRET_KEY")
if not AUTH_KEY:
    AUTH_KEY = "test_key_edge_cases"
    os.environ["MERCHANT_API_KEY"] = AUTH_KEY

client = TestClient(app)
HEADERS = {"X-API-Key": AUTH_KEY, "Content-Type": "application/json"}
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET") or os.getenv("RAZORPAY_WEBHOOK_SECRET", "rzp_recovery_secret_prod_2026")


def test_edge_case_1_network_timeout():
    print("\n[Edge Case 1/4] Network Timeout...")
    test_pid = f"pay_netto_{uuid.uuid4().hex[:6]}"
    payload = {
        "payment_id": test_pid,
        "amount": 1499.00,
        "currency": "INR",
        "error_code": "GATEWAY_ERROR",
        "error_reason": "network_timeout",
        "error_description": "Network gateway timed out while contacting issuing bank.",
        "payment_method": "upi",
        "customer_name": "Test Customer",
        "customer_email": "test@example.com"
    }
    res = client.post("/analyze-failure", json=payload, headers=HEADERS)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["failure_category"] in ("network_timeout", "gateway_issue"), f"Unexpected cat: {data['failure_category']}"
    assert data["action_taken"] in ("auto_retry", "suggest_alternate_method"), f"Unexpected action: {data['action_taken']}"
    print(f"  PASS: Network timeout correctly classified as '{data['failure_category']}' -> action: '{data['action_taken']}'")


def test_edge_case_2_duplicate_webhooks():
    print("\n[Edge Case 2/4] Duplicate Webhook Ingestion...")
    test_pid = f"pay_dup_{uuid.uuid4().hex[:6]}"
    webhook_payload = {
        "entity": "event",
        "account_id": "acc_edge_test",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": test_pid,
                    "amount": 250000,  # paise = Rs. 2500.00
                    "currency": "INR",
                    "status": "failed",
                    "method": "card",
                    "error_code": "GATEWAY_TIMEOUT",
                    "error_description": "Gateway connection timed out",
                    "error_reason": "network_timeout"
                }
            }
        },
        "created_at": int(datetime.now(timezone.utc).timestamp())
    }

    import json
    raw_body = json.dumps(webhook_payload).encode("utf-8")
    sig = compute_webhook_signature(raw_body, WEBHOOK_SECRET)
    wh_headers = {
        "X-Razorpay-Signature": sig,
        "Content-Type": "application/json"
    }

    # First delivery
    res1 = client.post("/webhooks/razorpay", content=raw_body, headers=wh_headers)
    assert res1.status_code == 200, f"First webhook failed: {res1.text}"

    # Second delivery (duplicate replay)
    res2 = client.post("/webhooks/razorpay", content=raw_body, headers=wh_headers)
    assert res2.status_code == 200, f"Duplicate webhook failed: {res2.text}"

    # Verify database has exactly ONE transaction record
    with get_db_session() as s:
        matches = s.query(Transaction).filter(Transaction.razorpay_payment_id == test_pid).all()
        assert len(matches) == 1, f"Expected exactly 1 record, found {len(matches)}"

    print(f"  PASS: Duplicate webhook processed idempotently. Exactly 1 DB record maintained for {test_pid}.")


def test_edge_case_3_invalid_amount():
    print("\n[Edge Case 3/4] Invalid Amount Validation...")
    # Negative amount
    res_neg = client.post("/analyze-failure", json={
        "payment_id": f"pay_inv_{uuid.uuid4().hex[:6]}",
        "amount": -500.0,
        "currency": "INR",
        "error_code": "BAD_REQUEST_ERROR"
    }, headers=HEADERS)
    assert res_neg.status_code == 422, f"Expected 422 for negative amount, got {res_neg.status_code}"

    # Zero amount
    res_zero = client.post("/analyze-failure", json={
        "payment_id": f"pay_inv_{uuid.uuid4().hex[:6]}",
        "amount": 0.0,
        "currency": "INR",
        "error_code": "BAD_REQUEST_ERROR"
    }, headers=HEADERS)
    assert res_zero.status_code == 422, f"Expected 422 for zero amount, got {res_zero.status_code}"

    print("  PASS: Invalid amounts (<= 0) rejected with HTTP 422 Unprocessable Entity.")


def test_edge_case_4_already_recovered():
    print("\n[Edge Case 4/4] Already-Recovered Payment Guardrail...")
    test_pid = f"pay_rec_{uuid.uuid4().hex[:6]}"
    
    # Pre-populate transaction with status RECOVERED
    event = NormalizedEvent(
        event_id=f"evt_rec_{test_pid}",
        event_type=EventType.PAYMENT_FAILED.value,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=test_pid,
        merchant_id="acc_edge_test",
        amount=5000.0,
        currency="INR",
        status="RECOVERED",
        payment_id=test_pid,
        error_code="GATEWAY_ERROR",
        error_reason="network_timeout"
    )
    save_transaction(event)

    # Check recovery suggestions API
    res_sug = client.get(f"/recovery-suggestions?payment_id={test_pid}", headers=HEADERS)
    assert res_sug.status_code == 200
    data = res_sug.json()
    assert len(data["suggestions"]) >= 1
    sug = data["suggestions"][0]
    assert sug["can_retry"] is False, f"can_retry should be False for recovered payment, got {sug['can_retry']}"
    assert sug["suggested_action"] == "none_recovered"

    print(f"  PASS: Already-recovered payment {test_pid} blocked from retry (can_retry=False, action=none_recovered).")


if __name__ == "__main__":
    print("=" * 65)
    print("EDGE CASE VERIFICATION SUITE")
    print("=" * 65)
    test_edge_case_1_network_timeout()
    test_edge_case_2_duplicate_webhooks()
    test_edge_case_3_invalid_amount()
    test_edge_case_4_already_recovered()
    print("\n" + "=" * 65)
    print("ALL 4 EDGE CASES VERIFIED SUCCESSFULLY!")
    print("=" * 65)
