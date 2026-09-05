"""
API Integration Tests — covers all REST endpoints with realistic Razorpay-style payloads.

Scenarios tested:
  1. POST /analyze-failure  — 5 failure types end-to-end (mocked pipeline)
  2. GET  /recovery-suggestions — list, filter, lookup
  3. POST /trigger-retry   — success, guardrail, force override, 404
  4. GET  /stats           — counter accuracy after inserting known records
  5. GET  /stream          — SSE connect + disconnect (no hang)
  6. POST /webhooks/razorpay — signed webhook ingestion for 3 event types
"""

import hashlib
import hmac
import json
import uuid
import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from main import app
from agent.db_writer import save_transaction
from api_integration.schemas import (
    NormalizedEvent,
    FailureCategory,
    EventType,
)

client = TestClient(app)

# ── Shared fixtures ──────────────────────────────────────────────────────────

AUTH_KEY = "int_test_merchant_key_abc"
HEADERS  = {"X-API-Key": AUTH_KEY, "Content-Type": "application/json"}
WEBHOOK_SECRET = "rzp_recovery_new_secret_2026"


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Inject test credentials into every test."""
    monkeypatch.setenv("MERCHANT_API_KEY", AUTH_KEY)
    monkeypatch.setenv("APP_SECRET_KEY", "alt_test_secret_xyz")
    monkeypatch.setenv("SIMULATION_MODE", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", WEBHOOK_SECRET)


def _uid() -> str:
    """Short unique suffix to avoid DB collisions."""
    return uuid.uuid4().hex[:8]


def _signed_body(payload: dict) -> tuple[bytes, str]:
    """Compute HMAC-SHA256 signature exactly as Razorpay does."""
    body = json.dumps(payload, separators=(",", ":")).encode()
    sig  = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return body, sig


def _insert_tx(payment_id: str, error_code: str = "GATEWAY_ERROR",
                error_reason: str = "network_timeout", amount: float = 999.0) -> str:
    """Helper: persist a NormalizedEvent and return its payment_id."""
    event = NormalizedEvent(
        event_id=f"evt_{payment_id}",
        event_type=EventType.PAYMENT_FAILED.value,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=payment_id,
        merchant_id="acc_int_test",
        amount=amount,
        currency="INR",
        status="FAILED",
        payment_id=payment_id,
        error_code=error_code,
        error_reason=error_reason,
    )
    save_transaction(event)
    return payment_id


# ─────────────────────────────────────────────────────────────────────────────
# 1. POST /analyze-failure — 5 failure scenario types
# ─────────────────────────────────────────────────────────────────────────────

FAILURE_SCENARIOS = [
    {
        "name": "insufficient_funds",
        "payload": {
            "payment_id": "pay_INSUF_{uid}",
            "amount": 4999.00,
            "currency": "INR",
            "error_code": "BAD_REQUEST_ERROR",
            "error_reason": "insufficient_funds",
            "error_description": "Your account has insufficient funds to complete this payment.",
            "payment_method": "card",
            "customer_name": "Priya Sharma",
            "customer_email": "priya@example.com",
        },
        "expected_category": "insufficient_funds",
    },
    {
        "name": "network_timeout",
        "payload": {
            "payment_id": "pay_NETTO_{uid}",
            "amount": 1299.00,
            "currency": "INR",
            "error_code": "GATEWAY_ERROR",
            "error_reason": "network_timeout",
            "error_description": "Payment gateway timed out. Please retry.",
            "payment_method": "upi",
            "customer_name": "Rahul Verma",
            "customer_email": "rahul@example.com",
        },
        "expected_category": "network_timeout",
    },
    {
        "name": "card_expired",
        "payload": {
            "payment_id": "pay_EXPRD_{uid}",
            "amount": 799.00,
            "currency": "INR",
            "error_code": "BAD_REQUEST_ERROR",
            "error_reason": "card_expired",
            "error_description": "The card has expired. Please use a different card.",
            "payment_method": "card",
            "customer_name": "Anjali Singh",
            "customer_email": "anjali@example.com",
        },
        "expected_category": "card_expired",
    },
    {
        "name": "authentication_failed",
        "payload": {
            "payment_id": "pay_AUTHF_{uid}",
            "amount": 2499.00,
            "currency": "INR",
            "error_code": "BAD_REQUEST_ERROR",
            "error_reason": "payment_failed",
            "error_description": "OTP authentication failed. Please try again.",
            "payment_method": "netbanking",
            "customer_name": "Suresh Kumar",
            "customer_email": "suresh@example.com",
        },
        "expected_category": None,  # may vary by classifier
    },
    {
        "name": "gateway_issue",
        "payload": {
            "payment_id": "pay_GATEW_{uid}",
            "amount": 9999.00,
            "currency": "INR",
            "error_code": "SERVER_ERROR",
            "error_reason": "gateway_error",
            "error_description": "Internal gateway error. Please retry after some time.",
            "payment_method": "card",
            "customer_name": "Meera Pillai",
            "customer_email": "meera@example.com",
        },
        "expected_category": "gateway_issue",
    },
]


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS, ids=[s["name"] for s in FAILURE_SCENARIOS])
@patch("api_integration.rest_router.run_recovery_pipeline")
@patch("api_integration.rest_router.broadcast", new_callable=AsyncMock)
def test_analyze_failure_scenarios(mock_broadcast, mock_pipeline, scenario):
    """Each of the 5 failure scenarios produces a valid, structured response."""
    uid = _uid()
    payload = {
        k: v.replace("{uid}", uid) if isinstance(v, str) and "{uid}" in v else v
        for k, v in scenario["payload"].items()
    }

    mock_pipeline.return_value = {
        "transaction_id": f"tx_{payload['payment_id']}",
        "payment_id": payload["payment_id"],
        "failure_category": scenario["expected_category"] or "unknown",
        "action_taken": "auto_retry",
        "priority": "high",
        "confidence": 0.91,
        "retry_after": 300,
        "alternate_method": None,
        "message": "Recovery action dispatched.",
        "reasoning": "Network-type failure — retry is the optimal path.",
        "db_record_id": 42,
        "elapsed_ms": 88.5,
    }

    response = client.post("/analyze-failure", json=payload, headers=HEADERS)

    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "success"
    assert data["payment_id"] == payload["payment_id"]
    assert data["action_taken"] in (
        "auto_retry", "suggest_alternate_method", "send_payment_link",
        "send_customer_notification", "no_action"
    )
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["priority"] in ("low", "medium", "high", "critical")
    mock_pipeline.assert_called_once()


@patch("api_integration.rest_router.run_recovery_pipeline")
@patch("api_integration.rest_router.broadcast", new_callable=AsyncMock)
def test_analyze_failure_missing_required_fields(mock_broadcast, mock_pipeline):
    """Omitting payment_id or amount returns HTTP 422 (validation error)."""
    response = client.post(
        "/analyze-failure",
        json={"currency": "INR", "error_code": "GATEWAY_ERROR"},
        headers=HEADERS,
    )
    assert response.status_code == 422


@patch("api_integration.rest_router.run_recovery_pipeline")
@patch("api_integration.rest_router.broadcast", new_callable=AsyncMock)
def test_analyze_failure_pipeline_error_returns_500(mock_broadcast, mock_pipeline):
    """Pipeline exceptions surface as HTTP 500 with a meaningful message."""
    mock_pipeline.side_effect = RuntimeError("Gemini API unreachable")
    uid = _uid()
    response = client.post(
        "/analyze-failure",
        json={"payment_id": f"pay_ERR_{uid}", "amount": 100.0},
        headers=HEADERS,
    )
    assert response.status_code == 500
    assert "Recovery pipeline failed" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /recovery-suggestions
# ─────────────────────────────────────────────────────────────────────────────

def test_recovery_suggestions_list():
    """Returns a list of suggestions with expected shape."""
    _insert_tx(f"pay_SUGLST_{_uid()}")
    response = client.get("/recovery-suggestions?limit=10", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["suggestions"], list)
    assert "count" in data


def test_recovery_suggestions_lookup_by_payment_id():
    """Lookup by exact payment_id returns the matching suggestion."""
    pid = f"pay_LOOKUP_{_uid()}"
    _insert_tx(pid, amount=3499.0)
    response = client.get(f"/recovery-suggestions?payment_id={pid}", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    item = next(s for s in data["suggestions"] if s["payment_id"] == pid)
    assert item["amount"] == 3499.0
    assert item["can_retry"] is True


def test_recovery_suggestions_auto_retry_for_network_timeout():
    """Network timeout → suggested_action should be auto_retry."""
    pid = f"pay_NETTO2_{_uid()}"
    _insert_tx(pid, error_code="GATEWAY_ERROR", error_reason="network_timeout")
    response = client.get(f"/recovery-suggestions?payment_id={pid}", headers=HEADERS)
    assert response.status_code == 200
    item = next(s for s in response.json()["suggestions"] if s["payment_id"] == pid)
    assert item["suggested_action"] == "auto_retry"


def test_recovery_suggestions_pagination():
    """Offset and limit are respected."""
    for i in range(3):
        _insert_tx(f"pay_PAG_{_uid()}")
    r1 = client.get("/recovery-suggestions?limit=2&offset=0", headers=HEADERS)
    r2 = client.get("/recovery-suggestions?limit=2&offset=2", headers=HEADERS)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert len(r1.json()["suggestions"]) <= 2


def test_recovery_suggestions_auth_rejected():
    """Requests without valid auth get HTTP 401."""
    response = client.get("/recovery-suggestions")
    assert response.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST /trigger-retry
# ─────────────────────────────────────────────────────────────────────────────

def test_trigger_retry_full_flow():
    """Three successful retries -> guardrail blocks 4th -> force=True overrides."""
    pid = f"pay_RETFL_{_uid()}"
    _insert_tx(pid)

    # Attempt 1 — scheduled
    r1 = client.post("/trigger-retry",
                     json={"payment_id": pid, "delay_seconds": 300},
                     headers=HEADERS)
    assert r1.status_code == 200
    assert r1.json()["attempt_number"] == 1
    assert r1.json()["result"] == "SCHEDULED"

    # Attempt 2 — immediate
    r2 = client.post("/trigger-retry",
                     json={"payment_id": pid, "delay_seconds": 0},
                     headers=HEADERS)
    assert r2.status_code == 200
    assert r2.json()["attempt_number"] == 2
    assert r2.json()["result"] == "TRIGGERED"

    # Attempt 3 — immediate
    r3 = client.post("/trigger-retry",
                     json={"payment_id": pid, "delay_seconds": 0},
                     headers=HEADERS)
    assert r3.status_code == 200
    assert r3.json()["attempt_number"] == 3
    assert r3.json()["result"] == "TRIGGERED"

    # Attempt 4 — blocked by max 3 attempts guardrail
    r4 = client.post("/trigger-retry",
                     json={"payment_id": pid},
                     headers=HEADERS)
    assert r4.status_code == 400
    assert "Maximum retry limit" in r4.json()["detail"]

    # Attempt 4 force — allowed
    r5 = client.post("/trigger-retry",
                     json={"payment_id": pid, "force": True},
                     headers=HEADERS)
    assert r5.status_code == 200
    assert r5.json()["attempt_number"] == 4


def test_trigger_retry_not_found():
    """Non-existent payment_id returns HTTP 404."""
    response = client.post(
        "/trigger-retry",
        json={"payment_id": "pay_GHOST_DOES_NOT_EXIST_9999"},
        headers=HEADERS,
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_trigger_retry_missing_id():
    """Omitting both payment_id and transaction_id returns HTTP 400."""
    response = client.post("/trigger-retry", json={}, headers=HEADERS)
    assert response.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /stats
# ─────────────────────────────────────────────────────────────────────────────

def test_stats_returns_expected_shape():
    """Stats endpoint returns all required keys with correct types."""
    response = client.get("/stats", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    for key in ("total", "recovered", "retry_scheduled", "failed", "success_rate"):
        assert key in data, f"Missing key: {key}"
    assert isinstance(data["success_rate"], float)
    assert 0.0 <= data["success_rate"] <= 100.0


def test_stats_reflects_new_transaction():
    """Inserting a transaction increases total by at least 1."""
    r_before = client.get("/stats", headers=HEADERS)
    total_before = r_before.json()["total"]

    _insert_tx(f"pay_STATS_{_uid()}")

    r_after = client.get("/stats", headers=HEADERS)
    assert r_after.json()["total"] >= total_before + 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /stream (SSE)
# ─────────────────────────────────────────────────────────────────────────────

def test_stream_requires_auth():
    """SSE endpoint without credentials returns 401."""
    response = client.get("/stream")
    assert response.status_code == 401


def test_stream_blocked_in_production(monkeypatch):
    """In ENVIRONMENT=production SSE without key returns 401."""
    monkeypatch.setenv("SIMULATION_MODE", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    response = client.get("/stream")
    assert response.status_code == 401


def test_broadcaster_subscribe_broadcast_unsubscribe():
    """Unit test for SSE in-memory pub/sub broadcaster."""
    import asyncio
    from agent.broadcaster import subscribe, unsubscribe, broadcast, _subscribers

    initial_count = len(_subscribers)
    q = subscribe()
    assert len(_subscribers) == initial_count + 1

    test_ev = {"type": "test_event", "data": "123"}
    asyncio.run(broadcast(test_ev))
    assert not q.empty()
    received = q.get_nowait()
    assert received == test_ev

    unsubscribe(q)
    assert len(_subscribers) == initial_count


# ─────────────────────────────────────────────────────────────────────────────
# 6. POST /webhooks/razorpay — signed payloads
# ─────────────────────────────────────────────────────────────────────────────

def _webhook_payload(event_type: str, payment_id: str) -> dict:
    return {
        "event": event_type,
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "amount": 150000,          # paise → ₹1500
                    "currency": "INR",
                    "status": "failed",
                    "method": "card",
                    "order_id": f"order_{payment_id}",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Your card's issuing bank is currently down.",
                    "error_source": "bank",
                    "error_step": "payment_authentication",
                    "error_reason": "bank_offline",
                    "notes": {},
                    "contact": "+919876543210",
                    "email": "customer@example.com",
                }
            }
        },
        "account_id": "acc_test_razorpay",
        "created_at": 1725000000,
    }


@pytest.mark.parametrize("event_type,expected_action", [
    ("payment.failed",       "payment_failure_routed"),
    ("subscription.halted",  "subscription_halted_routed"),
    ("invoice.overdue",      "invoice_overdue_routed"),
])
def test_webhook_signed_ingestion(event_type, expected_action):
    """Signed Razorpay webhook for each of the 3 failure event types is accepted."""
    pid = f"pay_WH_{_uid()}"
    payload = _webhook_payload(event_type, pid)
    # For subscription / invoice events, wrap appropriately
    if event_type == "subscription.halted":
        payload["payload"] = {
            "subscription": {
                "entity": {
                    "id": f"sub_{pid}", "plan_id": "plan_test",
                    "status": "halted", "amount": 49900, "charge_at": 1725000000,
                    "total_count": 12, "paid_count": 3,
                    "current_start": 1724000000, "current_end": 1726000000,
                }
            }
        }
    elif event_type == "invoice.overdue":
        payload["payload"] = {
            "invoice": {
                "entity": {
                    "id": f"inv_{pid}", "type": "invoice",
                    "status": "overdue", "amount": 150000,
                    "amount_due": 150000, "currency": "INR",
                    "due_by": 1724000000,
                }
            }
        }

    body, sig = _signed_body(payload)

    response = client.post(
        "/webhooks/razorpay",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": sig,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "success"
    assert data["action_taken"] == expected_action


def test_webhook_invalid_signature_rejected():
    """Tampered signature is rejected with HTTP 400."""
    pid = f"pay_BADSIG_{_uid()}"
    payload = _webhook_payload("payment.failed", pid)
    body = json.dumps(payload).encode()

    response = client.post(
        "/webhooks/razorpay",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": "deadbeefdeadbeef0000000000000000",
        },
    )
    assert response.status_code == 400


def test_webhook_missing_signature_rejected():
    """Missing X-Razorpay-Signature is rejected (non-simulation mode)."""
    pid = f"pay_NOSIG_{_uid()}"
    payload = _webhook_payload("payment.failed", pid)
    body = json.dumps(payload).encode()

    response = client.post(
        "/webhooks/razorpay",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_webhook_simulation_mode_bypasses_signature(monkeypatch):
    """In simulation mode, missing signature is allowed (dev convenience)."""
    monkeypatch.setenv("SIMULATION_MODE", "true")
    pid = f"pay_SIMWH_{_uid()}"
    payload = _webhook_payload("payment.failed", pid)

    response = client.post(
        "/webhooks/razorpay",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["action_taken"] == "payment_failure_routed"


def test_webhook_malformed_json_rejected():
    """Non-JSON body returns HTTP 400."""
    response = client.post(
        "/webhooks/razorpay",
        content=b"not valid json {{{",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
