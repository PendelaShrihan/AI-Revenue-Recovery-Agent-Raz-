# tests/test_llm_integration.py
"""Integration tests for GeminiAgent and RecoveryEngine.

These tests call the REAL Gemini API (gemini-2.0-flash). They require a valid
GEMINI_API_KEY in the environment or .env file.

Run with:
    pytest tests/test_llm_integration.py -v -s

The -s flag streams logging output so you can watch prompts and responses live.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv

# ── Load .env so GEMINI_API_KEY is available ──────────────────────────────
load_dotenv()

from api_integration.schemas import NormalizedEvent, FailureCategory, EventType
from agent.llm_agent import (
    GeminiAgent,
    GeminiAgentError,
    GeminiOutputParseError,
    RecoveryDecision,
    SYSTEM_PROMPT,
    build_cot_prompt,
)
from agent.recovery_engine import RecoveryEngine, get_recovery_decision

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ── Skip guard ────────────────────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set – skipping real-API integration tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    failure_category: str = "unknown",
    payment_method: str = "card",
    error_reason: str = "unknown",
    error_description: str = "Payment failed",
    error_code: str = "BAD_REQUEST_ERROR",
    error_source: str = "bank",
    error_step: str = "payment_authentication",
    amount: float = 999.00,
    subscription_id: str | None = None,
    customer_email: str = "customer@example.com",
    customer_phone: str = "+919876543210",
    hour: int = 10,
) -> NormalizedEvent:
    """Factory for building NormalizedEvent fixtures for each failure category."""
    created_at = datetime(2024, 8, 28, hour, 0, 0, tzinfo=timezone.utc)
    return NormalizedEvent(
        event_id=f"evt_{failure_category}_{payment_method}",
        event_type=EventType.PAYMENT_FAILED,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=f"pay_test_{failure_category}",
        merchant_id="merchant_test_001",
        amount=amount,
        currency="INR",
        status="FAILED",
        payment_id=f"pay_{failure_category}_001",
        subscription_id=subscription_id,
        customer_email=customer_email,
        customer_phone=customer_phone,
        payment_method=payment_method,
        error_code=error_code,
        error_description=error_description,
        error_source=error_source,
        error_step=error_step,
        error_reason=error_reason,
        created_at=created_at,
    )


def _assert_valid_decision(decision: RecoveryDecision, category_label: str) -> None:
    """Common assertions that every RecoveryDecision must satisfy."""
    print("=" * 60)
    print(f"  Category: {category_label}")
    print(f"  Action  : {decision.action}")
    print(f"  Priority: {decision.priority}")
    print(f"  Retry in: {decision.retry_after}s")
    print(f"  Alternate: {decision.alternate_method}")
    msg_safe = decision.message.encode("ascii", errors="backslashreplace").decode("ascii")
    reasoning_safe = decision.reasoning.encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"  Message : {msg_safe}")
    print(f"  Reasoning: {reasoning_safe}")
    print("=" * 60)

    assert decision.action in (
        "auto_retry", "suggest_alternate_method", "send_payment_link",
        "notify_customer", "manual_review", "no_action",
    ), f"Unexpected action: {decision.action}"
    assert decision.priority in ("high", "medium", "low")
    assert decision.alternate_method in ("upi", "wallet", "netbanking", "card", "none")
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.retry_after, int)
    assert isinstance(decision.message, str)
    if decision.action in ("suggest_alternate_method", "send_payment_link", "notify_customer"):
        assert len(decision.message) > 0, f"Customer message must not be empty for action '{decision.action}'"
    assert len(decision.reasoning) > 0, "reasoning must not be empty"

    # Verify raw_response is valid JSON (audit trail check)
    cleaned = decision.raw_response
    if cleaned.startswith("```"):
        lines = [l for l in cleaned.splitlines() if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    assert isinstance(parsed, dict), "raw_response must be a JSON object"


# ---------------------------------------------------------------------------
# Unit tests – no API call needed
# ---------------------------------------------------------------------------

class TestBuildCotPrompt:
    """Unit tests for the prompt builder – no API call needed."""

    def test_prompt_contains_amount(self):
        event = _make_event(amount=4999.50, failure_category="insufficient_funds")
        prompt = build_cot_prompt(event)
        assert "4999.5" in prompt or "4999.50" in prompt

    def test_prompt_contains_failure_category(self):
        event = _make_event(error_reason="insufficient_funds")
        event = event.model_copy(update={"failure_category": FailureCategory.CHECKOUT_FAILURE})
        prompt = build_cot_prompt(event)
        assert "checkout_failure" in prompt.lower() or "insufficient_funds" in prompt.lower()

    def test_prompt_marks_subscription_high_priority(self):
        event = _make_event(subscription_id="sub_abc123")
        prompt = build_cot_prompt(event)
        assert "SUBSCRIPTION" in prompt

    def test_prompt_contains_notification_channels(self):
        event = _make_event(customer_email="test@example.com", customer_phone="+919999999999")
        prompt = build_cot_prompt(event)
        assert "test@example.com" in prompt
        assert "+919999999999" in prompt

    def test_prompt_contains_hour_of_day(self):
        event = _make_event(hour=9)
        prompt = build_cot_prompt(event)
        assert "9" in prompt  # hour_of_day


class TestSystemPrompt:
    """Verify SYSTEM_PROMPT contains all eight categories."""

    def test_all_eight_categories_present(self):
        categories = [
            "insufficient_funds", "card_blocked", "network_timeout",
            "gateway_issue", "expired_card", "authentication_failed",
            "limit_exceeded", "unknown",
        ]
        for cat in categories:
            assert cat in SYSTEM_PROMPT, f"Category '{cat}' missing from SYSTEM_PROMPT"

    def test_json_schema_described(self):
        required_keys = [
            '"action"', '"priority"', '"message"',
            '"retry_after"', '"alternate_method"', '"confidence"', '"reasoning"',
        ]
        for key in required_keys:
            assert key in SYSTEM_PROMPT, f"Key {key} missing from SYSTEM_PROMPT schema description"


# ---------------------------------------------------------------------------
# Integration tests – real Gemini API
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent() -> GeminiAgent:
    return GeminiAgent()


@pytest.fixture(scope="module")
def engine() -> RecoveryEngine:
    return RecoveryEngine()


class TestGeminiAgentIntegration:
    """End-to-end Gemini API calls for all eight failure categories."""

    @pytest.fixture(autouse=True)
    def _rate_limit_throttle(self):
        time.sleep(1.5)

    # ── 1. insufficient_funds ────────────────────────────────────────────

    def test_insufficient_funds(self, engine: RecoveryEngine):
        event = _make_event(
            failure_category="checkout_failure",
            payment_method="card",
            error_reason="insufficient_funds",
            error_description="Your account does not have sufficient funds.",
            error_source="bank",
            error_step="payment_authentication",
            amount=2500.00,
            hour=14,
        )
        decision = engine.process(event, ml_failure_category="insufficient_funds")
        _assert_valid_decision(decision, "insufficient_funds")
        # Gemini should suggest an alternate method, not retry immediately
        assert decision.alternate_method in ("upi", "wallet", "netbanking"), (
            f"Expected alternate method for insufficient_funds, got {decision.alternate_method}"
        )

    # ── 2. card_blocked ──────────────────────────────────────────────────

    def test_card_blocked(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="card",
            error_reason="card_blocked",
            error_description="Your card has been blocked by the issuing bank.",
            error_source="bank",
            error_step="payment_authentication",
            amount=1200.00,
            hour=11,
        )
        decision = engine.process(event, ml_failure_category="card_blocked")
        _assert_valid_decision(decision, "card_blocked")
        # No retry expected for blocked card
        assert decision.retry_after == 0 or decision.action in (
            "suggest_alternate_method", "notify_customer", "send_payment_link", "manual_review"
        ), f"Unexpected decision for card_blocked: {decision.action}"

    # ── 3. network_timeout ───────────────────────────────────────────────

    def test_network_timeout(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="upi",
            error_reason="network_timeout",
            error_description="Request timed out while connecting to payment gateway.",
            error_source="gateway",
            error_step="payment_initiation",
            amount=599.00,
            hour=16,
        )
        decision = engine.process(event, ml_failure_category="network_timeout")
        _assert_valid_decision(decision, "network_timeout")
        assert decision.action == "auto_retry", (
            f"network_timeout should trigger auto_retry, got {decision.action}"
        )
        assert 600 <= decision.retry_after <= 1800, (
            f"network_timeout retry_after should be ~900s (15 min), got {decision.retry_after}"
        )

    # ── 4. gateway_issue ─────────────────────────────────────────────────

    def test_gateway_issue(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="netbanking",
            error_reason="gateway_issue",
            error_description="Gateway returned a 502 Bad Gateway error.",
            error_source="gateway",
            error_step="payment_routing",
            amount=3400.00,
            hour=20,
        )
        decision = engine.process(event, ml_failure_category="gateway_issue")
        _assert_valid_decision(decision, "gateway_issue")
        assert decision.action == "auto_retry", (
            f"gateway_issue should trigger auto_retry, got {decision.action}"
        )

    # ── 5. expired_card ──────────────────────────────────────────────────

    def test_expired_card(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="card",
            error_reason="expired_card",
            error_description="The card has expired. Please use a valid card.",
            error_source="bank",
            error_step="payment_authentication",
            amount=7999.00,
            hour=10,
        )
        decision = engine.process(event, ml_failure_category="expired_card")
        _assert_valid_decision(decision, "expired_card")
        assert decision.retry_after == 0, (
            f"expired_card must not retry, got retry_after={decision.retry_after}"
        )
        assert decision.priority == "high", (
            f"expired_card should be high priority, got {decision.priority}"
        )

    # ── 6. authentication_failed ─────────────────────────────────────────

    def test_authentication_failed(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="card",
            error_reason="authentication_failed",
            error_description="OTP expired before the customer could complete authentication.",
            error_source="customer",
            error_step="payment_authentication",
            amount=450.00,
            hour=19,
        )
        decision = engine.process(event, ml_failure_category="authentication_failed")
        _assert_valid_decision(decision, "authentication_failed")

    # ── 7. limit_exceeded ────────────────────────────────────────────────

    def test_limit_exceeded(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="card",
            error_reason="limit_exceeded",
            error_description="Transaction amount exceeds the daily limit set on the card.",
            error_source="bank",
            error_step="payment_authorization",
            amount=50000.00,
            hour=13,
        )
        decision = engine.process(event, ml_failure_category="limit_exceeded")
        _assert_valid_decision(decision, "limit_exceeded")
        assert decision.retry_after == 0 or decision.alternate_method != "none", (
            "limit_exceeded should either not retry or suggest an alternate method"
        )

    # ── 8. unknown ───────────────────────────────────────────────────────

    def test_unknown(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="wallet",
            error_reason="unknown",
            error_description="An unexpected error occurred during payment processing.",
            error_source="gateway",
            error_step="payment_initiation",
            amount=199.00,
            hour=3,
        )
        decision = engine.process(event, ml_failure_category="unknown")
        _assert_valid_decision(decision, "unknown")

    # ── 9. Subscription (higher priority) ────────────────────────────────

    def test_subscription_payment_insufficient_funds(self, engine: RecoveryEngine):
        event = _make_event(
            payment_method="card",
            error_reason="insufficient_funds",
            error_description="Insufficient balance for recurring subscription charge.",
            error_source="bank",
            error_step="payment_authentication",
            amount=999.00,
            subscription_id="sub_Abc123XyZ",
            hour=9,
        )
        decision = engine.process(event, ml_failure_category="insufficient_funds")
        _assert_valid_decision(decision, "insufficient_funds (subscription)")
        # Subscription payments should get high priority
        assert decision.priority == "high", (
            f"Subscription payment should be high priority, got {decision.priority}"
        )

    # ── 10. Convenience function ──────────────────────────────────────────

    def test_get_recovery_decision_convenience(self):
        event = _make_event(
            payment_method="upi",
            error_reason="network_timeout",
            error_description="UPI transaction timed out.",
            error_source="gateway",
            error_step="payment_initiation",
            amount=100.00,
            hour=12,
        )
        decision = get_recovery_decision(event, ml_failure_category="network_timeout")
        _assert_valid_decision(decision, "network_timeout (convenience fn)")
        assert isinstance(decision, RecoveryDecision)
