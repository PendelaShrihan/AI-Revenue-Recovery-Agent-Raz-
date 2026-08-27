"""
Unit Tests — ML Retry-Timing Predictor & Schedule Optimizer.

Tests:
1. Transient failures (network_timeout, gateway_issue) trigger auto-retry with fast backoff.
2. Terminal failures (card_blocked, expired_card) return is_retryable=False and recommend ALTERNATE_METHOD.
3. Authentication failures (authentication_failed) recommend SEND_PAYMENT_LINK.
4. Insufficient funds (insufficient_funds) provides extended delay (e.g. 4-6h) for fund reload.
5. Limit exceeded (limit_exceeded) schedules for daily limit reset or alternate method.
6. Bounded state machine rule: attempt_count >= 2 strictly stops auto-retry and escalates to MANUAL_REVIEW_REQUIRED.
7. IST banking maintenance window (01:00 AM - 04:00 AM IST) avoidance logic.
8. Batch prediction interface across multiple transaction objects.
9. Serialization via to_dict().
"""

from datetime import datetime, timezone, timedelta
import pytest

from ml.error_codes import FailureCategory
from ml.retry_predictor import (
    RetryTimingPredictor,
    RetryTimingRecommendation,
    RecoveryAction,
    BASE_RETRY_DELAYS,
)


@pytest.fixture
def predictor() -> RetryTimingPredictor:
    return RetryTimingPredictor(max_retries=2)


# ─── 1. Transient Failures ───────────────────────────────────────────────────

def test_transient_network_timeout_attempt_1(predictor: RetryTimingPredictor):
    # Noon UTC = 17:30 IST (Outside night maintenance window)
    dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.NETWORK_TIMEOUT.value,
        attempt_count=1,
        payment_method="upi",
        created_at=dt,
    )
    assert rec.is_retryable is True
    assert rec.recommended_action == RecoveryAction.AUTO_RETRY.value
    assert rec.optimal_delay_minutes == 5
    assert rec.confidence_score >= 0.90
    assert rec.attempt_count == 1
    assert rec.avoided_maintenance_window is False


def test_transient_gateway_issue_attempt_2(predictor: RetryTimingPredictor):
    dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.GATEWAY_ISSUE.value,
        attempt_count=2,
        payment_method="card",
        created_at=dt,
    )
    # Attempt 2 is the 2nd retry, which hits max_retries=2
    assert rec.is_retryable is False
    assert rec.recommended_action == RecoveryAction.MANUAL_REVIEW.value
    assert rec.optimal_delay_minutes == 0


def test_transient_gateway_issue_attempt_1(predictor: RetryTimingPredictor):
    dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.GATEWAY_ISSUE.value,
        attempt_count=1,
        payment_method="card",
        created_at=dt,
    )
    assert rec.is_retryable is True
    assert rec.recommended_action == RecoveryAction.AUTO_RETRY.value
    assert rec.optimal_delay_minutes > 0


# ─── 2. Terminal Failures ────────────────────────────────────────────────────

def test_terminal_card_blocked(predictor: RetryTimingPredictor):
    rec = predictor.predict(
        failure_category=FailureCategory.CARD_BLOCKED.value,
        attempt_count=1,
        payment_method="card",
    )
    assert rec.is_retryable is False
    assert rec.recommended_action == RecoveryAction.ALTERNATE_METHOD.value
    assert rec.optimal_delay_minutes == 0
    assert "permanently blocked" in rec.reasoning.lower()


def test_terminal_expired_card(predictor: RetryTimingPredictor):
    rec = predictor.predict(
        failure_category=FailureCategory.EXPIRED_CARD.value,
        attempt_count=1,
        payment_method="card",
    )
    assert rec.is_retryable is False
    assert rec.recommended_action == RecoveryAction.ALTERNATE_METHOD.value
    assert rec.optimal_delay_minutes == 0


# ─── 3. Authentication Failures ──────────────────────────────────────────────

def test_authentication_failed(predictor: RetryTimingPredictor):
    dt = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.AUTHENTICATION_FAILED.value,
        attempt_count=1,
        payment_method="card",
        created_at=dt,
    )
    assert rec.is_retryable is False
    assert rec.recommended_action == RecoveryAction.SEND_PAYMENT_LINK.value
    assert rec.optimal_delay_minutes == 15
    assert "otp" in rec.reasoning.lower() or "authentication" in rec.reasoning.lower()


# ─── 4. Insufficient Funds ───────────────────────────────────────────────────

def test_insufficient_funds_onetime(predictor: RetryTimingPredictor):
    dt = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.INSUFFICIENT_FUNDS.value,
        attempt_count=1,
        payment_method="upi",
        amount=1500.0,
        created_at=dt,
        is_subscription=False,
    )
    assert rec.is_retryable is True
    assert rec.recommended_action == RecoveryAction.SEND_PAYMENT_LINK.value
    assert rec.optimal_delay_minutes >= 240  # 4 hours minimum for balance reload


def test_insufficient_funds_subscription(predictor: RetryTimingPredictor):
    dt = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.INSUFFICIENT_FUNDS.value,
        attempt_count=1,
        payment_method="mandate",
        amount=999.0,
        created_at=dt,
        is_subscription=True,
    )
    assert rec.is_retryable is True
    assert rec.recommended_action == RecoveryAction.AUTO_RETRY.value
    assert rec.optimal_delay_minutes >= 240


# ─── 5. Limit Exceeded ───────────────────────────────────────────────────────

def test_limit_exceeded_onetime(predictor: RetryTimingPredictor):
    dt = datetime(2026, 8, 27, 10, 0, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.LIMIT_EXCEEDED.value,
        attempt_count=1,
        payment_method="upi",
        created_at=dt,
        is_subscription=False,
    )
    assert rec.recommended_action == RecoveryAction.ALTERNATE_METHOD.value
    assert rec.optimal_delay_minutes >= 720


# ─── 6. Bounded State Machine Max Retry Enforcement ─────────────────────────

def test_max_retries_enforced_at_limit(predictor: RetryTimingPredictor):
    rec = predictor.predict(
        failure_category=FailureCategory.NETWORK_TIMEOUT.value,
        attempt_count=2,
    )
    assert rec.is_retryable is False
    assert rec.recommended_action == RecoveryAction.MANUAL_REVIEW.value
    assert rec.optimal_delay_minutes == 0
    assert "Retry limit exceeded" in rec.reasoning


def test_max_retries_enforced_exceeded(predictor: RetryTimingPredictor):
    rec = predictor.predict(
        failure_category=FailureCategory.NETWORK_TIMEOUT.value,
        attempt_count=3,
    )
    assert rec.is_retryable is False
    assert rec.recommended_action == RecoveryAction.MANUAL_REVIEW.value
    assert rec.optimal_delay_minutes == 0


# ─── 7. Indian Banking Maintenance Window (IST) ─────────────────────────────

def test_ist_maintenance_window_avoidance(predictor: RetryTimingPredictor):
    # 20:30 UTC = 02:00 IST (Inside 01:00 - 04:00 IST maintenance window)
    # A 15-minute retry would land at 02:15 IST.
    dt = datetime(2026, 8, 27, 20, 30, 0, tzinfo=timezone.utc)
    rec = predictor.predict(
        failure_category=FailureCategory.GATEWAY_ISSUE.value,
        attempt_count=1,
        payment_method="card",
        created_at=dt,
    )
    assert rec.avoided_maintenance_window is True
    # The scheduled time should be pushed to 06:00 IST (00:30 UTC next day)
    scheduled = datetime.fromisoformat(rec.scheduled_at)
    try:
        from zoneinfo import ZoneInfo
        ist_tz = ZoneInfo("Asia/Kolkata")
    except Exception:
        import pytz
        ist_tz = pytz.timezone("Asia/Kolkata")
    
    ist_time = scheduled.astimezone(ist_tz)
    assert ist_time.hour == 6
    assert ist_time.minute == 0


# ─── 8. Batch Predictions & Serialization ───────────────────────────────────

def test_batch_predict(predictor: RetryTimingPredictor):
    records = [
        {"failure_category": "network_timeout", "attempt_count": 1, "payment_method": "upi"},
        {"error_code": "CARD_BLOCKED", "error_reason": "card_declined", "attempt_count": 1},
        {"failure_category": "insufficient_funds", "attempt_count": 2},
    ]
    results = predictor.batch_predict(records)
    assert len(results) == 3
    assert results[0].recommended_action == RecoveryAction.AUTO_RETRY.value
    assert results[1].recommended_action == RecoveryAction.ALTERNATE_METHOD.value
    assert results[2].recommended_action == RecoveryAction.MANUAL_REVIEW.value


def test_recommendation_to_dict(predictor: RetryTimingPredictor):
    rec = predictor.predict(failure_category="network_timeout", attempt_count=1)
    d = rec.to_dict()
    assert isinstance(d, dict)
    assert d["failure_category"] == "network_timeout"
    assert "optimal_delay_minutes" in d
    assert "recommended_action" in d
    assert "confidence_score" in d
    assert "reasoning" in d
    assert "scheduled_at" in d
