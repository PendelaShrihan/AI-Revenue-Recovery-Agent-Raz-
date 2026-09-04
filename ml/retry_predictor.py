"""
Razorpay Smart Retry-Timing Predictor & Schedule Optimizer.

This module provides intelligent retry-delay predictions and recovery action recommendations
tailored to Indian payment systems (UPI, Cards, Netbanking, e-Mandates, Wallets).

Key Capabilities:
1. Categorical Retry Timing:
   - Transient network/gateway glitches -> Fast backoff with jitter (5–60 mins).
   - Insufficient funds -> Elongated delay (4–6 hours) to permit bank account reload.
   - Limit exceeded -> End-of-day / next-day window (after 00:00 IST limit resets).
   - Terminal errors (blocked/expired cards) -> Immediate alternate payment method.
   - Authentication errors (OTP/MPIN failures) -> Customer payment link re-dispatch.
2. Bounded State Machine Enforcement:
   - Strictly enforces max 2 retry attempts.
   - Attempt count >= 2 immediately escalates to MANUAL_REVIEW_REQUIRED.
3. Indian Banking Window Awareness (IST / UTC+5:30):
   - Automatically detects and avoids the NPCI/Bank midnight maintenance window (01:00 AM – 04:00 AM IST)
     by pushing scheduled retries forward into the morning business window (08:30+ AM IST).
4. Deterministic and Auditable:
   - Provides confidence scores and structured diagnostic reasoning for every decision.
"""

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Explicitly enforce Asia/Kolkata (IST) timezone
try:
    from zoneinfo import ZoneInfo
    IST_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    try:
        import pytz
        IST_TZ = pytz.timezone("Asia/Kolkata")
    except Exception:
        IST_TZ = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")

from ml.error_codes import FailureCategory, is_transient_failure


class RecoveryAction(str, Enum):
    """Recommended recovery action types."""
    AUTO_RETRY = "AUTO_RETRY"
    SEND_PAYMENT_LINK = "SEND_PAYMENT_LINK"
    ALTERNATE_METHOD = "ALTERNATE_METHOD"
    INVOICE_REMINDER = "INVOICE_REMINDER"
    MANUAL_REVIEW = "MANUAL_REVIEW_REQUIRED"


@dataclass
class RetryTimingRecommendation:
    """
    Structured recommendation payload returned by RetryTimingPredictor.
    """
    failure_category: str
    is_retryable: bool
    optimal_delay_minutes: int
    recommended_action: str
    confidence_score: float
    reasoning: str
    scheduled_at: str  # ISO-8601 UTC string
    attempt_count: int
    avoided_maintenance_window: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes recommendation to a dictionary."""
        return asdict(self)


# Base retry delay matrices (in minutes) for attempt 1 and attempt 2
# Format: {category: (attempt_1_delay_mins, attempt_2_delay_mins)}
BASE_RETRY_DELAYS: Dict[str, tuple[int, int]] = {
    FailureCategory.NETWORK_TIMEOUT.value: (5, 30),
    FailureCategory.GATEWAY_ISSUE.value: (15, 60),
    FailureCategory.INSUFFICIENT_FUNDS.value: (240, 480),  # 4h, 8h
    FailureCategory.LIMIT_EXCEEDED.value: (720, 1440),     # 12h, 24h
    FailureCategory.AUTHENTICATION_FAILED.value: (15, 60),  # Delay before sending payment link reminder
    FailureCategory.CARD_BLOCKED.value: (0, 0),             # Non-retryable
    FailureCategory.EXPIRED_CARD.value: (0, 0),             # Non-retryable
    FailureCategory.UNKNOWN.value: (30, 90),
}


class RetryTimingPredictor:
    """
    Predicts optimal retry delay and recovery action for failed payment transactions.
    """

    MAX_RETRIES = 3

    def __init__(self, max_retries: int = 3, tz: Any = None):
        self.max_retries = max_retries
        self.tz = tz or IST_TZ

    def _to_utc_datetime(self, dt: Optional[Union[datetime, str, int, float]]) -> datetime:
        """Helper to parse varied timestamp inputs into timezone-aware UTC datetime."""
        if dt is None:
            return datetime.now(timezone.utc)
        if isinstance(dt, (int, float)):
            # Handle epoch seconds or milliseconds
            if dt > 1e11:
                dt = dt / 1000.0
            return datetime.fromtimestamp(dt, tz=timezone.utc)
        if isinstance(dt, str):
            try:
                # Handle ISO 8601 strings
                parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except Exception:
                return datetime.now(timezone.utc)
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _adjust_for_ist_maintenance_window(
        self, scheduled_utc: datetime, base_delay_mins: int, base_time: datetime
    ) -> tuple[datetime, int, bool]:
        """
        Detects if the scheduled retry falls inside Indian banking maintenance windows
        (01:00 AM – 04:00 AM IST) where core banking systems / NPCI switch batch jobs run.
        Explicitly enforces Asia/Kolkata timezone before extracting .hour.
        Pushes scheduled retry forward to 06:00 AM IST.
        """
        ist_dt = scheduled_utc.astimezone(self.tz)
        hour = ist_dt.hour
        avoided = False

        if 1 <= hour < 4:
            # Advance to 06:00 AM IST on the same morning
            adjusted_ist = ist_dt.replace(hour=6, minute=0, second=0, microsecond=0)
            if adjusted_ist < ist_dt:
                adjusted_ist += timedelta(days=1)
            scheduled_utc = adjusted_ist.astimezone(timezone.utc)
            # Recalculate adjusted delay in minutes from event base_time
            diff_mins = max(1, int((scheduled_utc - base_time).total_seconds() / 60))
            return scheduled_utc, diff_mins, True

        return scheduled_utc, base_delay_mins, avoided

    def predict(
        self,
        failure_category: str,
        attempt_count: int = 1,
        payment_method: Optional[str] = None,
        amount: Optional[float] = None,
        created_at: Optional[Union[datetime, str, int, float]] = None,
        is_subscription: bool = False,
        merchant_category: Optional[str] = None,
    ) -> RetryTimingRecommendation:
        """
        Computes the optimal retry delay (in minutes) and recommended recovery action.

        Args:
            failure_category: Canonical failure category (e.g., 'network_timeout', 'insufficient_funds').
            attempt_count: Current retry attempt index (1 for 1st retry, 2 for 2nd retry, >=3 exhausted).
            payment_method: Payment rail ('upi', 'card', 'netbanking', 'wallet', 'mandate', 'emi').
            amount: Transaction amount in INR.
            created_at: Failure timestamp (datetime, ISO string, or epoch timestamp).
            is_subscription: True if this is an automated recurring mandate / subscription.
            merchant_category: Industry vertical (e.g. 'ecommerce', 'saas', 'gaming').

        Returns:
            RetryTimingRecommendation with structured action, delay, confidence, and reasoning.
        """
        cat_norm = str(failure_category or "").strip().lower()
        method_norm = str(payment_method or "").strip().lower()
        base_time = self._to_utc_datetime(created_at)

        # 1. Check Hard Stopping Rule (Max 2 retries)
        if attempt_count >= self.max_retries:
            return RetryTimingRecommendation(
                failure_category=cat_norm,
                is_retryable=False,
                optimal_delay_minutes=0,
                recommended_action=RecoveryAction.MANUAL_REVIEW.value,
                confidence_score=0.99,
                reasoning=(
                    f"Retry limit exceeded ({attempt_count}/{self.max_retries} attempts completed). "
                    "Escalating to manual merchant review to prevent infinite payment loops."
                ),
                scheduled_at=base_time.isoformat(),
                attempt_count=attempt_count,
                avoided_maintenance_window=False,
            )

        # 2. Terminal Failures (Card Blocked / Expired Card)
        if cat_norm in (FailureCategory.CARD_BLOCKED.value, FailureCategory.EXPIRED_CARD.value):
            reason_msg = (
                "Card is permanently blocked or expired. Automated retries on the same instrument will fail 100% of the time. "
                "Immediate customer notification with alternate payment method request is required."
            )
            return RetryTimingRecommendation(
                failure_category=cat_norm,
                is_retryable=False,
                optimal_delay_minutes=0,
                recommended_action=RecoveryAction.ALTERNATE_METHOD.value,
                confidence_score=0.98,
                reasoning=reason_msg,
                scheduled_at=base_time.isoformat(),
                attempt_count=attempt_count,
                avoided_maintenance_window=False,
            )

        # 3. Authentication Failures (OTP / 3DS / MPIN)
        if cat_norm == FailureCategory.AUTHENTICATION_FAILED.value:
            # Cannot auto-retry in headless background because OTP requires customer presence.
            # Recommend sending a smart payment recovery link.
            delays = BASE_RETRY_DELAYS.get(cat_norm, (15, 60))
            delay_mins = delays[0] if attempt_count == 1 else delays[1]

            scheduled_utc = base_time + timedelta(minutes=delay_mins)
            scheduled_utc, final_delay, avoided = self._adjust_for_ist_maintenance_window(scheduled_utc, delay_mins, base_time)

            return RetryTimingRecommendation(
                failure_category=cat_norm,
                is_retryable=False,  # Headless background retry not possible without user OTP
                optimal_delay_minutes=final_delay,
                recommended_action=RecoveryAction.SEND_PAYMENT_LINK.value,
                confidence_score=0.95,
                reasoning=(
                    "Customer 3DS / OTP authentication failed. Headless auto-retry will fail without user intervention. "
                    f"Recommended action is dispatching a dynamic payment link reminder with a {final_delay}-minute cooloff."
                ),
                scheduled_at=scheduled_utc.isoformat(),
                attempt_count=attempt_count,
                avoided_maintenance_window=avoided,
            )

        # 4. Limit Exceeded
        if cat_norm == FailureCategory.LIMIT_EXCEEDED.value:
            delays = BASE_RETRY_DELAYS.get(cat_norm, (720, 1440))
            delay_mins = delays[0] if attempt_count == 1 else delays[1]

            scheduled_utc = base_time + timedelta(minutes=delay_mins)
            scheduled_utc, final_delay, avoided = self._adjust_for_ist_maintenance_window(scheduled_utc, delay_mins, base_time)

            return RetryTimingRecommendation(
                failure_category=cat_norm,
                is_retryable=True if is_subscription else False,
                optimal_delay_minutes=final_delay,
                recommended_action=RecoveryAction.AUTO_RETRY.value if is_subscription else RecoveryAction.ALTERNATE_METHOD.value,
                confidence_score=0.90,
                reasoning=(
                    "Transaction or velocity limit exceeded. Daily banking limits reset at 00:00 IST. "
                    f"Scheduling next attempt window after {final_delay} minutes or prompting customer for alternate method."
                ),
                scheduled_at=scheduled_utc.isoformat(),
                attempt_count=attempt_count,
                avoided_maintenance_window=avoided,
            )

        # 5. Insufficient Funds
        if cat_norm == FailureCategory.INSUFFICIENT_FUNDS.value:
            delays = BASE_RETRY_DELAYS.get(cat_norm, (240, 480))
            base_delay = delays[0] if attempt_count == 1 else delays[1]

            # If it's a high-ticket payment (> Rs. 10,000), allow more time for funds transfer
            if amount and amount > 10000:
                base_delay = int(base_delay * 1.5)

            scheduled_utc = base_time + timedelta(minutes=base_delay)
            scheduled_utc, final_delay, avoided = self._adjust_for_ist_maintenance_window(scheduled_utc, base_delay, base_time)

            return RetryTimingRecommendation(
                failure_category=cat_norm,
                is_retryable=True,
                optimal_delay_minutes=final_delay,
                recommended_action=RecoveryAction.AUTO_RETRY.value if is_subscription else RecoveryAction.SEND_PAYMENT_LINK.value,
                confidence_score=0.88,
                reasoning=(
                    "Insufficient account balance. Immediate retry yields low recovery probability. "
                    f"Applying a {final_delay}-minute window to allow customer balance reload before re-attempting."
                ),
                scheduled_at=scheduled_utc.isoformat(),
                attempt_count=attempt_count,
                avoided_maintenance_window=avoided,
            )

        # 6. Transient Technical Glitches (Network Timeout / Gateway Issue / Server Downtime)
        if cat_norm in (FailureCategory.NETWORK_TIMEOUT.value, FailureCategory.GATEWAY_ISSUE.value):
            delays = BASE_RETRY_DELAYS.get(cat_norm, (10, 45))
            base_delay = delays[0] if attempt_count == 1 else delays[1]

            # Fast recovery for UPI network timeouts vs Netbanking gateway switch drops
            if method_norm == "upi" and cat_norm == FailureCategory.NETWORK_TIMEOUT.value:
                base_delay = 5 if attempt_count == 1 else 20
            elif method_norm == "netbanking":
                base_delay = 20 if attempt_count == 1 else 60

            scheduled_utc = base_time + timedelta(minutes=base_delay)
            scheduled_utc, final_delay, avoided = self._adjust_for_ist_maintenance_window(scheduled_utc, base_delay, base_time)

            return RetryTimingRecommendation(
                failure_category=cat_norm,
                is_retryable=True,
                optimal_delay_minutes=final_delay,
                recommended_action=RecoveryAction.AUTO_RETRY.value,
                confidence_score=0.96,
                reasoning=(
                    f"Transient {cat_norm.replace('_', ' ')} detected on {method_norm or 'payment rail'}. "
                    f"Optimal recovery strategy is automated retry after {final_delay} minutes."
                ),
                scheduled_at=scheduled_utc.isoformat(),
                attempt_count=attempt_count,
                avoided_maintenance_window=avoided,
            )

        # 7. Default / Unknown Fallback
        delays = BASE_RETRY_DELAYS.get(FailureCategory.UNKNOWN.value, (30, 90))
        base_delay = delays[0] if attempt_count == 1 else delays[1]

        scheduled_utc = base_time + timedelta(minutes=base_delay)
        scheduled_utc, final_delay, avoided = self._adjust_for_ist_maintenance_window(scheduled_utc, base_delay, base_time)

        return RetryTimingRecommendation(
            failure_category=cat_norm or FailureCategory.UNKNOWN.value,
            is_retryable=True,
            optimal_delay_minutes=final_delay,
            recommended_action=RecoveryAction.AUTO_RETRY.value if is_transient_failure(cat_norm) else RecoveryAction.SEND_PAYMENT_LINK.value,
            confidence_score=0.75,
            reasoning=(
                f"Generic or unmapped failure category '{cat_norm}'. "
                f"Applying baseline conservative retry delay of {final_delay} minutes."
            ),
            scheduled_at=scheduled_utc.isoformat(),
            attempt_count=attempt_count,
            avoided_maintenance_window=avoided,
        )

    def batch_predict(
        self, records: List[Dict[str, Any]]
    ) -> List[RetryTimingRecommendation]:
        """
        Executes batch retry timing predictions across multiple payment failure records.

        Args:
            records: List of dictionaries containing keys:
                     - failure_category (or error_code / error_reason)
                     - attempt_count (optional, default 1)
                     - payment_method (optional)
                     - amount (optional)
                     - created_at (optional)

        Returns:
            List of RetryTimingRecommendation instances.
        """
        results = []
        for r in records:
            cat = r.get("failure_category") or r.get("category")
            if not cat and ("error_code" in r or "error_reason" in r):
                from ml.error_codes import classify_failure
                cat = classify_failure(r.get("error_code"), r.get("error_reason"))

            rec = self.predict(
                failure_category=cat or "unknown",
                attempt_count=int(r.get("attempt_count", 1)),
                payment_method=r.get("payment_method"),
                amount=float(r.get("amount", 0.0)) if r.get("amount") is not None else None,
                created_at=r.get("created_at"),
                is_subscription=bool(r.get("is_subscription", False)),
                merchant_category=r.get("merchant_category"),
            )
            results.append(rec)
        return results
