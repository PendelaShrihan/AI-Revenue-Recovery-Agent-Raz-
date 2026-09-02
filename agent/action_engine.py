# agent/action_engine.py
"""Action Engine – executes concrete recovery actions from Gemini decisions.

Provides three concrete action executors:
    - execute_auto_retry()            — schedules a retry attempt in DB
    - execute_alternate_suggestion()  — logs alternate payment method suggestion
    - execute_customer_notification() — logs a customer notification draft

And one dispatcher:
    - dispatch_recovery_action()      — reads Gemini JSON output and calls the
                                        correct action automatically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from agent.db_writer import (
    get_db_session,
    save_recovery_action,
    save_retry_attempt,
    update_transaction_status,
)
from agent.models import RecoveryAction, RetryAttempt, Transaction

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action 1 — Auto Retry
# ---------------------------------------------------------------------------

def execute_auto_retry(
    transaction: Transaction,
    retry_after_seconds: int,
) -> RetryAttempt:
    """Schedule an automatic retry for a failed transaction.

    Delegates to ``agent.retry_scheduler.schedule_retry`` to apply exponential
    backoff and enforce the maximum 3 retries limit with notification fallback.

    Args:
        transaction:         The parent Transaction ORM instance.
        retry_after_seconds: Base delay in seconds from Gemini.

    Returns:
        The newly created RetryAttempt instance (or existing/last attempt).
    """
    from agent.retry_scheduler import schedule_retry
    retry = schedule_retry(transaction, retry_after_seconds=retry_after_seconds)
    return retry


# ---------------------------------------------------------------------------
# Action 2 — Alternate Payment Method Suggestion
# ---------------------------------------------------------------------------

def execute_alternate_suggestion(
    transaction: Transaction,
    alternate_method: str,
) -> RecoveryAction:
    """Log a suggestion to use an alternate payment method.

    Creates a RecoveryAction record with ``action_type = 'alternate_method_suggested'``
    and stores ``alternate_method`` in the JSON payload.  Updates the transaction
    status to ``alternate_suggested``.

    Args:
        transaction:      The parent Transaction ORM instance.
        alternate_method: Suggested alternate method string (e.g. ``"upi"``,
                          ``"netbanking"``, ``"wallet"``).

    Returns:
        The newly created RecoveryAction instance.
    """
    _logger.info(
        "[ActionEngine] execute_alternate_suggestion | tx='%s' | alternate_method='%s'",
        transaction.id,
        alternate_method,
    )

    payload: Dict[str, Any] = {
        "alternate_method": alternate_method,
        "original_payment_method": transaction.failure_reason,
        "transaction_amount": transaction.amount,
        "currency": transaction.currency,
        "suggested_at": datetime.now(timezone.utc).isoformat(),
    }

    action = save_recovery_action(
        transaction_id=transaction.id,
        action_type="alternate_method_suggested",
        action_payload=payload,
        status="EXECUTED",
    )

    update_transaction_status(
        razorpay_payment_id=transaction.id,
        new_status="alternate_suggested",
    )

    _logger.info(
        "[ActionEngine] RecoveryAction created | id=%d | tx='%s' | type='alternate_method_suggested' | method='%s'",
        action.id,
        transaction.id,
        alternate_method,
    )
    return action


# ---------------------------------------------------------------------------
# Action 3 — Customer Notification Draft
# ---------------------------------------------------------------------------

def execute_customer_notification(
    transaction: Transaction,
    message: str,
    channel: str = "email",
    merchant_name: Optional[str] = None,
    payment_link: Optional[str] = None,
    alternate_method: Optional[str] = None,
    customer_name: Optional[str] = None,
) -> RecoveryAction:
    """Draft and log a personalized customer notification for a failed transaction.

    Creates a RecoveryAction record with ``action_type = 'customer_notified'``
    and stores rich personalization data (merchant name, formatted amount, payment link,
    alternate methods, and channel-specific copies) in the JSON payload. Updates the
    transaction status to ``customer_notified``.

    Args:
        transaction:      The parent Transaction ORM instance.
        message:          Customer-facing notification message text.
        channel:          Delivery channel — one of ``"email"``, ``"sms"``,
                          ``"whatsapp"`` (default: ``"email"``).
        merchant_name:    Optional merchant/brand name.
        payment_link:     Optional Razorpay payment link.
        alternate_method: Optional alternate payment method suggestion.
        customer_name:    Optional customer name.

    Returns:
        The newly created RecoveryAction instance.
    """
    from agent.notification_engine import generate_personalized_notification

    VALID_CHANNELS = {"email", "sms", "whatsapp"}
    if channel not in VALID_CHANNELS:
        _logger.warning(
            "[ActionEngine] Unknown notification channel '%s'. Defaulting to 'email'.", channel
        )
        channel = "email"

    _logger.info(
        "[ActionEngine] execute_customer_notification | tx='%s' | channel='%s' | message_len=%d",
        transaction.id,
        channel,
        len(message),
    )

    # Generate complete multi-channel personalized package
    notif_data = generate_personalized_notification(
        transaction=transaction,
        failure_category=transaction.failure_code or "unknown",
        channel=channel,
        merchant_name=merchant_name,
        payment_link=payment_link,
        alternate_method=alternate_method,
        customer_name=customer_name,
        use_llm=False,  # Keep action engine synchronous and deterministic
    )
    # Ensure raw message passed from caller is preserved in primary payload
    notif_data["message"] = message
    notif_data["channel"] = channel

    action = save_recovery_action(
        transaction_id=transaction.id,
        action_type="customer_notified",
        action_payload=notif_data,
        status="EXECUTED",
    )

    update_transaction_status(
        razorpay_payment_id=transaction.id,
        new_status="customer_notified",
    )

    _logger.info(
        "[ActionEngine] RecoveryAction created | id=%d | tx='%s' | type='customer_notified' | channel='%s'",
        action.id,
        transaction.id,
        channel,
    )
    return action


# ---------------------------------------------------------------------------
# Dispatcher — routes Gemini decision → correct action
# ---------------------------------------------------------------------------

def dispatch_recovery_action(
    decision: Dict[str, Any],
    transaction: Transaction,
) -> Dict[str, Any]:
    """Route a Gemini recovery decision to the correct concrete action executor.

    Reads the ``action`` field from the Gemini JSON output and calls one of:
      - ``execute_auto_retry``           for action = ``auto_retry``
      - ``execute_alternate_suggestion`` for action = ``suggest_alternate_method``
      - ``execute_customer_notification`` for all other actionable decisions
        (``send_payment_link``, ``notify_customer``, ``manual_review``)
      - No-op log for action = ``no_action``

    Args:
        decision:    Parsed Gemini JSON decision dict.  Must contain at minimum
                     the ``action`` key.  Recognised keys: ``action``,
                     ``retry_after``, ``alternate_method``, ``message``,
                     ``priority``, ``reasoning``.
        transaction: The parent Transaction ORM instance to act upon.

    Returns:
        Summary dict describing what was dispatched::

            {
                "action_taken": str,
                "retry_after":  int | None,
                "alternate_method": str | None,
                "db_record_id": int | None,
                "status": str,
            }
    """
    action = str(decision.get("action", "no_action")).strip().lower()
    retry_after: int = int(decision.get("retry_after", 0))
    alternate_method: str = str(decision.get("alternate_method", "none")).strip().lower()
    message: str = str(decision.get("message", "")).strip()
    priority: str = str(decision.get("priority", "medium")).strip().lower()
    reasoning: str = str(decision.get("reasoning", "")).strip()

    _logger.info(
        "[ActionEngine] dispatch_recovery_action | tx='%s' | action='%s' | priority='%s'",
        transaction.id,
        action,
        priority,
    )

    result: Dict[str, Any] = {
        "action_taken": action,
        "retry_after": retry_after if retry_after > 0 else None,
        "alternate_method": alternate_method if alternate_method not in ("none", "") else None,
        "db_record_id": None,
        "status": "dispatched",
    }

    # ── Route to the correct executor ──────────────────────────────────────

    if action == "auto_retry":
        # Use retry_after from Gemini; default to 900s (15 min) if not specified
        delay = retry_after if retry_after > 0 else 900
        retry = execute_auto_retry(transaction, retry_after_seconds=delay)
        result["db_record_id"] = retry.id
        result["retry_after"] = delay

    elif action == "suggest_alternate_method":
        # Prefer Gemini's alternate_method; fall back to "upi"
        method = alternate_method if alternate_method not in ("none", "", "card") else "upi"
        ra = execute_alternate_suggestion(transaction, alternate_method=method)
        result["db_record_id"] = ra.id
        result["alternate_method"] = method

    elif action in ("send_payment_link", "notify_customer", "manual_review"):
        # All notification-type actions send a customer message via email by default
        channel = "email"
        if not message:
            message = (
                f"We noticed an issue with your payment of {transaction.currency} "
                f"{transaction.amount:.2f}. Please try again or contact support."
            )
        ra = execute_customer_notification(transaction, message=message, channel=channel)
        result["db_record_id"] = ra.id

    elif action == "no_action":
        _logger.info(
            "[ActionEngine] no_action decided by Gemini for tx='%s'. Reason: %s",
            transaction.id,
            reasoning,
        )
        result["status"] = "no_action"

    else:
        # Unknown action — fall back to customer notification
        _logger.warning(
            "[ActionEngine] Unknown action '%s' for tx='%s'. Falling back to customer notification.",
            action,
            transaction.id,
        )
        ra = execute_customer_notification(
            transaction,
            message=message or "We encountered an issue with your payment. Please try again.",
            channel="email",
        )
        result["db_record_id"] = ra.id
        result["action_taken"] = "customer_notified"

    _logger.info(
        "[ActionEngine] dispatch complete | tx='%s' | result=%s",
        transaction.id,
        result,
    )
    return result


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "execute_auto_retry",
    "execute_alternate_suggestion",
    "execute_customer_notification",
    "dispatch_recovery_action",
]
