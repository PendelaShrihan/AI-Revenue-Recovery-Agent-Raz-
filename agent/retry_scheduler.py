# agent/retry_scheduler.py
"""Intelligent Retry Scheduler – schedules bounded retries with exponential backoff.

Provides:
    - schedule_retry()   — schedules a retry attempt applying exponential backoff
                           (1x, 2x, 4x delay), capped at 3 attempts maximum.
                           Escalates to execute_customer_notification() when limit is exceeded.
    - get_retry_status() — retrieves the current retry progress and attempt counts for a transaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from agent.action_engine import execute_customer_notification
from agent.db_writer import (
    get_db_session,
    save_retry_attempt,
    update_transaction_status,
)
from agent.models import RetryAttempt, Transaction

_logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def schedule_retry(
    transaction: Transaction,
    retry_after_seconds: int,
) -> Optional[RetryAttempt]:
    """Schedule an intelligent retry attempt for a failed transaction.

    Rules:
    - Check retry_attempts table — if transaction already has 3 attempts,
      stop and call execute_customer_notification() instead — never retry more than 3 times.
    - Apply exponential backoff on top of Gemini's suggested delay:
        Attempt 1: retry_after_seconds as-is (1x)
        Attempt 2: retry_after_seconds * 2   (2x)
        Attempt 3: retry_after_seconds * 4   (4x)
    - Save RetryAttempt to database with result/status="SCHEDULED" and correct next_retry_at.
    - Log clearly: "Retry X/3 scheduled for tx='pay_xxx' in Ys"

    Args:
        transaction:         The parent Transaction ORM instance.
        retry_after_seconds: Base delay in seconds (from Gemini decision or caller).

    Returns:
        The created RetryAttempt instance, or None if max attempts exceeded.
    """
    # Defensive check on base delay
    base_delay = int(retry_after_seconds) if retry_after_seconds > 0 else 900

    with get_db_session() as session:
        existing_attempts = (
            session.query(RetryAttempt)
            .filter_by(transaction_id=transaction.id)
            .order_by(RetryAttempt.attempt_number.asc())
            .all()
        )
        attempts_made = len(existing_attempts)

    # 1. Enforce Max 3 Retries Guardrail
    if attempts_made >= MAX_RETRIES:
        _logger.warning(
            "[RetryScheduler] Max retry limit (%d) reached for tx='%s' (payment_id='%s'). "
            "Blocking further retries and triggering customer notification.",
            MAX_RETRIES,
            transaction.id,
            transaction.razorpay_payment_id,
        )
        notification_message = (
            f"We were unable to process your payment of {transaction.currency} {transaction.amount:.2f} "
            f"after {MAX_RETRIES} attempts. Please update your payment method or try a different option."
        )
        execute_customer_notification(
            transaction=transaction,
            message=notification_message,
            channel="email",
        )
        return None

    # 2. Determine attempt number (1-indexed: 1, 2, or 3)
    attempt_number = attempts_made + 1

    # 3. Calculate Exponential Backoff Delay
    # Attempt 1: multiplier 1 (2^0) -> base_delay * 1
    # Attempt 2: multiplier 2 (2^1) -> base_delay * 2
    # Attempt 3: multiplier 4 (2^2) -> base_delay * 4
    backoff_multiplier = 2 ** (attempt_number - 1)
    actual_delay_seconds = base_delay * backoff_multiplier

    now = datetime.now(timezone.utc)
    next_retry_at = now + timedelta(seconds=actual_delay_seconds)

    # 4. Save RetryAttempt to database with status="SCHEDULED"
    retry = save_retry_attempt(
        transaction_id=transaction.id,
        attempt_number=attempt_number,
        result="SCHEDULED",
        next_retry_at=next_retry_at,
    )

    # 5. Update transaction status
    update_transaction_status(
        razorpay_payment_id=transaction.id,
        new_status="retry_scheduled",
    )

    # 6. Log clearly as specified
    _logger.info(
        "Retry %d/%d scheduled for tx='%s' in %ds",
        attempt_number,
        MAX_RETRIES,
        transaction.razorpay_payment_id or transaction.id,
        actual_delay_seconds,
    )

    return retry


def get_retry_status(transaction_id: str) -> Dict[str, Any]:
    """Retrieve the current retry status and counts for a transaction.

    Returns dict schema:
    {
      "transaction_id": "tx_pay_xxx",
      "attempts_made": 2,
      "attempts_remaining": 1,
      "next_retry_at": "2026-09-01T10:30:00Z",
      "status": "retry_scheduled"
    }

    Args:
        transaction_id: Internal transaction ID (e.g. 'tx_pay_xxx') or Razorpay payment ID.

    Returns:
        Structured retry status dictionary.
    """
    with get_db_session() as session:
        tx = (
            session.query(Transaction)
            .filter(
                (Transaction.id == transaction_id)
                | (Transaction.razorpay_payment_id == transaction_id)
            )
            .first()
        )

        if not tx:
            return {
                "transaction_id": transaction_id,
                "attempts_made": 0,
                "attempts_remaining": MAX_RETRIES,
                "next_retry_at": None,
                "status": "not_found",
            }

        attempts = (
            session.query(RetryAttempt)
            .filter_by(transaction_id=tx.id)
            .order_by(RetryAttempt.attempt_number.asc())
            .all()
        )
        attempts_made = len(attempts)
        attempts_remaining = max(0, MAX_RETRIES - attempts_made)

        # Get next_retry_at from latest scheduled attempt if any
        next_retry_at_str = None
        if attempts:
            latest_attempt = attempts[-1]
            if latest_attempt.next_retry_at:
                # Format to ISO 8601 string
                if latest_attempt.next_retry_at.tzinfo is None:
                    next_retry_at_str = latest_attempt.next_retry_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
                else:
                    next_retry_at_str = latest_attempt.next_retry_at.isoformat().replace("+00:00", "Z")

        status_str = tx.status or "retry_scheduled"

        return {
            "transaction_id": tx.id,
            "attempts_made": attempts_made,
            "attempts_remaining": attempts_remaining,
            "next_retry_at": next_retry_at_str,
            "status": status_str,
        }


__all__ = [
    "schedule_retry",
    "get_retry_status",
    "MAX_RETRIES",
]
