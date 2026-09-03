# agent/retry_executor.py
"""Retry Executor – executes payment retries via real Razorpay API.

Provides:
    - execute_retry() — Fetches transaction, executes payment retry/capture via Razorpay SDK/API,
                        updates attempt result, updates transaction status, and either schedules
                        the next retry attempt or escalates to customer notification if exhausted.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import razorpay
import razorpay.errors

from agent.action_engine import execute_customer_notification
from agent.db_writer import (
    get_db_session,
    update_transaction_status,
)
from agent.models import RetryAttempt, Transaction

_logger = logging.getLogger(__name__)


def _get_razorpay_client() -> razorpay.Client:
    """Instantiate a real Razorpay Client using credentials from environment."""
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()

    if not key_id or not key_secret:
        raise RuntimeError(
            "Missing Razorpay credentials in environment. "
            "Please ensure RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET are set."
        )

    return razorpay.Client(auth=(key_id, key_secret))


def execute_retry(retry_attempt: RetryAttempt) -> Dict[str, Any]:
    """Execute a payment recovery retry using real Razorpay API.

    Workflow:
    1. Fetch the transaction from database using retry_attempt.transaction_id.
    2. Call Razorpay API:
       - If payment is in authorized / capturable state or retry capture is invoked:
         Call client.payment.capture(payment_id, amount_in_paise) or client.payment.fetch(payment_id).
    3. If payment succeeds:
       - Update retry_attempt.result = "SUCCESS"
       - Update transaction.status = "recovered"
       - Log: "Payment recovered! tx='pay_xxx' amount=₹2499.00"
       - Return {"status": "recovered", "payment_id": "pay_xxx", "amount": 2499.00}
    4. If payment fails again:
       - Update retry_attempt.result = "FAILED"
       - Call schedule_retry() for next attempt if attempts remaining (< 3)
       - If no attempts remaining, call execute_customer_notification()
       - Return {"status": "failed", "reason": "...", "next_retry_at": "..."}

    Args:
        retry_attempt: The RetryAttempt instance to execute.

    Returns:
        Dictionary reporting the outcome.
    """
    from agent.retry_scheduler import schedule_retry, MAX_RETRIES

    tx_id = retry_attempt.transaction_id
    with get_db_session() as session:
        tx = session.query(Transaction).filter_by(id=tx_id).first()
        if not tx:
            # Fallback search by payment_id
            tx = session.query(Transaction).filter_by(razorpay_payment_id=tx_id).first()

        if not tx:
            raise ValueError(f"Transaction not found for ID: {tx_id}")

        # Extract values for thread-safe operations
        payment_id = tx.razorpay_payment_id
        amount = float(tx.amount)
        currency = tx.currency or "INR"
        amount_in_subunits = int(round(amount * 100))  # Convert to paise for INR

    _logger.info(
        "[RetryExecutor] Executing retry attempt #%d for tx='%s' (payment_id='%s', amount=%s %s)",
        retry_attempt.attempt_number,
        tx_id,
        payment_id,
        amount,
        currency,
    )

    client = _get_razorpay_client()
    payment_recovered = False
    failure_reason = ""
    is_transient_error = True

    try:
        # Call Real Razorpay API
        # Attempt to fetch payment state or trigger capture if authorized
        payment_data = client.payment.fetch(payment_id)
        current_status = str(payment_data.get("status", "")).lower()

        if current_status in ("captured", "paid"):
            payment_recovered = True
        elif current_status == "authorized":
            capture_res = client.payment.capture(payment_id, amount_in_subunits, {"currency": currency})
            if str(capture_res.get("status", "")).lower() == "captured":
                payment_recovered = True
            else:
                failure_reason = f"Payment capture status: {capture_res.get('status')}"
        else:
            failure_reason = payment_data.get("error_description") or f"Payment status is {current_status}"

    except (TimeoutError, Exception) as err:
        import requests
        err_msg = str(err)
        err_type = type(err).__name__

        if isinstance(err, (TimeoutError, requests.exceptions.Timeout)) or "timeout" in err_msg.lower():
            # Razorpay API timeout → log and mark retry as FAILED, schedule next attempt (transient)
            failure_reason = f"Razorpay API timeout: {err_msg}"
            is_transient_error = True
            _logger.warning("[RetryExecutor] Razorpay API timeout on tx='%s' (attempt #%d): %s", payment_id, retry_attempt.attempt_number, failure_reason)

        elif isinstance(err, razorpay.errors.BadRequestError) or "400" in err_msg or "bad_request" in err_msg.lower():
            # Razorpay 4xx error → log full error code and reason, do not retry
            failure_reason = f"Razorpay 4xx client error: {getattr(err, 'message', err_msg)}"
            is_transient_error = False
            _logger.warning(
                "[RetryExecutor] Razorpay 4xx non-retryable error on tx='%s' (type=%s): %s",
                payment_id,
                err_type,
                failure_reason,
            )

        elif isinstance(err, (razorpay.errors.ServerError, razorpay.errors.GatewayError)) or "500" in err_msg or "502" in err_msg or "503" in err_msg or "504" in err_msg:
            # Razorpay 5xx error → treat as transient, schedule retry
            failure_reason = f"Razorpay 5xx gateway/server error: {getattr(err, 'message', err_msg)}"
            is_transient_error = True
            _logger.warning("[RetryExecutor] Razorpay 5xx transient error on tx='%s': %s", payment_id, failure_reason)

        else:
            failure_reason = f"Unexpected error ({err_type}): {err_msg}"
            is_transient_error = False
            _logger.error("[RetryExecutor] Unexpected exception calling Razorpay API for tx='%s': %s", payment_id, err, exc_info=True)

    # ── Outcome Handling ─────────────────────────────────────────────────────

    if payment_recovered:
        # 1. Update retry_attempt result
        with get_db_session() as session:
            db_attempt = session.query(RetryAttempt).filter_by(id=retry_attempt.id).first()
            if db_attempt:
                db_attempt.result = "SUCCESS"
                retry_attempt.result = "SUCCESS"

        # 2. Update transaction status to "recovered"
        update_transaction_status(
            razorpay_payment_id=payment_id,
            new_status="recovered",
        )

        # 3. Log clearly as specified
        _logger.info("Payment recovered! tx='%s' amount=₹%.2f", payment_id, amount)

        result_dict = {
            "status": "recovered",
            "payment_id": payment_id,
            "amount": amount,
        }
        try:
            from agent.observability import record_retry_execution
            record_retry_execution(result_dict)
        except Exception:
            pass

        return result_dict

    else:
        # 1. Update retry_attempt result to FAILED
        with get_db_session() as session:
            db_attempt = session.query(RetryAttempt).filter_by(id=retry_attempt.id).first()
            if db_attempt:
                db_attempt.result = "FAILED"
                retry_attempt.result = "FAILED"

            # Check attempts count in DB
            attempts_count = session.query(RetryAttempt).filter_by(transaction_id=tx_id).count()

        next_retry_iso: Optional[str] = None

        # Only schedule next retry if the failure is transient AND attempts remaining (< MAX_RETRIES)
        if is_transient_error and attempts_count < MAX_RETRIES:
            with get_db_session() as session:
                active_tx = session.query(Transaction).filter_by(id=tx_id).first()
                # Schedule next retry using base backoff (900s)
                next_attempt = schedule_retry(active_tx, retry_after_seconds=900)
                if next_attempt and next_attempt.next_retry_at:
                    next_retry_iso = next_attempt.next_retry_at.isoformat()
        else:
            # Terminal error (4xx or exhausted retries) — notify customer immediately
            with get_db_session() as session:
                active_tx = session.query(Transaction).filter_by(id=tx_id).first()
                notification_message = (
                    f"Your payment of {currency} {amount:.2f} could not be completed ({failure_reason}). "
                    f"Please update your payment method or contact support."
                )
                execute_customer_notification(
                    transaction=active_tx,
                    message=notification_message,
                    channel="email",
                )
            update_transaction_status(
                razorpay_payment_id=payment_id,
                new_status="customer_notified",
                failure_reason=failure_reason,
            )

        result_dict = {
            "status": "failed",
            "reason": failure_reason,
            "next_retry_at": next_retry_iso,
        }
        try:
            from agent.observability import record_retry_execution
            record_retry_execution(result_dict)
        except Exception:
            pass

        return result_dict


__all__ = [
    "execute_retry",
]

