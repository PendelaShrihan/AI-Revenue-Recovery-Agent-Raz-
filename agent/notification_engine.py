# agent/notification_engine.py
"""Customer Notification & Communication Engine.

Provides multi-channel message drafting (WhatsApp, SMS, Email) for failed digital
payments, checkout drop-offs, mandate failures, and overdue receivables.

Key Capabilities:
- Generates context-aware, personalized messages using Gemini LLM (or intelligent fallbacks).
- Supports 3 primary channels:
    * WhatsApp: Rich formatting, emojis, clear CTA link, optional Hinglish tone.
    * SMS: Concise, DLT template compliant (<160 chars) with short link.
    * Email: Structured HTML/text with Subject line, salutation, diagnosis, and CTA.
- Personalization parameters:
    * Merchant Name / Brand
    * Formatted Amount (₹ / INR / currency)
    * Dynamic Razorpay Payment Link
    * Recommended Alternate Payment Method (UPI, Netbanking, Cards)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from agent.db_writer import (
    get_db_session,
    save_recovery_action,
    update_transaction_status,
)
from agent.models import RecoveryAction, Transaction

_logger = logging.getLogger(__name__)

# Canonical failure reasons mapped to default suggestions and human friendly reasons
FAILURE_GUIDANCE: Dict[str, Dict[str, str]] = {
    "insufficient_funds": {
        "reason_summary": "insufficient account balance",
        "default_method": "UPI or another bank account",
        "action_tip": "Complete payment instantly via UPI or choose another bank account.",
    },
    "card_blocked": {
        "reason_summary": "card blocked by your issuing bank",
        "default_method": "Netbanking or UPI",
        "action_tip": "Use Netbanking/UPI while you contact your bank to unblock the card.",
    },
    "network_timeout": {
        "reason_summary": "temporary bank network timeout",
        "default_method": "UPI or Instant Retry",
        "action_tip": "Our automated retry is processing, or you can complete it instantly using the link below.",
    },
    "gateway_issue": {
        "reason_summary": "temporary payment gateway glitch",
        "default_method": "UPI or Netbanking",
        "action_tip": "We have switched to an alternate gateway. Click below to retry securely.",
    },
    "expired_card": {
        "reason_summary": "card has expired or details need updating",
        "default_method": "UPI or an active Card",
        "action_tip": "Update your card details or pay seamlessly via UPI.",
    },
    "authentication_failed": {
        "reason_summary": "OTP / 3D-Secure authentication timeout",
        "default_method": "UPI 1-Click",
        "action_tip": "Avoid OTP delays by completing your payment with 1-click UPI.",
    },
    "limit_exceeded": {
        "reason_summary": "card transaction or daily limit exceeded",
        "default_method": "Netbanking or split payment",
        "action_tip": "Try Netbanking or use an alternate card with a higher transaction limit.",
    },
    "mandate_inactive": {
        "reason_summary": "auto-debit recurring mandate expired or paused",
        "default_method": "Mandate Re-authorization",
        "action_tip": "Re-authorize your payment mandate in 30 seconds to avoid service interruption.",
    },
    "unknown": {
        "reason_summary": "temporary technical issue",
        "default_method": "UPI or Netbanking",
        "action_tip": "Please use the secure link below to retry your payment.",
    },
}


def _format_currency(amount: float, currency: str = "INR") -> str:
    """Formats amount into standard currency representation."""
    if currency.upper() == "INR":
        return f"₹{amount:,.2f}"
    return f"{currency.upper()} {amount:,.2f}"


def draft_whatsapp_message(
    merchant_name: str,
    amount_str: str,
    payment_link: str,
    failure_category: str,
    alternate_method: Optional[str] = None,
    customer_name: Optional[str] = None,
) -> str:
    """Drafts a rich, personalized WhatsApp message for payment recovery."""
    info = FAILURE_GUIDANCE.get(failure_category, FAILURE_GUIDANCE["unknown"])
    method_text = alternate_method or info["default_method"]
    salutation = f"Hi {customer_name}," if customer_name else "Hello,"

    msg = (
        f"👋 {salutation}\n\n"
        f"Your payment of *{amount_str}* to *{merchant_name}* could not be completed due to {info['reason_summary']}.\n\n"
        f"💡 *Recommended Action:* {info['action_tip']}\n"
        f"⚡ *Suggested Method:* {method_text}\n\n"
        f"👉 Complete your payment securely here:\n{payment_link}\n\n"
        f"Need help? Reply directly to this message."
    )
    return msg.strip()


def draft_sms_message(
    merchant_name: str,
    amount_str: str,
    payment_link: str,
    failure_category: str,
) -> str:
    """Drafts a concise, DLT-compliant SMS message (<160 chars when possible)."""
    info = FAILURE_GUIDANCE.get(failure_category, FAILURE_GUIDANCE["unknown"])
    # SMS requires extreme conciseness
    return f"Payment of {amount_str} to {merchant_name} failed due to {info['reason_summary']}. Complete now: {payment_link} - {merchant_name}"


def draft_email_message(
    merchant_name: str,
    amount_str: str,
    payment_link: str,
    failure_category: str,
    alternate_method: Optional[str] = None,
    customer_name: Optional[str] = None,
) -> Dict[str, str]:
    """Drafts a structured email with Subject and Body."""
    info = FAILURE_GUIDANCE.get(failure_category, FAILURE_GUIDANCE["unknown"])
    method_text = alternate_method or info["default_method"]
    greeting = f"Dear {customer_name}," if customer_name else "Hello,"

    subject = f"Action Required: Complete your payment of {amount_str} to {merchant_name}"
    
    body = (
        f"{greeting}\n\n"
        f"We noticed that your recent payment of {amount_str} to {merchant_name} was unsuccessful "
        f"due to {info['reason_summary']}.\n\n"
        f"To ensure uninterrupted order fulfillment or access to your subscription, please use the secure link below to retry:\n\n"
        f"Payment Link: {payment_link}\n\n"
        f"Suggested Alternative: We recommend completing this via {method_text} for instant confirmation.\n\n"
        f"If you have already resolved this or have questions, please reach out to {merchant_name} support.\n\n"
        f"Warm regards,\n"
        f"{merchant_name} Payment Support (Powered by Razorpay AI Recovery)"
    )
    return {
        "subject": subject,
        "body": body.strip(),
    }


def generate_llm_notification_message(
    merchant_name: str,
    amount_str: str,
    payment_link: str,
    failure_category: str,
    channel: str,
    alternate_method: Optional[str] = None,
    customer_name: Optional[str] = None,
) -> Optional[str]:
    """Optionally queries Gemini LLM to draft an ultra-tailored notification message."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        prompt = (
            f"You are a professional customer retention copywriter for {merchant_name}. "
            f"Write a courteous, high-converting payment recovery notification for channel: {channel}.\n"
            f"Customer: {customer_name or 'Valued Customer'}\n"
            f"Amount: {amount_str}\n"
            f"Failure Reason Category: {failure_category}\n"
            f"Recommended Alternate Method: {alternate_method or 'UPI'}\n"
            f"Payment Link: {payment_link}\n\n"
            f"Constraints for channel '{channel}':\n"
            f"- If whatsapp: Use polite emojis, clear bullet points, call to action with link.\n"
            f"- If sms: Under 160 characters, direct and clear.\n"
            f"- If email: Return a professional subject line on line 1 formatted as 'Subject: <title>' followed by body.\n"
            f"Return plain text only."
        )

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        if resp and resp.text:
            return resp.text.strip()
    except Exception as exc:
        _logger.warning("[NotificationEngine] LLM message generation failed (%s). Falling back to template.", exc)

    return None


def generate_personalized_notification(
    transaction: Transaction,
    failure_category: str,
    channel: str = "whatsapp",
    merchant_name: Optional[str] = None,
    payment_link: Optional[str] = None,
    alternate_method: Optional[str] = None,
    customer_name: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Generates a complete personalized recovery message across requested or all channels.

    Args:
        transaction: The parent Transaction ORM instance.
        failure_category: Normalized error classification (e.g. 'insufficient_funds', 'network_timeout').
        channel: Target channel ('whatsapp', 'sms', 'email', or 'all').
        merchant_name: Name of merchant (defaults to transaction.merchant_id).
        payment_link: Razorpay payment link (defaults to dynamic URL).
        alternate_method: Recommended method override (e.g. 'UPI', 'Netbanking').
        customer_name: Customer's name if available.
        use_llm: If True, attempts LLM synthesis before template fallback.

    Returns:
        Structured notification package dict.
    """
    m_name = merchant_name or transaction.merchant_id or "Razorpay Merchant"
    amt_str = _format_currency(transaction.amount, transaction.currency or "INR")
    pay_url = payment_link or f"https://rzp.io/i/pay_{transaction.razorpay_payment_id}"
    cat = (failure_category or transaction.failure_code or "unknown").lower()

    whatsapp_msg = None
    sms_msg = None
    email_data = None

    if use_llm:
        llm_draft = generate_llm_notification_message(
            merchant_name=m_name,
            amount_str=amt_str,
            payment_link=pay_url,
            failure_category=cat,
            channel=channel,
            alternate_method=alternate_method,
            customer_name=customer_name,
        )
        if llm_draft:
            if channel == "whatsapp":
                whatsapp_msg = llm_draft
            elif channel == "sms":
                sms_msg = llm_draft
            elif channel == "email":
                lines = llm_draft.split("\n", 1)
                subject = lines[0].replace("Subject:", "").strip() if len(lines) > 1 else f"Complete your payment to {m_name}"
                body = lines[1].strip() if len(lines) > 1 else llm_draft
                email_data = {"subject": subject, "body": body}

    # Deterministic fallback / default generators
    if not whatsapp_msg:
        whatsapp_msg = draft_whatsapp_message(
            merchant_name=m_name,
            amount_str=amt_str,
            payment_link=pay_url,
            failure_category=cat,
            alternate_method=alternate_method,
            customer_name=customer_name,
        )

    if not sms_msg:
        sms_msg = draft_sms_message(
            merchant_name=m_name,
            amount_str=amt_str,
            payment_link=pay_url,
            failure_category=cat,
        )

    if not email_data:
        email_data = draft_email_message(
            merchant_name=m_name,
            amount_str=amt_str,
            payment_link=pay_url,
            failure_category=cat,
            alternate_method=alternate_method,
            customer_name=customer_name,
        )

    return {
        "transaction_id": transaction.id,
        "payment_id": transaction.razorpay_payment_id,
        "merchant_name": m_name,
        "amount": transaction.amount,
        "amount_formatted": amt_str,
        "payment_link": pay_url,
        "failure_category": cat,
        "alternate_method": alternate_method or FAILURE_GUIDANCE.get(cat, FAILURE_GUIDANCE["unknown"])["default_method"],
        "channels": {
            "whatsapp": whatsapp_msg,
            "sms": sms_msg,
            "email": email_data,
        },
        "selected_channel": channel,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def dispatch_customer_notification(
    transaction: Transaction,
    failure_category: str,
    channel: str = "whatsapp",
    merchant_name: Optional[str] = None,
    payment_link: Optional[str] = None,
    alternate_method: Optional[str] = None,
    customer_name: Optional[str] = None,
) -> RecoveryAction:
    """Generates personalized messages and logs a RecoveryAction in DB with full audit trail."""
    notif_data = generate_personalized_notification(
        transaction=transaction,
        failure_category=failure_category,
        channel=channel,
        merchant_name=merchant_name,
        payment_link=payment_link,
        alternate_method=alternate_method,
        customer_name=customer_name,
    )

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
        "[NotificationEngine] Logged customer notification action_id=%d for tx='%s' (channel=%s)",
        action.id,
        transaction.id,
        channel,
    )
    return action


__all__ = [
    "draft_whatsapp_message",
    "draft_sms_message",
    "draft_email_message",
    "generate_personalized_notification",
    "dispatch_customer_notification",
    "FAILURE_GUIDANCE",
]
