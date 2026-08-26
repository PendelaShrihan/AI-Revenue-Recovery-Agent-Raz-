"""
Razorpay Webhook Payload Normalizer.
Converts heterogeneous raw Razorpay webhook payloads into standard NormalizedEvent instances.
"""

from datetime import datetime, timezone
import uuid
import logging
from typing import Dict, Any, Optional

from api_integration.schemas import (
    NormalizedEvent,
    EventType,
    FailureCategory
)

logger = logging.getLogger(__name__)


def _parse_timestamp(ts: Optional[Any]) -> datetime:
    """Safely converts an epoch timestamp (int/float) or ISO string to UTC datetime."""
    if ts is None:
        return datetime.now(timezone.utc)
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _to_rupees(paise_amount: Optional[Any]) -> float:
    """Converts Razorpay amount in paise to standard currency units (INR)."""
    if paise_amount is None:
        return 0.0
    try:
        return round(float(paise_amount) / 100.0, 2)
    except (ValueError, TypeError):
        return 0.0


def normalize_webhook_payload(raw_payload: Dict[str, Any]) -> NormalizedEvent:
    """
    Normalizes raw Razorpay webhook JSON into a unified NormalizedEvent.

    Supports:
    1. payment.failed -> One-time checkout drop / card decline / OTP freeze
    2. subscription.halted -> Recurring auto-debit / mandate drop
    3. invoice.overdue -> B2B commercial invoice overdue
    4. Informational events -> payment.authorized, order.paid

    Raises:
        ValueError: If payload is missing essential webhook envelope attributes.
    """
    if not isinstance(raw_payload, dict):
        raise ValueError("Invalid payload: Webhook body must be a JSON object.")

    event_type = raw_payload.get("event")
    if not event_type:
        raise ValueError("Malformed Razorpay webhook payload: missing 'event' field.")

    payload_container = raw_payload.get("payload")
    if not isinstance(payload_container, dict):
        raise ValueError("Malformed Razorpay webhook payload: missing or invalid 'payload' object.")

    merchant_id = (
        raw_payload.get("account_id")
        or raw_payload.get("merchant_id")
        or "unknown_merchant"
    )
    event_timestamp = _parse_timestamp(raw_payload.get("created_at"))
    synthetic_event_id = raw_payload.get("id") or f"evt_{uuid.uuid4().hex[:16]}"

    # 1. Handle payment.failed
    if event_type == EventType.PAYMENT_FAILED.value:
        payment_data = payload_container.get("payment", {})
        entity = payment_data.get("entity", {})
        if not entity:
            raise ValueError("Malformed 'payment.failed' payload: missing 'payload.payment.entity'.")

        payment_id = entity.get("id") or f"pay_unknown_{uuid.uuid4().hex[:8]}"
        amount_in_rupees = _to_rupees(entity.get("amount"))
        notes = entity.get("notes") or {}
        if not isinstance(notes, dict):
            notes = {"raw_notes": str(notes)}

        return NormalizedEvent(
            event_id=synthetic_event_id,
            event_type=event_type,
            failure_category=FailureCategory.CHECKOUT_FAILURE,
            entity_type="payment",
            entity_id=payment_id,
            merchant_id=merchant_id,
            amount=amount_in_rupees,
            currency=entity.get("currency", "INR"),
            status="FAILED",
            payment_id=payment_id,
            order_id=entity.get("order_id"),
            invoice_id=entity.get("invoice_id"),
            customer_id=entity.get("customer_id"),
            customer_name=entity.get("notes", {}).get("customer_name") or entity.get("name"),
            customer_email=entity.get("email"),
            customer_phone=entity.get("contact"),
            payment_method=entity.get("method"),
            error_code=entity.get("error_code") or "PAYMENT_FAILED",
            error_description=entity.get("error_description"),
            error_source=entity.get("error_source"),
            error_step=entity.get("error_step"),
            error_reason=entity.get("error_reason"),
            notes=notes,
            created_at=_parse_timestamp(entity.get("created_at")) if entity.get("created_at") else event_timestamp,
            raw_payload=raw_payload
        )

    # 2. Handle subscription.halted
    elif event_type == EventType.SUBSCRIPTION_HALTED.value:
        sub_data = payload_container.get("subscription", {})
        sub_entity = sub_data.get("entity", {})
        if not sub_entity:
            raise ValueError("Malformed 'subscription.halted' payload: missing 'payload.subscription.entity'.")

        sub_id = sub_entity.get("id") or f"sub_unknown_{uuid.uuid4().hex[:8]}"
        notes = sub_entity.get("notes") or {}
        if not isinstance(notes, dict):
            notes = {"raw_notes": str(notes)}

        # Subscription halted events frequently bundle the failed payment entity
        payment_data = payload_container.get("payment", {})
        payment_entity = payment_data.get("entity", {}) if isinstance(payment_data, dict) else {}

        payment_id = payment_entity.get("id")
        amount_paise = payment_entity.get("amount") or sub_entity.get("plan", {}).get("item", {}).get("amount") or 0
        amount_in_rupees = _to_rupees(amount_paise)

        error_code = payment_entity.get("error_code") or "MANDATE_HALTED"
        error_description = (
            payment_entity.get("error_description")
            or f"Subscription {sub_id} halted after recurring mandate debit failures."
        )
        error_reason = payment_entity.get("error_reason") or "mandate_halted"

        if sub_entity.get("short_url"):
            notes["short_url"] = sub_entity.get("short_url")

        return NormalizedEvent(
            event_id=synthetic_event_id,
            event_type=event_type,
            failure_category=FailureCategory.MANDATE_FAILURE,
            entity_type="subscription",
            entity_id=sub_id,
            merchant_id=merchant_id,
            amount=amount_in_rupees,
            currency=payment_entity.get("currency") or "INR",
            status="HALTED",
            payment_id=payment_id,
            subscription_id=sub_id,
            customer_id=sub_entity.get("customer_id") or payment_entity.get("customer_id"),
            customer_name=payment_entity.get("name") or notes.get("customer_name"),
            customer_email=payment_entity.get("email"),
            customer_phone=payment_entity.get("contact"),
            payment_method=payment_entity.get("method") or sub_entity.get("payment_method"),
            error_code=error_code,
            error_description=error_description,
            error_source=payment_entity.get("error_source") or "mandate",
            error_step=payment_entity.get("error_step") or "recurring_charge",
            error_reason=error_reason,
            notes=notes,
            created_at=_parse_timestamp(sub_entity.get("charge_at") or sub_entity.get("created_at")) or event_timestamp,
            raw_payload=raw_payload
        )

    # 3. Handle invoice.overdue and invoice.expired (B2B overdue receivables)
    elif event_type in (EventType.INVOICE_OVERDUE.value, EventType.INVOICE_EXPIRED.value):
        invoice_data = payload_container.get("invoice", {})
        entity = invoice_data.get("entity", {})
        if not entity:
            raise ValueError(f"Malformed '{event_type}' payload: missing 'payload.invoice.entity'.")

        invoice_id = entity.get("id") or f"inv_unknown_{uuid.uuid4().hex[:8]}"
        customer_details = entity.get("customer_details") or {}
        if not isinstance(customer_details, dict):
            customer_details = {}

        amount_due = entity.get("amount_due") if entity.get("amount_due") is not None else entity.get("amount")
        amount_in_rupees = _to_rupees(amount_due)

        notes = entity.get("notes") or {}
        if not isinstance(notes, dict):
            notes = {"raw_notes": str(notes)}

        if entity.get("short_url"):
            notes["invoice_url"] = entity.get("short_url")

        return NormalizedEvent(
            event_id=synthetic_event_id,
            event_type=event_type,
            failure_category=FailureCategory.INVOICE_OVERDUE,
            entity_type="invoice",
            entity_id=invoice_id,
            merchant_id=merchant_id,
            amount=amount_in_rupees,
            currency=entity.get("currency", "INR"),
            status="OVERDUE",
            order_id=entity.get("order_id"),
            invoice_id=invoice_id,
            customer_id=entity.get("customer_id"),
            customer_name=customer_details.get("name"),
            customer_email=customer_details.get("email") or entity.get("email"),
            customer_phone=customer_details.get("contact") or entity.get("contact"),
            payment_method=None,
            error_code="INVOICE_OVERDUE",
            error_description=f"Commercial invoice {invoice_id} is overdue with unpaid balance of ₹{amount_in_rupees:.2f}",
            error_source="customer",
            error_step="receivable_collection",
            error_reason="invoice_overdue",
            notes=notes,
            created_at=_parse_timestamp(entity.get("date") or entity.get("created_at")) or event_timestamp,
            raw_payload=raw_payload
        )

    # 4. Handle other informational / success events (order.paid, payment.authorized, etc.)
    else:
        # Generic fallback normalization
        primary_entity_key = next(iter(payload_container.keys()), "unknown")
        entity = payload_container.get(primary_entity_key, {}).get("entity", {})

        entity_id = entity.get("id") or f"ent_{uuid.uuid4().hex[:8]}"
        amount_in_rupees = _to_rupees(entity.get("amount"))
        notes = entity.get("notes") or {}
        if not isinstance(notes, dict):
            notes = {"raw_notes": str(notes)}

        status_str = str(entity.get("status", "RECEIVED")).upper()

        return NormalizedEvent(
            event_id=synthetic_event_id,
            event_type=event_type,
            failure_category=FailureCategory.INFORMATIONAL,
            entity_type=primary_entity_key,
            entity_id=entity_id,
            merchant_id=merchant_id,
            amount=amount_in_rupees,
            currency=entity.get("currency", "INR"),
            status=status_str,
            payment_id=entity.get("id") if primary_entity_key == "payment" else None,
            order_id=entity.get("order_id"),
            customer_id=entity.get("customer_id"),
            customer_email=entity.get("email"),
            customer_phone=entity.get("contact"),
            payment_method=entity.get("method"),
            notes=notes,
            created_at=_parse_timestamp(entity.get("created_at")) or event_timestamp,
            raw_payload=raw_payload
        )
