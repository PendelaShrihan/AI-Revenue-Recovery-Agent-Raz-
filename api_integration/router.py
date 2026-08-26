"""
FastAPI Webhook Router for Razorpay Ingestion.
Exposes POST /webhooks/razorpay, verifies HMAC-SHA256 signatures, normalizes payloads, and routes events.
"""

import os
import json
import logging
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from api_integration.verifier import verify_webhook_signature
from api_integration.normalizer import normalize_webhook_payload
from api_integration.schemas import WebhookResponse, EventType, FailureCategory

logger = logging.getLogger(__name__)

webhook_router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@webhook_router.get(
    "/razorpay",
    summary="Webhook Endpoint Status",
    description="Browser info endpoint displaying webhook listener status."
)
async def get_webhook_info():
    """Returns status and instructions when visited via a browser (GET request)."""
    return {
        "status": "online",
        "listener": "Razorpay Webhook Ingestion Engine",
        "expected_method": "POST",
        "supported_events": [
            "payment.failed",
            "subscription.halted",
            "invoice.overdue",
            "invoice.expired"
        ],
        "message": "Webhook listener is healthy and ready to receive POST events from Razorpay."
    }


@webhook_router.post(
    "/razorpay",
    response_model=WebhookResponse,
    status_code=status.HTTP_200_OK,
    summary="Receive and route Razorpay Webhooks",
    description="Validates cryptographic HMAC-SHA256 signature and normalizes/routes failure events."
)
async def handle_razorpay_webhook(request: Request):
    """
    Core Razorpay Webhook Ingestion Listener.
    1. Extracts raw request body bytes for HMAC-SHA256 signature verification.
    2. Validates X-Razorpay-Signature against configured RAZORPAY_WEBHOOK_SECRET.
    3. Normalizes raw payload into internal NormalizedEvent model.
    4. Routes the 3 core failure types (payment.failed, subscription.halted, invoice.overdue).
    """
    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "").strip()
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "").strip()
    simulation_mode = os.getenv("SIMULATION_MODE", "false").lower() in ("true", "1", "yes")

    # Signature verification logic:
    # 1. In Simulation Mode without a signature header: allow developer testing (Swagger UI / curl / Postman)
    if simulation_mode and not signature:
        logger.info("Simulation mode active: Webhook signature verification bypassed.")
    else:
        # Strict verification: Webhook secret and matching signature are mandatory
        if not webhook_secret:
            logger.error("RAZORPAY_WEBHOOK_SECRET is not configured on the server.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Webhook secret is not configured on the server."
            )
        if not signature or not verify_webhook_signature(raw_body, signature, webhook_secret):
            logger.warning("Rejected webhook: Invalid or missing X-Razorpay-Signature.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or missing X-Razorpay-Signature header."
            )

    # Parse JSON payload
    try:
        payload_data = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to decode webhook JSON: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed JSON body."
        )

    # Normalize payload into canonical NormalizedEvent
    try:
        normalized_event = normalize_webhook_payload(payload_data)
    except ValueError as val_err:
        logger.error(f"Payload normalization failed: {str(val_err)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Payload normalization error: {str(val_err)}"
        )

    # Persist transaction to database (PostgreSQL)
    try:
        from agent.db_writer import save_transaction
        save_transaction(normalized_event)
    except Exception as db_err:
        logger.error(f"Database persistence error in webhook handler: {db_err}", exc_info=True)

    # Route event based on type and category
    event_type = normalized_event.event_type
    action_taken = "event_acknowledged"
    message = f"Event '{event_type}' processed successfully."

    if event_type == EventType.PAYMENT_FAILED.value:
        action_taken = "payment_failure_routed"
        message = (
            f"Payment failure '{normalized_event.entity_id}' (₹{normalized_event.amount:.2f}) "
            f"captured and routed to LLM diagnostic & recovery pipeline."
        )
        logger.info(message)

    elif event_type == EventType.SUBSCRIPTION_HALTED.value:
        action_taken = "subscription_halted_routed"
        message = (
            f"Subscription halt '{normalized_event.entity_id}' (₹{normalized_event.amount:.2f}) "
            f"captured and routed to mandate recovery pipeline."
        )
        logger.info(message)

    elif event_type in (EventType.INVOICE_OVERDUE.value, EventType.INVOICE_EXPIRED.value):
        action_taken = "invoice_overdue_routed"
        message = (
            f"Overdue invoice '{normalized_event.entity_id}' (₹{normalized_event.amount:.2f}) "
            f"captured and routed to B2B invoice chaser pipeline."
        )
        logger.info(message)

    elif normalized_event.failure_category == FailureCategory.INFORMATIONAL:
        action_taken = "informational_event_recorded"
        message = f"Informational event '{event_type}' received and recorded."
        logger.info(message)

    else:
        action_taken = "unhandled_event_ignored"
        message = f"Unhandled event type '{event_type}' received; no recovery action taken."
        logger.warning(message)

    return WebhookResponse(
        status="success",
        event_id=normalized_event.event_id,
        event_type=normalized_event.event_type,
        action_taken=action_taken,
        message=message,
        normalized_event=normalized_event
    )
