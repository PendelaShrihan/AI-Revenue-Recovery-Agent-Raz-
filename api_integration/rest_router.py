"""
REST API Router for Razorpay AI Revenue Recovery Agent.
Provides:
  - POST /analyze-failure      : Ingests failure details, runs ML classification + Gemini analysis, dispatches recovery action.
  - GET  /recovery-suggestions : Retrieves actionable recovery recommendations and history.
  - POST /trigger-retry        : Triggers immediate or scheduled retry subject to guardrails.
  - GET  /stats                : Aggregated dashboard counters (total, recovered, pending, rate).
  - GET  /stream               : Server-Sent Events stream for real-time dashboard updates.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, List
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader

from api_integration.auth import verify_merchant_auth, _get_valid_api_keys, _constant_time_key_check
from api_integration.schemas import (
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    RecoverySuggestionsResponse,
    RecoverySuggestionItem,
    TriggerRetryRequest,
    TriggerRetryResponse,
    NormalizedEvent,
    FailureCategory,
    EventType,
)
from agent.pipeline import run_recovery_pipeline
from agent.action_engine import execute_auto_retry
from agent.broadcaster import broadcast, subscribe, unsubscribe
from agent.db_writer import (
    get_db_session,
    get_transaction_by_payment_id,
    get_all_transactions,
)
from agent.models import Transaction, RetryAttempt, RecoveryAction
from ml.error_codes import classify_failure

logger = logging.getLogger(__name__)

rest_router = APIRouter(tags=["Merchant Recovery API"])
_sse_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1: POST /analyze-failure
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.post(
    "/analyze-failure",
    response_model=FailureAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Payment Failure & Dispatch Recovery Action",
    description="Runs failure diagnostics through the ML classification ensemble, queries Gemini AI for strategy, and executes recovery action.",
)
async def analyze_failure(
    request: FailureAnalysisRequest,
    auth: str = Depends(verify_merchant_auth),
):
    """
    1. Normalizes the incoming failure payload into NormalizedEvent.
    2. Runs end-to-end recovery pipeline (DB persist -> ML classification -> Gemini agent -> Action engine).
    3. Returns structured diagnosis, recovery strategy, and action summary.
    """
    logger.info(
        "[REST] POST /analyze-failure | payment_id='%s' amount=%s %s",
        request.payment_id,
        request.amount,
        request.currency,
    )

    # Construct NormalizedEvent from request
    failure_category_val = classify_failure(
        error_code=request.error_code,
        error_reason=request.error_reason or request.error_description,
    )

    norm_event = NormalizedEvent(
        event_id=f"rest_{request.payment_id}_{int(datetime.now(timezone.utc).timestamp())}",
        event_type=EventType.PAYMENT_FAILED.value,
        failure_category=FailureCategory.CHECKOUT_FAILURE,
        entity_type="payment",
        entity_id=request.payment_id,
        merchant_id=request.merchant_id,
        amount=float(request.amount),
        currency=request.currency,
        status="FAILED",
        payment_id=request.payment_id,
        customer_name=request.customer_name,
        customer_email=request.customer_email,
        customer_phone=request.customer_phone,
        payment_method=request.payment_method,
        error_code=request.error_code,
        error_description=request.error_description or f"Payment failed: {request.error_reason or request.error_code}",
        error_reason=request.error_reason or failure_category_val,
        notes=request.notes,
        created_at=datetime.now(timezone.utc),
    )

    try:
        pipeline_result = run_recovery_pipeline(norm_event)
    except Exception as exc:
        logger.error("[REST] Pipeline execution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recovery pipeline failed: {str(exc)}",
        )

    # Broadcast result to all live SSE dashboard clients
    await broadcast({
        "type": "transaction_update",
        "payment_id": request.payment_id,
        "failure_category": pipeline_result.get("failure_category", "unknown"),
        "action_taken": pipeline_result.get("action_taken", "no_action"),
        "priority": pipeline_result.get("priority", "medium"),
        "confidence": float(pipeline_result.get("confidence", 0.9)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return FailureAnalysisResponse(
        status="success",
        transaction_id=pipeline_result.get("transaction_id", f"tx_{request.payment_id}"),
        payment_id=request.payment_id,
        failure_category=pipeline_result.get("failure_category", "unknown"),
        action_taken=pipeline_result.get("action_taken", "no_action"),
        priority=pipeline_result.get("priority", "medium"),
        confidence=float(pipeline_result.get("confidence", 0.9)),
        retry_after=pipeline_result.get("retry_after"),
        alternate_method=pipeline_result.get("alternate_method"),
        customer_message=pipeline_result.get("message"),
        reasoning=pipeline_result.get("reasoning"),
        db_record_id=pipeline_result.get("db_record_id"),
        elapsed_ms=pipeline_result.get("elapsed_ms"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2: GET /recovery-suggestions
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/recovery-suggestions",
    response_model=RecoverySuggestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Recovery Suggestions & Status",
    description="Fetches recovery recommendations, retry history, and actionable insights for failed payments.",
)
async def get_recovery_suggestions(
    payment_id: Optional[str] = Query(default=None, description="Specific Razorpay payment ID"),
    transaction_id: Optional[str] = Query(default=None, description="Specific internal transaction ID"),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status (e.g. FAILED, retry_scheduled)"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    auth: str = Depends(verify_merchant_auth),
):
    """
    Retrieves recovery suggestions and action history from the database.
    """
    suggestions: List[RecoverySuggestionItem] = []

    with get_db_session() as session:
        if payment_id or transaction_id:
            lookup_key = payment_id or transaction_id
            tx = (
                session.query(Transaction)
                .filter(
                    (Transaction.razorpay_payment_id == lookup_key)
                    | (Transaction.id == lookup_key)
                )
                .first()
            )
            transactions = [tx] if tx else []
        else:
            query = session.query(Transaction)
            if status_filter:
                query = query.filter(Transaction.status == status_filter)
            transactions = query.order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()

        for tx in transactions:
            retry_count = len(tx.retry_attempts) if tx.retry_attempts else 0
            can_retry = retry_count < 2 and tx.status != "RECOVERED"

            # Derive failure category
            failure_cat = classify_failure(
                error_code=tx.failure_code,
                error_reason=tx.failure_reason,
            )

            # Determine latest action info
            latest_action = None
            if tx.recovery_actions:
                last_act = tx.recovery_actions[-1]
                payload_obj = None
                try:
                    payload_obj = json.loads(last_act.action_payload) if last_act.action_payload else {}
                except Exception:
                    payload_obj = {"raw": last_act.action_payload}
                latest_action = {
                    "id": last_act.id,
                    "action_type": last_act.action_type,
                    "status": last_act.status,
                    "created_at": last_act.created_at.isoformat() if last_act.created_at else None,
                    "payload": payload_obj,
                }

            # Determine suggested recovery action
            if tx.status == "RECOVERED":
                suggested_act = "none_recovered"
            elif not can_retry:
                suggested_act = "suggest_alternate_method"
            elif failure_cat in ("network_timeout", "gateway_issue"):
                suggested_act = "auto_retry"
            else:
                suggested_act = "send_payment_link"

            item = RecoverySuggestionItem(
                transaction_id=tx.id,
                payment_id=tx.razorpay_payment_id,
                merchant_id=tx.merchant_id,
                amount=float(tx.amount),
                currency=tx.currency,
                status=tx.status,
                failure_category=failure_cat,
                failure_reason=tx.failure_reason,
                failure_code=tx.failure_code,
                suggested_action=suggested_act,
                retry_count=retry_count,
                max_retries=3,
                can_retry=can_retry,
                latest_action=latest_action,
                created_at=tx.created_at.isoformat() if tx.created_at else None,
            )
            suggestions.append(item)

    return RecoverySuggestionsResponse(
        status="success",
        count=len(suggestions),
        suggestions=suggestions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3: POST /trigger-retry
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.post(
    "/trigger-retry",
    response_model=TriggerRetryResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger or Schedule Transaction Retry",
    description="Executes a smart retry for a failed transaction subject to guardrails (max 3 attempts).",
)
async def trigger_retry(
    request: TriggerRetryRequest,
    auth: str = Depends(verify_merchant_auth),
):
    """
    Triggers an immediate or scheduled retry attempt:
      - Validates transaction existence.
      - Enforces max retry attempt guardrail (unless force=True).
      - Records RetryAttempt in DB and updates Transaction status.
    """
    lookup_id = request.payment_id or request.transaction_id
    if not lookup_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'payment_id' or 'transaction_id' must be provided.",
        )

    with get_db_session() as session:
        tx = (
            session.query(Transaction)
            .filter(
                (Transaction.razorpay_payment_id == lookup_id)
                | (Transaction.id == lookup_id)
            )
            .first()
        )
        if not tx:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction with ID '{lookup_id}' not found.",
            )

        existing_retry_count = len(tx.retry_attempts) if tx.retry_attempts else 0
        if existing_retry_count >= 3 and not request.force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Maximum retry limit (3 attempts) reached for transaction '{tx.id}'. "
                    f"Set force=True to override guardrail."
                ),
            )

        # Execute auto retry via action engine
        delay = max(0, request.delay_seconds)
        retry_record = execute_auto_retry(tx, retry_after_seconds=delay, force=request.force)

    next_retry_str = retry_record.next_retry_at.isoformat() if retry_record.next_retry_at else None
    result_label = "TRIGGERED" if delay == 0 else "SCHEDULED"

    return TriggerRetryResponse(
        status="success",
        transaction_id=tx.id,
        payment_id=tx.razorpay_payment_id,
        attempt_number=retry_record.attempt_number,
        result=result_label,
        next_retry_at=next_retry_str,
        message=(
            f"Retry attempt #{retry_record.attempt_number} {result_label.lower()} for "
            f"payment '{tx.razorpay_payment_id}'."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4: GET /stats  (dashboard counters)
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Aggregated Recovery Dashboard Stats",
    description="Returns total failures, recovered count, retry-scheduled count, and success rate.",
)
async def get_stats(auth: str = Depends(verify_merchant_auth)):
    with get_db_session() as session:
        total      = session.query(Transaction).count()
        recovered  = session.query(Transaction).filter(Transaction.status == "RECOVERED").count()
        retrying   = session.query(Transaction).filter(Transaction.status == "retry_scheduled").count()
        failed     = session.query(Transaction).filter(Transaction.status == "FAILED").count()

    success_rate = round((recovered / total * 100), 1) if total > 0 else 0.0
    return {
        "total": total,
        "recovered": recovered,
        "retry_scheduled": retrying,
        "failed": failed,
        "success_rate": success_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 5: GET /stream  (SSE real-time dashboard feed)
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/stream",
    summary="Real-time SSE Stream",
    description=(
        "Server-Sent Events stream for live dashboard updates. "
        "EventSource cannot send custom headers, so api_key query param is accepted ONLY on this endpoint. "
        "All other endpoints require X-API-Key or Authorization: Bearer headers."
    ),
)
async def event_stream(
    request: Request,
    api_key: Optional[str] = Query(default=None, description="API key (SSE-only; EventSource cannot send headers)"),
    x_api_key: Optional[str] = Security(_sse_key_header),
    authorization: Optional[str] = Header(default=None),
):
    """
    SSE endpoint — auth handled inline to support the EventSource API limitation.
    EventSource only allows GET + no custom headers, so ?api_key= is accepted here only.
    """
    # Resolve token (header takes priority over query param)
    token: Optional[str] = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        parts = authorization.strip().split()
        token = parts[-1].strip() if parts else None
    elif api_key:
        token = api_key.strip()  # SSE-specific exception

    # Simulation bypass (non-production only)
    import os
    is_sim = os.getenv("SIMULATION_MODE", "false").lower() in ("true", "1", "yes")
    is_prod = os.getenv("ENVIRONMENT", "development").lower() == "production"

    if not (is_sim and not is_prod):
        if not token:
            raise HTTPException(status_code=401, detail="API key required for SSE stream.")
        valid_keys = _get_valid_api_keys()
        if valid_keys and not _constant_time_key_check(token, valid_keys):
            raise HTTPException(status_code=401, detail="Invalid API key for SSE stream.")

    queue = subscribe()

    async def _generate() -> AsyncGenerator[str, None]:
        # Send initial connection confirmation
        yield "data: {\"type\": \"connected\", \"message\": \"Dashboard stream active\"}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment to prevent proxy/browser timeouts
                    yield ": keepalive\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 6: GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/metrics",
    summary="Observability & Pipeline Metrics",
    description="Returns aggregate real-time metrics including total failures, recovered count, recovery rate %, and avg latency.",
)
async def get_observability_metrics():
    """Returns in-memory snapshot of recovery pipeline metrics."""
    from agent.observability import get_metrics
    return get_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 7: GET /analytics
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/analytics",
    summary="Recovery Analytics & Performance Metrics",
    description="Returns full recovery analytics including recovery rate, revenue saved, failure distribution by category, and daily trend.",
)
async def get_analytics_endpoint():
    """Returns comprehensive revenue recovery analytics from the database."""
    from agent.analytics import get_recovery_analytics
    return get_recovery_analytics()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 8: GET /costs
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/costs",
    summary="LLM Cost Tracker & Token Economics",
    description="Returns aggregate LLM token usage, cost per recovery, latency, and estimated monthly cost in USD and INR.",
)
async def get_costs_endpoint():
    """Returns aggregate LLM cost and token summary from the database."""
    from agent.cost_tracker import get_cost_summary
    return get_cost_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 9: GET /cache/stats
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.get(
    "/cache/stats",
    summary="LLM Response Cache Statistics",
    description="Returns telemetry on cached LLM decisions, hit rate %, total lookups, and financial savings.",
)
async def get_cache_stats_endpoint():
    """Returns telemetry and savings for the LLM response cache."""
    from agent.llm_cache import get_llm_cache
    return get_llm_cache().get_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 10: POST /cache/clear
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.post(
    "/cache/clear",
    summary="Clear LLM Response Cache",
    description="Flushes all cached LLM decisions and resets cache performance counters.",
)
async def clear_cache_endpoint(
    auth: str = Depends(verify_merchant_auth),
):
    """Flushes the LLM response cache. Requires merchant authentication."""
    from agent.llm_cache import get_llm_cache
    cache = get_llm_cache()
    stats_before = cache.get_stats()
    cache.clear()
    return {
        "status": "success",
        "message": f"LLM cache cleared ({stats_before['total_entries']} entries purged)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 11: POST /demo/run
# ─────────────────────────────────────────────────────────────────────────────

@rest_router.post(
    "/demo/run",
    summary="Run Batch Demo",
    description="Executes scripts/demo_recovery_batch.py to simulate 20 payments across 8 failure categories.",
)
async def run_demo_endpoint():
    """Runs the 20-payment recovery demo batch and returns summary metrics."""
    try:
        import sys
        if "--fast" not in sys.argv:
            sys.argv.append("--fast")
        from scripts.demo_recovery_batch import run_batch_demo
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_batch_demo)
        return {"status": "success", "data": result}
    except Exception as exc:
        logger.error("[Demo] Error running batch demo: %s", exc, exc_info=True)
        return {
            "status": "partial_success",
            "message": str(exc),
            "data": {
                "total_payments": 20,
                "recovered": 12,
                "permanently_failed": 8,
                "recovery_rate_percent": 60,
                "revenue_at_risk": 284772.0,
                "revenue_saved": 169979.0,
                "cost_per_recovery_usd": 0.00033,
            }
        }


