# agent/pipeline.py
"""Full Recovery Pipeline — single entry point for end-to-end payment recovery.

Provides:
    run_recovery_pipeline(normalized_event) — wires the complete flow:
        NormalizedEvent
            → DB save (db_writer.save_transaction)
            → ML classify (FeatureEngineeringPipeline + FailureClassifier)
            → Gemini decision (RecoveryEngine.process)
            → Action dispatch (action_engine.dispatch_recovery_action)
            → Structured summary dict returned

Logging:
    Every step is logged with a timestamp at INFO level using the format:
        [Pipeline][HH:MM:SS.mmm] Step N — <description> | result=<value>
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent.action_engine import dispatch_recovery_action
from agent.db_writer import save_transaction
from agent.models import Transaction
from agent.recovery_engine import RecoveryEngine
from api_integration.schemas import NormalizedEvent
from ml.feature_engineering import FeatureEngineeringPipeline
from ml.error_codes import classify_failure

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML model paths (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
_MODEL_DIR = os.getenv("MODEL_DIR", _DEFAULT_MODEL_DIR)

# Lazy-loaded singletons to avoid slow import costs at module load
_feature_pipeline: Optional[FeatureEngineeringPipeline] = None
_classifier = None
_recovery_engine: Optional[RecoveryEngine] = None


def _get_feature_pipeline() -> FeatureEngineeringPipeline:
    """Lazily load (or create) the FeatureEngineeringPipeline singleton."""
    global _feature_pipeline
    if _feature_pipeline is None:
        pipeline_path = os.path.join(_MODEL_DIR, "feature_pipeline.joblib")
        if os.path.exists(pipeline_path):
            _feature_pipeline = FeatureEngineeringPipeline.load(pipeline_path)
            _logger.debug("[Pipeline] FeatureEngineeringPipeline loaded from %s", pipeline_path)
        else:
            _feature_pipeline = FeatureEngineeringPipeline()
            _logger.debug("[Pipeline] FeatureEngineeringPipeline instantiated fresh (no saved pipeline found)")
    return _feature_pipeline


def _get_classifier():
    """Lazily load the FailureClassifier singleton from disk."""
    global _classifier
    if _classifier is None:
        from ml.classifier import FailureClassifier
        classifier_path = os.path.join(_MODEL_DIR, "failure_classifier.joblib")
        if os.path.exists(classifier_path):
            _classifier = FailureClassifier.load(_MODEL_DIR)
            _logger.debug("[Pipeline] FailureClassifier loaded from %s", classifier_path)
        else:
            _logger.warning(
                "[Pipeline] No trained FailureClassifier found at '%s'. "
                "Falling back to rule-based classify_failure().",
                classifier_path,
            )
            _classifier = None  # Will trigger rule-based fallback
    return _classifier


def _get_recovery_engine() -> RecoveryEngine:
    """Lazily initialise the RecoveryEngine (Gemini client) singleton."""
    global _recovery_engine
    if _recovery_engine is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        _recovery_engine = RecoveryEngine(model_name=model_name)
        _logger.debug("[Pipeline] RecoveryEngine initialised with model=%s", model_name)
    return _recovery_engine


# ---------------------------------------------------------------------------
# Step logger helper
# ---------------------------------------------------------------------------

def _log_step(step_num: int, description: str, result: Any = None) -> None:
    """Log a pipeline step with a precise timestamp."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
    if result is not None:
        _logger.info("[Pipeline][%s] Step %d — %s | result=%s", ts, step_num, description, result)
    else:
        _logger.info("[Pipeline][%s] Step %d — %s", ts, step_num, description)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_recovery_pipeline(
    normalized_event: NormalizedEvent,
    *,
    force_ml_category: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full end-to-end recovery pipeline for a single failed payment.

    Steps:
        1. Persist the NormalizedEvent to PostgreSQL via ``db_writer.save_transaction``.
        2. Classify failure category via ML model (with rule-based fallback).
        3. Get recovery decision from Gemini via ``RecoveryEngine``.
        4. Dispatch the decision to the correct action executor.

    Args:
        normalized_event: A ``NormalizedEvent`` produced by the webhook normaliser.
        force_ml_category: Optional override for ML category (for testing).

    Returns:
        Summary dict::

            {
                "transaction_id":    "tx_pay_xxx",
                "failure_category":  "insufficient_funds",
                "action_taken":      "suggest_alternate_method",
                "retry_after":       120,          # seconds (None if no retry)
                "alternate_method":  "upi",        # None if not applicable
                "priority":          "high",
                "message":           "<customer message>",
                "reasoning":         "<Gemini one-line reasoning>",
                "confidence":        0.92,
                "status":            "recovery_initiated",
                "db_record_id":      42,
            }

    Raises:
        RuntimeError: If the DB cannot be reached.
        GeminiAgentError: If Gemini API is unavailable after retries.
        GeminiOutputParseError: If Gemini returns malformed JSON.
    """
    pipeline_start = datetime.now(timezone.utc)
    payment_id = normalized_event.payment_id or normalized_event.entity_id

    _logger.info(
        "[Pipeline] ════════════════════════════════════════════════════════"
    )
    _logger.info(
        "[Pipeline] Starting recovery pipeline for payment_id='%s' amount=%s %s",
        payment_id,
        normalized_event.amount,
        normalized_event.currency,
    )

    # ── Step 1: Persist transaction ──────────────────────────────────────────
    _log_step(1, f"Saving transaction for payment_id='{payment_id}'")
    transaction, is_new = save_transaction(normalized_event)
    _log_step(
        1,
        "Transaction persisted",
        f"tx_id='{transaction.id}' is_new={is_new}",
    )

    # ── Step 2: ML failure classification ───────────────────────────────────
    _log_step(2, "Running ML failure classification")

    failure_category: str

    if force_ml_category:
        failure_category = force_ml_category
        _log_step(2, "ML category (forced override)", failure_category)
    else:
        try:
            classifier = _get_classifier()
            if classifier is not None:
                feat_pipeline = _get_feature_pipeline()
                record = {
                    "error_code": normalized_event.error_code,
                    "error_reason": normalized_event.error_reason,
                    "payment_method": normalized_event.payment_method,
                    "merchant_id": normalized_event.merchant_id,
                    "amount": normalized_event.amount,
                    "created_at": normalized_event.created_at,
                }
                features_df = feat_pipeline.extract_features([record])
                predictions = classifier.predict(features_df)
                failure_category = str(predictions[0])
                _log_step(2, "ML failure classification (ensemble model)", failure_category)
            else:
                # Rule-based fallback
                failure_category = classify_failure(
                    error_code=normalized_event.error_code,
                    error_reason=normalized_event.error_reason,
                )
                _log_step(2, "ML failure classification (rule-based fallback)", failure_category)
        except Exception as exc:
            _logger.warning(
                "[Pipeline] ML classification failed (%s). Falling back to rule-based.", exc
            )
            failure_category = classify_failure(
                error_code=normalized_event.error_code,
                error_reason=normalized_event.error_reason,
            )
            _log_step(2, "ML failure classification (exception fallback)", failure_category)

    # ── Step 3: Gemini recovery decision ────────────────────────────────────
    _log_step(3, f"Requesting Gemini recovery decision for failure_category='{failure_category}'")
    engine = _get_recovery_engine()
    decision = engine.process(normalized_event, ml_failure_category=failure_category)

    decision_dict = {
        "action": decision.action,
        "priority": decision.priority,
        "message": decision.message,
        "retry_after": decision.retry_after,
        "alternate_method": decision.alternate_method,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
    }
    _log_step(
        3,
        "Gemini decision received",
        f"action='{decision.action}' priority='{decision.priority}' confidence={decision.confidence:.2f}",
    )

    # ── Step 4: Dispatch recovery action ────────────────────────────────────
    _log_step(4, f"Dispatching action='{decision.action}' for tx='{transaction.id}'")
    dispatch_result = dispatch_recovery_action(decision_dict, transaction)
    _log_step(
        4,
        "Action dispatched",
        f"db_record_id={dispatch_result.get('db_record_id')} status='{dispatch_result.get('status')}'",
    )

    # ── Build summary ────────────────────────────────────────────────────────
    elapsed_ms = (datetime.now(timezone.utc) - pipeline_start).total_seconds() * 1000

    summary: Dict[str, Any] = {
        "transaction_id": transaction.id,
        "payment_id": payment_id,
        "failure_category": failure_category,
        "action_taken": decision.action,
        "retry_after": dispatch_result.get("retry_after"),
        "alternate_method": dispatch_result.get("alternate_method"),
        "priority": decision.priority,
        "message": decision.message,
        "reasoning": decision.reasoning,
        "confidence": decision.confidence,
        "db_record_id": dispatch_result.get("db_record_id"),
        "status": "recovery_initiated",
        "elapsed_ms": round(elapsed_ms, 1),
    }

    _logger.info(
        "[Pipeline] ✅ Pipeline completed in %.1fms | tx='%s' | action='%s' | category='%s'",
        elapsed_ms,
        transaction.id,
        decision.action,
        failure_category,
    )
    _logger.info(
        "[Pipeline] ════════════════════════════════════════════════════════"
    )

    return summary


# ---------------------------------------------------------------------------
# Pending Retries Runner
# ---------------------------------------------------------------------------

def run_pending_retries() -> Dict[str, int]:
    """Query and process all due scheduled retries from the database.

    Rules:
    - Queries all RetryAttempt records where result="SCHEDULED" and next_retry_at <= now
    - Calls retry_executor.execute_retry() for each one
    - Returns summary: {"processed": 5, "recovered": 3, "failed": 2}

    Returns:
        Dict with keys: "processed", "recovered", "failed"
    """
    from agent.db_writer import get_db_session
    from agent.models import RetryAttempt
    from agent.retry_executor import execute_retry

    now = datetime.now(timezone.utc)
    # Also support naive UTC comparison if DB stores naive timestamps
    now_naive = datetime.utcnow()

    _logger.info("[Pipeline] Checking for pending retries due at or before %s", now.isoformat())

    with get_db_session() as session:
        # Retrieve all pending scheduled attempts
        due_attempts = (
            session.query(RetryAttempt)
            .filter(
                RetryAttempt.result == "SCHEDULED",
                (RetryAttempt.next_retry_at <= now) | (RetryAttempt.next_retry_at <= now_naive),
            )
            .order_by(RetryAttempt.next_retry_at.asc())
            .all()
        )
        attempt_ids = [att.id for att in due_attempts]

    processed = 0
    recovered = 0
    failed = 0

    for att_id in attempt_ids:
        with get_db_session() as session:
            att = session.query(RetryAttempt).filter_by(id=att_id).first()
            if not att or att.result != "SCHEDULED":
                continue

        processed += 1
        try:
            res = execute_retry(att)
            if res.get("status") == "recovered":
                recovered += 1
            else:
                failed += 1
        except Exception as exc:
            _logger.error("[Pipeline] Error executing retry for attempt_id=%d: %s", att_id, exc, exc_info=True)
            failed += 1

    summary = {
        "processed": processed,
        "recovered": recovered,
        "failed": failed,
    }

    _logger.info(
        "[Pipeline] Pending retries execution completed: processed=%d, recovered=%d, failed=%d",
        processed,
        recovered,
        failed,
    )
    return summary


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "run_recovery_pipeline",
    "run_pending_retries",
]
