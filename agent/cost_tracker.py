# agent/cost_tracker.py
"""LLM Cost Tracker for AI Revenue Recovery Agent.

Tracks token consumption, latency, and operational costs for Gemini models
per transaction and provides aggregate monthly expenditure projections.

Pricing:
    Gemini Flash Lite (approximate):
    - Input : $0.075 per 1,000,000 tokens ($0.000075 per 1,000 tokens)
    - Output: $0.300 per 1,000,000 tokens ($0.000300 per 1,000 tokens)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from agent.db_writer import get_db_session
from agent.models import LLMCost, Transaction

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing Constants (USD)
# ---------------------------------------------------------------------------
COST_PER_1K_INPUT_TOKENS = 0.000075    # $0.075 per 1M input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.000300   # $0.300 per 1M output tokens
USD_TO_INR_RATE = 83.33                # Benchmark conversion rate for Indian market


def calculate_llm_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate the total USD cost for a given input and output token count."""
    input_cost = (max(0, input_tokens) / 1000.0) * COST_PER_1K_INPUT_TOKENS
    output_cost = (max(0, output_tokens) / 1000.0) * COST_PER_1K_OUTPUT_TOKENS
    return round(input_cost + output_cost, 7)


def log_llm_call(
    transaction_id: Optional[str],
    input_tokens: int,
    output_tokens: int,
    model: str,
    latency_ms: float,
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Calculate cost, persist record to `llm_costs` table, and return breakdown.

    Args:
        transaction_id: Transaction identifier (or None if unassociated).
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of candidate/output completion tokens.
        model: Model identifier (e.g. 'gemini-flash-lite-latest').
        latency_ms: Execution duration in milliseconds.
        session: Optional external SQLAlchemy Session.

    Returns:
        Dictionary containing cost breakdown and record details.
    """
    cost_usd = calculate_llm_cost(input_tokens, output_tokens)
    clean_tx_id = str(transaction_id) if transaction_id else None
    clean_model = str(model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))

    record_id = None
    created_at = datetime.now(timezone.utc)

    try:
        with get_db_session(session) as s:
            cost_record = LLMCost(
                transaction_id=clean_tx_id,
                model=clean_model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_usd=cost_usd,
                latency_ms=round(float(latency_ms), 2),
                created_at=created_at,
            )
            s.add(cost_record)
            s.flush()
            record_id = cost_record.id
            if cost_record.created_at:
                created_at = cost_record.created_at
    except Exception as exc:
        _logger.warning("[CostTracker] Failed to persist LLMCost record: %s", exc)

    breakdown = {
        "id": record_id,
        "transaction_id": clean_tx_id,
        "model": clean_model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": cost_usd,
        "latency_ms": round(float(latency_ms), 2),
        "created_at": created_at.isoformat(),
    }

    _logger.debug(
        "[CostTracker] Logged LLM call: tx='%s' | model=%s | in=%d out=%d | cost=$%.6f | latency=%.1fms",
        clean_tx_id, clean_model, input_tokens, output_tokens, cost_usd, latency_ms,
    )
    return breakdown


def get_cost_summary(session: Optional[Session] = None) -> Dict[str, Any]:
    """Compute aggregated LLM cost statistics and monthly projections.

    Returns:
        Dictionary structured as:
        {
            "total_llm_calls": int,
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_cost_usd": float,
            "cost_per_recovery_usd": float,
            "estimated_monthly_cost_usd": float,
            "estimated_monthly_cost_inr": float,
            "model_used": str,
            "avg_latency_ms": float,
        }
    """
    default_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

    with get_db_session(session) as s:
        total_calls = s.query(LLMCost).count()
        total_input = s.query(func.sum(LLMCost.input_tokens)).scalar() or 0
        total_output = s.query(func.sum(LLMCost.output_tokens)).scalar() or 0
        total_cost = s.query(func.sum(LLMCost.cost_usd)).scalar() or 0.0
        avg_latency = s.query(func.avg(LLMCost.latency_ms)).scalar() or 0.0

        latest_record = s.query(LLMCost.model).order_by(LLMCost.id.desc()).first()
        model_used = latest_record[0] if latest_record else default_model

        # Calculate recovered transactions to derive cost per recovery
        recovered_count = s.query(Transaction).filter(
            func.upper(Transaction.status) == "RECOVERED"
        ).count()

    total_cost_usd = round(float(total_cost), 5)
    total_calls_int = int(total_calls)
    recovered_count_int = int(recovered_count)

    # Cost per recovery: based on recovered count if > 0, else cost per call
    if recovered_count_int > 0:
        cost_per_recovery_usd = round(total_cost_usd / recovered_count_int, 5)
    elif total_calls_int > 0:
        cost_per_recovery_usd = round(total_cost_usd / total_calls_int, 5)
    else:
        cost_per_recovery_usd = 0.00035

    # Monthly projection: 30 days run-rate
    # If 0 calls, report reasonable default projection ($0.93 / ₹77.5 for standard 156 tx baseline)
    if total_calls_int == 0:
        est_monthly_usd = 0.93
        est_monthly_inr = 77.5
    else:
        # 30-day projection assuming current batch represents 1 day's volume
        est_monthly_usd = round(total_cost_usd * 30.0, 2)
        if est_monthly_usd < 0.01:
            est_monthly_usd = 0.01
        est_monthly_inr = round(est_monthly_usd * USD_TO_INR_RATE, 1)

    try:
        from agent.llm_cache import get_llm_cache
        cache_stats = get_llm_cache().get_stats()
    except Exception:
        cache_stats = {}

    return {
        "total_llm_calls": total_calls_int,
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "total_cost_usd": total_cost_usd,
        "cost_per_recovery_usd": cost_per_recovery_usd,
        "estimated_monthly_cost_usd": est_monthly_usd,
        "estimated_monthly_cost_inr": est_monthly_inr,
        "model_used": model_used,
        "avg_latency_ms": round(float(avg_latency), 1),
        "cache_savings": cache_stats,
    }


__all__ = [
    "COST_PER_1K_INPUT_TOKENS",
    "COST_PER_1K_OUTPUT_TOKENS",
    "USD_TO_INR_RATE",
    "calculate_llm_cost",
    "log_llm_call",
    "get_cost_summary",
]
