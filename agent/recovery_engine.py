# agent/recovery_engine.py
"""Recovery Engine – orchestrates NormalizedEvent → ML classifier → Gemini → action.

This module is the single entry-point for all payment recovery decisions.
It takes a ``NormalizedEvent``, optionally accepts an ML-classifier-predicted
failure category override, calls the ``GeminiAgent``, and returns a
``RecoveryDecision`` ready to be persisted or acted on.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from api_integration.schemas import NormalizedEvent, FailureCategory
from agent.llm_agent import (
    GeminiAgent,
    GeminiAgentError,
    GeminiOutputParseError,
    RecoveryDecision,
    build_cot_prompt,
)
from agent.llm_cache import LLMResponseCache, get_llm_cache
from ml.feature_engineering import _derive_merchant_category

_logger = logging.getLogger(__name__)


class RecoveryEngine:
    """Orchestrates the full recovery pipeline for a single failed payment event.

    Usage::

        engine = RecoveryEngine()
        decision = engine.process(event, ml_failure_category="insufficient_funds")
        print(decision.action, decision.message)

    Args:
        model_name: Gemini model name (default: ``gemini-flash-lite-latest``).
        cache: Optional LLMResponseCache instance (defaults to global singleton).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache: Optional[LLMResponseCache] = None,
    ) -> None:
        effective_model = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self._agent = GeminiAgent(model_name=effective_model)
        self._cache = cache if cache is not None else get_llm_cache()
        _logger.debug("RecoveryEngine initialised (model=%s, cache_enabled=%s)", effective_model, self._cache is not None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        event: NormalizedEvent,
        ml_failure_category: Optional[str] = None,
        use_cache: bool = True,
    ) -> RecoveryDecision:
        """Produce a recovery decision for *event*.

        If caching is active and a prior decision with the same failure type and
        merchant category exists, it is reused without an external LLM call.

        Args:
            event: Normalised payment failure event.
            ml_failure_category: Optional failure category string from the ML model
                (e.g. ``"insufficient_funds"``). When provided, this is injected
                into the CoT prompt in place of the enum-level failure_category.
            use_cache: If True, look up and write to the LLMResponseCache.

        Returns:
            A validated ``RecoveryDecision``.

        Raises:
            GeminiAgentError: Gemini API unreachable after retries.
            GeminiOutputParseError: Gemini response did not match expected schema.
        """
        base_failure_type = ml_failure_category or event.error_reason or str(
            event.failure_category.value if hasattr(event.failure_category, "value") else event.failure_category
        )

        is_sub = bool(
            event.subscription_id
            or event.entity_type == "subscription"
            or (event.event_type and "subscription" in str(event.event_type))
        )
        is_inv = bool(
            event.invoice_id
            or event.entity_type == "invoice"
            or (event.event_type and "invoice" in str(event.event_type))
        )
        stream_tag = ":subscription" if is_sub else (":invoice" if is_inv else "")
        effective_failure_type = f"{base_failure_type}{stream_tag}"

        merchant_cat = None
        if isinstance(event.notes, dict):
            merchant_cat = event.notes.get("merchant_category")
        if not merchant_cat:
            merchant_cat = getattr(event, "merchant_category", None)
        if not merchant_cat:
            merchant_cat = _derive_merchant_category(event.merchant_id)

        _logger.debug(
            "[RecoveryEngine] Processing event_id=%s failure_type=%s merchant_cat=%s amount=%s %s",
            event.event_id,
            effective_failure_type,
            merchant_cat,
            event.amount,
            event.currency,
        )

        # ── Step 1: Cache Lookup ───────────────────────────────────────────────
        if use_cache and self._cache is not None:
            cached_decision = self._cache.get(effective_failure_type, merchant_cat)
            if cached_decision is not None:
                _logger.info(
                    "[RecoveryEngine] Returning CACHED decision for failure_type='%s' merchant_category='%s' (action=%s)",
                    effective_failure_type, merchant_cat, cached_decision.action,
                )
                return cached_decision

        # ── Step 2: Prompt Construction & LLM Invocation ───────────────────────
        prompt_event = event
        if ml_failure_category is not None:
            prompt_event = event.model_copy(deep=False)
            object.__setattr__(prompt_event, "_ml_failure_category", ml_failure_category)

        prompt = build_cot_prompt(prompt_event, ml_failure_category=ml_failure_category)
        tx_id = event.payment_id or event.entity_id
        decision = self._agent.generate(prompt, transaction_id=tx_id)

        # ── Step 3: Populate Cache on Success ──────────────────────────────────
        if use_cache and self._cache is not None:
            self._cache.set(effective_failure_type, merchant_cat, decision)

        _logger.debug(
            "[RecoveryEngine] Decision for event_id=%s: action=%s priority=%s confidence=%.2f",
            event.event_id,
            decision.action,
            decision.priority,
            decision.confidence,
        )
        return decision


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_recovery_decision(
    event: NormalizedEvent,
    ml_failure_category: Optional[str] = None,
    model_name: Optional[str] = None,
    use_cache: bool = True,
    cache: Optional[LLMResponseCache] = None,
) -> RecoveryDecision:
    """One-shot convenience wrapper around ``RecoveryEngine.process``.

    Args:
        event: Normalised payment failure event.
        ml_failure_category: Optional ML classifier output.
        model_name: Gemini model name.
        use_cache: Whether to check/populate the response cache.
        cache: Optional custom cache instance.

    Returns:
        A validated ``RecoveryDecision``.
    """
    engine = RecoveryEngine(model_name=model_name, cache=cache)
    return engine.process(event, ml_failure_category=ml_failure_category, use_cache=use_cache)


__all__ = [
    "RecoveryEngine",
    "get_recovery_decision",
]
