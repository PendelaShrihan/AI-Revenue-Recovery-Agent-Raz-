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

_logger = logging.getLogger(__name__)


class RecoveryEngine:
    """Orchestrates the full recovery pipeline for a single failed payment event.

    Usage::

        engine = RecoveryEngine()
        decision = engine.process(event, ml_failure_category="insufficient_funds")
        print(decision.action, decision.message)

    Args:
        model_name: Gemini model name (default: ``gemini-3.5-flash-lite``).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        effective_model = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self._agent = GeminiAgent(model_name=effective_model)
        _logger.debug("RecoveryEngine initialised (model=%s)", effective_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        event: NormalizedEvent,
        ml_failure_category: Optional[str] = None,
    ) -> RecoveryDecision:
        """Produce a recovery decision for *event*.

        If *ml_failure_category* is provided (output from the ML classifier), it is
        injected into the CoT prompt as the ``failure_category`` field. The ML
        classifier emits raw reason strings such as ``"insufficient_funds"`` or
        ``"network_timeout"`` — these are **different** from the ``FailureCategory``
        enum values (which are stream-level categories like ``checkout_failure``).
        We therefore pass the ML string directly through to the prompt without
        coercing it through the enum.

        Args:
            event: Normalised payment failure event.
            ml_failure_category: Optional failure category string from the ML model
                (e.g. ``"insufficient_funds"``). When provided, this is injected
                into the CoT prompt in place of the enum-level failure_category.

        Returns:
            A validated ``RecoveryDecision``.

        Raises:
            GeminiAgentError: Gemini API unreachable after retries.
            GeminiOutputParseError: Gemini response did not match expected schema.
        """
        _logger.debug(
            "[RecoveryEngine] Processing event_id=%s failure_category=%s ml_override=%s amount=%s %s",
            event.event_id,
            event.failure_category,
            ml_failure_category,
            event.amount,
            event.currency,
        )

        # Build the prompt, optionally injecting the ML raw category string.
        # build_cot_prompt reads event.failure_category; we pass the ML string
        # as a side-channel by monkey-patching a copy of the event's attribute
        # so the prompt builder can embed it.
        prompt_event = event
        if ml_failure_category is not None:
            # Use object.__setattr__ to set a non-model attribute used by the
            # prompt builder without breaking Pydantic validation.
            prompt_event = event.model_copy(deep=False)
            object.__setattr__(prompt_event, "_ml_failure_category", ml_failure_category)

        prompt = build_cot_prompt(prompt_event, ml_failure_category=ml_failure_category)
        decision = self._agent.generate(prompt)

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
) -> RecoveryDecision:
    """One-shot convenience wrapper around ``RecoveryEngine.process``.

    Creates a ``RecoveryEngine``, processes the event, and returns the decision.
    Use ``RecoveryEngine`` directly if you need to reuse the Gemini client across
    multiple events (avoids re-initialisation overhead).

    Args:
        event: Normalised payment failure event.
        ml_failure_category: Optional ML classifier output.
        model_name: Gemini model name.

    Returns:
        A validated ``RecoveryDecision``.
    """
    engine = RecoveryEngine(model_name=model_name)
    return engine.process(event, ml_failure_category=ml_failure_category)


__all__ = [
    "RecoveryEngine",
    "get_recovery_decision",
]
