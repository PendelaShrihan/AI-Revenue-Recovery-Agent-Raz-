# agent/llm_agent.py
"""Gemini LLM wrapper for payment recovery.

Provides:
- ``GeminiAgent``        – thin wrapper around the Gemini API with retry + logging.
- ``SYSTEM_PROMPT``      – detailed payment-recovery expert system prompt covering
                           all eight failure categories and their exact recovery actions.
- ``build_cot_prompt()`` – builds a numbered chain-of-thought prompt from a
                           ``NormalizedEvent``, injecting all contextual fields.
- ``RecoveryDecision``   – dataclass that holds the validated, parsed Gemini output.
- ``GeminiAgentError``   – raised when the Gemini API cannot be reached after retries.
- ``GeminiOutputParseError`` – raised when Gemini returns output that does not match
                               the required JSON schema.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

from api_integration.schemas import NormalizedEvent

# ---------------------------------------------------------------------------
# Bootstrap – load .env so GEMINI_API_KEY is available locally
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior payment-recovery expert agent for Razorpay. Your sole job is \
to analyse a failed payment event and decide the single best recovery action. \
You have deep knowledge of all eight failure categories and their prescribed \
recovery actions as detailed below.

=== FAILURE CATEGORIES AND RECOVERY ACTIONS ===

1. insufficient_funds
   - Do NOT retry immediately.
   - Suggest UPI or wallet as an alternate payment method.
   - Send the customer a payment link with the alternate method pre-selected.
   - Schedule an automatic retry the following morning between 09:00–11:00 IST
     if no customer action has been taken.

2. card_blocked
   - Do NOT retry at all – every retry will fail.
   - Suggest net banking or UPI as alternate methods.
   - Send the customer a message advising them to contact their bank to unblock
     the card.
   - Log the transaction as requiring manual intervention.

3. network_timeout
   - Auto-retry after exactly 15 minutes – no customer notification needed.
   - If the first retry fails, retry once more after 30 minutes.
   - If the second retry also fails, send the customer a payment link.
   - This is a transient failure with a high success rate on retry.

4. gateway_issue
   - Auto-retry after 30 minutes.
   - Switch to an alternate gateway if one is available.
   - Do NOT notify the customer unless both retries fail.
   - This is a transient failure – retry is always the right first action.

5. expired_card
   - Do NOT retry – it will always fail with an expired card.
   - Send the customer a message asking them to update their card details.
   - Include a payment link in the message.
   - Mark priority HIGH – customer action is required.
   - CRITICAL: Never suggest "card" as the alternate_method for expired_card failures.
     The customer's card is expired so suggesting the same card method is useless.
     Always set alternate_method to "upi" or "netbanking" instead.

6. authentication_failed
   - Retry once after 10 minutes using the same payment method.
   - If the retry fails, suggest UPI as a simpler alternate method.
   - Send the customer a gentle nudge message.
   - Common root causes: OTP timeout, bank 2FA issue.

7. limit_exceeded
   - Do NOT retry with the same card.
   - Suggest splitting the payment across two methods if the platform supports it.
   - Suggest using a different card or net banking.
   - Send a notification to the customer with the available options.

8. unknown
   - Retry once after 60 minutes.
   - If the retry fails, notify the customer with a payment link.
   - Flag the transaction for manual review in the recovery dashboard.
   - Log full error details for the engineering team.

=== IMPORTANT INSTRUCTIONS ===

- Read the provided payment context carefully before deciding.
- Consider the payment method, error reason, hour of day, subscription priority,
  and notification channels available (email / phone).
- Subscriptions (subscription_id is set) are HIGHER priority than one-off payments.
- Always output EXACTLY the following JSON object and NOTHING else – no markdown,
  no commentary, no surrounding text:

{
  "action": "<one of: auto_retry | suggest_alternate_method | send_payment_link | notify_customer | manual_review | no_action>",
  "priority": "<one of: high | medium | low>",
  "message": "<customer-facing message in plain English, max 2 sentences>",
  "retry_after": <integer seconds until next retry, 0 if no retry>,
  "alternate_method": "<one of: upi | wallet | netbanking | card | none>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the decision>"
}

If you cannot determine a confident action, still output the JSON with action=\
"manual_review" and your best confidence estimate. Never output anything other \
than this JSON object.
"""

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GeminiAgentError(RuntimeError):
    """Raised when the Gemini API encounters a generic or unrecoverable error."""


class GeminiTimeoutError(GeminiAgentError):
    """Raised when the Gemini API times out (>10s) after retries."""


class GeminiRateLimitError(GeminiAgentError):
    """Raised when Gemini returns 429 rate limit or quota exceeded after retries."""


class GeminiOutputParseError(ValueError):
    """Raised when Gemini's response cannot be parsed into the required schema."""


# ---------------------------------------------------------------------------
# Parsed output dataclass
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({
    "auto_retry",
    "suggest_alternate_method",
    "send_payment_link",
    "notify_customer",
    "manual_review",
    "no_action",
})

VALID_PRIORITIES = frozenset({"high", "medium", "low"})
VALID_METHODS = frozenset({"upi", "wallet", "netbanking", "card", "none"})


@dataclass
class RecoveryDecision:
    """Validated, structured output from Gemini for a single failed payment."""

    action: str
    priority: str
    message: str
    retry_after: int
    alternate_method: str
    confidence: float
    reasoning: str

    # The raw text returned by Gemini (for audit / debugging)
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw: str = "") -> "RecoveryDecision":
        """Validate and construct a RecoveryDecision from a parsed JSON dict."""
        missing = [
            k for k in (
                "action", "priority", "message",
                "retry_after", "alternate_method", "confidence", "reasoning"
            ) if k not in data
        ]
        if missing:
            raise GeminiOutputParseError(
                f"Gemini response is missing required fields: {missing}. "
                f"Raw response: {raw!r}"
            )

        action = str(data["action"]).strip().lower()
        priority = str(data["priority"]).strip().lower()
        alternate_method = str(data["alternate_method"]).strip().lower()

        if action not in VALID_ACTIONS:
            raise GeminiOutputParseError(
                f"Invalid action {action!r}. Must be one of {sorted(VALID_ACTIONS)}. "
                f"Raw: {raw!r}"
            )
        if priority not in VALID_PRIORITIES:
            raise GeminiOutputParseError(
                f"Invalid priority {priority!r}. Must be one of {sorted(VALID_PRIORITIES)}. "
                f"Raw: {raw!r}"
            )
        if alternate_method not in VALID_METHODS:
            raise GeminiOutputParseError(
                f"Invalid alternate_method {alternate_method!r}. "
                f"Must be one of {sorted(VALID_METHODS)}. Raw: {raw!r}"
            )

        try:
            retry_after = int(data["retry_after"])
        except (TypeError, ValueError) as exc:
            raise GeminiOutputParseError(
                f"retry_after must be an integer. Got {data['retry_after']!r}. Raw: {raw!r}"
            ) from exc

        try:
            confidence = float(data["confidence"])
        except (TypeError, ValueError) as exc:
            raise GeminiOutputParseError(
                f"confidence must be a float. Got {data['confidence']!r}. Raw: {raw!r}"
            ) from exc

        if not (0.0 <= confidence <= 1.0):
            raise GeminiOutputParseError(
                f"confidence must be between 0.0 and 1.0. Got {confidence}. Raw: {raw!r}"
            )

        return cls(
            action=action,
            priority=priority,
            message=str(data["message"]).strip(),
            retry_after=retry_after,
            alternate_method=alternate_method,
            confidence=confidence,
            reasoning=str(data["reasoning"]).strip(),
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# Chain-of-thought prompt builder
# ---------------------------------------------------------------------------

def build_cot_prompt(event: NormalizedEvent, ml_failure_category: Optional[str] = None) -> str:
    """Build a numbered chain-of-thought prompt from a ``NormalizedEvent``.

    Injects all contextual fields required by the system prompt so that Gemini
    can make an informed, step-by-step recovery decision.

    Args:
        event: A ``NormalizedEvent`` produced by the webhook normaliser.
        ml_failure_category: Optional raw failure category string from the ML
            classifier (e.g. ``"insufficient_funds"``). This is the fine-grained
            error-reason category used by the recovery system prompt, distinct
            from the coarser ``FailureCategory`` enum on the event.

    Returns:
        A fully-formed prompt string ready to be sent to the Gemini model.
    """
    # Extract hour of day for retry-timing logic
    created_at = event.created_at
    if isinstance(created_at, datetime):
        hour_of_day = created_at.hour
        created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        hour_of_day = "unknown"
        created_at_str = str(created_at)

    # Subscription priority flag
    is_subscription = bool(event.subscription_id)
    priority_flag = "SUBSCRIPTION (high priority)" if is_subscription else "one-off payment"

    # Notification channels available
    channels: list[str] = []
    if event.customer_email:
        channels.append(f"email ({event.customer_email})")
    if event.customer_phone:
        channels.append(f"phone/SMS ({event.customer_phone})")
    notification_channels = ", ".join(channels) if channels else "none available"

    # Effective failure category: prefer ML classifier output over enum-level category
    effective_failure_category = ml_failure_category or event.error_reason or str(event.failure_category)

    prompt = f"""\
=== PAYMENT FAILURE CONTEXT ===

Entity:
  - entity_type   : {event.entity_type}
  - event_type    : {event.event_type}
  - entity_id     : {event.entity_id}
  - payment_id    : {event.payment_id or "N/A"}
  - subscription_id: {event.subscription_id or "N/A"}
  - invoice_id    : {event.invoice_id or "N/A"}
  - payment_type  : {priority_flag}
  - merchant_id   : {event.merchant_id}

Amount & Currency:
  - amount        : {event.amount} {event.currency}

Payment Method:
  - payment_method: {event.payment_method or "unknown"}

Failure Details:
  - failure_category  : {effective_failure_category}
  - error_code        : {event.error_code or "N/A"}
  - error_reason      : {event.error_reason or "N/A"}
  - error_description : {event.error_description or "N/A"}
  - error_source      : {event.error_source or "N/A"}
  - error_step        : {event.error_step or "N/A"}

Timing:
  - created_at    : {created_at_str}
  - hour_of_day   : {hour_of_day} (24h IST – relevant for retry scheduling)

Customer:
  - customer_id   : {event.customer_id or "N/A"}
  - customer_name : {event.customer_name or "N/A"}
  - notification_channels: {notification_channels}

=== RECOVERY DECISION TASK ===
Evaluate the failure context above against the failure categories in your system instructions.
Output ONLY the JSON object with the exact fields:
{{
  "action": "<auto_retry | suggest_alternate_method | send_payment_link | notify_customer | manual_review | no_action>",
  "priority": "<high | medium | low>",
  "message": "<friendly customer message, max 2 sentences>",
  "retry_after": <integer seconds, 0 if no retry>,
  "alternate_method": "<upi | wallet | netbanking | card | none>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<concise 1-sentence explanation of decision>"
}}
"""
    return prompt


# ---------------------------------------------------------------------------
# GeminiAgent
# ---------------------------------------------------------------------------

class GeminiAgent:
    """Wrapper around the Google Gemini API for payment recovery decisions.

    - Reads the API key from the ``GEMINI_API_KEY`` environment variable.
    - Uses the ``gemini-3.6-flash`` / ``gemini-2.0-flash`` model by default.
    - Retries up to 3 times with exponential back-off (2 s → 4 s → 8 s) for rate limits.
    - Times out if API response exceeds 10s per call and retries up to 3 times before raising GeminiTimeoutError.
    - Logs every prompt at DEBUG, every response at DEBUG, every error at ERROR.
    - Raises ``GeminiTimeoutError`` on timeouts, ``GeminiRateLimitError`` on 429s, or ``GeminiAgentError`` on unrecoverable errors.
    - Raises ``GeminiOutputParseError`` if the response cannot be parsed into the required JSON schema.
    """

    _MAX_RETRIES = 3
    _TIMEOUT_SECONDS = 10.0

    def __init__(self, model_name: Optional[str] = None) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAgentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your .env file or export it in your shell."
            )
        self._client = genai.Client(api_key=api_key, http_options={"timeout": 10000})
        self._model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self._config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.1,
        )
        self._last_call_meta: Dict[str, Any] = {}
        _logger.debug("GeminiAgent initialised with model=%s", self._model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str, transaction_id: Optional[str] = None) -> str:
        """Call the Gemini API with retry, 10s timeout, and exponential back-off.

        Returns the raw text from the model on success.
        Raises:
            GeminiTimeoutError: If all 3 attempts exceed 10s timeout.
            GeminiRateLimitError: If rate limited (429) after all backoff retries (2s, 4s, 8s).
            GeminiAgentError: If unrecoverable API error occurs after retries.
        """
        last_error: Optional[Exception] = None
        is_rate_limit = False
        is_timeout = False

        backoff_delays = [2, 4, 8]

        for attempt in range(1, self._MAX_RETRIES + 1):
            _logger.debug(
                "[Gemini] Request attempt=%d/%d model=%s prompt_length=%d chars\nPROMPT:\n%s",
                attempt, self._MAX_RETRIES, self._model_name, len(prompt), prompt,
            )

            try:
                start_time = time.perf_counter()
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=self._config,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                raw_text = response.text.strip()

                # Extract token usage and record cost
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                if input_tokens == 0:
                    input_tokens = max(1, len(prompt) // 4)
                if output_tokens == 0:
                    output_tokens = max(1, len(raw_text) // 4)

                try:
                    from agent.cost_tracker import log_llm_call
                    cost_info = log_llm_call(
                        transaction_id=transaction_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=self._model_name,
                        latency_ms=latency_ms,
                    )
                    cost_usd = cost_info.get("cost_usd", 0.0)
                except Exception as log_err:
                    _logger.warning("[Gemini] Failed to log LLM call cost: %s", log_err)
                    cost_usd = 0.0

                self._last_call_meta = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "model": self._model_name,
                }

                _logger.debug(
                    "[Gemini] Response attempt=%d response_length=%d chars | tokens: in=%d out=%d | latency=%.1fms\nRESPONSE:\n%s",
                    attempt, len(raw_text), input_tokens, output_tokens, latency_ms, raw_text,
                )
                return raw_text

            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                if "timeout" in err_str or "timed out" in err_str:
                    is_timeout = True
                    _logger.error("[Gemini] API call timed out on attempt=%d/%d (>10s): %s", attempt, self._MAX_RETRIES, exc)
                elif "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
                    is_rate_limit = True
                    _logger.error("[Gemini] API call rate limited (429) attempt=%d/%d: %s", attempt, self._MAX_RETRIES, exc)
                else:
                    _logger.error("[Gemini] API call failed attempt=%d/%d error=%s", attempt, self._MAX_RETRIES, exc)

            if attempt >= self._MAX_RETRIES:
                if is_timeout:
                    raise GeminiTimeoutError(
                        f"Gemini API timed out (>10s) after {self._MAX_RETRIES} attempts. Last error: {last_error}"
                    ) from last_error
                if is_rate_limit:
                    raise GeminiRateLimitError(
                        f"Gemini API rate limit (429) exceeded after {self._MAX_RETRIES} attempts. Last error: {last_error}"
                    ) from last_error
                raise GeminiAgentError(
                    f"Gemini API failed after {self._MAX_RETRIES} attempts. Last error: {last_error}"
                ) from last_error

            # Exponential backoff delay (2s, 4s, 8s) or dynamic API retryDelay on 429
            delay = backoff_delays[attempt - 1] if attempt - 1 < len(backoff_delays) else 8
            if is_rate_limit:
                import re
                match = re.search(r"retry\s*in\s*([0-9.]+)\s*s", str(last_error), re.IGNORECASE)
                if not match:
                    match = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"]([0-9]+)s['\"]", str(last_error))
                if match:
                    delay = max(int(float(match.group(1))) + 2, delay)
                else:
                    delay = max(delay, 6 * attempt)
                _logger.warning("[Gemini] 429 Rate limit encountered. Backing off for %ds before attempt %d…", delay, attempt + 1)
            else:
                _logger.debug("[Gemini] Retrying in %d seconds before attempt %d…", delay, attempt + 1)

            time.sleep(delay)

        raise GeminiAgentError("Unexpected termination in GeminiAgent._call_api")

    @staticmethod
    def _parse_response(raw: str) -> RecoveryDecision:
        """Parse and validate Gemini's raw text response into a ``RecoveryDecision``.

        Strips Markdown fences if present before JSON parsing.
        Logs raw response at DEBUG and raises ``GeminiOutputParseError`` on any parsing or validation failure.
        """
        # Strip optional ```json … ``` fences that some model versions emit
        cleaned = raw
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            _logger.debug("[Gemini] Malformed JSON response received from model: %r", raw)
            raise GeminiOutputParseError(
                f"Gemini response is not valid JSON. "
                f"Parse error: {exc}. Raw response: {raw!r}"
            ) from exc

        if not isinstance(data, dict):
            _logger.debug("[Gemini] Malformed JSON structure (not a dict): %r", raw)
            raise GeminiOutputParseError(
                f"Gemini response parsed to {type(data).__name__} instead of a dict. "
                f"Raw: {raw!r}"
            )

        try:
            return RecoveryDecision.from_dict(data, raw=raw)
        except GeminiOutputParseError:
            _logger.debug("[Gemini] Malformed JSON schema in response: %r", raw)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, transaction_id: Optional[str] = None) -> RecoveryDecision:
        """Send a prompt to Gemini and return a validated ``RecoveryDecision``.

        Args:
            prompt: The full user prompt (typically built by ``build_cot_prompt``).
            transaction_id: Optional transaction ID for cost logging.

        Returns:
            A ``RecoveryDecision`` with all fields validated.

        Raises:
            GeminiTimeoutError: If the call times out >10s after retries.
            GeminiRateLimitError: If rate limit 429 persists after retries.
            GeminiAgentError: If the API cannot be reached after retries.
            GeminiOutputParseError: If the response does not match the required schema.
        """
        raw = self._call_api(prompt, transaction_id=transaction_id)
        decision = self._parse_response(raw)
        if self._last_call_meta:
            decision.input_tokens = int(self._last_call_meta.get("input_tokens", 0))
            decision.output_tokens = int(self._last_call_meta.get("output_tokens", 0))
            decision.latency_ms = float(self._last_call_meta.get("latency_ms", 0.0))
            decision.cost_usd = float(self._last_call_meta.get("cost_usd", 0.0))
            decision.model = str(self._last_call_meta.get("model", self._model_name))
        return decision

    def decide(
        self,
        event: NormalizedEvent,
        ml_failure_category: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> RecoveryDecision:
        """End-to-end convenience method: build CoT prompt → call API → parse output.

        Args:
            event: A ``NormalizedEvent`` from the webhook normaliser / ML classifier.
            ml_failure_category: Optional raw ML classifier failure category string.
            transaction_id: Optional transaction ID for cost tracking.

        Returns:
            A validated ``RecoveryDecision``.
        """
        prompt = build_cot_prompt(event, ml_failure_category=ml_failure_category)
        tx_id = transaction_id or event.payment_id or event.entity_id
        return self.generate(prompt, transaction_id=tx_id)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "GeminiAgent",
    "GeminiAgentError",
    "GeminiTimeoutError",
    "GeminiRateLimitError",
    "GeminiOutputParseError",
    "RecoveryDecision",
    "SYSTEM_PROMPT",
    "build_cot_prompt",
]
