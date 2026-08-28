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
    """Raised when the Gemini API cannot be reached after all retries."""


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

=== CHAIN-OF-THOUGHT STEPS ===

Step 1 – Summarise the failure:
  Briefly describe what happened based on the context above.

Step 2 – Identify the failure category:
  Match the failure_category field to one of the eight categories in the system
  prompt. If the category is unknown, infer the most likely one from the error
  details.

Step 3 – Choose the recovery action:
  Select the appropriate action from the system prompt for the identified
  category. Factor in:
    - Whether a subscription is involved (higher urgency).
    - The hour of day (relevant for retry scheduling).
    - Available notification channels.
    - The payment method (some methods cannot be switched trivially).

Step 4 – Compose the customer message:
  Write a friendly, concise customer-facing message (max 2 sentences) that:
    - Explains the issue simply (no technical jargon).
    - Tells the customer what action they need to take (if any).

Step 5 – Output the JSON:
  Output ONLY the JSON object described in the system prompt. No extra text.
"""
    return prompt


# ---------------------------------------------------------------------------
# GeminiAgent
# ---------------------------------------------------------------------------

class GeminiAgent:
    """Wrapper around the Google Gemini API for payment recovery decisions.

    - Reads the API key from the ``GEMINI_API_KEY`` environment variable.
    - Uses the ``gemini-2.0-flash`` model by default.
    - Retries up to 3 times with exponential back-off (2 s → 4 s → 8 s).
    - Logs every prompt at DEBUG, every response at DEBUG, every error at ERROR.
    - Raises ``GeminiAgentError`` if all retries are exhausted.
    - Raises ``GeminiOutputParseError`` if the response cannot be parsed into the
      required JSON schema.
    """

    _MAX_RETRIES = 5
    _INITIAL_BACKOFF = 3  # seconds

    def __init__(self, model_name: str = "gemini-3.5-flash-lite") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAgentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your .env file or export it in your shell."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        )
        _logger.debug("GeminiAgent initialised with model=%s", model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """Call the Gemini API with retry + exponential back-off.

        Returns the raw text from the model on success.
        Raises ``GeminiAgentError`` after ``_MAX_RETRIES`` consecutive failures.
        """
        backoff = self._INITIAL_BACKOFF

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                _logger.debug(
                    "[Gemini] Request attempt=%d model=%s prompt_length=%d chars\nPROMPT:\n%s",
                    attempt, self._model_name, len(prompt), prompt,
                )
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=self._config,
                )
                raw_text = response.text.strip()
                _logger.debug(
                    "[Gemini] Response attempt=%d response_length=%d chars\nRESPONSE:\n%s",
                    attempt, len(raw_text), raw_text,
                )
                return raw_text

            except Exception as exc:  # noqa: BLE001  (broad – Gemini SDK raises various types)
                _logger.error(
                    "[Gemini] API call failed attempt=%d/%d error=%s",
                    attempt, self._MAX_RETRIES, exc,
                )
                if attempt >= self._MAX_RETRIES:
                    raise GeminiAgentError(
                        f"Gemini API failed after {self._MAX_RETRIES} attempts. "
                        f"Last error: {exc}"
                    ) from exc

                sleep_time = backoff
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    sleep_time = max(backoff, 6)
                    _logger.warning("[Gemini] Rate limit detected. Backing off for %ds …", sleep_time)
                else:
                    _logger.debug("[Gemini] Retrying in %d seconds …", sleep_time)

                time.sleep(sleep_time)
                backoff *= 2

        # Unreachable, but satisfies type checkers
        raise GeminiAgentError("Unexpected state in GeminiAgent._call_api")

    @staticmethod
    def _parse_response(raw: str) -> RecoveryDecision:
        """Parse and validate Gemini's raw text response into a ``RecoveryDecision``.

        Strips Markdown fences if present before JSON parsing.
        Raises ``GeminiOutputParseError`` on any parsing or validation failure.
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
            raise GeminiOutputParseError(
                f"Gemini response is not valid JSON. "
                f"Parse error: {exc}. Raw response: {raw!r}"
            ) from exc

        if not isinstance(data, dict):
            raise GeminiOutputParseError(
                f"Gemini response parsed to {type(data).__name__} instead of a dict. "
                f"Raw: {raw!r}"
            )

        return RecoveryDecision.from_dict(data, raw=raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> RecoveryDecision:
        """Send a prompt to Gemini and return a validated ``RecoveryDecision``.

        Args:
            prompt: The full user prompt (typically built by ``build_cot_prompt``).

        Returns:
            A ``RecoveryDecision`` with all fields validated.

        Raises:
            GeminiAgentError: If the API cannot be reached after retries.
            GeminiOutputParseError: If the response does not match the required schema.
        """
        raw = self._call_api(prompt)
        return self._parse_response(raw)

    def decide(self, event: NormalizedEvent, ml_failure_category: Optional[str] = None) -> RecoveryDecision:
        """End-to-end convenience method: build CoT prompt → call API → parse output.

        Args:
            event: A ``NormalizedEvent`` from the webhook normaliser / ML classifier.
            ml_failure_category: Optional raw ML classifier failure category string.

        Returns:
            A validated ``RecoveryDecision``.
        """
        prompt = build_cot_prompt(event, ml_failure_category=ml_failure_category)
        return self.generate(prompt)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "GeminiAgent",
    "GeminiAgentError",
    "GeminiOutputParseError",
    "RecoveryDecision",
    "SYSTEM_PROMPT",
    "build_cot_prompt",
]
