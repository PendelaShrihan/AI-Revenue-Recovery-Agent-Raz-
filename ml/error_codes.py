"""
Razorpay Error Code Mapper & Failure Classifier.

Maps Razorpay API/Webhook error codes, error reasons, and technical diagnostics
into 8 standardized canonical recovery categories:
1. insufficient_funds
2. card_blocked
3. network_timeout
4. gateway_issue
5. expired_card
6. authentication_failed
7. limit_exceeded
8. unknown
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Set, Any


class FailureCategory(str, Enum):
    """Canonical payment failure categories."""
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_BLOCKED = "card_blocked"
    NETWORK_TIMEOUT = "network_timeout"
    GATEWAY_ISSUE = "gateway_issue"
    EXPIRED_CARD = "expired_card"
    AUTHENTICATION_FAILED = "authentication_failed"
    LIMIT_EXCEEDED = "limit_exceeded"
    UNKNOWN = "unknown"


# List of all valid categories
CATEGORIES: List[str] = [cat.value for cat in FailureCategory]

# Direct Exact Mappings for Razorpay Error Codes
ERROR_CODE_MAP: Dict[str, str] = {
    # Insufficient Funds
    "INSUFFICIENT_FUNDS": FailureCategory.INSUFFICIENT_FUNDS.value,
    "BAD_REQUEST_INSUFFICIENT_FUNDS": FailureCategory.INSUFFICIENT_FUNDS.value,
    "PAYMENT_INSUFFICIENT_BALANCE": FailureCategory.INSUFFICIENT_FUNDS.value,
    "ACCOUNT_INSUFFICIENT_FUNDS": FailureCategory.INSUFFICIENT_FUNDS.value,

    # Card Blocked / Account Inactive / Declined
    "CARD_INACTIVE_OR_CLOSED": FailureCategory.CARD_BLOCKED.value,
    "CARD_BLOCKED": FailureCategory.CARD_BLOCKED.value,
    "ACCOUNT_BLOCKED": FailureCategory.CARD_BLOCKED.value,
    "CARD_BLACKLISTED": FailureCategory.CARD_BLOCKED.value,
    "CARD_DECLINED": FailureCategory.CARD_BLOCKED.value,
    "PAYMENT_DECLINED": FailureCategory.CARD_BLOCKED.value,
    "CARD_RESTRICTED": FailureCategory.CARD_BLOCKED.value,
    "DO_NOT_HONOUR": FailureCategory.CARD_BLOCKED.value,
    "PICK_UP_CARD": FailureCategory.CARD_BLOCKED.value,
    "LOST_CARD": FailureCategory.CARD_BLOCKED.value,
    "STOLEN_CARD": FailureCategory.CARD_BLOCKED.value,

    # Network / Timeout
    "TRANSACTION_TIMED_OUT": FailureCategory.NETWORK_TIMEOUT.value,
    "NETWORK_ERROR": FailureCategory.NETWORK_TIMEOUT.value,
    "TIMEDOUT": FailureCategory.NETWORK_TIMEOUT.value,
    "GATEWAY_TIMED_OUT": FailureCategory.NETWORK_TIMEOUT.value,
    "REQUEST_TIMEOUT": FailureCategory.NETWORK_TIMEOUT.value,
    "CONNECTION_TIMEOUT": FailureCategory.NETWORK_TIMEOUT.value,
    "UPI_COLLECT_EXPIRED": FailureCategory.NETWORK_TIMEOUT.value,
    "SESSION_EXPIRED": FailureCategory.NETWORK_TIMEOUT.value,

    # Gateway / Bank / Server Issues
    "GATEWAY_ERROR": FailureCategory.GATEWAY_ISSUE.value,
    "SERVER_ERROR": FailureCategory.GATEWAY_ISSUE.value,
    "INTERNAL_SERVER_ERROR": FailureCategory.GATEWAY_ISSUE.value,
    "ISSUER_DOWN": FailureCategory.GATEWAY_ISSUE.value,
    "ISSUING_BANK_DOWN": FailureCategory.GATEWAY_ISSUE.value,
    "PAYMENT_METHOD_TEMPORARILY_UNAVAILABLE": FailureCategory.GATEWAY_ISSUE.value,
    "GATEWAY_SERVICE_UNAVAILABLE": FailureCategory.GATEWAY_ISSUE.value,
    "SWITCH_DOWN": FailureCategory.GATEWAY_ISSUE.value,
    "BANK_SYSTEM_ERROR": FailureCategory.GATEWAY_ISSUE.value,
    "ACQUIRER_DOWN": FailureCategory.GATEWAY_ISSUE.value,

    # Expired Card
    "CARD_EXPIRED": FailureCategory.EXPIRED_CARD.value,
    "EXPIRED_CARD": FailureCategory.EXPIRED_CARD.value,
    "CARD_VALIDITY_EXPIRED": FailureCategory.EXPIRED_CARD.value,
    "INVALID_EXPIRY_DATE": FailureCategory.EXPIRED_CARD.value,

    # Authentication / OTP / 3DS / PIN
    "AUTHENTICATION_FAILED": FailureCategory.AUTHENTICATION_FAILED.value,
    "PAYMENT_AUTHENTICATION_FAILED": FailureCategory.AUTHENTICATION_FAILED.value,
    "OTP_EXPIRED": FailureCategory.AUTHENTICATION_FAILED.value,
    "OTP_INCORRECT": FailureCategory.AUTHENTICATION_FAILED.value,
    "INCORRECT_OTP": FailureCategory.AUTHENTICATION_FAILED.value,
    "INVALID_PIN": FailureCategory.AUTHENTICATION_FAILED.value,
    "PIN_INCORRECT": FailureCategory.AUTHENTICATION_FAILED.value,
    "3DS_AUTHENTICATION_FAILED": FailureCategory.AUTHENTICATION_FAILED.value,
    "MPIN_INCORRECT": FailureCategory.AUTHENTICATION_FAILED.value,
    "INVALID_CVV": FailureCategory.AUTHENTICATION_FAILED.value,

    # Limits Exceeded
    "LIMIT_EXCEEDED": FailureCategory.LIMIT_EXCEEDED.value,
    "EXCEEDED_MAX_AMOUNT_PER_TRANSACTION": FailureCategory.LIMIT_EXCEEDED.value,
    "EXCEEDED_DAILY_AMOUNT_LIMIT": FailureCategory.LIMIT_EXCEEDED.value,
    "MAX_AMOUNT_LIMIT_EXCEEDED": FailureCategory.LIMIT_EXCEEDED.value,
    "VELOCITY_EXCEEDED": FailureCategory.LIMIT_EXCEEDED.value,
    "TRANSACTION_AMOUNT_LIMIT_EXCEEDED": FailureCategory.LIMIT_EXCEEDED.value,
    "CARD_LIMIT_EXCEEDED": FailureCategory.LIMIT_EXCEEDED.value,
}

# Direct Exact Mappings for Razorpay Error Reasons (lowercase / snake_case)
ERROR_REASON_MAP: Dict[str, str] = {
    # Insufficient Funds
    "insufficient_funds": FailureCategory.INSUFFICIENT_FUNDS.value,
    "low_balance": FailureCategory.INSUFFICIENT_FUNDS.value,
    "not_enough_balance": FailureCategory.INSUFFICIENT_FUNDS.value,
    "balance_insufficient": FailureCategory.INSUFFICIENT_FUNDS.value,
    "insufficient_balance": FailureCategory.INSUFFICIENT_FUNDS.value,

    # Card Blocked / Declined
    "card_blocked": FailureCategory.CARD_BLOCKED.value,
    "card_inactive": FailureCategory.CARD_BLOCKED.value,
    "card_closed": FailureCategory.CARD_BLOCKED.value,
    "card_declined": FailureCategory.CARD_BLOCKED.value,
    "card_blacklisted": FailureCategory.CARD_BLOCKED.value,
    "account_blocked": FailureCategory.CARD_BLOCKED.value,
    "account_frozen": FailureCategory.CARD_BLOCKED.value,
    "stolen_card": FailureCategory.CARD_BLOCKED.value,
    "lost_card": FailureCategory.CARD_BLOCKED.value,
    "card_restricted": FailureCategory.CARD_BLOCKED.value,
    "declined_by_bank": FailureCategory.CARD_BLOCKED.value,
    "do_not_honour": FailureCategory.CARD_BLOCKED.value,
    "pick_up_card": FailureCategory.CARD_BLOCKED.value,

    # Network / Timeout
    "network_timeout": FailureCategory.NETWORK_TIMEOUT.value,
    "connection_timeout": FailureCategory.NETWORK_TIMEOUT.value,
    "timed_out": FailureCategory.NETWORK_TIMEOUT.value,
    "gateway_timed_out": FailureCategory.NETWORK_TIMEOUT.value,
    "timeout": FailureCategory.NETWORK_TIMEOUT.value,
    "network_error": FailureCategory.NETWORK_TIMEOUT.value,
    "connection_reset": FailureCategory.NETWORK_TIMEOUT.value,
    "upi_collect_expired": FailureCategory.NETWORK_TIMEOUT.value,
    "request_timeout": FailureCategory.NETWORK_TIMEOUT.value,
    "otp_timeout": FailureCategory.NETWORK_TIMEOUT.value,

    # Gateway / Bank / Server Issues
    "gateway_error": FailureCategory.GATEWAY_ISSUE.value,
    "gateway_issue": FailureCategory.GATEWAY_ISSUE.value,
    "server_error": FailureCategory.GATEWAY_ISSUE.value,
    "bank_error": FailureCategory.GATEWAY_ISSUE.value,
    "issuer_down": FailureCategory.GATEWAY_ISSUE.value,
    "issuing_bank_down": FailureCategory.GATEWAY_ISSUE.value,
    "bank_down": FailureCategory.GATEWAY_ISSUE.value,
    "switch_down": FailureCategory.GATEWAY_ISSUE.value,
    "internal_error": FailureCategory.GATEWAY_ISSUE.value,
    "service_unavailable": FailureCategory.GATEWAY_ISSUE.value,
    "system_error": FailureCategory.GATEWAY_ISSUE.value,
    "acquirer_down": FailureCategory.GATEWAY_ISSUE.value,

    # Expired Card
    "card_expired": FailureCategory.EXPIRED_CARD.value,
    "expired_card": FailureCategory.EXPIRED_CARD.value,
    "card_validity_expired": FailureCategory.EXPIRED_CARD.value,
    "invalid_expiry": FailureCategory.EXPIRED_CARD.value,

    # Authentication / Security
    "authentication_failed": FailureCategory.AUTHENTICATION_FAILED.value,
    "payment_authentication_failed": FailureCategory.AUTHENTICATION_FAILED.value,
    "incorrect_otp": FailureCategory.AUTHENTICATION_FAILED.value,
    "otp_incorrect": FailureCategory.AUTHENTICATION_FAILED.value,
    "invalid_otp": FailureCategory.AUTHENTICATION_FAILED.value,
    "pin_incorrect": FailureCategory.AUTHENTICATION_FAILED.value,
    "invalid_pin": FailureCategory.AUTHENTICATION_FAILED.value,
    "3ds_authentication_failed": FailureCategory.AUTHENTICATION_FAILED.value,
    "3ds_failed": FailureCategory.AUTHENTICATION_FAILED.value,
    "auth_failed": FailureCategory.AUTHENTICATION_FAILED.value,
    "mpin_incorrect": FailureCategory.AUTHENTICATION_FAILED.value,
    "invalid_cvv": FailureCategory.AUTHENTICATION_FAILED.value,
    "customer_authentication_failed": FailureCategory.AUTHENTICATION_FAILED.value,

    # Limit Exceeded
    "limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "daily_limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "transaction_limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "maximum_amount_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "velocity_limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "per_transaction_limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "credit_limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
    "amount_limit_exceeded": FailureCategory.LIMIT_EXCEEDED.value,
}

# Keyword Heuristics for Substring / Fuzzy Detection
KEYWORD_PATTERNS = [
    # High Priority: Specific causes
    (FailureCategory.INSUFFICIENT_FUNDS.value, [
        r"insufficient[_\s]*funds?",
        r"low[_\s]*balance",
        r"not[_\s]*enough[_\s]*balance",
        r"short[_\s]*of[_\s]*balance",
        r"insufficient[_\s]*balance",
    ]),
    (FailureCategory.EXPIRED_CARD.value, [
        r"expired[_\s]*card",
        r"card[_\s]*expired",
        r"validity[_\s]*expired",
        r"invalid[_\s]*expiry",
    ]),
    (FailureCategory.LIMIT_EXCEEDED.value, [
        r"limit[_\s]*exceeded",
        r"daily[_\s]*limit",
        r"velocity[_\s]*limit",
        r"maximum[_\s]*amount",
        r"exceeded[_\s]*amount",
        r"credit[_\s]*limit",
        r"per[_\s]*transaction[_\s]*limit",
    ]),
    (FailureCategory.AUTHENTICATION_FAILED.value, [
        r"auth(?:entication)?[_\s]*failed",
        r"incorrect[_\s]*otp",
        r"otp[_\s]*incorrect",
        r"invalid[_\s]*otp",
        r"invalid[_\s]*pin",
        r"pin[_\s]*incorrect",
        r"mpin",
        r"3ds(?:ecure)?[_\s]*(?:auth|failed)",
        r"invalid[_\s]*cvv",
        r"password[_\s]*incorrect",
    ]),
    (FailureCategory.CARD_BLOCKED.value, [
        r"card[_\s]*blocked",
        r"card[_\s]*inactive",
        r"card[_\s]*closed",
        r"card[_\s]*declined",
        r"account[_\s]*blocked",
        r"account[_\s]*frozen",
        r"blacklisted",
        r"stolen",
        r"lost[_\s]*card",
        r"do[_\s]*not[_\s]*honou?r",
        r"pick[_\s]*up[_\s]*card",
        r"restricted[_\s]*card",
    ]),
    (FailureCategory.NETWORK_TIMEOUT.value, [
        r"timed?[_\s]*out",
        r"network[_\s]*timeout",
        r"connection[_\s]*timeout",
        r"connection[_\s]*reset",
        r"socket[_\s]*timeout",
        r"request[_\s]*timeout",
        r"collect[_\s]*expired",
        r"otp[_\s]*timeout",
    ]),
    (FailureCategory.GATEWAY_ISSUE.value, [
        r"gateway[_\s]*(?:error|issue|down|unavailable)",
        r"issuer[_\s]*down",
        r"bank[_\s]*(?:down|error|unavailable)",
        r"switch[_\s]*down",
        r"server[_\s]*error",
        r"internal[_\s]*error",
        r"acquirer[_\s]*down",
        r"temporarily[_\s]*unavailable",
        r"service[_\s]*unavailable",
    ]),
]


def _normalize_string(val: Optional[Any]) -> str:
    """Helper to clean and normalize error string input safely."""
    if val is None:
        return ""
    try:
        s = str(val).strip().lower()
        s = re.sub(r"[\s\-]+", "_", s)
        return s
    except Exception:
        return ""


def classify_failure(error_code: Optional[Any] = None, error_reason: Optional[Any] = None) -> str:
    """
    Classifies a payment failure into one of 8 canonical categories:
    - insufficient_funds
    - card_blocked
    - network_timeout
    - gateway_issue
    - expired_card
    - authentication_failed
    - limit_exceeded
    - unknown

    Guarantees that None, empty string, or non-string inputs are handled gracefully
    without raising exceptions, returning 'unknown'.

    Strategy:
    1. Check exact matches on `error_reason` (which is typically the most granular signal).
    2. Check exact matches on `error_code`.
    3. Run regex / keyword heuristics across `error_reason` then `error_code`.
    4. Fall back to `unknown` if no confident match is found.

    Args:
        error_code: Razorpay error code (e.g., 'BAD_REQUEST_ERROR', 'GATEWAY_ERROR', 'INSUFFICIENT_FUNDS')
        error_reason: Razorpay granular reason (e.g., 'insufficient_funds', 'card_declined', 'otp_timeout')

    Returns:
        Canonical category string (e.g. 'insufficient_funds', 'unknown')
    """
    if error_code is None and error_reason is None:
        return FailureCategory.UNKNOWN.value

    norm_code = _normalize_string(error_code)
    norm_reason = _normalize_string(error_reason)
    upper_code = str(error_code or "").strip().upper()

    # Step 1: Check exact matches on error_reason
    if norm_reason and norm_reason in ERROR_REASON_MAP:
        return ERROR_REASON_MAP[norm_reason]

    # Step 2: Check exact matches on error_code (both upper and normalized)
    if upper_code and upper_code in ERROR_CODE_MAP:
        return ERROR_CODE_MAP[upper_code]
    if norm_code and norm_code in ERROR_REASON_MAP:
        return ERROR_REASON_MAP[norm_code]

    # Step 3: Combined text heuristic scan (reason prioritized over code)
    combined_text = f"{norm_reason} {norm_code}".strip()
    if not combined_text:
        return FailureCategory.UNKNOWN.value

    for category, patterns in KEYWORD_PATTERNS:
        for pattern in patterns:
            if norm_reason and re.search(pattern, norm_reason, re.IGNORECASE):
                return category

    for category, patterns in KEYWORD_PATTERNS:
        for pattern in patterns:
            if norm_code and re.search(pattern, norm_code, re.IGNORECASE):
                return category

    return FailureCategory.UNKNOWN.value


def is_transient_failure(category: str) -> bool:
    """
    Determines whether a failure category is transient (auto-retry recommended)
    versus terminal (requires user intervention or alternative payment method).

    Transient:
        - network_timeout
        - gateway_issue
    Non-transient / User action needed:
        - insufficient_funds (requires funds/UPI app)
        - card_blocked (requires alternative card)
        - expired_card (requires new card)
        - authentication_failed (requires re-auth/OTP)
        - limit_exceeded (requires bank limit increase or split payment)
    """
    return category in (FailureCategory.NETWORK_TIMEOUT.value, FailureCategory.GATEWAY_ISSUE.value)


def get_category_description(category: str) -> str:
    """Returns human-readable explanation of failure category."""
    descriptions = {
        FailureCategory.INSUFFICIENT_FUNDS.value: "Customer account or card has insufficient balance to complete the transaction.",
        FailureCategory.CARD_BLOCKED.value: "Card or bank account has been blocked, deactivated, blacklisted, or declined by the issuing bank.",
        FailureCategory.NETWORK_TIMEOUT.value: "Temporary network timeout or connection reset occurred between Razorpay, bank, or customer session.",
        FailureCategory.GATEWAY_ISSUE.value: "Issuing bank switch, payment gateway, or server is experiencing downtime or internal errors.",
        FailureCategory.EXPIRED_CARD.value: "Customer attempted payment using an expired card or invalid expiration date.",
        FailureCategory.AUTHENTICATION_FAILED.value: "Customer failed 3D-Secure authentication, entered incorrect OTP, or invalid PIN/CVV.",
        FailureCategory.LIMIT_EXCEEDED.value: "Transaction amount exceeds customer's daily limit, per-transaction velocity, or credit limit.",
        FailureCategory.UNKNOWN.value: "Unclassified or generic failure reason."
    }
    return descriptions.get(category, "Unclassified failure.")
