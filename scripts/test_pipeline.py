#!/usr/bin/env python3
# scripts/test_pipeline.py
"""
Manual end-to-end pipeline test — 5 real Razorpay failure scenarios.

Runs the complete recovery pipeline (DB → ML → Gemini → action engine) for
five real-world payment failure scenarios using authentic Razorpay webhook
payloads.  Requires a live PostgreSQL connection and a valid GEMINI_API_KEY.

Usage:
    python scripts/test_pipeline.py

Expected output per scenario:
    Scenario:        insufficient_funds
    Amount:          ₹2499.00
    Action:          suggest_alternate_method
    Priority:        high
    Message:         <customer message>
    Retry After:     None
    Alternate Method: upi
    Reasoning:       <one line from Gemini>
    Status:          PASSED ✅
"""

import os
import sys
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any

# ── Windows console UTF-8 safety ──────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Add project root to path ──────────────────────────────────────────────────
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

# ── Logging — show pipeline steps ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Quieten noisy low-level loggers from dependencies
for noisy in ("httpx", "httpcore", "google", "sqlalchemy.engine"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from api_integration.normalizer import normalize_webhook_payload
from agent.pipeline import run_recovery_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 5 Real Razorpay Webhook Payloads
# ─────────────────────────────────────────────────────────────────────────────

# Base Unix timestamp: 2025-08-28 10:30:00 UTC (working hours IST)
_BASE_TS = 1724840200

SCENARIOS: Dict[str, Dict[str, Any]] = {

    # ── Scenario 1: Insufficient Funds — ₹2,499 card payment ─────────────────
    "insufficient_funds": {
        "entity": "event",
        "account_id": "acc_TestMerchant01",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_InsFunds24990001",
                    "entity": "payment",
                    "amount": 249900,          # ₹2,499.00 in paise
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_INS001",
                    "method": "card",
                    "card_id": "card_hdfc_101",
                    "bank": "HDFC",
                    "email": "priya.sharma@gmail.com",
                    "contact": "+919876543210",
                    "customer_id": "cust_PRIYA_001",
                    "notes": {
                        "customer_name": "Priya Sharma",
                        "merchant_category": "fashion",
                    },
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Your payment failed because your account has insufficient funds.",
                    "error_source": "bank",
                    "error_step": "payment_authorization",
                    "error_reason": "insufficient_funds",
                    "created_at": _BASE_TS,
                }
            }
        },
        "created_at": _BASE_TS,
    },

    # ── Scenario 2: Network Timeout — ₹899 UPI payment ───────────────────────
    "network_timeout": {
        "entity": "event",
        "account_id": "acc_TestMerchant02",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_NetTimeout89900002",
                    "entity": "payment",
                    "amount": 89900,           # ₹899.00 in paise
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_NET002",
                    "method": "upi",
                    "vpa": "rahul.kumar@okicici",
                    "email": "rahul.kumar@yahoo.com",
                    "contact": "+918765432109",
                    "customer_id": "cust_RAHUL_002",
                    "notes": {
                        "customer_name": "Rahul Kumar",
                        "merchant_category": "food_delivery",
                    },
                    "error_code": "GATEWAY_ERROR",
                    "error_description": "Payment failed due to UPI collect request timeout. Please try again.",
                    "error_source": "customer",
                    "error_step": "payment_authentication",
                    "error_reason": "network_timeout",
                    "created_at": _BASE_TS + 300,
                }
            }
        },
        "created_at": _BASE_TS + 300,
    },

    # ── Scenario 3: Card Blocked — ₹4,999 card payment ───────────────────────
    "card_blocked": {
        "entity": "event",
        "account_id": "acc_TestMerchant03",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_CardBlk49990003",
                    "entity": "payment",
                    "amount": 499900,          # ₹4,999.00 in paise
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_CBLK003",
                    "method": "card",
                    "card_id": "card_axis_202",
                    "bank": "AXIS",
                    "email": "sneha.patel@outlook.com",
                    "contact": "+917654321098",
                    "customer_id": "cust_SNEHA_003",
                    "notes": {
                        "customer_name": "Sneha Patel",
                        "merchant_category": "electronics",
                    },
                    "error_code": "CARD_BLOCKED",
                    "error_description": "Your card has been blocked by your issuing bank. Please contact your bank.",
                    "error_source": "bank",
                    "error_step": "payment_authorization",
                    "error_reason": "card_blocked",
                    "created_at": _BASE_TS + 600,
                }
            }
        },
        "created_at": _BASE_TS + 600,
    },

    # ── Scenario 4: Expired Card — ₹1,200 card payment ───────────────────────
    "expired_card": {
        "entity": "event",
        "account_id": "acc_TestMerchant04",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_ExpCard12000004",
                    "entity": "payment",
                    "amount": 120000,          # ₹1,200.00 in paise
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_EXP004",
                    "method": "card",
                    "card_id": "card_sbi_303",
                    "bank": "SBI",
                    "email": "ankit.verma@gmail.com",
                    "contact": "+916543210987",
                    "customer_id": "cust_ANKIT_004",
                    "notes": {
                        "customer_name": "Ankit Verma",
                        "merchant_category": "travel",
                    },
                    "error_code": "CARD_EXPIRED",
                    "error_description": "Your card has expired. Please update your card details.",
                    "error_source": "customer",
                    "error_step": "payment_authorization",
                    "error_reason": "expired_card",
                    "created_at": _BASE_TS + 900,
                }
            }
        },
        "created_at": _BASE_TS + 900,
    },

    # ── Scenario 5: Authentication Failed — ₹599 UPI payment ─────────────────
    "authentication_failed": {
        "entity": "event",
        "account_id": "acc_TestMerchant05",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_AuthFail59900005",
                    "entity": "payment",
                    "amount": 59900,           # ₹599.00 in paise
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_AUTH005",
                    "method": "upi",
                    "vpa": "kavya.nair@paytm",
                    "email": "kavya.nair@hotmail.com",
                    "contact": "+915432109876",
                    "customer_id": "cust_KAVYA_005",
                    "notes": {
                        "customer_name": "Kavya Nair",
                        "merchant_category": "entertainment",
                    },
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Your UPI PIN was entered incorrectly. Please try again.",
                    "error_source": "customer",
                    "error_step": "payment_authentication",
                    "error_reason": "authentication_failed",
                    "created_at": _BASE_TS + 1200,
                }
            }
        },
        "created_at": _BASE_TS + 1200,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

_DIVIDER = "─" * 70
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"


def _fmt_amount(amount: float, currency: str = "INR") -> str:
    symbol = "₹" if currency == "INR" else currency
    return f"{symbol}{amount:.2f}"


def _print_result(scenario_name: str, result: Dict[str, Any], passed: bool) -> None:
    status_str = f"{_GREEN}PASSED ✅{_RESET}" if passed else f"{_RED}FAILED ❌{_RESET}"

    retry_after = result.get("retry_after")
    retry_display = f"{retry_after}s" if retry_after else "None"

    alt_method = result.get("alternate_method")
    alt_display = alt_method if alt_method else "None"

    print(_DIVIDER)
    print(f"{_BOLD}Scenario:{_RESET}        {_CYAN}{scenario_name}{_RESET}")
    print(f"{_BOLD}Amount:{_RESET}          {result.get('amount_display', 'N/A')}")
    print(f"{_BOLD}Action:{_RESET}          {result.get('action_taken', 'N/A')}")
    print(f"{_BOLD}Priority:{_RESET}        {result.get('priority', 'N/A')}")
    print(f"{_BOLD}Message:{_RESET}         {result.get('message', 'N/A')}")
    print(f"{_BOLD}Retry After:{_RESET}     {retry_display}")
    print(f"{_BOLD}Alternate Method:{_RESET} {alt_display}")
    print(f"{_BOLD}Reasoning:{_RESET}       {result.get('reasoning', 'N/A')}")
    print(f"{_BOLD}Status:{_RESET}          {status_str}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(f"{_BOLD}{'═' * 70}{_RESET}")
    print(f"{_BOLD}  AI Revenue Recovery Agent — End-to-End Pipeline Test{_RESET}")
    print(f"{_BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 5 Scenarios{_RESET}")
    print(f"{_BOLD}{'═' * 70}{_RESET}")
    print()

    passed_count = 0
    failed_count = 0
    errors = []

    for scenario_name, raw_payload in SCENARIOS.items():
        print(f"\n{_YELLOW}▶ Running: {scenario_name}{_RESET}")

        try:
            # Step A: Normalise the raw webhook payload
            normalized_event = normalize_webhook_payload(raw_payload)

            # Stash amount for display (before pipeline runs)
            amount_display = _fmt_amount(normalized_event.amount, normalized_event.currency)

            # Step B: Run the full pipeline
            result = run_recovery_pipeline(normalized_event)

            # Inject display helpers
            result["amount_display"] = amount_display

            # Minimal sanity checks
            assert result.get("transaction_id"), "Missing transaction_id in result"
            assert result.get("action_taken"),   "Missing action_taken in result"
            assert result.get("failure_category"), "Missing failure_category in result"
            assert result.get("status") == "recovery_initiated", "Unexpected pipeline status"

            _print_result(scenario_name, result, passed=True)
            passed_count += 1

        except Exception as exc:
            failed_count += 1
            tb = traceback.format_exc()
            errors.append((scenario_name, str(exc), tb))

            # Still print a partial result block so failures are obvious
            print(_DIVIDER)
            print(f"{_BOLD}Scenario:{_RESET}        {_CYAN}{scenario_name}{_RESET}")
            print(f"{_BOLD}Status:{_RESET}          {_RED}FAILED ❌  — {exc}{_RESET}")
            print()
            print(f"{_YELLOW}Traceback:{_RESET}")
            print(tb)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"{'═' * 70}")
    print(f"{_BOLD}  Results: {_GREEN}{passed_count} passed{_RESET}  {_RED}{failed_count} failed{_RESET}  (out of {len(SCENARIOS)} scenarios)")
    print(f"{'═' * 70}\n")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
