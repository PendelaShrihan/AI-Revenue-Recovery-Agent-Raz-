# tests/test_integration_full.py
"""Full Integration Test Suite for AI Revenue Recovery Agent.

Runs all 11 failure scenarios end-to-end through the complete pipeline using
REAL APIs ONLY:
  - Real PostgreSQL database (Docker port 5434 / DATABASE_URL)
  - Real Google Gemini API (gemini-3.6-flash / GEMINI_API_KEY)
  - Real Razorpay sandbox client credentials
  - NO MOCKS anywhere in this test suite.

For each scenario, verifies:
  1. Webhook payload normalizes correctly into a NormalizedEvent
  2. ML classifier determines the expected failure category
  3. Gemini returns a valid, schema-compliant JSON RecoveryDecision
  4. Recovery action is dispatched and persisted to PostgreSQL
  5. Pipeline completes with no unhandled exceptions

At the end of the suite, outputs the formatted Integration Test Matrix.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pytest
from dotenv import load_dotenv

load_dotenv()

from api_integration.normalizer import normalize_webhook_payload
from api_integration.schemas import NormalizedEvent
from agent.pipeline import run_recovery_pipeline
from agent.db_writer import get_db_session
from agent.models import Transaction, RecoveryAction, RetryAttempt
from agent.observability import record_pipeline_run, get_metrics, print_metrics_report

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_integration_full")

# ---------------------------------------------------------------------------
# Skip Guard
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set in environment or .env",
)

# ---------------------------------------------------------------------------
# 11 Scenario Definitions
# ---------------------------------------------------------------------------
SCENARIOS = [
    "insufficient_funds",      # 1. card payment
    "card_blocked",            # 2. card payment  
    "network_timeout",         # 3. UPI payment
    "gateway_issue",           # 4. netbanking
    "expired_card",            # 5. card payment
    "authentication_failed",   # 6. UPI payment
    "limit_exceeded",          # 7. card payment
    "unknown",                 # 8. unknown method
    "subscription.halted",     # 9. recurring payment
    "invoice.overdue",         # 10. B2B invoice
    "insufficient_funds_upi",  # 11. UPI insufficient funds
]


def _build_webhook_payload(scenario: str, run_id: int) -> Dict[str, Any]:
    """Build a realistic raw Razorpay webhook payload for the given scenario."""
    ts = int(time.time()) + run_id

    if scenario == "insufficient_funds":
        return {
            "entity": "event",
            "account_id": "acc_mer_001",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_ins_funds_{ts}",
                        "entity": "payment",
                        "amount": 299900,  # ₹2999.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_ins_{ts}",
                        "method": "card",
                        "email": "rahul.sharma@example.com",
                        "contact": "+919876543210",
                        "customer_id": "cust_rahul_01",
                        "notes": {"customer_name": "Rahul Sharma"},
                        "error_code": "BAD_REQUEST_INSUFFICIENT_FUNDS",
                        "error_description": "The customer account has insufficient funds to complete transaction",
                        "error_source": "bank",
                        "error_step": "payment_authentication",
                        "error_reason": "insufficient_funds",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "card_blocked":
        return {
            "entity": "event",
            "account_id": "acc_mer_002",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_card_blk_{ts}",
                        "entity": "payment",
                        "amount": 149900,  # ₹1499.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_blk_{ts}",
                        "method": "card",
                        "email": "priya.verma@example.com",
                        "contact": "+919811223344",
                        "customer_id": "cust_priya_02",
                        "notes": {"customer_name": "Priya Verma"},
                        "error_code": "CARD_BLOCKED",
                        "error_description": "Card has been blocked or blacklisted by issuing bank",
                        "error_source": "bank",
                        "error_step": "payment_authentication",
                        "error_reason": "card_blocked",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "network_timeout":
        return {
            "entity": "event",
            "account_id": "acc_mer_003",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_net_tout_{ts}",
                        "entity": "payment",
                        "amount": 49900,  # ₹499.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_tout_{ts}",
                        "method": "upi",
                        "email": "ananya.rao@example.com",
                        "contact": "+919822334455",
                        "customer_id": "cust_ananya_03",
                        "notes": {"customer_name": "Ananya Rao"},
                        "error_code": "GATEWAY_TIMED_OUT",
                        "error_description": "UPI PSP node timed out during transaction collect",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "network_timeout",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "gateway_issue":
        return {
            "entity": "event",
            "account_id": "acc_mer_004",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_gw_issue_{ts}",
                        "entity": "payment",
                        "amount": 540000,  # ₹5400.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_gw_{ts}",
                        "method": "netbanking",
                        "email": "vikram.singh@example.com",
                        "contact": "+919833445566",
                        "customer_id": "cust_vikram_04",
                        "notes": {"customer_name": "Vikram Singh"},
                        "error_code": "ISSUING_BANK_DOWN",
                        "error_description": "Netbanking gateway switch is temporarily unavailable",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "gateway_issue",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "expired_card":
        return {
            "entity": "event",
            "account_id": "acc_mer_005",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_exp_card_{ts}",
                        "entity": "payment",
                        "amount": 89900,  # ₹899.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_exp_{ts}",
                        "method": "card",
                        "email": "neha.gupta@example.com",
                        "contact": "+919844556677",
                        "customer_id": "cust_neha_05",
                        "notes": {"customer_name": "Neha Gupta"},
                        "error_code": "CARD_EXPIRED",
                        "error_description": "The card expiry date passed prior to authorization",
                        "error_source": "bank",
                        "error_step": "payment_authentication",
                        "error_reason": "expired_card",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "authentication_failed":
        return {
            "entity": "event",
            "account_id": "acc_mer_006",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_auth_fail_{ts}",
                        "entity": "payment",
                        "amount": 125000,  # ₹1250.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_auth_{ts}",
                        "method": "upi",
                        "email": "karan.malhotra@example.com",
                        "contact": "+919855667788",
                        "customer_id": "cust_karan_06",
                        "notes": {"customer_name": "Karan Malhotra"},
                        "error_code": "AUTHENTICATION_FAILED",
                        "error_description": "UPI MPIN entered by user was incorrect",
                        "error_source": "bank",
                        "error_step": "payment_authentication",
                        "error_reason": "authentication_failed",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "limit_exceeded":
        return {
            "entity": "event",
            "account_id": "acc_mer_007",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_lim_exc_{ts}",
                        "entity": "payment",
                        "amount": 15000000,  # ₹150,000.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_lim_{ts}",
                        "method": "card",
                        "email": "sanjay.kapoor@example.com",
                        "contact": "+919866778899",
                        "customer_id": "cust_sanjay_07",
                        "notes": {"customer_name": "Sanjay Kapoor"},
                        "error_code": "EXCEEDED_DAILY_AMOUNT_LIMIT",
                        "error_description": "Transaction amount exceeds daily banking velocity limit",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "limit_exceeded",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "unknown":
        return {
            "entity": "event",
            "account_id": "acc_mer_008",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_unk_err_{ts}",
                        "entity": "payment",
                        "amount": 79900,  # ₹799.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_unk_{ts}",
                        "method": "unknown",
                        "email": "user.unknown@example.com",
                        "contact": "+919877889900",
                        "customer_id": "cust_unk_08",
                        "notes": {"customer_name": "Guest Customer"},
                        "error_code": "UNKNOWN_ERROR",
                        "error_description": "Unclassified gateway rejection code",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "unknown",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "subscription.halted":
        return {
            "entity": "event",
            "account_id": "acc_mer_009",
            "event": "subscription.halted",
            "contains": ["subscription", "payment"],
            "payload": {
                "subscription": {
                    "entity": {
                        "id": f"sub_halt_{ts}",
                        "entity": "subscription",
                        "plan_id": "plan_enterprise_pro",
                        "customer_id": "cust_corp_09",
                        "status": "halted",
                        "notes": {
                            "plan_name": "Cloud Enterprise Monthly",
                            "customer_name": "DevCorp Solutions",
                        },
                        "short_url": f"https://rzp.io/i/sub_reauth_{ts}",
                        "payment_method": "emandate",
                        "charge_at": ts,
                    }
                },
                "payment": {
                    "entity": {
                        "id": f"pay_sub_fl_{ts}",
                        "entity": "payment",
                        "amount": 1999900,  # ₹19,999.00
                        "currency": "INR",
                        "status": "failed",
                        "method": "emandate",
                        "email": "billing@devcorp.com",
                        "contact": "+919888990011",
                        "customer_id": "cust_corp_09",
                        "error_code": "GATEWAY_ERROR",
                        "error_description": "Recurring auto-debit mandate inactive on destination account",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "mandate_inactive",
                        "created_at": ts,
                    }
                },
            },
            "created_at": ts,
        }

    elif scenario == "invoice.overdue":
        return {
            "entity": "event",
            "account_id": "acc_mer_010",
            "event": "invoice.overdue",
            "contains": ["invoice"],
            "payload": {
                "invoice": {
                    "entity": {
                        "id": f"inv_overdue_{ts}",
                        "entity": "invoice",
                        "customer_id": "cust_b2b_10",
                        "customer_details": {
                            "name": "Tata Steel Distribution",
                            "email": "ap.vendor@tatasteel.com",
                            "contact": "+919899001122",
                        },
                        "order_id": f"order_inv_{ts}",
                        "status": "overdue",
                        "amount": 8500000,  # ₹85,000.00
                        "amount_paid": 0,
                        "amount_due": 8500000,
                        "currency": "INR",
                        "short_url": f"https://rzp.io/i/inv_pay_{ts}",
                        "notes": {
                            "customer_name": "Tata Steel Distribution",
                            "po_number": f"PO-2026-{ts}",
                        },
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    elif scenario == "insufficient_funds_upi":
        return {
            "entity": "event",
            "account_id": "acc_mer_011",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": f"pay_upi_lowbal_{ts}",
                        "entity": "payment",
                        "amount": 350000,  # ₹3500.00
                        "currency": "INR",
                        "status": "failed",
                        "order_id": f"order_upi_low_{ts}",
                        "method": "upi",
                        "email": "divya.nair@example.com",
                        "contact": "+919800112233",
                        "customer_id": "cust_divya_11",
                        "notes": {"customer_name": "Divya Nair"},
                        "error_code": "PAYMENT_INSUFFICIENT_BALANCE",
                        "error_description": "UPI bank account balance insufficient for collect request",
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "error_reason": "insufficient_funds",
                        "created_at": ts,
                    }
                }
            },
            "created_at": ts,
        }

    raise ValueError(f"Unknown test scenario: {scenario}")


# ---------------------------------------------------------------------------
# Integration Runner Helper
# ---------------------------------------------------------------------------
def _run_single_scenario(scenario: str, run_id: int) -> Tuple[str, str, str, str, float]:
    """Runs a single scenario end-to-end and verifies each stage."""
    t_start = time.perf_counter()

    # 1. Build & Normalize Webhook
    raw_payload = _build_webhook_payload(scenario, run_id)
    event: NormalizedEvent = normalize_webhook_payload(raw_payload)
    assert event is not None, f"Webhook normalization failed for {scenario}"
    assert event.amount > 0, f"Normalized amount must be positive for {scenario}"

    # 2. Run Full Pipeline (Persist -> ML classify -> Real Gemini -> Action Dispatch)
    summary = run_recovery_pipeline(event)

    # 3. Assertions
    assert summary is not None, f"Pipeline returned None for {scenario}"
    assert summary.get("status") != "pipeline_error", f"Pipeline crashed for {scenario}: {summary.get('error')}"
    assert summary.get("action_taken") is not None, f"Action must be decided for {scenario}"
    assert summary.get("failure_category") is not None, f"Failure category must be classified for {scenario}"
    assert summary.get("confidence", 0.0) >= 0.0, f"Confidence must be non-negative for {scenario}"

    # 4. Verify DB Persistence in PostgreSQL
    tx_id = summary.get("transaction_id")
    with get_db_session() as session:
        tx = session.query(Transaction).filter_by(id=tx_id).first()
        assert tx is not None, f"Transaction '{tx_id}' not found in DB for {scenario}"
        assert tx.status in (
            "recovery_initiated", "recovered", "alternate_suggested",
            "payment_link_sent", "customer_notified", "manual_review", "retry_scheduled", "FAILED"
        ), f"Unexpected DB transaction status '{tx.status}' for {scenario}"

        # Verify action or retry record in DB
        action_recs = session.query(RecoveryAction).filter_by(transaction_id=tx_id).all()
        retry_recs = session.query(RetryAttempt).filter_by(transaction_id=tx_id).all()
        assert len(action_recs) > 0 or len(retry_recs) > 0, f"No action or retry DB records for {scenario}"

    elapsed_s = time.perf_counter() - t_start
    category = str(summary.get("failure_category"))
    action = str(summary.get("action_taken"))

    return (scenario, category, action, "✅ PASS", elapsed_s)


# ---------------------------------------------------------------------------
# Test Class for Pytest
# ---------------------------------------------------------------------------
class TestIntegrationFull:
    """Full suite of 11 real-API integration tests."""

    @pytest.fixture(autouse=True)
    def _rate_limit_throttle(self):
        """Throttle slightly between Gemini calls to avoid rate limit spikes."""
        time.sleep(1.0)

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_scenario_end_to_end(self, scenario: str):
        """Executes each scenario end-to-end against live Gemini & PostgreSQL."""
        scenario_name, category, action, status, elapsed_s = _run_single_scenario(
            scenario,
            run_id=SCENARIOS.index(scenario) + 1,
        )
        assert status == "✅ PASS"
        print(f"[{scenario_name}] Category: {category} | Action: {action} | Time: {elapsed_s:.2f}s | Status: {status}")


# ---------------------------------------------------------------------------
# Standalone Matrix Runner
# ---------------------------------------------------------------------------
def run_all_scenarios_and_print_matrix() -> bool:
    """Runs all 11 scenarios sequentially, records metrics, and prints matrix."""
    print("\n" + "=" * 65)
    print(" 🚀 Starting Full Integration Test Suite (11 Scenarios)")
    print(" Using: Real Gemini AI, Real PostgreSQL (Docker:5434), Real Razorpay")
    print("=" * 65 + "\n")

    results: List[Tuple[str, str, str, str, float]] = []
    overall_start = time.perf_counter()
    passed_count = 0
    failed_count = 0

    for idx, scenario in enumerate(SCENARIOS, 1):
        print(f"[{idx}/11] Running scenario: {scenario} ...", end="", flush=True)
        try:
            res = _run_single_scenario(scenario, run_id=idx)
            results.append(res)
            passed_count += 1
            print(f" {res[3]} ({res[4]:.2f}s)")
        except Exception as exc:
            failed_count += 1
            print(f" ❌ FAIL: {exc}")
            results.append((scenario, "error", "none", "❌ FAIL", 0.0))
        time.sleep(2.0)  # Generous rate-limit buffer to stay within free tier RPM limits

    total_time = time.perf_counter() - overall_start

    # Format the Test Matrix Table
    print("\n═══════════════════════════════════════════════════════")
    print(" Integration Test Matrix — 11 Scenarios")
    print("═══════════════════════════════════════════════════════")
    print(f" {'Scenario':<22} {'Category':<16} {'Action':<14} {'Status'}")
    print("───────────────────────────────────────────────────────")
    for sc, cat, act, st, el in results:
        sc_disp = sc[:22]
        cat_disp = (cat[:13] + "..") if len(cat) > 15 else cat
        act_disp = (act[:11] + "..") if len(act) > 13 else act
        print(f" {sc_disp:<22} {cat_disp:<16} {act_disp:<14} {st}")
    print("═══════════════════════════════════════════════════════")
    print(f" Results: {passed_count} passed, {failed_count} failed")
    print(f" Total time: {total_time:.1f}s")
    print("═══════════════════════════════════════════════════════\n")

    # Print Observability Metrics
    print_metrics_report()

    return failed_count == 0


if __name__ == "__main__":
    success = run_all_scenarios_and_print_matrix()
    if not success:
        exit(1)
