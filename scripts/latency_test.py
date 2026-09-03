# scripts/latency_test.py
"""Latency Performance Benchmark for AI Revenue Recovery Agent.

Runs 5 payments end-to-end through the complete recovery pipeline:
  - normalize_webhook_payload()
  - save_transaction() (PostgreSQL)
  - ML classification
  - Gemini AI Recovery Decision
  - dispatch_recovery_action()

Evaluates latency against target threshold (< 3000ms per payment).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List, Dict, Any

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

from dotenv import load_dotenv

load_dotenv()

from api_integration.normalizer import normalize_webhook_payload
from api_integration.schemas import NormalizedEvent
from agent.pipeline import run_recovery_pipeline

logging.basicConfig(level=logging.WARNING)

TARGET_LATENCY_MS = 3000.0


def _build_test_payload(index: int) -> Dict[str, Any]:
    """Generates realistic payment failure payload for performance testing."""
    ts = int(time.time()) + index
    return {
        "entity": "event",
        "account_id": "acc_mer_perf_01",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": f"pay_perf_{index}_{ts}",
                    "entity": "payment",
                    "amount": 199900,  # ₹1999.00
                    "currency": "INR",
                    "status": "failed",
                    "order_id": f"order_perf_{index}_{ts}",
                    "method": "card",
                    "email": f"user{index}@example.com",
                    "contact": f"+91987654321{index}",
                    "customer_id": f"cust_perf_{index}",
                    "notes": {"customer_name": f"Perf User {index}"},
                    "error_code": "BAD_REQUEST_INSUFFICIENT_FUNDS",
                    "error_description": "Insufficient funds in customer card account",
                    "error_source": "bank",
                    "error_step": "payment_authentication",
                    "error_reason": "insufficient_funds",
                    "created_at": ts,
                }
            }
        },
        "created_at": ts,
    }


def run_latency_test(num_payments: int = 5) -> bool:
    """Runs N payments and measures end-to-end processing latency."""
    print("\n" + "=" * 55)
    print(" ⏱️  AI Revenue Recovery Agent — Pipeline Latency Test")
    print(f" Target: Under {TARGET_LATENCY_MS:.0f}ms per payment | Runs: {num_payments}")
    print("=" * 55 + "\n")

    latencies_ms: List[float] = []

    for i in range(1, num_payments + 1):
        raw_payload = _build_test_payload(i)

        # Start timer: from normalize_webhook_payload()
        t_start = time.perf_counter()

        event: NormalizedEvent = normalize_webhook_payload(raw_payload)
        summary = run_recovery_pipeline(event)

        # End timer: after dispatch completion
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0
        latencies_ms.append(elapsed_ms)

        status_tag = "✅" if elapsed_ms <= TARGET_LATENCY_MS else "⚠️ SLOW"
        print(f"Payment {i}: {elapsed_ms:.0f}ms {status_tag}")

        time.sleep(1.0)  # Gentle gap between API requests

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    print("───────────────────────────────────────────────────────")
    print(f"Average: {avg_latency:.0f}ms")
    print("=" * 55 + "\n")

    return avg_latency <= TARGET_LATENCY_MS


if __name__ == "__main__":
    success = run_latency_test(5)
    if not success:
        print("Note: Average latency exceeded target threshold.")
