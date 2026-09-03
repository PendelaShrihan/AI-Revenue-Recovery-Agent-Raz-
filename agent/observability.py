# agent/observability.py
"""Structured Logging & Observability for AI Revenue Recovery Agent.

Tracks real-time recovery metrics in-memory and provides formatting
and reporting helpers.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Dict

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-Memory Metrics Store
# ---------------------------------------------------------------------------

_METRICS_LOCK = threading.Lock()

METRICS: Dict[str, Any] = {
    "total_failures_received": 0,
    "total_recovered": 0,
    "total_failed_permanently": 0,
    "recovery_rate_percent": 0.0,
    "avg_pipeline_latency_ms": 0.0,
    "gemini_calls": 0,
    "gemini_errors": 0,
    "retries_scheduled": 0,
    "retries_executed": 0,
}


def reset_metrics() -> None:
    """Reset all in-memory metrics to zero (useful for test isolation)."""
    with _METRICS_LOCK:
        METRICS["total_failures_received"] = 0
        METRICS["total_recovered"] = 0
        METRICS["total_failed_permanently"] = 0
        METRICS["recovery_rate_percent"] = 0.0
        METRICS["avg_pipeline_latency_ms"] = 0.0
        METRICS["gemini_calls"] = 0
        METRICS["gemini_errors"] = 0
        METRICS["retries_scheduled"] = 0
        METRICS["retries_executed"] = 0


def get_metrics() -> Dict[str, Any]:
    """Return a thread-safe copy of the current metrics snapshot."""
    with _METRICS_LOCK:
        return dict(METRICS)


def record_pipeline_run(summary: Dict[str, Any]) -> None:
    """Record a pipeline execution run and update aggregate metrics.

    Args:
        summary: Result dictionary returned by ``run_recovery_pipeline()``.
    """
    with _METRICS_LOCK:
        METRICS["total_failures_received"] += 1
        total_runs = METRICS["total_failures_received"]

        # Track pipeline latency moving average
        latency = float(summary.get("elapsed_ms", 0.0))
        if total_runs == 1:
            METRICS["avg_pipeline_latency_ms"] = round(latency, 2)
        else:
            prev_avg = METRICS["avg_pipeline_latency_ms"]
            # Cumulative moving average
            new_avg = prev_avg + (latency - prev_avg) / total_runs
            METRICS["avg_pipeline_latency_ms"] = round(new_avg, 2)

        # Track LLM calls
        if summary.get("status") != "pipeline_error" or summary.get("confidence") is not None:
            METRICS["gemini_calls"] += 1

        if summary.get("status") == "pipeline_error":
            # Check if it was an LLM error or general pipeline error
            err_msg = str(summary.get("error", "")).lower()
            if "gemini" in err_msg or "llm" in err_msg:
                METRICS["gemini_errors"] += 1
            METRICS["total_failed_permanently"] += 1

        action = summary.get("action_taken") or summary.get("action")
        status = summary.get("status")

        if action == "auto_retry" or summary.get("retry_after"):
            METRICS["retries_scheduled"] += 1
        elif status == "recovered":
            METRICS["total_recovered"] += 1
        elif action in ("manual_review", "no_action") or status == "customer_notified":
            METRICS["total_failed_permanently"] += 1

        # Calculate recovery rate percent
        if total_runs > 0:
            rate = (METRICS["total_recovered"] / total_runs) * 100.0
            METRICS["recovery_rate_percent"] = round(rate, 2)

    _logger.debug(
        "[Observability] Recorded pipeline run for tx='%s' | Latency=%.1fms | Total received=%d",
        summary.get("transaction_id"),
        summary.get("elapsed_ms", 0.0),
        METRICS["total_failures_received"],
    )


def record_retry_execution(result: Dict[str, Any]) -> None:
    """Record the outcome of a retry execution.

    Args:
        result: Outcome dict from ``execute_retry()``.
    """
    with _METRICS_LOCK:
        METRICS["retries_executed"] += 1
        status = result.get("status", "").lower()
        if status == "recovered":
            METRICS["total_recovered"] += 1
        elif status == "failed" and not result.get("next_retry_at"):
            # Permanently exhausted retries
            METRICS["total_failed_permanently"] += 1

        total = METRICS["total_failures_received"]
        if total > 0:
            rate = (METRICS["total_recovered"] / total) * 100.0
            METRICS["recovery_rate_percent"] = round(rate, 2)


def print_metrics_report() -> None:
    """Print a clean formatted observability metrics table to console/logger."""
    m = get_metrics()
    report = (
        "\n═══════════════════════════════════════════════════════\n"
        " 📊 Razorpay Recovery Agent Observability Metrics\n"
        "═══════════════════════════════════════════════════════\n"
        f" Total Failures Received    : {m['total_failures_received']}\n"
        f" Total Recovered Payments   : {m['total_recovered']}\n"
        f" Total Permanently Failed   : {m['total_failed_permanently']}\n"
        f" Recovery Rate              : {m['recovery_rate_percent']:.2f}%\n"
        f" Avg Pipeline Latency       : {m['avg_pipeline_latency_ms']:.2f} ms\n"
        f" Gemini AI Calls Total      : {m['gemini_calls']}\n"
        f" Gemini AI Errors           : {m['gemini_errors']}\n"
        f" Retries Scheduled          : {m['retries_scheduled']}\n"
        f" Retries Executed           : {m['retries_executed']}\n"
        "═══════════════════════════════════════════════════════\n"
    )
    print(report)
    _logger.info(report)


__all__ = [
    "METRICS",
    "reset_metrics",
    "get_metrics",
    "record_pipeline_run",
    "record_retry_execution",
    "print_metrics_report",
]
