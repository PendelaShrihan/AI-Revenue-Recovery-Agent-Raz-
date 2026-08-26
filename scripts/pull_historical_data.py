#!/usr/bin/env python3
"""
Razorpay Historical Failed Payments Ingestion Pipeline.

Fetches the last 90 days of failed payments from the real Razorpay /v1/payments API,
normalizes each payload into a standard NormalizedEvent, persists records
to the real PostgreSQL database via db_writer, and prints a comprehensive execution summary.

Strict Policy:
- Always uses the real Razorpay API (api.razorpay.com) with credentials from .env.
- Always uses the real PostgreSQL database configured in DATABASE_URL.
- Never uses mocks, stubs, or synthetic data as substitutes for real API calls.
- If a real API call fails, stops execution immediately and prints the exact error and status code.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from api_integration.normalizer import normalize_webhook_payload
from api_integration.schemas import NormalizedEvent
from agent.db_writer import save_transaction, init_db, get_database_url
from ml.error_codes import classify_failure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pull_historical_data")

# Configure UTF-8 stdout encoding where possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def fetch_from_razorpay_api(
    days: int = 90,
    count: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetches real failed payments directly from live Razorpay REST API (https://api.razorpay.com/v1/payments).
    
    Raises:
        RuntimeError: If credentials are missing, or if the API call fails / is rejected.
    """
    key_id = os.getenv("RAZORPAY_KEY_ID", "").strip()
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()

    if not key_id or not key_secret:
        raise RuntimeError(
            "Missing Razorpay API credentials in .env!\n"
            "Please ensure both RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET are set."
        )

    try:
        import razorpay
        import razorpay.errors
    except ImportError as e:
        raise RuntimeError(f"Required package 'razorpay' is not installed: {e}") from e

    client = razorpay.Client(auth=(key_id, key_secret))

    to_ts = int(datetime.now(timezone.utc).timestamp())
    from_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    logger.info(f"Connecting to real Razorpay API (api.razorpay.com)...")
    logger.info(f"Querying payments window: {datetime.fromtimestamp(from_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} to {datetime.fromtimestamp(to_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    all_failed_payments = []
    skip = 0
    batch_size = min(100, count)

    try:
        while len(all_failed_payments) < count:
            # Query Razorpay Payments API
            response = client.payment.all({
                "from": from_ts,
                "to": to_ts,
                "count": batch_size,
                "skip": skip
            })

            items = response.get("items", [])
            if not items:
                logger.info("No further payments found in this time range.")
                break

            # Filter for failed transactions
            failed_in_batch = [p for p in items if str(p.get("status", "")).lower() == "failed"]
            all_failed_payments.extend(failed_in_batch)
            skip += len(items)

            logger.info(f"Retrieved page with {len(items)} total payments ({len(failed_in_batch)} failed) [skip={skip}]")

            # If fewer items than batch_size returned, end of data reached
            if len(items) < batch_size:
                break

    except razorpay.errors.BadRequestError as err:
        error_msg = getattr(err, "message", str(err))
        status_code = getattr(err, "status_code", 400)
        raise RuntimeError(
            f"Razorpay API Bad Request [HTTP {status_code}]: {error_msg}\n"
            f"Endpoint: https://api.razorpay.com/v1/payments"
        ) from err
    except razorpay.errors.ServerError as err:
        status_code = getattr(err, "status_code", 500)
        raise RuntimeError(
            f"Razorpay Gateway Server Error [HTTP {status_code}]: {err}"
        ) from err
    except Exception as err:
        status_code = getattr(err, "status_code", "N/A")
        raise RuntimeError(
            f"Failed to communicate with Razorpay API [Status: {status_code}]: {err}\n"
            f"Please verify network connectivity and credentials in .env (Key ID: {key_id[:8]}...)"
        ) from err

    logger.info(f"Successfully retrieved {len(all_failed_payments)} failed payment records from Razorpay API.")

    # Wrap real payment entities into standard webhook envelope format for normalizer
    wrapped_payloads = []
    account_id = os.getenv("RAZORPAY_ACCOUNT_ID") or key_id
    for payment_entity in all_failed_payments:
        wrapped_payloads.append({
            "entity": "event",
            "account_id": account_id,
            "event": "payment.failed",
            "contains": ["payment"],
            "created_at": payment_entity.get("created_at"),
            "id": f"evt_{payment_entity.get('id', '')}",
            "payload": {
                "payment": {
                    "entity": payment_entity
                }
            }
        })

    return wrapped_payloads


def pull_and_ingest_historical_data(
    days: int = 90,
    count: int = 100
) -> Dict[str, Any]:
    """
    Orchestrates historical payment pull from real Razorpay API,
    normalization, deduplication, and database persistence into real PostgreSQL.

    Raises:
        RuntimeError: If API call or database connection fails.
    """
    start_time = time.time()
    db_url = get_database_url()
    
    logger.info(f"Initializing database schema against configured DATABASE_URL: {db_url}")
    try:
        init_db(db_url)
    except Exception as e:
        raise RuntimeError(
            f"Database initialization failed against '{db_url}': {e}\n"
            f"Please ensure PostgreSQL server is running and accessible."
        ) from e

    # Fetch Real Payments from Razorpay API
    raw_payloads = fetch_from_razorpay_api(days=days, count=count)

    total_fetched = len(raw_payloads)
    total_saved = 0
    duplicates_skipped = 0
    total_revenue_at_risk = 0.0
    category_counts: Dict[str, int] = {}
    payment_method_counts: Dict[str, int] = {}

    if total_fetched > 0:
        logger.info(f"Processing and normalizing {total_fetched} real failed payment events...")

        for raw in raw_payloads:
            try:
                normalized_event: NormalizedEvent = normalize_webhook_payload(raw)
                
                # Categorize using ML error codes classifier
                category = classify_failure(
                    error_code=normalized_event.error_code,
                    error_reason=normalized_event.error_reason
                )
                category_counts[category] = category_counts.get(category, 0) + 1

                method = normalized_event.payment_method or "unknown"
                payment_method_counts[method] = payment_method_counts.get(method, 0) + 1

                # Save to real database via db_writer
                tx, is_created = save_transaction(normalized_event)

                if is_created:
                    total_saved += 1
                    total_revenue_at_risk += normalized_event.amount
                else:
                    duplicates_skipped += 1

            except Exception as e:
                logger.error(f"Error persisting record: {e}", exc_info=True)
                raise RuntimeError(f"Failed to process and persist record: {e}") from e
    else:
        logger.info("Razorpay API returned 0 failed payments for the specified window.")

    elapsed_sec = round(time.time() - start_time, 3)

    summary = {
        "days_window": days,
        "source": "REAL_RAZORPAY_API (api.razorpay.com)",
        "total_fetched": total_fetched,
        "total_saved": total_saved,
        "duplicates_skipped": duplicates_skipped,
        "total_revenue_at_risk_inr": round(total_revenue_at_risk, 2),
        "category_breakdown": category_counts,
        "method_breakdown": payment_method_counts,
        "elapsed_seconds": elapsed_sec,
        "database_url": db_url
    }

    return summary


def print_summary_report(summary: Dict[str, Any]) -> None:
    """Prints a structured terminal report of the real historical ingestion results."""
    print("\n" + "=" * 70)
    print(" [>] RAZORPAY HISTORICAL DATA INGESTION REPORT (REAL API)")
    print("=" * 70)
    print(f" * Time Window:          Last {summary['days_window']} Days")
    print(f" * Data Source:          {summary['source']}")
    print(f" * Target Database:      {summary['database_url']}")
    print(f" * Execution Duration:   {summary['elapsed_seconds']}s")
    print("-" * 70)
    print(f" [IN]  Total Failed Fetched: {summary['total_fetched']}")
    print(f" [DB]  Total Saved (New):    {summary['total_saved']}")
    print(f" [DUP] Duplicates Skipped:   {summary['duplicates_skipped']}")
    print(f" [INR] Total Amount at Risk: Rs. {summary['total_revenue_at_risk_inr']:,.2f} INR")
    
    if summary["category_breakdown"]:
        print("-" * 70)
        print(" [CLASS] Classification Breakdown (by ml.error_codes):")
        for cat, count in sorted(summary["category_breakdown"].items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(count * 20 / max(summary['total_fetched'], 1))
            print(f"   - {cat:<24} : {count:>3} records  {bar}")

    if summary["method_breakdown"]:
        print("-" * 70)
        print(" [PAY] Payment Method Distribution:")
        for method, count in sorted(summary["method_breakdown"].items(), key=lambda x: x[1], reverse=True):
            print(f"   - {method:<16} : {count:>3} records")
            
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pull last 90 days of failed payments from real Razorpay API and persist to PostgreSQL.")
    parser.add_argument("--days", type=int, default=90, help="Historical time window in days (default: 90)")
    parser.add_argument("--count", type=int, default=100, help="Maximum records to fetch (default: 100)")
    parser.add_argument("--db-url", type=str, default=None, help="Override database connection URL")

    args = parser.parse_args()

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    try:
        summary = pull_and_ingest_historical_data(
            days=args.days,
            count=args.count
        )
        print_summary_report(summary)
    except Exception as exc:
        print("\n" + "!" * 70, file=sys.stderr)
        print(f" [ERROR] Historical Data Ingestion Failed!", file=sys.stderr)
        print(f" Reason: {exc}", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
