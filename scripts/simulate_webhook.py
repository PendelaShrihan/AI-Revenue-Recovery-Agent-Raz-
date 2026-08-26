"""
Webhook Simulation & Testing Tool
Simulates live Razorpay webhook events by signing payloads with HMAC-SHA256
and dispatching them to the local or remote listener endpoint.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

# Safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

# Import signature computer from api_integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api_integration.verifier import compute_webhook_signature

try:
    import httpx
except ImportError:
    import requests as httpx


SAMPLE_PAYLOADS = {
    "payment.failed": {
        "entity": "event",
        "account_id": "acc_sim_merchant_01",
        "event": "payment.failed",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_SIM_FAIL_901",
                    "entity": "payment",
                    "amount": 249900,  # ₹2,499.00 in paise
                    "currency": "INR",
                    "status": "failed",
                    "order_id": "order_ORD_901",
                    "method": "card",
                    "card_id": "card_901",
                    "bank": "HDFC",
                    "email": "customer@example.com",
                    "contact": "+919876543210",
                    "customer_id": "cust_SIM_901",
                    "notes": {
                        "customer_name": "Aditi Sharma",
                        "friction_note": "Customer bank OTP page froze during checkout",
                        "merchant_category": "electronics"
                    },
                    "error_code": "GATEWAY_ERROR",
                    "error_description": "Payment failed due to bank OTP verification timeout",
                    "error_source": "bank",
                    "error_step": "payment_authentication",
                    "error_reason": "network_timeout",
                    "created_at": 1756200000
                }
            }
        },
        "created_at": 1756200000
    },
    "subscription.halted": {
        "entity": "event",
        "account_id": "acc_sim_merchant_02",
        "event": "subscription.halted",
        "contains": ["subscription", "payment"],
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_SIM_HALT_702",
                    "entity": "subscription",
                    "plan_id": "plan_PRO_MONTHLY",
                    "customer_id": "cust_CORP_702",
                    "status": "halted",
                    "auth_attempts": 3,
                    "total_count": 12,
                    "paid_count": 2,
                    "notes": {
                        "plan_name": "SaaS Pro Enterprise Monthly",
                        "customer_name": "Rohan Verma"
                    },
                    "short_url": "https://rzp.io/i/sub_sim_reauth",
                    "charge_at": 1756200000
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_recurring_fail_702",
                    "amount": 1499900,  # ₹14,999.00
                    "currency": "INR",
                    "status": "failed",
                    "method": "emandate",
                    "email": "finance@clientcorp.com",
                    "contact": "+919811223344",
                    "error_code": "GATEWAY_ERROR",
                    "error_description": "Recurring auto-debit mandate expired or rejected by issuing bank",
                    "error_reason": "mandate_inactive"
                }
            }
        },
        "created_at": 1756200000
    },
    "invoice.overdue": {
        "entity": "event",
        "account_id": "acc_sim_merchant_03",
        "event": "invoice.overdue",
        "contains": ["invoice"],
        "payload": {
            "invoice": {
                "entity": {
                    "id": "inv_SIM_OVERDUE_503",
                    "entity": "invoice",
                    "customer_id": "cust_B2B_503",
                    "customer_details": {
                        "name": "Tata Consultancy Subcontracting",
                        "email": "billing@tata.com",
                        "contact": "+919822334455"
                    },
                    "status": "overdue",
                    "amount": 7500000,
                    "amount_due": 7500000,  # ₹75,000.00
                    "currency": "INR",
                    "short_url": "https://rzp.io/i/inv_sim_503",
                    "notes": {
                        "po_number": "PO-2026-AUG-9912",
                        "client_tier": "Enterprise A+",
                        "relationship_notes": "Awaiting VP Procurement sign-off on Net-30 invoice"
                    },
                    "created_at": 1755000000
                }
            }
        },
        "created_at": 1756200000
    }
}


def send_simulated_webhook(event_type: str, target_url: str = "http://localhost:8000/webhooks/razorpay"):
    """Signs and sends a simulated Razorpay webhook to the target listener."""
    if event_type not in SAMPLE_PAYLOADS:
        print(f"❌ Error: Unknown event '{event_type}'. Supported: {list(SAMPLE_PAYLOADS.keys())}")
        return False

    payload = SAMPLE_PAYLOADS[event_type]
    payload_bytes = json.dumps(payload).encode("utf-8")
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "demo_webhook_secret_mock")

    # Compute HMAC-SHA256 signature
    signature = compute_webhook_signature(payload_bytes, webhook_secret)

    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature
    }

    print(f"\n============================================================")
    print(f" 🚀 Dispatching Simulated Razorpay Webhook")
    print(f"============================================================")
    print(f" • Target URL       : {target_url}")
    print(f" • Event Type       : {event_type}")
    print(f" • Signature        : {signature[:12]}...{signature[-8:]}")
    print(f" • Using Secret     : {'(set in .env)' if os.getenv('RAZORPAY_WEBHOOK_SECRET') else '(demo fallback)'}")

    try:
        if hasattr(httpx, "post"):
            response = httpx.post(target_url, content=payload_bytes, headers=headers, timeout=10.0)
            status_code = response.status_code
            try:
                res_data = response.json()
            except Exception:
                res_data = response.text
        else:
            import urllib.request
            req = urllib.request.Request(target_url, data=payload_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status_code = resp.status
                res_data = json.loads(resp.read().decode("utf-8"))

        print(f"\n📥 Response:")
        print(f" • Status Code      : {status_code} ({'SUCCESS' if status_code == 200 else 'FAILED'})")
        if isinstance(res_data, dict):
            print(f" • Action Taken     : {res_data.get('action_taken')}")
            print(f" • Message          : {res_data.get('message')}")
            norm = res_data.get("normalized_event", {})
            if norm:
                print(f" • Entity ID        : {norm.get('entity_id')}")
                print(f" • Normalized Amount: ₹{norm.get('amount'):,.2f} {norm.get('currency')}")
                print(f" • Failure Stream   : {norm.get('failure_category')}")
                print(f" • Diagnostic Error : {norm.get('error_code')} ({norm.get('error_reason')})")
        else:
            print(f" • Body             : {res_data}")
        print(f"============================================================\n")
        return status_code == 200

    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        print(f"   Make sure 'uvicorn main:app --reload --port 8000' is running!\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simulate Razorpay Webhook Ingestion")
    parser.add_argument(
        "--event",
        choices=["payment.failed", "subscription.halted", "invoice.overdue", "all"],
        default="all",
        help="Event type to simulate (default: all)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/webhooks/razorpay",
        help="Target Webhook URL (default: http://localhost:8000/webhooks/razorpay)"
    )
    args = parser.parse_args()

    if args.event == "all":
        print("\n🧪 Running full simulation of all 3 failure event streams...")
        for evt in ["payment.failed", "subscription.halted", "invoice.overdue"]:
            send_simulated_webhook(evt, target_url=args.url)
    else:
        send_simulated_webhook(args.event, target_url=args.url)


if __name__ == "__main__":
    main()
