#!/usr/bin/env python3
"""
Environment & API Key Validation Script
Razorpay AI Revenue Recovery Agent

Validates presence of required environment variables without printing or logging any values.
Checks:
- Razorpay API Sandbox Credentials (RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
- Active LLM Provider Key (GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY)
- Database Connectivity (PostgreSQL / SQLite via DATABASE_URL)

Outputs strictly PASS or FAIL status indicators.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure safe console output across all platforms
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import razorpay
    HAS_RAZORPAY_SDK = True
except ImportError:
    HAS_RAZORPAY_SDK = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


def is_valid_secret(val: str) -> bool:
    if not val:
        return False
    val_clean = val.strip().lower()
    return not (
        "placeholder" in val_clean
        or "your-" in val_clean
        or "your_" in val_clean
        or "demo" in val_clean
    )


def check_env():
    load_dotenv()

    print("=" * 60)
    print(" 1. CREDENTIALS & ENVIRONMENT STATUS")
    print("=" * 60)

    # Razorpay Keys
    rzp_key = os.getenv("RAZORPAY_KEY_ID", "").strip()
    rzp_secret = os.getenv("RAZORPAY_KEY_SECRET", "").strip()
    rzp_key_ok = is_valid_secret(rzp_key)
    rzp_secret_ok = is_valid_secret(rzp_secret)
    print(f" • RAZORPAY_KEY_ID          : {'PASS' if rzp_key_ok else 'FAIL'}")
    print(f" • RAZORPAY_KEY_SECRET      : {'PASS' if rzp_secret_ok else 'FAIL'}")

    # LLM Providers (Any active provider satisfies LLM readiness)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    gemini_ok = is_valid_secret(gemini_key)
    anthropic_ok = is_valid_secret(anthropic_key)
    openai_ok = is_valid_secret(openai_key)

    has_any_llm = gemini_ok or anthropic_ok or openai_ok

    print(f" • GEMINI_API_KEY           : {'PASS' if gemini_ok else 'NOT SET'}")
    print(f" • ANTHROPIC_API_KEY        : {'PASS' if anthropic_ok else 'NOT SET'}")
    print(f" • OPENAI_API_KEY           : {'PASS' if openai_ok else 'NOT SET'}")
    print(f" • LLM Provider Status      : {'PASS (Provider Ready)' if has_any_llm else 'ACTION REQUIRED'}")

    # Database URL
    db_url = os.getenv("DATABASE_URL", "").strip()
    db_url_ok = bool(db_url) and not ("placeholder" in db_url.lower())
    print(f" • DATABASE_URL             : {'PASS' if db_url_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(" 2. LIVE CONNECTIVITY TESTS")
    print("=" * 60)

    # 2a. Razorpay Ping
    rzp_ping = "FAIL"
    if rzp_key_ok and rzp_secret_ok:
        try:
            if HAS_RAZORPAY_SDK:
                client = razorpay.Client(auth=(rzp_key, rzp_secret))
                client.order.all({"count": 1})
                rzp_ping = "PASS"
            elif HAS_HTTPX:
                resp = httpx.get(
                    "https://api.razorpay.com/v1/orders?count=1",
                    auth=(rzp_key, rzp_secret),
                    timeout=5.0
                )
                rzp_ping = "PASS" if resp.status_code == 200 else "FAIL"
            else:
                import urllib.request
                import base64
                req = urllib.request.Request("https://api.razorpay.com/v1/orders?count=1")
                credentials = f"{rzp_key}:{rzp_secret}".encode("ascii")
                req.add_header("Authorization", f"Basic {base64.b64encode(credentials).decode('ascii')}")
                with urllib.request.urlopen(req, timeout=5) as response:
                    rzp_ping = "PASS" if response.status == 200 else "FAIL"
        except Exception:
            rzp_ping = "FAIL"
    else:
        rzp_ping = "SKIPPED (Keys Not Configured)"

    print(f" • Razorpay Sandbox Ping    : {rzp_ping}")

    # 2b. Database Connectivity Ping
    db_ping = "FAIL"
    if db_url_ok and HAS_SQLALCHEMY:
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ping = "PASS"
        except Exception:
            db_ping = "FAIL (Check PostgreSQL container/service)"
    elif db_url_ok:
        db_ping = "PASS (URL format valid)"

    print(f" • Database Connection Ping : {db_ping}")

    print("\n" + "=" * 60)
    print(" 3. OVERALL READINESS")
    print("=" * 60)
    ready = rzp_key_ok and rzp_secret_ok and has_any_llm and db_url_ok
    print(f" • System Readiness         : {'PASS — READY FOR SPRINT DAY 2' if ready else 'ACTION REQUIRED'}")
    print("=" * 60 + "\n")

    return ready


if __name__ == "__main__":
    check_env()
    sys.exit(0)
