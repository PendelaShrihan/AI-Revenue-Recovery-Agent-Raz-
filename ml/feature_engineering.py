"""
Feature Engineering Pipeline for Razorpay Failure Classification.

Converts raw payment records (list of dicts from DB / NormalizedEvent / webhook)
into a structured feature DataFrame for sklearn Pipeline consumption.

Design principle:
    This class is intentionally STATELESS (no fit step). It performs only
    deterministic feature extraction:
        - Rule-based error category derivation (from ml.error_codes)
        - Payment method normalization
        - Cyclical hour-of-day encoding (sin/cos — no fitting required)
        - Merchant volume-tier derivation (deterministic hash)

    Statistical transforms (StandardScaler, OrdinalEncoder) are deliberately
    excluded here and live inside FailureClassifier's sklearn.pipeline.Pipeline
    so they fit only on training data during each CV fold — preventing data leakage.
"""

import math
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ml.error_codes import FailureCategory, classify_failure

logger = logging.getLogger(__name__)

# ─── Feature Schema ─────────────────────────────────────────────────────────────

#: Recognized payment methods (anything else maps to 'unknown')
PAYMENT_METHODS: List[str] = ["card", "upi", "netbanking", "wallet", "emi", "unknown"]

#: Merchant volume tiers derived from merchant_id hash
MERCHANT_CATEGORIES: List[str] = ["low_volume", "medium_volume", "high_volume"]

#: All 8 canonical failure categories from error_codes.FailureCategory
ERROR_CATEGORIES: List[str] = [cat.value for cat in FailureCategory]

# Column name groups — referenced by FailureClassifier's ColumnTransformer
CATEGORICAL_FEATURES: List[str] = ["error_category", "payment_method", "merchant_category"]
NUMERICAL_FEATURES: List[str] = ["amount"]
PASSTHROUGH_FEATURES: List[str] = ["hour_sin", "hour_cos", "is_weekend"]
ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + PASSTHROUGH_FEATURES


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _derive_merchant_category(merchant_id: str) -> str:
    """
    Assigns a deterministic volume tier to a merchant from its ID.

    Uses SHA-256 hash mod 3 → low / medium / high_volume.
    This is a stable proxy until real MCC codes are available via the
    Razorpay Account API (GET /v1/accounts/{id}).

    Args:
        merchant_id: Razorpay account ID (e.g. 'acc_XXXXXXXXXX') or any string.

    Returns:
        One of: 'low_volume', 'medium_volume', 'high_volume'
    """
    digest = hashlib.sha256(merchant_id.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 3
    return MERCHANT_CATEGORIES[bucket]


def _parse_datetime(value: Any) -> datetime:
    """
    Safely coerces various timestamp representations to datetime.

    Handles:
        - datetime objects (returned as-is)
        - Unix integer / float timestamps
        - ISO 8601 strings (with or without 'Z' suffix)
        - None / unrecognized types (falls back to datetime.utcnow())
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            return datetime.utcnow()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.utcnow()


# ─── Pipeline Class ────────────────────────────────────────────────────────────

class FeatureEngineeringPipeline:
    """
    Stateless feature extractor for Razorpay payment failure records.

    Accepts records as list-of-dicts in any of these shapes:
        - Raw SQLite row dicts (from scripts/train_classifier.py)
        - NormalizedEvent.model_dump() (from api_integration/normalizer.py)
        - Webhook payload flattened dicts

    Recognized input keys (all optional, missing values are handled gracefully):
        error_code, error_reason, failure_code, failure_reason,
        payment_method, merchant_id, amount, created_at

    Returns a pd.DataFrame with exactly these columns (ALL_FEATURES):
        error_category, payment_method, merchant_category,   ← categorical
        amount,                                               ← numerical
        hour_sin, hour_cos, is_weekend                        ← passthrough
    """

    def extract_features(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extracts feature values from a list of payment record dicts.

        Args:
            records: List of payment record dicts.

        Returns:
            pd.DataFrame with columns matching ALL_FEATURES.
            Returns an empty DataFrame with correct columns if records is empty.
        """
        if not records:
            return pd.DataFrame(columns=ALL_FEATURES)

        rows = []
        for r in records:
            # ── Error category (rule-based classifier as feature signal) ──
            error_cat = classify_failure(
                error_code=r.get("error_code") or r.get("failure_code"),
                error_reason=r.get("error_reason") or r.get("failure_reason"),
            )

            # ── Payment method normalization ──────────────────────────────
            raw_method = str(r.get("payment_method") or "unknown").lower().strip()
            payment_method = raw_method if raw_method in PAYMENT_METHODS else "unknown"

            # ── Merchant volume tier (deterministic, no fit needed) ───────
            merchant_id = str(r.get("merchant_id") or "unknown")
            merchant_category = _derive_merchant_category(merchant_id)

            # ── Amount (INR float, clamped to non-negative) ───────────────
            try:
                amount = max(0.0, float(r.get("amount") or 0.0))
            except (TypeError, ValueError):
                amount = 0.0

            # ── Cyclical hour-of-day encoding ─────────────────────────────
            # sin/cos preserves periodicity: hour 23 is adjacent to hour 0.
            # This is a deterministic transform — no fitting required.
            created_at = _parse_datetime(r.get("created_at"))
            hour = created_at.hour
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
            is_weekend = int(created_at.weekday() >= 5)

            rows.append({
                "error_category": error_cat,
                "payment_method": payment_method,
                "merchant_category": merchant_category,
                "amount": amount,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "is_weekend": is_weekend,
            })

        return pd.DataFrame(rows, columns=ALL_FEATURES)

    def extract_labels(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts target label strings from records using the rule-based classifier.

        The rule-based classifier's output serves as the training label — the ML
        model learns to generalize this classification using payment context features
        (amount, payment method, time patterns) beyond just error codes alone.

        Args:
            records: List of payment record dicts.

        Returns:
            np.ndarray of FailureCategory string labels (shape: [n_samples]).
        """
        return np.array([
            classify_failure(
                error_code=r.get("error_code") or r.get("failure_code"),
                error_reason=r.get("error_reason") or r.get("failure_reason"),
            )
            for r in records
        ])

    def save(self, path: str) -> None:
        """Serializes this pipeline instance to disk via joblib."""
        joblib.dump(self, path)
        logger.info(f"FeatureEngineeringPipeline saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineeringPipeline":
        """Loads a previously saved FeatureEngineeringPipeline from disk."""
        instance = joblib.load(path)
        logger.info(f"FeatureEngineeringPipeline loaded ← {path}")
        return instance
