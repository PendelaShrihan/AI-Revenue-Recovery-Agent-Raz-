#!/usr/bin/env python3
"""
Razorpay Failure Classification Model — Training Script.

End-to-end training workflow:
    1. Load payment records from SQLite (default) or fall back to synthetic data.
    2. Extract features via FeatureEngineeringPipeline (stateless, no leakage risk).
    3. Strict stratified 80/20 train/test split before any fitting.
    4. 5-fold stratified CV on TRAIN set only — sklearn Pipeline ensures
       StandardScaler / OrdinalEncoder fit only on each fold's training split.
    5. Final fit on full training partition.
    6. Evaluate on held-out test set (never seen by model during training or CV).
    7. Save model artifacts to ml/models/.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --db-path data/recovery_agent.db
    python scripts/train_classifier.py --model-dir ml/models --cv-folds 5
    python scripts/train_classifier.py --test-size 0.2 --synthetic
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sklearn.model_selection import train_test_split

from ml.classifier import FailureClassifier
from ml.error_codes import FailureCategory
from ml.feature_engineering import FeatureEngineeringPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_classifier")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_from_sqlite(db_path: str) -> List[Dict[str, Any]]:
    """
    Loads all transaction records from the local SQLite database.

    Queries the 'transactions' table created by scripts/init_db.sql.
    Maps failure_code → error_code and failure_reason → error_reason so the
    FeatureEngineeringPipeline can apply classify_failure() correctly.

    Args:
        db_path: Absolute or relative path to the SQLite .db file.

    Returns:
        List of row dicts.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT
                id,
                merchant_id,
                amount,
                currency,
                status,
                failure_reason  AS error_reason,
                failure_code    AS error_code,
                created_at
            FROM transactions
            """
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    logger.info(f"Loaded {len(rows)} records from SQLite: {os.path.abspath(db_path)}")
    return rows


def generate_synthetic_records(n: int = 600) -> List[Dict[str, Any]]:
    """
    Generates a labeled synthetic dataset when the database has insufficient data.

    Ensures balanced class representation and realistic feature distributions
    so the full training pipeline can run end-to-end in CI / dev environments
    without live Razorpay API credentials.

    Args:
        n: Number of synthetic records to generate (default 600).

    Returns:
        List of synthetic payment record dicts.
    """
    rng = random.Random(42)

    # Map each failure category to realistic (error_code, error_reason) pairs
    cat_to_signals = {
        "insufficient_funds":   [("INSUFFICIENT_FUNDS", "insufficient_funds"),
                                  ("PAYMENT_INSUFFICIENT_BALANCE", "low_balance")],
        "card_blocked":         [("CARD_BLOCKED", "card_declined"),
                                  ("DO_NOT_HONOUR", "do_not_honour"),
                                  ("CARD_DECLINED", "card_blocked")],
        "network_timeout":      [("TRANSACTION_TIMED_OUT", "network_timeout"),
                                  ("GATEWAY_TIMED_OUT", "timed_out"),
                                  ("CONNECTION_TIMEOUT", "connection_timeout")],
        "gateway_issue":        [("GATEWAY_ERROR", "gateway_error"),
                                  ("ISSUER_DOWN", "issuer_down"),
                                  ("SERVER_ERROR", "server_error")],
        "expired_card":         [("CARD_EXPIRED", "card_expired"),
                                  ("EXPIRED_CARD", "expired_card")],
        "authentication_failed":[("AUTHENTICATION_FAILED", "authentication_failed"),
                                  ("OTP_INCORRECT", "incorrect_otp"),
                                  ("INVALID_PIN", "invalid_pin")],
        "limit_exceeded":       [("LIMIT_EXCEEDED", "daily_limit_exceeded"),
                                  ("VELOCITY_EXCEEDED", "limit_exceeded")],
        "unknown":              [("BAD_REQUEST_ERROR", None),
                                  ("BAD_REQUEST_ERROR", "")],
    }

    payment_methods = ["card", "upi", "netbanking", "wallet", "emi"]
    merchant_ids    = [f"acc_{i:08d}" for i in range(30)]
    categories      = list(cat_to_signals.keys())
    base_time       = datetime.utcnow() - timedelta(days=90)

    records: List[Dict[str, Any]] = []
    for i in range(n):
        cat = categories[i % len(categories)]  # even distribution across classes
        code, reason = rng.choice(cat_to_signals[cat])
        created = base_time + timedelta(
            days=rng.randint(0, 89),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        records.append({
            "id":             f"tx_syn_{i:06d}",
            "merchant_id":    rng.choice(merchant_ids),
            "amount":         round(rng.uniform(50.0, 75_000.0), 2),
            "currency":       "INR",
            "status":         "FAILED",
            "error_code":     code,
            "error_reason":   reason,
            "payment_method": rng.choice(payment_methods),
            "created_at":     created.isoformat(),
        })

    # Shuffle so stratify= doesn't see sorted label blocks
    rng.shuffle(records)
    logger.info(f"Generated {n} synthetic records (evenly distributed across 8 classes).")
    return records


# ─── Training Workflow ────────────────────────────────────────────────────────

def run_training(
    db_path: str,
    model_dir: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    min_records: int = 50,
    force_synthetic: bool = False,
) -> None:
    """
    Orchestrates the full train → CV → evaluate → save pipeline.

    Data leakage prevention steps:
        1. train/test split is done BEFORE fit_transform (pure extraction here).
        2. CV uses a fresh sklearn Pipeline — scaler fits per fold.
        3. Test set is touched only once, after the final model is trained.

    Args:
        db_path:         Path to SQLite database.
        model_dir:       Directory to save trained model artifacts.
        test_size:       Fraction for held-out test set (default 0.2).
        cv_folds:        Number of stratified CV folds (default 5).
        min_records:     Minimum DB records before falling back to synthetic data.
        force_synthetic: If True, skip DB and use synthetic data regardless.
    """
    print("\n" + "=" * 70)
    print("  Razorpay Failure Classification — Training Pipeline")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if force_synthetic:
        records = generate_synthetic_records(n=600)
    else:
        try:
            records = load_from_sqlite(db_path)
        except FileNotFoundError as e:
            logger.warning(str(e))
            records = []

        if len(records) < min_records:
            logger.warning(
                f"Only {len(records)} records in DB (minimum: {min_records}). "
                "Generating synthetic dataset..."
            )
            records = generate_synthetic_records(n=max(600, min_records * 12))

    # ── 2. Feature extraction (stateless — no leakage risk) ──────────────────
    fp = FeatureEngineeringPipeline()
    X_df = fp.extract_features(records)
    y    = fp.extract_labels(records)

    class_dist = {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"\n  Records:           {len(X_df)}")
    print(f"  Features:          {X_df.shape[1]} ({', '.join(X_df.columns.tolist())})")
    print(f"  Class distribution:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt * 25 / max(class_dist.values()))
        print(f"    {cls:<24} {cnt:>4}  {bar}")

    # ── 3. Strict train/test split — BEFORE any model fitting ─────────────────
    # stratify= ensures all 8 classes appear in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )
    print(f"\n  Train set: {len(X_train)} samples ({int((1 - test_size) * 100)}%)")
    print(f"  Test set:  {len(X_test)} samples ({int(test_size * 100)}%)")

    # ── 4. Cross-validation on train set only (leakage-free via sklearn Pipeline)
    clf = FailureClassifier()
    cv_results = clf.cross_validate(X_train, y_train, cv=cv_folds)
    _print_cv_results(cv_results)

    # ── 5. Final fit on full training partition ───────────────────────────────
    print(f"\n  Fitting final model on {len(X_train)} training samples...")
    clf.train(X_train, y_train)

    # ── 6. Held-out test evaluation (touched exactly once) ────────────────────
    report = clf.evaluate_holdout(X_test, y_test)
    print("\n" + "=" * 70)
    print("  [HELD-OUT TEST SET] Per-Class Classification Report")
    print("=" * 70)
    print(report)

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    clf.save(model_dir)
    fp.save(os.path.join(model_dir, "feature_pipeline.joblib"))
    _save_metadata(model_dir, cv_results, len(records), X_df.shape)

    print("=" * 70)
    print(f"  Model artifacts saved → {os.path.abspath(model_dir)}/")
    print(f"    failure_classifier.joblib")
    print(f"    feature_pipeline.joblib")
    print(f"    model_metadata.json")
    print("=" * 70 + "\n")


# ─── Reporting Helpers ────────────────────────────────────────────────────────

def _print_cv_results(cv: Dict[str, Any]) -> None:
    """Prints a formatted cross-validation results table."""
    print("\n" + "─" * 70)
    print(f"  [{cv['cv_folds']}-Fold Stratified CV] Results (Train partition only)")
    print("─" * 70)
    print(f"  Test  Accuracy    : {cv['test_accuracy_mean']:.4f} ± {cv['test_accuracy_std']:.4f}")
    print(f"  Test  F1 Weighted : {cv['test_f1_weighted_mean']:.4f}")
    print(f"  Test  F1 Macro    : {cv['test_f1_macro_mean']:.4f}")
    print(f"  Train Accuracy    : {cv['train_accuracy_mean']:.4f}  ← overfitting gap check")
    print(f"  Per-Fold Scores   : {[f'{s:.4f}' for s in cv['per_fold_test_accuracy']]}")
    overfitting_gap = cv["train_accuracy_mean"] - cv["test_accuracy_mean"]
    print(f"  Overfitting Gap   : {overfitting_gap:+.4f}")
    print("─" * 70)


def _save_metadata(
    model_dir: str,
    cv_results: Dict[str, Any],
    n_records: int,
    shape: tuple,
) -> None:
    """Saves training metadata to model_metadata.json for audit / reproducibility."""
    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_training_records": n_records,
        "feature_shape": list(shape),
        "model_architecture": "VotingClassifier(soft) [BalancedXGBClassifier + LogisticRegression]",
        "cv_results": cv_results,
    }
    path = os.path.join(model_dir, "model_metadata.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Model metadata saved → {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Razorpay failure classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db-path",
        default="data/recovery_agent.db",
        help="Path to SQLite database (default: data/recovery_agent.db)",
    )
    parser.add_argument(
        "--model-dir",
        default="ml/models",
        help="Directory to save trained model artifacts (default: ml/models)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for held-out test set (default: 0.2)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Skip DB and use synthetic data (useful for CI / smoke tests)",
    )
    args = parser.parse_args()

    try:
        run_training(
            db_path=args.db_path,
            model_dir=args.model_dir,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            force_synthetic=args.synthetic,
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
