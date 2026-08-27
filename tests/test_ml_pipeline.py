"""
Unit Tests — ML Feature Engineering Pipeline & Failure Classifier.

Tests are entirely self-contained (no DB, no Razorpay API).
A small synthetic dataset is built inline for each test.

Run:
    python -m pytest tests/test_ml_pipeline.py -v
"""

import math
import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from ml.error_codes import FailureCategory, classify_failure
from ml.feature_engineering import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ERROR_CATEGORIES,
    MERCHANT_CATEGORIES,
    NUMERICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    FeatureEngineeringPipeline,
    _derive_merchant_category,
    _parse_datetime,
)
from ml.classifier import BalancedXGBClassifier, FailureClassifier


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_record(
    error_code: str = "INSUFFICIENT_FUNDS",
    error_reason: str = "insufficient_funds",
    payment_method: str = "card",
    merchant_id: str = "acc_test0001",
    amount: float = 1500.0,
    created_at: Any = None,
) -> Dict[str, Any]:
    if created_at is None:
        created_at = datetime(2024, 6, 15, 14, 30, 0)  # Saturday 14:30
    return {
        "error_code": error_code,
        "error_reason": error_reason,
        "payment_method": payment_method,
        "merchant_id": merchant_id,
        "amount": amount,
        "created_at": created_at,
    }


def _make_synthetic_records(n: int = 200) -> List[Dict[str, Any]]:
    """Generates a balanced synthetic dataset across all 8 failure categories."""
    import random
    rng = random.Random(42)
    cat_signals = {
        "insufficient_funds":    ("INSUFFICIENT_FUNDS", "insufficient_funds"),
        "card_blocked":          ("CARD_BLOCKED", "card_declined"),
        "network_timeout":       ("TRANSACTION_TIMED_OUT", "network_timeout"),
        "gateway_issue":         ("GATEWAY_ERROR", "gateway_error"),
        "expired_card":          ("CARD_EXPIRED", "card_expired"),
        "authentication_failed": ("AUTHENTICATION_FAILED", "incorrect_otp"),
        "limit_exceeded":        ("LIMIT_EXCEEDED", "daily_limit_exceeded"),
        "unknown":               ("BAD_REQUEST_ERROR", None),
    }
    categories = list(cat_signals.keys())
    methods = ["card", "upi", "netbanking", "wallet", "emi"]
    base = datetime(2024, 1, 1)
    records = []
    for i in range(n):
        cat = categories[i % len(categories)]
        code, reason = cat_signals[cat]
        created = base + timedelta(days=rng.randint(0, 89), hours=rng.randint(0, 23))
        records.append({
            "error_code":     code,
            "error_reason":   reason,
            "payment_method": rng.choice(methods),
            "merchant_id":    f"acc_{rng.randint(1, 20):04d}",
            "amount":         round(rng.uniform(50.0, 50000.0), 2),
            "created_at":     created,
        })
    return records


# ─── FeatureEngineeringPipeline Tests ─────────────────────────────────────────

class TestFeatureEngineeringPipeline:
    """Tests for the stateless FeatureEngineeringPipeline."""

    def test_extract_features_returns_dataframe(self):
        fp = FeatureEngineeringPipeline()
        records = [_make_record()]
        df = fp.extract_features(records)
        assert isinstance(df, pd.DataFrame)

    def test_extract_features_correct_columns(self):
        fp = FeatureEngineeringPipeline()
        df = fp.extract_features([_make_record()])
        assert list(df.columns) == ALL_FEATURES

    def test_extract_features_empty_input(self):
        fp = FeatureEngineeringPipeline()
        df = fp.extract_features([])
        assert df.empty
        assert list(df.columns) == ALL_FEATURES

    def test_error_category_classification(self):
        fp = FeatureEngineeringPipeline()
        r = _make_record(error_code="INSUFFICIENT_FUNDS", error_reason="insufficient_funds")
        df = fp.extract_features([r])
        assert df["error_category"].iloc[0] == "insufficient_funds"

    def test_unknown_payment_method_normalized(self):
        fp = FeatureEngineeringPipeline()
        r = _make_record(payment_method="crypto_pay")
        df = fp.extract_features([r])
        assert df["payment_method"].iloc[0] == "unknown"

    def test_known_payment_methods_preserved(self):
        fp = FeatureEngineeringPipeline()
        for method in ["card", "upi", "netbanking", "wallet", "emi"]:
            r = _make_record(payment_method=method)
            df = fp.extract_features([r])
            assert df["payment_method"].iloc[0] == method

    def test_amount_non_negative(self):
        fp = FeatureEngineeringPipeline()
        r = _make_record(amount=-500.0)
        df = fp.extract_features([r])
        assert df["amount"].iloc[0] == 0.0

    def test_amount_correct_value(self):
        fp = FeatureEngineeringPipeline()
        r = _make_record(amount=9999.75)
        df = fp.extract_features([r])
        assert df["amount"].iloc[0] == pytest.approx(9999.75)

    def test_hour_cyclical_encoding_range(self):
        """hour_sin and hour_cos must both be in [-1, 1]."""
        fp = FeatureEngineeringPipeline()
        for hour in range(24):
            t = datetime(2024, 1, 1, hour, 0)
            df = fp.extract_features([_make_record(created_at=t)])
            assert -1.0 <= df["hour_sin"].iloc[0] <= 1.0
            assert -1.0 <= df["hour_cos"].iloc[0] <= 1.0

    def test_hour_cyclical_continuity_midnight(self):
        """
        sin/cos encoding must be continuous across midnight:
        hour 23 and hour 0 should be close in cyclical space.
        """
        fp = FeatureEngineeringPipeline()
        t23 = datetime(2024, 1, 1, 23, 59)
        t0  = datetime(2024, 1, 2, 0, 0)
        df23 = fp.extract_features([_make_record(created_at=t23)])
        df0  = fp.extract_features([_make_record(created_at=t0)])

        sin23, cos23 = df23["hour_sin"].iloc[0], df23["hour_cos"].iloc[0]
        sin0,  cos0  = df0["hour_sin"].iloc[0],  df0["hour_cos"].iloc[0]

        # Euclidean distance in cyclical space should be small (< 0.3)
        dist = math.sqrt((sin23 - sin0)**2 + (cos23 - cos0)**2)
        assert dist < 0.3, f"Cyclical gap too large at midnight: {dist:.4f}"

    def test_weekend_detection(self):
        fp = FeatureEngineeringPipeline()
        sat = datetime(2024, 6, 15)  # Saturday
        sun = datetime(2024, 6, 16)  # Sunday
        mon = datetime(2024, 6, 17)  # Monday
        df_sat = fp.extract_features([_make_record(created_at=sat)])
        df_sun = fp.extract_features([_make_record(created_at=sun)])
        df_mon = fp.extract_features([_make_record(created_at=mon)])
        assert df_sat["is_weekend"].iloc[0] == 1
        assert df_sun["is_weekend"].iloc[0] == 1
        assert df_mon["is_weekend"].iloc[0] == 0

    def test_merchant_category_deterministic(self):
        """Same merchant_id must always produce the same category."""
        cat1 = _derive_merchant_category("acc_12345678")
        cat2 = _derive_merchant_category("acc_12345678")
        assert cat1 == cat2
        assert cat1 in MERCHANT_CATEGORIES

    def test_merchant_category_distribution(self):
        """Hash-based assignment should distribute across all 3 tiers."""
        cats = {_derive_merchant_category(f"acc_{i:08d}") for i in range(100)}
        assert cats == set(MERCHANT_CATEGORIES)

    def test_extract_labels_shape(self):
        fp = FeatureEngineeringPipeline()
        records = _make_synthetic_records(n=50)
        y = fp.extract_labels(records)
        assert isinstance(y, np.ndarray)
        assert len(y) == 50

    def test_extract_labels_valid_categories(self):
        fp = FeatureEngineeringPipeline()
        records = _make_synthetic_records(n=80)
        y = fp.extract_labels(records)
        for label in y:
            assert label in ERROR_CATEGORIES, f"Unexpected label: {label}"

    def test_failure_code_field_fallback(self):
        """
        FeatureEngineeringPipeline must read 'failure_code' / 'failure_reason'
        (SQLite column names) in addition to 'error_code' / 'error_reason'.
        """
        fp = FeatureEngineeringPipeline()
        r = {
            "failure_code":   "CARD_EXPIRED",
            "failure_reason": "card_expired",
            "payment_method": "card",
            "merchant_id":    "acc_00000001",
            "amount":         500.0,
            "created_at":     datetime(2024, 3, 10, 10, 0),
        }
        df = fp.extract_features([r])
        assert df["error_category"].iloc[0] == "expired_card"

    def test_multiple_records_consistent_shape(self):
        fp = FeatureEngineeringPipeline()
        records = _make_synthetic_records(n=120)
        df = fp.extract_features(records)
        assert df.shape == (120, len(ALL_FEATURES))
        assert df.isnull().sum().sum() == 0, "Feature matrix must have no NaN values"

    def test_pipeline_save_load_roundtrip(self, tmp_path):
        """Saved and loaded pipeline must produce identical feature values."""
        fp = FeatureEngineeringPipeline()
        records = _make_synthetic_records(n=20)
        df_before = fp.extract_features(records)

        save_path = str(tmp_path / "fp.joblib")
        fp.save(save_path)
        fp_loaded = FeatureEngineeringPipeline.load(save_path)
        df_after = fp_loaded.extract_features(records)

        pd.testing.assert_frame_equal(df_before, df_after)


# ─── _parse_datetime Tests ────────────────────────────────────────────────────

class TestParseDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2024, 5, 1, 12, 0)
        assert _parse_datetime(dt) == dt

    def test_unix_timestamp(self):
        ts = 1_700_000_000
        result = _parse_datetime(ts)
        assert isinstance(result, datetime)

    def test_iso_string(self):
        result = _parse_datetime("2024-06-15T14:30:00")
        assert result.hour == 14
        assert result.minute == 30

    def test_iso_string_with_z(self):
        result = _parse_datetime("2024-06-15T14:30:00Z")
        assert isinstance(result, datetime)

    def test_none_returns_datetime(self):
        result = _parse_datetime(None)
        assert isinstance(result, datetime)

    def test_invalid_string_returns_datetime(self):
        result = _parse_datetime("not-a-date")
        assert isinstance(result, datetime)


# ─── BalancedXGBClassifier Tests ──────────────────────────────────────────────

class TestBalancedXGBClassifier:
    def test_fit_predict(self):
        X = np.random.RandomState(0).randn(100, 5)
        y = np.array([0, 1, 2, 3] * 25)
        clf = BalancedXGBClassifier(n_estimators=10, max_depth=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_predict_proba_shape(self):
        X = np.random.RandomState(1).randn(60, 4)
        y = np.array([0, 1, 2] * 20)
        clf = BalancedXGBClassifier(n_estimators=10, max_depth=2)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (60, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_attribute_set_after_fit(self):
        X = np.random.RandomState(2).randn(40, 3)
        y = np.array([0, 1] * 20)
        clf = BalancedXGBClassifier(n_estimators=5)
        clf.fit(X, y)
        assert hasattr(clf, "classes_")

    def test_get_params_returns_dict(self):
        clf = BalancedXGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1)
        params = clf.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 4
        assert params["learning_rate"] == pytest.approx(0.1)

    def test_set_params(self):
        clf = BalancedXGBClassifier()
        clf.set_params(n_estimators=300, max_depth=8)
        assert clf.n_estimators == 300
        assert clf.max_depth == 8


# ─── FailureClassifier Tests ──────────────────────────────────────────────────

class TestFailureClassifier:
    """Tests for FailureClassifier end-to-end pipeline."""

    @pytest.fixture
    def trained_classifier(self):
        """Returns a FailureClassifier trained on synthetic data."""
        fp = FeatureEngineeringPipeline()
        records = _make_synthetic_records(n=240)
        X_df = fp.extract_features(records)
        y    = fp.extract_labels(records)
        clf  = FailureClassifier()
        clf.train(X_df, y)
        return clf, X_df, y

    @pytest.fixture
    def data(self):
        """Returns (X_df, y) for 200 synthetic records."""
        fp      = FeatureEngineeringPipeline()
        records = _make_synthetic_records(n=200)
        X_df    = fp.extract_features(records)
        y       = fp.extract_labels(records)
        return X_df, y

    def test_predict_before_train_raises(self, data):
        X_df, _ = data
        clf = FailureClassifier()
        with pytest.raises(RuntimeError, match="called before training"):
            clf.predict(X_df)

    def test_evaluate_holdout_before_train_raises(self, data):
        X_df, y = data
        clf = FailureClassifier()
        with pytest.raises(RuntimeError, match="called before training"):
            clf.evaluate_holdout(X_df, y)

    def test_train_sets_is_trained(self, data):
        X_df, y = data
        clf = FailureClassifier()
        clf.train(X_df, y)
        assert clf._is_trained is True

    def test_predict_output_shape(self, trained_classifier):
        clf, X_df, _ = trained_classifier
        preds = clf.predict(X_df)
        assert preds.shape == (len(X_df),)

    def test_predict_valid_categories(self, trained_classifier):
        clf, X_df, _ = trained_classifier
        preds = clf.predict(X_df)
        valid = set(ERROR_CATEGORIES)
        for p in preds:
            assert p in valid, f"Unknown category predicted: {p}"

    def test_predict_proba_shape_and_sums(self, trained_classifier):
        clf, X_df, _ = trained_classifier
        proba = clf.predict_proba(X_df)
        assert proba.shape[0] == len(X_df)
        assert proba.shape[1] == 8  # 8 failure categories
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_attribute(self, trained_classifier):
        clf, _, _ = trained_classifier
        assert clf.classes_ is not None
        assert len(clf.classes_) == 8
        for c in clf.classes_:
            assert c in ERROR_CATEGORIES

    def test_evaluate_holdout_returns_string(self, trained_classifier):
        clf, X_df, y = trained_classifier
        report = clf.evaluate_holdout(X_df, y)
        assert isinstance(report, str)
        assert "precision" in report.lower()
        assert "recall" in report.lower()

    def test_cross_validate_returns_expected_keys(self, data):
        X_df, y = data
        clf = FailureClassifier()
        cv = clf.cross_validate(X_df, y, cv=3)
        for key in [
            "cv_folds",
            "test_accuracy_mean",
            "test_accuracy_std",
            "test_f1_weighted_mean",
            "test_f1_macro_mean",
            "train_accuracy_mean",
            "per_fold_test_accuracy",
        ]:
            assert key in cv, f"Missing CV key: {key}"

    def test_cross_validate_accuracy_in_range(self, data):
        X_df, y = data
        clf = FailureClassifier()
        cv = clf.cross_validate(X_df, y, cv=3)
        assert 0.0 <= cv["test_accuracy_mean"] <= 1.0
        assert 0.0 <= cv["test_f1_weighted_mean"] <= 1.0

    def test_cross_validate_does_not_fit_classifier(self, data):
        """cross_validate() must not leave the classifier in trained state."""
        X_df, y = data
        clf = FailureClassifier()
        clf.cross_validate(X_df, y, cv=3)
        assert clf._is_trained is False

    def test_save_load_roundtrip(self, trained_classifier, tmp_path):
        """Loaded model must produce identical predictions to the original."""
        clf, X_df, _ = trained_classifier
        clf.save(str(tmp_path))
        clf_loaded = FailureClassifier.load(str(tmp_path))
        preds_orig   = clf.predict(X_df)
        preds_loaded = clf_loaded.predict(X_df)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_save_before_train_raises(self, tmp_path):
        clf = FailureClassifier()
        with pytest.raises(RuntimeError, match="called before training"):
            clf.save(str(tmp_path))

    def test_model_file_created_on_save(self, trained_classifier, tmp_path):
        clf, _, _ = trained_classifier
        clf.save(str(tmp_path))
        assert (tmp_path / "failure_classifier.joblib").exists()


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """Integration test covering the full train_classifier.py workflow."""

    def test_full_pipeline_synthetic(self, tmp_path):
        """
        Runs the complete extract → split → CV → train → evaluate → save
        workflow with synthetic data and verifies all artifacts are created.
        """
        from sklearn.model_selection import train_test_split

        records = _make_synthetic_records(n=240)
        fp  = FeatureEngineeringPipeline()
        X   = fp.extract_features(records)
        y   = fp.extract_labels(records)

        # Strict 80/20 split before any fitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # CV on train set only
        clf = FailureClassifier()
        cv = clf.cross_validate(X_train, y_train, cv=3)
        assert cv["test_accuracy_mean"] > 0.0

        # Final fit
        clf.train(X_train, y_train)

        # Holdout evaluation
        report = clf.evaluate_holdout(X_test, y_test)
        assert len(report) > 0

        # Save & load
        model_dir = str(tmp_path / "models")
        clf.save(model_dir)
        fp.save(os.path.join(model_dir, "feature_pipeline.joblib"))

        assert os.path.exists(os.path.join(model_dir, "failure_classifier.joblib"))
        assert os.path.exists(os.path.join(model_dir, "feature_pipeline.joblib"))

        # Inference with loaded model
        clf2 = FailureClassifier.load(model_dir)
        preds = clf2.predict(X_test)
        assert len(preds) == len(X_test)
        for p in preds:
            assert p in ERROR_CATEGORIES
