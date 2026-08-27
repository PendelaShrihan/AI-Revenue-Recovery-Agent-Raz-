"""
Failure Classification Model for Razorpay Payment Recovery Agent.

Architecture
────────────
VotingClassifier(soft) combining two mathematically diverse estimators:
    1. BalancedXGBClassifier  — gradient-boosted tree ensemble
    2. LogisticRegression     — linear probabilistic model (class_weight='balanced')

Ensembling a non-linear tree ensemble with a linear model provides complementary
inductive biases: XGBoost captures feature interactions and non-linearity;
Logistic Regression contributes calibrated linear decision boundaries, preventing
the soft vote from being entirely dominated by one learning paradigm.

Imbalance handling
──────────────────
    BalancedXGBClassifier.fit():
        Calls compute_sample_weight('balanced', y) inside every fit() invocation.
        Because weights are derived from y within fit(), cross_validate() gets
        correct per-fold weights automatically — no external fit_params needed.

    LogisticRegression:
        class_weight='balanced' rescales the loss function using inverse class
        frequency, equivalent to oversampling minority classes.

Leakage prevention
──────────────────
    ColumnTransformer (OrdinalEncoder + StandardScaler) is wrapped inside a
    sklearn.pipeline.Pipeline. During cross_validate(), sklearn calls
    pipeline.fit() on each training fold — so the scaler's mean/variance are
    computed only from training data and never see held-out fold samples.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml.error_codes import FailureCategory
from ml.feature_engineering import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ERROR_CATEGORIES,
    MERCHANT_CATEGORIES,
    NUMERICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    PAYMENT_METHODS,
)

logger = logging.getLogger(__name__)

#: All possible target label strings — used to fit LabelEncoder on known universe
CATEGORY_LABELS: List[str] = [cat.value for cat in FailureCategory]


# ─── BalancedXGBClassifier ─────────────────────────────────────────────────────

class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """
    XGBClassifier wrapper that automatically computes balanced sample weights.

    Note: _estimator_type is set explicitly so sklearn's VotingClassifier
    _validate_estimators() recognises this as a proper classifier in all
    sklearn 1.4+ validation paths (ClassifierMixin alone is not sufficient
    when the estimator is instantiated dynamically inside a Pipeline).

    Problem solved:
        cross_validate() calls estimator.fit(X_fold, y_fold) with no fit_params.
        Passing sample_weight externally requires manual per-fold weight slicing,
        which is error-prone and not supported by cross_validate()'s API cleanly.

    Solution:
        Compute sample_weight = compute_sample_weight('balanced', y) inside fit().
        Each CV fold receives correctly balanced weights derived from that fold's
        class distribution — no external orchestration needed.

    Parameters:
        n_estimators:  Number of boosting rounds (default 200).
        max_depth:     Maximum tree depth (default 6).
        learning_rate: Step size shrinkage (default 0.05).
        random_state:  Reproducibility seed (default 42).
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BalancedXGBClassifier":
        """Fits XGBoost with balanced sample weights computed from y."""
        self._xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="mlogloss",
            random_state=self.random_state,
            verbosity=0,
        )
        sample_weight = compute_sample_weight("balanced", y)
        self._xgb.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._xgb.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._xgb.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._xgb.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "BalancedXGBClassifier":
        for key, val in params.items():
            setattr(self, key, val)
        return self


# ─── FailureClassifier ────────────────────────────────────────────────────────

class FailureClassifier:
    """
    End-to-end payment failure classifier with leakage-free cross-validation.

    Wraps a full sklearn Pipeline:

        ColumnTransformer (preprocessing)
            ├── OrdinalEncoder  → error_category, payment_method, merchant_category
            ├── StandardScaler  → amount
            └── passthrough     → hour_sin, hour_cos, is_weekend
        │
        VotingClassifier (soft)
            ├── BalancedXGBClassifier  (XGBoost + auto balanced sample_weight)
            └── LogisticRegression     (class_weight='balanced')

    Usage:
        fp = FeatureEngineeringPipeline()
        X_df = fp.extract_features(records)
        y    = fp.extract_labels(records)

        X_train, X_test, y_train, y_test = train_test_split(X_df, y, stratify=y)

        clf = FailureClassifier()
        cv  = clf.cross_validate(X_train, y_train)   # leakage-free CV
        clf.train(X_train, y_train)                  # final fit on full train set
        report = clf.evaluate_holdout(X_test, y_test)
        clf.save("ml/models")

        # Inference
        labels = clf.predict(X_new_df)
        probs  = clf.predict_proba(X_new_df)
    """

    MODEL_FILENAME = "failure_classifier.joblib"

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._cv_results: Optional[Dict[str, Any]] = None
        self._is_trained: bool = False

    # ── Pipeline construction ────────────────────────────────────────────────

    def _build_sklearn_pipeline(self) -> Pipeline:
        """
        Constructs a fresh sklearn Pipeline instance.

        Called separately for:
            - cross_validate()  → fresh pipeline; scaler fits per fold
            - train()           → pipeline fitted on full train set

        Building fresh instances avoids accidentally sharing fitted state
        between CV runs and the final training fit.
        """
        # OrdinalEncoder categories must be specified explicitly so unknown
        # values at inference time are handled gracefully (→ -1).
        cat_categories = [
            ERROR_CATEGORIES,    # error_category
            PAYMENT_METHODS,     # payment_method
            MERCHANT_CATEGORIES, # merchant_category
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat_enc",
                    OrdinalEncoder(
                        categories=cat_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    CATEGORICAL_FEATURES,
                ),
                ("num_scale", StandardScaler(), NUMERICAL_FEATURES),
                ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        xgb_clf = BalancedXGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )
        lr_clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )

        ensemble = VotingClassifier(
            estimators=[("xgb", xgb_clf), ("lr", lr_clf)],
            voting="soft",
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", ensemble),
            ]
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X_df: pd.DataFrame, y: np.ndarray) -> None:
        """
        Fits the full sklearn Pipeline on training data.

        This should be called AFTER cross_validate() and AFTER a strict
        train/test split, using only the training partition.

        Args:
            X_df: Feature DataFrame from FeatureEngineeringPipeline.extract_features().
            y:    Label array of FailureCategory strings (shape: [n_samples]).
        """
        self._label_encoder = LabelEncoder()
        # Fit on all known category strings so the encoder is stable even if
        # some classes are absent in the training partition.
        self._label_encoder.fit(CATEGORY_LABELS)
        y_enc = self._label_encoder.transform(y)

        self._pipeline = self._build_sklearn_pipeline()
        logger.info(
            "Fitting XGBoost + LogisticRegression ensemble pipeline on "
            f"{len(X_df)} training samples..."
        )
        self._pipeline.fit(X_df, y_enc)
        self._is_trained = True
        logger.info("Training complete.")

    # ── Cross-Validation ────────────────────────────────────────────────────

    def cross_validate(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        cv: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs stratified k-fold CV on the TRAINING set with a fresh pipeline.

        Leakage guarantee:
            A brand-new sklearn Pipeline is created here. sklearn's cross_validate()
            calls pipeline.fit() on each fold's training split, so OrdinalEncoder
            and StandardScaler never see held-out fold data.

        Args:
            X_df: Feature DataFrame (training partition only).
            y:    Label array of FailureCategory strings.
            cv:   Number of stratified folds (default 5).

        Returns:
            Dict with mean/std accuracy, weighted F1, macro F1, and per-fold scores.
        """
        # Fresh LabelEncoder for CV — does not share state with final training encoder
        le = LabelEncoder()
        le.fit(CATEGORY_LABELS)
        y_enc = le.transform(y)

        cv_pipeline = self._build_sklearn_pipeline()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        logger.info(f"Running {cv}-fold stratified cross-validation on {len(X_df)} samples...")
        raw = cross_validate(
            cv_pipeline,
            X_df,
            y_enc,
            cv=skf,
            scoring=["accuracy", "f1_weighted", "f1_macro"],
            return_train_score=True,
            n_jobs=-1,
        )

        self._cv_results = {
            "cv_folds": cv,
            "test_accuracy_mean": float(np.mean(raw["test_accuracy"])),
            "test_accuracy_std": float(np.std(raw["test_accuracy"])),
            "test_f1_weighted_mean": float(np.mean(raw["test_f1_weighted"])),
            "test_f1_macro_mean": float(np.mean(raw["test_f1_macro"])),
            "train_accuracy_mean": float(np.mean(raw["train_accuracy"])),
            "per_fold_test_accuracy": raw["test_accuracy"].tolist(),
            "per_fold_f1_weighted": raw["test_f1_weighted"].tolist(),
        }
        logger.info(
            f"CV complete. Mean test accuracy: {self._cv_results['test_accuracy_mean']:.4f} "
            f"± {self._cv_results['test_accuracy_std']:.4f}"
        )
        return self._cv_results

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_holdout(self, X_df: pd.DataFrame, y: np.ndarray) -> str:
        """
        Generates a per-class classification report on the held-out test set.

        The test set must never have been used for training or CV selection.

        Args:
            X_df: Feature DataFrame (held-out test partition).
            y:    True label array of FailureCategory strings.

        Returns:
            sklearn classification_report string (precision, recall, F1 per class).

        Raises:
            RuntimeError: If called before train().
        """
        self._assert_trained("evaluate_holdout")
        y_enc = self._label_encoder.transform(y)
        y_pred = self._pipeline.predict(X_df)
        return classification_report(
            y_enc,
            y_pred,
            target_names=self._label_encoder.classes_,
            zero_division=0,
        )

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Returns predicted FailureCategory strings for one or more records.

        Args:
            X_df: Feature DataFrame from FeatureEngineeringPipeline.extract_features().

        Returns:
            np.ndarray of FailureCategory string labels (shape: [n_samples]).
        """
        self._assert_trained("predict")
        y_enc = self._pipeline.predict(X_df)
        return self._label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Returns class probability matrix for all FailureCategory classes.

        Args:
            X_df: Feature DataFrame from FeatureEngineeringPipeline.extract_features().

        Returns:
            np.ndarray of shape (n_samples, n_classes).
            Column order matches self.classes_.
        """
        self._assert_trained("predict_proba")
        return self._pipeline.predict_proba(X_df)

    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Returns the label encoder's class array (FailureCategory strings)."""
        if self._label_encoder is None:
            return None
        return self._label_encoder.classes_

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, model_dir: str) -> str:
        """
        Serializes the trained FailureClassifier (pipeline + label encoder) to disk.

        Args:
            model_dir: Directory path. Created if it does not exist.

        Returns:
            Absolute path of the saved model file.

        Raises:
            RuntimeError: If called before train().
        """
        self._assert_trained("save")
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, self.MODEL_FILENAME)
        joblib.dump(self, path)
        logger.info(f"FailureClassifier saved → {path}")
        return os.path.abspath(path)

    @classmethod
    def load(cls, model_dir: str) -> "FailureClassifier":
        """
        Loads a FailureClassifier from a previously saved model directory.

        Args:
            model_dir: Directory containing 'failure_classifier.joblib'.

        Returns:
            Loaded FailureClassifier instance ready for inference.
        """
        path = os.path.join(model_dir, cls.MODEL_FILENAME)
        instance: "FailureClassifier" = joblib.load(path)
        logger.info(f"FailureClassifier loaded ← {path}")
        return instance

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _assert_trained(self, method_name: str) -> None:
        """Raises RuntimeError if the model has not been trained yet."""
        if not self._is_trained:
            raise RuntimeError(
                f"FailureClassifier.{method_name}() called before training. "
                "Call train(X_train_df, y_train) first."
            )
