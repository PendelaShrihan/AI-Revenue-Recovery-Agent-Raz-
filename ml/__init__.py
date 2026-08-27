"""
Razorpay AI Revenue Recovery Agent - Machine Learning Module

Modules:
    error_codes          — Rule-based failure classifier & canonical category definitions
    feature_engineering  — Stateless feature extraction pipeline (FeatureEngineeringPipeline)
    classifier           — XGBoost + Logistic Regression ensemble (FailureClassifier)
"""

from ml.error_codes import (
    FailureCategory,
    CATEGORIES,
    classify_failure,
    is_transient_failure,
    get_category_description,
    ERROR_CODE_MAP,
    ERROR_REASON_MAP,
)
from ml.feature_engineering import (
    FeatureEngineeringPipeline,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    PAYMENT_METHODS,
    ERROR_CATEGORIES,
    MERCHANT_CATEGORIES,
)
from ml.classifier import (
    FailureClassifier,
    BalancedXGBClassifier,
    CATEGORY_LABELS,
)
from ml.retry_predictor import (
    RetryTimingPredictor,
    RetryTimingRecommendation,
    RecoveryAction,
    BASE_RETRY_DELAYS,
)

__version__ = "0.3.0"

__all__ = [
    # Error codes & rule-based classifier
    "FailureCategory",
    "CATEGORIES",
    "classify_failure",
    "is_transient_failure",
    "get_category_description",
    "ERROR_CODE_MAP",
    "ERROR_REASON_MAP",
    # Feature engineering
    "FeatureEngineeringPipeline",
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "PASSTHROUGH_FEATURES",
    "PAYMENT_METHODS",
    "ERROR_CATEGORIES",
    "MERCHANT_CATEGORIES",
    # Classification model
    "FailureClassifier",
    "BalancedXGBClassifier",
    "CATEGORY_LABELS",
    # Retry timing predictor
    "RetryTimingPredictor",
    "RetryTimingRecommendation",
    "RecoveryAction",
    "BASE_RETRY_DELAYS",
]
