"""
Razorpay AI Revenue Recovery Agent - Machine Learning Module
Contains failure classification models, feature engineering pipelines, and retry timing predictors.
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

__version__ = "0.1.0"

__all__ = [
    "FailureCategory",
    "CATEGORIES",
    "classify_failure",
    "is_transient_failure",
    "get_category_description",
    "ERROR_CODE_MAP",
    "ERROR_REASON_MAP",
]
