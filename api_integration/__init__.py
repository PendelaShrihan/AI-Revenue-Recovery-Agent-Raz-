"""
Razorpay AI Revenue Recovery Agent - API Integration Module
Contains Razorpay webhook ingestion, signature verification, payload normalization, and API simulators.
"""

from api_integration.schemas import (
    EventType,
    FailureCategory,
    NormalizedEvent,
    WebhookResponse,
)
from api_integration.verifier import (
    verify_webhook_signature,
    compute_webhook_signature,
)
from api_integration.normalizer import (
    normalize_webhook_payload,
)
from api_integration.router import (
    webhook_router,
)

__version__ = "0.1.0"

__all__ = [
    "EventType",
    "FailureCategory",
    "NormalizedEvent",
    "WebhookResponse",
    "verify_webhook_signature",
    "compute_webhook_signature",
    "normalize_webhook_payload",
    "webhook_router",
]
