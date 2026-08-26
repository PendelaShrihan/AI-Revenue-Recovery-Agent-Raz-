"""
Razorpay Webhook Cryptographic Signature Verification.
Validates HMAC SHA-256 signatures passed in the X-Razorpay-Signature header.
"""

import hmac
import hashlib
import logging
from typing import Union

logger = logging.getLogger(__name__)


def compute_webhook_signature(payload_body: Union[bytes, str], secret: str) -> str:
    """
    Computes the HMAC-SHA256 hex digest for a given payload body and secret key.
    Useful for testing, webhook simulation, and internal verification.
    """
    if isinstance(payload_body, str):
        body_bytes = payload_body.encode("utf-8")
    elif isinstance(payload_body, bytes):
        body_bytes = payload_body
    else:
        raise TypeError("payload_body must be either bytes or str")

    secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(secret_bytes, body_bytes, hashlib.sha256).hexdigest()


def verify_webhook_signature(
    payload_body: Union[bytes, str],
    signature: str,
    secret: str
) -> bool:
    """
    Verifies that the provided X-Razorpay-Signature matches the HMAC-SHA256 digest
    of the raw webhook request body.

    Args:
        payload_body: The raw unparsed request body bytes or string.
        signature: Hexadecimal signature string from X-Razorpay-Signature header.
        secret: Razorpay webhook secret configured in merchant dashboard.

    Returns:
        bool: True if signature is cryptographically valid, False otherwise.
    """
    if not signature or not secret:
        logger.warning("Verification failed: Signature or webhook secret is missing.")
        return False

    if payload_body is None:
        logger.warning("Verification failed: Payload body is None.")
        return False

    try:
        expected_signature = compute_webhook_signature(payload_body, secret)
        # Constant-time comparison prevents timing attacks
        return hmac.compare_digest(expected_signature, signature.strip())
    except Exception as e:
        logger.error(f"Error while verifying webhook signature: {str(e)}")
        return False
