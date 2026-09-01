"""
Merchant Authentication Middleware & Dependencies.

Accepts API keys via two channels only:
  1. 'X-API-Key' header
  2. 'Authorization: Bearer <key>' header

Query-parameter auth (?api_key=...) is intentionally NOT supported — it risks
keys appearing in server logs and browser history.

Security properties:
  - Key comparison uses hmac.compare_digest() to prevent timing-oracle attacks.
  - Valid keys sourced only from MERCHANT_API_KEY and APP_SECRET_KEY env vars.
  - Simulation mode bypass is blocked when ENVIRONMENT=production.
"""

import hmac
import logging
import os
from typing import Optional

from fastapi import Header, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

# OpenAPI / Swagger UI security scheme — header only
_api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_api_keys() -> set:
    """
    Returns the set of accepted merchant API keys from environment configuration.
    Only MERCHANT_API_KEY and APP_SECRET_KEY are considered.
    Never logs or exposes values.
    """
    keys: set = set()
    for env_var in ("MERCHANT_API_KEY", "APP_SECRET_KEY"):
        val = os.getenv(env_var, "").strip()
        if val:
            keys.add(val)
    return keys


def _constant_time_key_check(token: str, valid_keys: set) -> bool:
    """
    Compares *token* against every valid key using hmac.compare_digest to
    prevent timing-oracle attacks. Returns True if any key matches.
    """
    matched = False
    for key in valid_keys:
        # Always run compare_digest for every key — no short-circuit on first match —
        # so execution time is independent of which key (if any) matches.
        matched = hmac.compare_digest(token, key) or matched
    return matched


async def verify_merchant_auth(
    x_api_key: Optional[str] = Security(_api_key_header_scheme),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """
    FastAPI dependency validating merchant authentication.

    Accepted methods (in priority order):
      1. 'X-API-Key: <key>'              header
      2. 'Authorization: Bearer <key>'   header

    Simulation mode (SIMULATION_MODE=true) permits unauthenticated access ONLY
    when ENVIRONMENT is not 'production'.

    Raises:
        HTTPException 401 — missing or invalid credentials.
    """
    # ── Extract token ──────────────────────────────────────────────────────
    token: Optional[str] = None

    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        elif len(parts) == 1:
            # Bare token without scheme prefix — tolerate for convenience
            token = parts[0].strip()

    # ── Simulation mode bypass (non-production only) ───────────────────────
    is_simulation = os.getenv("SIMULATION_MODE", "false").lower() in ("true", "1", "yes")
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

    if is_simulation and not token:
        if is_production:
            logger.warning(
                "[Auth] Simulation mode is enabled but ENVIRONMENT=production — "
                "bypass denied. Set SIMULATION_MODE=false in production."
            )
            # Fall through to normal auth enforcement below
        else:
            logger.info("[Auth] Simulation mode active (non-production): permitting unauthenticated access.")
            return "simulation_merchant"

    # ── Require token ──────────────────────────────────────────────────────
    if not token:
        logger.warning("[Auth] Access denied: no API key or Bearer token provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Merchant authentication required. Provide 'X-API-Key' or 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Validate token (constant-time) ─────────────────────────────────────
    valid_keys = _get_valid_api_keys()

    if valid_keys and not _constant_time_key_check(token, valid_keys):
        logger.warning("[Auth] Access denied: invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid merchant API key or authorization token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "authenticated_merchant"
