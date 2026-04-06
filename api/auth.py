"""API key authentication via X-API-Key header.

Keys are stored in the API_KEYS environment variable (comma-separated).
Each key maps to a user_id derived from the key itself (first 8 chars).
"""

from __future__ import annotations

import hashlib
import logging
import os

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_keys() -> dict[str, str]:
    """Parse API_KEYS env var into {key: user_id} mapping.

    Format: comma-separated keys. User ID is derived from SHA-256 hash
    of the key (first 8 hex chars) to avoid exposing raw keys in logs.
    """
    raw = os.environ.get("API_KEYS", "")
    if not raw.strip():
        return {}
    keys: dict[str, str] = {}
    for key in raw.split(","):
        key = key.strip()
        if key:
            user_id = hashlib.sha256(key.encode()).hexdigest()[:8]
            keys[key] = user_id
    return keys


async def get_current_user(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate API key and return the associated user_id.

    Raises 401 if the key is missing or invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    valid_keys = _get_valid_keys()
    user_id = valid_keys.get(api_key)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return user_id
