"""Internal API key guard.

Every endpoint except ``GET /health`` requires the ``X-Internal-Api-Key``
header to match the ``INTERNAL_API_KEY`` environment variable
(see docs/API_CONTRACT.md, section 1 — internal auth).
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    """FastAPI dependency: reject the request with 401 unless the key matches."""
    expected = get_settings().internal_api_key
    if (
        not expected
        or x_internal_api_key is None
        or not hmac.compare_digest(x_internal_api_key.encode("utf-8"), expected.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing X-Internal-Api-Key header",
            },
        )
