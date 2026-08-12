#!/usr/bin/env python3
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


def require_admin_key(
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> None:
    """Authenticate admin mutations using either supported operator credential."""
    admin_api_key = os.getenv("ADMIN_API_KEY", "")
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_api_key and not admin_token:
        raise HTTPException(status_code=500, detail="Admin authentication is not configured")

    api_key_ok = bool(admin_api_key and x_api_key and hmac.compare_digest(x_api_key, admin_api_key))
    bearer = ""
    if authorization:
        bearer = authorization[7:] if authorization.startswith("Bearer ") else authorization
    token_ok = bool(admin_token and bearer and hmac.compare_digest(bearer, admin_token))
    if not api_key_ok and not token_ok:
        raise HTTPException(status_code=401, detail="Invalid or missing admin credential")
