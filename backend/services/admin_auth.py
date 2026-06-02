#!/usr/bin/env python3
from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_admin_key(x_api_key: str = Header(None)) -> None:
    admin = os.getenv("ADMIN_API_KEY", "")
    if not admin:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    if not x_api_key or x_api_key != admin:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
