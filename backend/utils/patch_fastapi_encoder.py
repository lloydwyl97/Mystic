"""
FastAPI jsonable_encoder hardening shim.

This module wraps fastapi.encoders.jsonable_encoder with a pre-sanitizing pass
that removes reference cycles and non-serializable objects using backend.utils.safe_encoder.decycle.
"""

from __future__ import annotations

import contextlib
from typing import Any

import fastapi.encoders as _fe

from backend.utils.safe_encoder import decycle

# Preserve the original only once to avoid stacking wrappers on repeated imports
if not hasattr(_fe, "_ORIG_jsonable_encoder"):
    _fe._ORIG_jsonable_encoder = _fe.jsonable_encoder


def _safe_jsonable_encoder(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Drop-in replacement for fastapi.encoders.jsonable_encoder that first
    normalizes objects via decycle() to avoid recursive walks or cycles.
    """
    with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # If decycle fails for any reason, fall back to the original object
        obj = decycle(obj)
    orig_encoder = _fe._ORIG_jsonable_encoder
    return orig_encoder(obj, *args, **kwargs)


# Apply the shim globally so any subsequent imports use the safe encoder.
_fe.jsonable_encoder = _safe_jsonable_encoder
