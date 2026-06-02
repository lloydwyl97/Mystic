from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

__all__ = ["decycle"]

_PRIMS = (str, int, float, bool, type(None))


def _to_primitive(obj: Any) -> Any | None:
    if isinstance(obj, _PRIMS):
        return obj
    if isinstance(obj, (datetime, date, time)):
        try:
            return obj.isoformat()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return str(obj)
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return str(obj)
    if isinstance(obj, Enum):
        return obj.value if isinstance(obj.value, _PRIMS) else str(obj.value)
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return repr(obj)
    return None


def decycle(obj: Any, _seen: set[int] | None = None, _depth: int = 64) -> Any:
    if _seen is None:
        _seen = set()
    if _depth <= 0:
        return "<max_depth>"
    if obj is None:
        return None

    prim = _to_primitive(obj)
    if prim is not None:
        return prim

    oid = id(obj)
    if oid in _seen:
        return "<cycle>"

    if hasattr(obj, "model_dump"):
        try:
            return decycle(obj.model_dump(), _seen, _depth - 1)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
    if hasattr(obj, "dict"):
        try:
            return decycle(obj.dict(), _seen, _depth - 1)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
    if is_dataclass(obj):
        try:
            return decycle(asdict(obj), _seen, _depth - 1)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass

    if isinstance(obj, Mapping):
        _seen.add(oid)
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = k if isinstance(k, _PRIMS) else str(k)
            out[str(ks)] = decycle(v, _seen, _depth - 1)
        return out

    if isinstance(obj, (Sequence, AbstractSet)) and not isinstance(obj, (str, bytes, bytearray)):
        _seen.add(oid)
        seq = list(obj) if isinstance(obj, AbstractSet) else obj
        return [decycle(v, _seen, _depth - 1) for v in seq]

    try:
        return str(obj)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return "<unserializable>"
