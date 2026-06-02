"""
Services package.

Heavy submodules are not imported at package load time so lightweight imports
(e.g. ai_decision_contract) work in minimal environments.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_authoritative_writer",
    "get_cache_reader",
    "start_authoritative_writer",
    "stop_authoritative_writer",
]


def __getattr__(name: str) -> Any:
    if name in ("get_authoritative_writer", "start_authoritative_writer", "stop_authoritative_writer"):
        from . import authoritative_writer as _aw

        return getattr(_aw, name)
    if name == "get_cache_reader":
        from . import cache_only_reader as _cr

        return _cr.get_cache_reader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
