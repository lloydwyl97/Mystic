"""Shared sentiment source status labels for collector + diagnostics."""

from __future__ import annotations

from typing import Any

SOURCE_FETCH_FAILED = "source_fetch_failed"
SOURCE_OK_NO_SYMBOL_MATCH = "source_ok_no_symbol_match"
SOURCE_OK_MATCHED = "source_ok_matched"


def apply_source_status(
    *,
    source_name: str,
    status: str,
    breakdown: dict[str, Any],
    sources_active: list[str],
    sources_missing: list[str],
    score_fields: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    breakdown[f"{source_name}_status"] = status
    if error:
        breakdown[f"{source_name}_error"] = error
    if score_fields:
        for k, v in score_fields.items():
            if v is not None:
                breakdown[k] = v
    if status == SOURCE_OK_MATCHED:
        sources_active.append(source_name)
    elif status == SOURCE_FETCH_FAILED:
        sources_missing.append(source_name)


__all__ = [
    "SOURCE_FETCH_FAILED",
    "SOURCE_OK_MATCHED",
    "SOURCE_OK_NO_SYMBOL_MATCH",
    "apply_source_status",
]
