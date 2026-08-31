"""Explicit SCALP structural process modes. Fail-closed. Exchange-live is impossible."""

from __future__ import annotations

from typing import Any

MODE_DISABLED = "DISABLED"
MODE_PAPER = "STRUCTURAL_PAPER"
MODE_SHADOW = "STRUCTURAL_SHADOW"
ALLOWED_MODES = frozenset({MODE_DISABLED, MODE_PAPER, MODE_SHADOW})
FILL_MODEL_VERSION = "structural_event_queue_v1"


class StructuralModeError(RuntimeError):
    """Startup refused. Do not fall through to paper or live."""


def normalize_mode(raw: Any) -> str:
    mode = str(raw or "").strip().upper()
    aliases = {
        "": MODE_PAPER,
        "PAPER": MODE_PAPER,
        "STRUCTURAL": MODE_PAPER,
        "SHADOW": MODE_SHADOW,
        "OFF": MODE_DISABLED,
        "FALSE": MODE_DISABLED,
    }
    mode = aliases.get(mode, mode)
    if mode not in ALLOWED_MODES:
        raise StructuralModeError(f"STRUCTURAL_MODE_REFUSED: unsupported mode {raw!r}")
    return mode


def resolve_structural_mode(
    *,
    env_mode: Any,
    scalp_live: bool,
    scalp_live_armed: bool,
    scalp_paper_enabled: bool,
    scalp_thesis: str,
    legacy_prediction_entries: bool,
    allow_market_orders: bool,
) -> str:
    if bool(scalp_live) or bool(scalp_live_armed) or bool(allow_market_orders):
        raise StructuralModeError("STRUCTURAL_MODE_REFUSED: exchange-live SCALP is impossible")
    thesis = str(scalp_thesis or "structural").strip().lower()
    if thesis == "legacy_prediction" or bool(legacy_prediction_entries):
        raise StructuralModeError("STRUCTURAL_MODE_REFUSED: legacy prediction flags cannot activate in the structural process")
    raw = env_mode
    if raw is None or str(raw).strip() == "":
        return MODE_PAPER if scalp_paper_enabled else MODE_DISABLED
    return normalize_mode(raw)


def ledger_writes_enabled(mode: str) -> bool:
    return mode == MODE_PAPER


def quoting_enabled(mode: str) -> bool:
    return mode in {MODE_PAPER, MODE_SHADOW}
