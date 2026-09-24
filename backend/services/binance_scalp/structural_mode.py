"""Explicit SCALP structural process modes.

Allowed modes
─────────────
STRUCTURAL_PAPER  — paper simulation (event-queue LP fill model)
STRUCTURAL_SHADOW — shadow evaluation only; no fills, no ledger writes
DISABLED          — SCALP inactive
LIVE              — genuine exchange execution via portfolio engine live path

LIVE requires SCALP_LIVE=true AND SCALP_LIVE_ARMED=true in the environment.
Order placement routes through the portfolio engine execute_buy_fifo live path
(ScalpOrderBridge / ccxt Binance.US) — not through the retired structural LP
paper-fill model.

Legacy prediction flags (SCALP_THESIS=legacy_prediction,
SCALP_LEGACY_PREDICTION_ENTRIES=true) remain permanently refused in all modes.
"""

from __future__ import annotations

from typing import Any

MODE_DISABLED = "DISABLED"
MODE_PAPER = "STRUCTURAL_PAPER"
MODE_SHADOW = "STRUCTURAL_SHADOW"
MODE_LIVE = "LIVE"
ALLOWED_MODES = frozenset({MODE_DISABLED, MODE_PAPER, MODE_SHADOW, MODE_LIVE})
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
        "LIVE": MODE_LIVE,
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
    thesis = str(scalp_thesis or "structural").strip().lower()
    if thesis == "legacy_prediction" or bool(legacy_prediction_entries):
        raise StructuralModeError("STRUCTURAL_MODE_REFUSED: legacy prediction flags cannot activate in the structural process")

    # LIVE mode: both SCALP_LIVE and SCALP_LIVE_ARMED must be true.
    # allow_market_orders is permanently refused (use execute_buy_fifo path).
    if bool(allow_market_orders):
        raise StructuralModeError("STRUCTURAL_MODE_REFUSED: SCALP_ALLOW_MARKET_ORDERS is refused — use portfolio engine live path")

    if bool(scalp_live) and bool(scalp_live_armed):
        # Explicit env override still allowed within the LIVE subset.
        raw = str(env_mode or "").strip().upper()
        if raw in ("", "LIVE"):
            return MODE_LIVE
        # Any other explicit env setting (e.g. STRUCTURAL_PAPER for testing) is
        # honoured even when scalp_live=true — the operator knows what they want.
        return normalize_mode(raw)

    raw = env_mode
    if raw is None or str(raw).strip() == "":
        return MODE_PAPER if scalp_paper_enabled else MODE_DISABLED
    return normalize_mode(raw)


def ledger_writes_enabled(mode: str) -> bool:
    """True for paper simulation only (LP fill model ledger writes)."""
    return mode == MODE_PAPER


def live_entry_enabled(mode: str) -> bool:
    """True when SCALP entries should route through the portfolio engine live path."""
    return mode == MODE_LIVE


def quoting_enabled(mode: str) -> bool:
    """True for modes that produce a quote/fill (paper or shadow evaluation)."""
    return mode in {MODE_PAPER, MODE_SHADOW}
