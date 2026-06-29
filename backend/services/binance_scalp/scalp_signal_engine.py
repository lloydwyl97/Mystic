"""
Scalp signal engine (production paper path) — structured like the day all-weather engine.

Goal: make the scalper actually run and accumulate bounded roundtrips on the local
paper sleeve (separate DB/tables, own PnL, $25 notional, strict after-cost economics).

Current v1 implementation:
* Delegates entry selection to the existing ScalpStrategyRouter + enabled strategies
  (breakout_momentum is the primary researched continuation setup; the other three
  from phase 3/4 replays are also available when not explicitly disabled).
* Enforces the existing high-quality bounded exits (evaluate_exit):
    - NET_PROFIT_TARGET (after realistic taker + slippage + spread)
    - SETUP_INVALIDATED (per-strategy rules, e.g. lost breakout level + negative mom)
    - MOMENTUM_FAILED (adverse move with no recovery signs)
    - MAX_HOLD_HARD_LIMIT (safety, default ~20-30 min range via stale timeout * N)
* The strict entry gate (spread caps, impact, projected gross edge, momentum_confirmed
  15/30/60s rising, data sufficiency, surplus buffer) remains in force for paper.
  This keeps the "only take real edge" discipline while letting the engine actually
  see opportunities after the runner warms the MomentumTracker.

Enable/disable with SCALP_SIGNAL_ENGINE_ENABLED (default true for paper runs so the
local testbed actually produces trades and learning rows; set false to fall back to
raw router path with identical behavior).

When the flag is off, the public helpers are light no-ops / pass-through so the
paper engine keeps its pre-existing behavior unchanged.

This module is the place future "lab validated" entry rules (from fresh replays that
prove positive expectancy + consistency on paper notional with max hold << 1h and
acceptable bad rate) can be dropped in with minimal changes to paper_engine.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.exit_manager import (
    DECISION_SELL,
    evaluate_exit,
    track_from_row,
)
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies.kline_cache import KlineCache

# Exit reasons we surface (kept in sync with exit_manager / strategy_exit_rules)
EXIT_NET_PROFIT_TARGET = "NET_PROFIT_TARGET"
EXIT_SETUP_INVALIDATED = "SETUP_INVALIDATED_EXIT"
EXIT_MOMENTUM_FAILED = "MOMENTUM_FAILED_EXIT"
EXIT_MAX_HOLD_HARD_LIMIT = "MAX_HOLD_HARD_LIMIT"


def scalp_signal_engine_enabled() -> bool:
    """True when the clean signal + bounded exit path should be used for paper."""
    raw = os.getenv("SCALP_SIGNAL_ENGINE_ENABLED", "true")
    if raw is None:
        return True
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# --------------------------- thin production facade ---------------------------

# Lazy singletons so importing the module has zero side effects when the flag is off
# and we are not in a paper run.
_config = None
_econ = None
_reader = None
_momentum = None
_klines = None
_router = None


def _ensure_components():
    global _config, _econ, _reader, _momentum, _klines, _router
    if _config is None:
        _config = get_scalp_config()
        _econ = economics_for_config(_config)
        _reader = ScalpMarketReader(_config)
        _momentum = MomentumTracker()
        _klines = KlineCache()
        _router = ScalpStrategyRouter(
            config=_config,
            econ=_econ,
            reader=_reader,
            momentum=_momentum,
            klines=_klines,
        )


def get_router() -> ScalpStrategyRouter | None:
    """Return the strategy router (or None if engine disabled / not initialized)."""
    if not scalp_signal_engine_enabled():
        return None
    _ensure_components()
    return _router


def entry_candidates(epoch: float, notional_usd: float) -> list[dict[str, Any]]:
    """
    Return ranked candidate dicts (same shape the paper engine already consumes)
    using the router when the signal engine flag is on. Empty list otherwise.
    The caller (paper_engine) still applies arming, cash, max positions, preflight etc.
    """
    if not scalp_signal_engine_enabled():
        return []
    _ensure_components()
    try:
        ranked = _router.evaluate_all(epoch=epoch, notional_usd=notional_usd)
        return ranked or []
    except Exception:
        return []


def exit_decision(
    *,
    track_row: Any,
    snap: Any,
    mom: Any,
    hold_sec: float,
    executable_net_pct: float,
    profit_hit: bool,
    exit_spread_ok: bool,
    perform_review: bool,
) -> dict[str, Any]:
    """
    Bounded exit decision using the production exit manager.

    Returns a small dict the paper engine can use:
      {"decision": "SELL" or "HOLD", "reason": "...", "exit_reason": "..." or None, "diagnostics": {...}}
    """
    if not scalp_signal_engine_enabled():
        # When disabled, let the caller fall through to its prior exit path.
        return {"decision": "HOLD", "reason": "engine_disabled", "exit_reason": None, "diagnostics": {}}

    _ensure_components()
    try:
        pos_diag = {}
        raw = getattr(track_row, "diagnostics_json", None) or (track_row.get("diagnostics_json") if isinstance(track_row, dict) else None)
        if raw:
            import json

            pos_diag = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

        track = track_from_row(track_row, pos_diag)
        review = evaluate_exit(
            track=track,
            snap=snap,
            mom=mom,
            econ=_econ,
            config=_config,
            trade_id=str(getattr(track_row, "trade_id", "") or (track_row.get("trade_id") if isinstance(track_row, dict) else "")),
            hold_sec=hold_sec,
            executable_net_pct=executable_net_pct,
            profit_hit=profit_hit,
            exit_spread_ok=exit_spread_ok,
            perform_review=perform_review,
        )
        return {
            "decision": review.decision,
            "reason": review.reason,
            "exit_reason": review.exit_reason,
            "diagnostics": review.diagnostics,
            "updated_track": review.updated_track,
        }
    except Exception:
        return {"decision": "HOLD", "reason": "exit_engine_error", "exit_reason": None, "diagnostics": {}}


__all__ = [
    "EXIT_MAX_HOLD_HARD_LIMIT",
    "EXIT_MOMENTUM_FAILED",
    "EXIT_NET_PROFIT_TARGET",
    "EXIT_SETUP_INVALIDATED",
    "entry_candidates",
    "exit_decision",
    "get_router",
    "scalp_signal_engine_enabled",
]
