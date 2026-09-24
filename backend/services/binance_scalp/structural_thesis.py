"""Structural SCALP policy. Ranking/prediction cannot activate in this process."""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.structural_mode import (
    FILL_MODEL_VERSION,
    MODE_DISABLED,
    MODE_LIVE,
    MODE_PAPER,
    MODE_SHADOW,
    ledger_writes_enabled,
    live_entry_enabled,
    quoting_enabled,
)

THESIS_STRUCTURAL = "structural"
THESIS_LEGACY_PREDICTION = "legacy_prediction"
PREDICTION_RETIRED = "SCALP_PREDICTION_THESIS_RETIRED"
STRUCTURAL_RANKING_BLOCKED = "SCALP_STRUCTURAL_NOT_EXECUTABLE"
STRUCTURAL_NOT_EXECUTABLE = STRUCTURAL_RANKING_BLOCKED


def normalize_thesis(raw: Any) -> str:
    thesis = str(raw or THESIS_STRUCTURAL).strip().lower()
    if thesis == THESIS_LEGACY_PREDICTION:
        return THESIS_LEGACY_PREDICTION
    return THESIS_STRUCTURAL


def prediction_entries_permitted(_config: Any) -> bool:
    return False


def ranking_eval_permitted(config: Any) -> bool:
    return False


def prediction_circuit_breaker_applies(config: Any) -> bool:
    """Leftover ranking-breaker arithmetic only. Structural process refuses these flags at startup."""
    thesis = str(getattr(config, "scalp_thesis", "") or "").strip().lower()
    return thesis == THESIS_LEGACY_PREDICTION and bool(getattr(config, "legacy_prediction_entries", False))


def structural_mode_of(config: Any) -> str:
    try:
        return config.resolved_structural_mode()
    except Exception:
        raw = getattr(config, "structural_mode", "") or ""
        if raw in {MODE_DISABLED, MODE_PAPER, MODE_SHADOW, MODE_LIVE}:
            return str(raw)
        return MODE_PAPER if bool(getattr(config, "scalp_paper_enabled", False)) else MODE_DISABLED


def structural_lp_executable(config: Any) -> bool:
    """True only for the paper simulation LP model (MODE_PAPER/MODE_SHADOW).

    MODE_LIVE bypasses the LP model entirely — entries go through the portfolio
    engine's execute_buy_fifo live path (real Binance.US orders).
    """
    mode = structural_mode_of(config)
    if live_entry_enabled(mode):
        return False
    return quoting_enabled(mode)


def new_entry_block_reason(config: Any) -> str | None:
    """None when live entries are enabled; STRUCTURAL_RANKING_BLOCKED otherwise."""
    if live_entry_enabled(structural_mode_of(config)):
        return None
    return STRUCTURAL_RANKING_BLOCKED


def status_fields(config: Any) -> dict[str, Any]:
    mode = structural_mode_of(config)
    live = live_entry_enabled(mode)
    return {
        "scalp_thesis": THESIS_STRUCTURAL,
        "structural_mode": mode,
        "legacy_prediction_entries": False,
        "prediction_entries_permitted": False,
        "ranking_eval_permitted": False,
        "prediction_circuit_breaker_applies": False,
        "structural_entries_executable": structural_lp_executable(config) and ledger_writes_enabled(mode),
        "live_entries_enabled": live,
        "structural_shadow": mode == MODE_SHADOW,
        "structural_arb_executable": False,
        "structural_fill_model": FILL_MODEL_VERSION,
        "fee_assumption_label": "exchange_actual" if live else "simulation_assumption",
        "exchange_live_impossible": False,
        "new_entry_block_reason": new_entry_block_reason(config),
        "scalp_engine_version": FILL_MODEL_VERSION,
    }
