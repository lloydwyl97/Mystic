"""Synthetic unit tests for Phase 3l scalp exit replay (in-memory, no DB writes)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config  # noqa: E402
from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from scripts.replay_scalp_exit_logic import (  # noqa: E402
    LOOP_INTERVAL_SEC,
    _new_sell_gate,
    _snapshot_from_bid,
    _target_bid_new,
    run_synthetic_tests,
)


def test_synthetic_cross_target_exits_profit():
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    result = run_synthetic_tests(econ, config)
    assert result["cross_target"]["passed"], result
    assert result["cross_target"]["exit"]["reason"] == "NET_PROFIT_TARGET"


def test_synthetic_below_target_stays_until_stale():
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    result = run_synthetic_tests(econ, config)
    assert result["below_target"]["passed"], result
    assert result["below_target"]["exit_reason"] == "STALE_SCALP_TIMEOUT"


def test_executable_exit_net_at_target_bid():
    econ = ScalpEconomics.from_env()
    entry = 1602.33
    entry_buy_i = 0.0
    sell_i = 0.0
    qty = 0.0156
    target_bid = _target_bid_new(entry, entry_buy_i, sell_i, econ)
    snap = _snapshot_from_bid("ETHUSDT", target_bid * 1.0001, 0.0003, qty)
    passed, net, _ = _new_sell_gate(
        snap,
        econ,
        get_scalp_config(),
        entry_price=entry,
        qty=qty,
        entry_buy_impact_pct=entry_buy_i,
    )
    assert passed, f"net_pct={net} target={econ.net_profit_target_pct}"
    assert net >= econ.net_profit_target_pct


def test_stale_timeout_interval():
    econ = ScalpEconomics.from_env()
    ticks_to_stale = int(econ.stale_scalp_timeout_sec / LOOP_INTERVAL_SEC)
    # At least 3 minutes of 5s exit-monitor ticks (180s operator floor).
    assert ticks_to_stale >= int(180 / LOOP_INTERVAL_SEC)
    assert ticks_to_stale == int(econ.stale_scalp_timeout_sec // LOOP_INTERVAL_SEC)
