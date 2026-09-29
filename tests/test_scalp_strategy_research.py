from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research import scalp_micro_impulse_research as smi
from scripts.research import scalp_strategy_research as ssr

FEATURE_COLS = ("atr5", "atr15", "c5", "c15", "ema20_5m", "ema20_15m", "ema50_15m", "ema20_1h_slope", "hh20", "volx", "flow5", "vwap60", "rv15", "rv240", "a", "a15")


def _klines(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0008, n)) * close
    vol = rng.uniform(1, 5, n)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": vol,
            "quote_volume": vol * close,
            "trades": 10,
            "taker_buy_base": vol * rng.uniform(0.2, 0.8, n),
            "taker_buy_quote": 0.0,
        },
        index=pd.Index(np.arange(n) * 60_000 + 1_775_000_040_000, name="open_ms"),
    )
    return df


def test_features_and_signals_use_only_closed_bars():
    raw = _klines()
    cut = 1500
    base = ssr.build_features(raw.copy())
    future = raw.copy()
    future.iloc[cut + 1 :, :] = future.iloc[cut + 1 :, :] * 3.0
    shifted = ssr.build_features(future)
    for col in FEATURE_COLS:
        np.testing.assert_allclose(base[col].to_numpy()[: cut + 1], shifted[col].to_numpy()[: cut + 1], equal_nan=True, err_msg=col)
    for fam in ssr.FAMILIES:
        for lv in ssr.LEVELS:
            assert (ssr.signals(base, fam, lv)[: cut + 1] == ssr.signals(shifted, fam, lv)[: cut + 1]).all(), (fam, lv)


def test_higher_timeframe_value_is_last_fully_closed_bar():
    df = ssr.build_features(_klines())
    b15 = ssr._clock_bars(df, 15)
    i = 1000
    close_ms = df.index[i] + 60_000
    last_closed_start = (close_ms // 900_000) * 900_000 - 900_000
    assert df["c15"].iloc[i] == pytest.approx(b15["close"].loc[last_closed_start])
    assert last_closed_start + 900_000 <= close_ms


def _arr(rows: list[tuple[float, float, float, float]], a: float = 0.003) -> dict[str, np.ndarray]:
    o, h, lo, c = (np.array(x, dtype=float) for x in zip(*rows, strict=True))
    n = len(o)
    return {"open": o, "high": h, "low": lo, "close": c, "a": np.full(n, a), "a15": np.full(n, a), "t": np.arange(n) * 60_000}


def test_entry_pays_ask_and_target_sells_at_bid_net_of_fees():
    ct = ssr.Contract("T", stop_a=1.0, target_a=2.0, max_bars=10)
    arr = _arr([(100, 100, 100, 100), (100, 100, 100, 100), (100, 101, 100, 100.5)])
    tr = ssr.simulate(arr, 0, ct, "ETHUSDT")
    hs = ssr.SPREAD_BPS["ETHUSDT"] / 2e4
    entry = 100 * (1 + hs + ssr.SLIP)
    qty = ssr._floor_qty(ssr.NOTIONAL / entry, ssr.STEP["ETHUSDT"])
    exit_px = entry * 1.006 * (1 - hs - ssr.SLIP)
    assert tr["reason"] == "TARGET"
    assert tr["net"] == pytest.approx(qty * (exit_px - entry) - ssr.FEE * qty * (entry + exit_px))
    assert tr["entry_ms"] == arr["t"][1]


def test_stop_fills_first_when_bar_touches_both_and_gaps_fill_at_open():
    ct = ssr.Contract("T", stop_a=1.0, target_a=1.0, max_bars=10)
    both = ssr.simulate(_arr([(100, 100, 100, 100), (100, 101, 99, 100)]), 0, ct, "BTCUSDT")
    assert both["reason"] == "STOP" and both["net"] < 0
    gap = ssr.simulate(_arr([(100, 100, 100, 100), (100, 100, 100, 100), (98, 98, 97, 97.5)]), 0, ct, "BTCUSDT")
    hs = ssr.SPREAD_BPS["BTCUSDT"] / 2e4
    entry = 100 * (1 + hs + ssr.SLIP)
    qty = ssr._floor_qty(ssr.NOTIONAL / entry, ssr.STEP["BTCUSDT"])
    exit_px = 98 * (1 - hs - ssr.STOP_SLIP)
    assert gap["net"] == pytest.approx(qty * (exit_px - entry) - ssr.FEE * qty * (entry + exit_px))


def test_quantity_floors_to_venue_step():
    assert ssr._floor_qty(0.123456, 1e-3) == pytest.approx(0.123)
    assert ssr._floor_qty(0.1, 0.1) == pytest.approx(0.1)


def test_fold_fit_and_validation_never_share_a_trade():
    edges = ssr.block_edges(0, 5_000)
    trades = [{"entry_ms": e, "exit_ms": e + d, "net": 1.0} for e in range(0, 5_000, 37) for d in (10, 400, 1500)]
    for k in range(1, ssr.N_BLOCKS):
        fit = ssr.fit_window(trades, edges[0], edges[k])
        ev = ssr.in_block(trades, edges[k], edges[k + 1])
        assert all(t["exit_ms"] <= edges[k] for t in fit)
        assert all(edges[k] <= t["entry_ms"] for t in ev)
        assert not {id(t) for t in fit} & {id(t) for t in ev}


def test_metrics_payoff_math():
    m = ssr.metrics([{"net": v, "exit_ms": i} for i, v in enumerate([0.3, -0.1, 0.3, -0.2])])
    assert m["avg_win"] == pytest.approx(0.3) and m["avg_loss"] == pytest.approx(-0.15)
    assert m["payoff"] == pytest.approx(2.0)
    assert m["breakeven_wr"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["pf"] == pytest.approx(2.0)
    assert m["max_dd"] == pytest.approx(-0.2)


def test_micro_replay_buys_next_ask_sells_recorded_bid():
    g = {
        "ts": np.array([0.0, 5.0, 10.0, 15.0]),
        "best_bid": np.array([100.0, 100.0, 100.0, 100.2]),
        "best_ask": np.array([100.01, 100.01, 100.01, 100.21]),
        "agg_flow_imbalance_5s": np.zeros(4),
        "ofi_5s": np.zeros(4),
    }
    ct = smi.Contract("T", target_bps=8.0, stop_bps=8.0, max_sec=60.0)
    tr = smi.simulate(g, 0, ct, "ETH")
    entry = 100.01 * (1 + smi.SLIP)
    assert tr["reason"] == "TARGET"
    qty = np.floor(smi.NOTIONAL / entry / smi.STEP["ETH"] + 1e-9) * smi.STEP["ETH"]
    exit_px = 100.2 * (1 - smi.SLIP)
    assert tr["net"] == pytest.approx(qty * (exit_px - entry) - smi.FEE * qty * (entry + exit_px))


def test_micro_replay_skips_stale_next_snapshot():
    g = {
        "ts": np.array([0.0, 60.0, 65.0]),
        "best_bid": np.full(3, 100.0),
        "best_ask": np.full(3, 100.01),
        "agg_flow_imbalance_5s": np.zeros(3),
        "ofi_5s": np.zeros(3),
    }
    assert smi.simulate(g, 0, smi.CONTRACTS[0], "BTC") is None
