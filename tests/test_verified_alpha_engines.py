"""Tests for verified causal alpha strategies (DAY_V2 + SCALP_V2) and calibration gates."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.services.day_v2.live_exit_evaluator import day_v2_recorded_exit_reason, evaluate_day_v2_exit
from backend.services.portfolio_engine import ExitType, strategy_exit_type
from backend.services.scalp_v2.exit_evaluator import (
    SCALP_V2_EXIT_CATASTROPHIC,
    SCALP_V2_EXIT_VERIFIED_HORIZON,
    evaluate_scalp_v2_exit,
    scalp_v2_recorded_exit_reason,
)
from backend.services.strategy_version import CURRENT_EXIT_AUTHORITIES, DAY_ENGINE, SCALP_ENGINE
from backend.services.verified_alpha_signals import (
    DAY_VERIFIED_LEAD_LAG_SETUP,
    SCALP_VERIFIED_CAPITULATION_SETUP,
    SCALP_VERIFIED_LEAD_LAG_SETUP,
    evaluate_day_verified_signals,
    evaluate_scalp_verified_signals,
    is_symbol_on_verified_cooldown,
)


def _init_ohlcv_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE feature_ohlcv (
            symbol TEXT,
            interval TEXT,
            ts TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE paper_trades (
            trade_id TEXT,
            engine_id TEXT,
            symbol TEXT,
            side TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn


def _insert_bars(
    conn: sqlite3.Connection,
    symbol: str,
    start_ts: int,
    closes: list[float],
    volumes: list[float] | None = None,
) -> None:
    vols = volumes if volumes is not None else [10.0] * len(closes)
    rows = []
    for idx, (px, vol) in enumerate(zip(closes, vols, strict=True)):
        t = start_ts + idx * 60
        iso = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
        rows.append((symbol, "1m", iso, px, px * 1.001, px * 0.999, px, vol))
    conn.executemany(
        "INSERT INTO feature_ohlcv(symbol, interval, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def test_day_verified_lead_lag_triggers_and_ranks_by_lag(tmp_path: Path) -> None:
    db = tmp_path / "alpha.db"
    conn = _init_ohlcv_db(db)
    base_ts = 1_760_000_000
    n = 70
    # BTC jumps +0.90% in the final 15 minutes
    btc = [100.0] * 55 + [100.0 + 0.90 * ((i + 1) / 15.0) for i in range(15)]
    # ETH flat (+0.00% -> lag +90 bps), SOL rises +0.30% (lag +60 bps), XRP rises +0.80% (lag +10 bps < 30 bps)
    eth = [2000.0] * n
    sol = [100.0] * 55 + [100.0 + 0.30 * ((i + 1) / 15.0) for i in range(15)]
    xrp = [2.0] * 55 + [2.0 * (1.0 + 0.0080 * ((i + 1) / 15.0)) for i in range(15)]

    _insert_bars(conn, "BTC-USDT", base_ts, btc)
    _insert_bars(conn, "ETH-USDT", base_ts, eth)
    _insert_bars(conn, "SOL-USDT", base_ts, sol)
    _insert_bars(conn, "XRP-USDT", base_ts, xrp)
    conn.close()

    # Exact close of final bar = base_ts + (n-1)*60 + 60
    as_of = base_ts + n * 60 + 5
    sigs = evaluate_day_verified_signals(str(db), now_ts=as_of)
    assert [s.symbol for s in sigs] == ["ETHUSDT", "SOLUSDT"]
    assert all(s.setup == DAY_VERIFIED_LEAD_LAG_SETUP for s in sigs)
    assert sigs[0].score > sigs[1].score


def test_verified_causality_and_halt_guards(tmp_path: Path) -> None:
    db = tmp_path / "guard.db"
    conn = _init_ohlcv_db(db)
    base_ts = 1_760_000_000
    n = 70
    btc = [100.0] * 69 + [102.0]  # Only the very last bar jumps +2%
    eth = [2000.0] * n
    _insert_bars(conn, "BTC-USDT", base_ts, btc)
    _insert_bars(conn, "ETH-USDT", base_ts, eth, volumes=[10.0] * 69 + [0.0])  # zero vol on last bar
    conn.close()

    # 1. Before final bar closes -> strictly causal loader ignores final bar -> no signal
    mid_final_bar = base_ts + (n - 1) * 60 + 30
    assert evaluate_day_verified_signals(str(db), now_ts=mid_final_bar) == []

    # 2. After final bar closes -> ETH latest volume is 0.0 -> rejected as stale quote
    after_close = base_ts + n * 60 + 5
    assert evaluate_day_verified_signals(str(db), now_ts=after_close) == []

    # 3. Halt check: BTC with only 3 active bars in last 30m -> blocked
    db2 = tmp_path / "halt.db"
    conn2 = _init_ohlcv_db(db2)
    _insert_bars(conn2, "BTC-USDT", base_ts, btc, volumes=[0.0] * 67 + [1.0, 1.0, 1.0])
    _insert_bars(conn2, "ETH-USDT", base_ts, eth)
    conn2.close()
    assert evaluate_day_verified_signals(str(db2), now_ts=after_close) == []


def test_scalp_verified_capitulation_and_lead_lag(tmp_path: Path) -> None:
    db = tmp_path / "scalp.db"
    conn = _init_ohlcv_db(db)
    base_ts = 1_760_000_000
    n = 70
    # BTC flat then +0.70% in last 5m
    btc = [100.0] * 65 + [100.0 + 0.70 * ((i + 1) / 5.0) for i in range(5)]
    # ETH drops -3.5% over 60m -> CAPITULATION_FLUSH_REBOUND
    eth = [2000.0] * 10 + [2000.0 * (1.0 - 0.035 * ((i + 1) / 60.0)) for i in range(60)]
    # SOL flat -> triggers BTC_LEAD_LAG_SCALP (+70 bps lag >= 25 bps)
    sol = [100.0] * n
    # XRP moves +0.60% with BTC -> lag 10 bps < 25 bps -> no trigger
    xrp = [2.0] * 65 + [2.0 * (1.0 + 0.0060 * ((i + 1) / 5.0)) for i in range(5)]

    _insert_bars(conn, "BTC-USDT", base_ts, btc)
    _insert_bars(conn, "ETH-USDT", base_ts, eth)
    _insert_bars(conn, "SOL-USDT", base_ts, sol)
    _insert_bars(conn, "XRP-USDT", base_ts, xrp)
    conn.close()

    as_of = base_ts + n * 60 + 5
    sigs = evaluate_scalp_verified_signals(str(db), now_ts=as_of)
    by_sym = {s.symbol: s.setup for s in sigs}
    assert by_sym["ETHUSDT"] == SCALP_VERIFIED_CAPITULATION_SETUP
    assert by_sym["SOLUSDT"] == SCALP_VERIFIED_LEAD_LAG_SETUP
    assert "XRPUSDT" not in by_sym


def test_day_verified_exit_bypasses_continuation_and_honors_horizon_and_catastrophic() -> None:
    entry_ts = 1_760_000_000.0
    # Hold at 30m even when learned continuation surface says sell (-0.01)
    dec_hold = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=100.4,
        bar_low=99.5,
        structural_anchor=97.5,
        target_price=101.5,
        atr_at_entry=1.0,
        highest_price=100.6,
        entry_time=entry_ts,
        estimated_roundtrip_cost=0.00066,
        now=entry_ts + 30 * 60,
        setup=DAY_VERIFIED_LEAD_LAG_SETUP,
        expected_terminal_net=-0.0100,
    )
    assert dec_hold is None

    # Exit at 60m horizon
    dec_horizon = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=100.8,
        bar_low=99.5,
        structural_anchor=97.5,
        target_price=101.5,
        atr_at_entry=1.0,
        highest_price=100.8,
        entry_time=entry_ts,
        estimated_roundtrip_cost=0.00066,
        now=entry_ts + 60 * 60,
        setup=DAY_VERIFIED_LEAD_LAG_SETUP,
        expected_terminal_net=0.0200,
    )
    assert dec_horizon is not None
    assert dec_horizon["action"] == "sell"
    assert dec_horizon["reason"] == "DAY_V2_VERIFIED_HORIZON_EXIT"
    assert day_v2_recorded_exit_reason(dec_horizon["reason"]) == "DAY_V2_VERIFIED_HORIZON_EXIT"
    assert strategy_exit_type(dec_horizon["reason"]) == ExitType.STRATEGY
    assert dec_horizon["reason"] in CURRENT_EXIT_AUTHORITIES[DAY_ENGINE]

    # Catastrophic exit at >= 2.5% adverse move
    dec_cat = evaluate_day_v2_exit(
        engine_id="DAY_V2",
        entry_price=100.0,
        current_price=97.4,
        bar_low=97.4,
        structural_anchor=97.5,
        target_price=101.5,
        atr_at_entry=1.0,
        highest_price=100.0,
        entry_time=entry_ts,
        estimated_roundtrip_cost=0.00066,
        now=entry_ts + 10 * 60,
        setup=DAY_VERIFIED_LEAD_LAG_SETUP,
    )
    assert dec_cat is not None
    assert dec_cat["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"


def test_scalp_verified_exit_bypasses_continuation_and_honors_horizon_and_catastrophic() -> None:
    pos = SimpleNamespace(
        engine_id="SCALP_V2",
        symbol="ETH/USDT",
        entry_price=2000.0,
        entry_thesis=SCALP_VERIFIED_CAPITULATION_SETUP,
        adaptive_decision={"setup": SCALP_VERIFIED_CAPITULATION_SETUP},
    )
    # Hold at 15m even with negative expected_terminal_net
    res_hold = evaluate_scalp_v2_exit(
        position=pos,
        current_price=2010.0,
        bar_low=1980.0,
        hold_minutes=15.0,
        net_pnl_pct=0.0043,
        expected_terminal_net=-0.0100,
    )
    assert res_hold["action"] == "hold"

    # Exit at 30m horizon
    res_time = evaluate_scalp_v2_exit(
        position=pos,
        current_price=2015.0,
        bar_low=1990.0,
        hold_minutes=30.0,
        net_pnl_pct=0.0068,
        expected_terminal_net=0.0200,
    )
    assert res_time["action"] == "sell"
    assert res_time["reason"] == SCALP_V2_EXIT_VERIFIED_HORIZON
    assert scalp_v2_recorded_exit_reason(res_time["reason"]) == "SCALP_V2_VERIFIED_HORIZON_EXIT"
    assert strategy_exit_type(res_time["reason"]) == ExitType.STRATEGY
    assert res_time["reason"] in CURRENT_EXIT_AUTHORITIES[SCALP_ENGINE]

    # Catastrophic stop at >= 3.0% adverse move
    res_cat = evaluate_scalp_v2_exit(
        position=pos,
        current_price=1938.0,
        bar_low=1938.0,
        hold_minutes=8.0,
        net_pnl_pct=-0.031,
    )
    assert res_cat["action"] == "sell"
    assert res_cat["reason"] == SCALP_V2_EXIT_CATASTROPHIC


def test_verified_cooldown_is_engine_scoped_and_restart_safe(tmp_path: Path) -> None:
    db = tmp_path / "cd.db"
    conn = _init_ohlcv_db(db)
    t_trade = 1_760_000_000.0
    iso = datetime.fromtimestamp(t_trade, tz=timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO paper_trades(trade_id, engine_id, symbol, side, timestamp) VALUES (?,?,?,?,?)",
        ("t1", "DAY_V2", "ETH-USDT", "BUY", iso),
    )
    conn.commit()
    conn.close()

    assert is_symbol_on_verified_cooldown(str(db), "DAY_V2", "ETHUSDT", now_ts=t_trade + 600.0) is True
    assert is_symbol_on_verified_cooldown(str(db), "DAY_V2", "ETHUSDT", now_ts=t_trade + 1805.0) is False
    assert is_symbol_on_verified_cooldown(str(db), "SCALP_V2", "ETHUSDT", now_ts=t_trade + 600.0) is False
