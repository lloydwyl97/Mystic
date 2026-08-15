"""evaluate_all must finish for all four coins even if the money DB is write-locked."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from pathlib import Path

from backend.services.binance_scalp.scalp_opportunity_dataset import (
    label_due_opportunities,
    record_opportunity_cycle,
)
from backend.services.binance_scalp.schema import init_scalp_schema
from backend.services.binance_scalp.strategies import ALL_STRATEGIES


class _Snap:
    symbol = "BTCUSDT"
    symbol_bus = "BTCUSDT"
    mid = 100.0
    best_bid = 99.99
    best_ask = 100.01
    spread_pct = 0.0002
    order_book_imbalance = 0.1
    orderbook_age_sec = 0.2
    bids = [[99.99, 20.0]]
    asks = [[100.01, 20.0]]


class _Mom:
    mid_change_15s = 0.0001
    mid_change_30s = 0.0002
    mid_change_60s = 0.0001
    bid_change_15s = 0.0001
    bid_change_30s = 0.0001
    bid_change_60s = 0.0001
    realized_volatility_pct = 0.001


class _Reader:
    def read(self, symbol: str):
        snap = _Snap()
        snap.symbol = symbol
        snap.symbol_bus = symbol
        return snap


class _Klines:
    def __init__(self) -> None:
        self.bars = [
            {"open": 99.8, "high": 100.2, "low": 99.7, "close": 100.0 + i * 0.01, "volume": 10 + i, "ts": 1_700_000_000 + i * 60}
            for i in range(40)
        ]

    def get(self, symbol: str, *, minutes: int = 120):
        return list(self.bars)

    def get_5m(self, symbol: str, *, minutes: int = 240):
        return list(self.bars)

    def get_15m(self, symbol: str, *, minutes: int = 720):
        return list(self.bars)

    def get_1h(self, symbol: str, *, minutes: int = 2880):
        return list(self.bars)


def _router(tmp_path: Path):
    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.binance_scalp.momentum_tracker import MomentumTracker
    from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter

    cfg = replace(
        ScalpConfig.from_env(),
        products=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
        database_path=str(tmp_path / "scalp.db"),
    )
    init_scalp_schema(cfg.database_path, principal=1000.0)
    return ScalpStrategyRouter(
        config=cfg,
        econ=ScalpEconomics.from_env(),
        reader=_Reader(),
        momentum=MomentumTracker(),
        klines=_Klines(),
    )


def test_evaluate_all_completes_all_four_under_write_lock(tmp_path: Path):
    router = _router(tmp_path)
    holder = sqlite3.connect(router.config.database_path, timeout=10)
    holder.execute("BEGIN IMMEDIATE")
    t0 = time.perf_counter()
    rows = router.evaluate_all(epoch=time.time(), notional_usd=50.0)
    elapsed = time.perf_counter() - t0
    holder.commit()
    holder.close()
    assert elapsed < 20.0
    assert {r["symbol"] for r in rows} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    for row in rows:
        signals = row.get("all_signals") or []
        assert len(signals) == len(ALL_STRATEGIES)
        assert "reject_reason" in signals[0] or signals[0].get("passed") in (True, False)


def test_nine_strategies_return_control(tmp_path: Path):
    from backend.services.binance_scalp.strategies.base import StrategyMarketContext

    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.economics import ScalpEconomics

    bars = _Klines().bars
    ctx = StrategyMarketContext(
        symbol="BTCUSDT",
        snap=_Snap(),
        mom=_Mom(),
        bars_1m=bars,
        econ=ScalpEconomics.from_env(),
        config=ScalpConfig.from_env(),
        notional_usd=50,
    )
    for strat in ALL_STRATEGIES:
        t0 = time.perf_counter()
        sig = strat.evaluate(ctx)
        assert (time.perf_counter() - t0) < 1.0
        assert sig.setup_name
        assert sig.passed in (True, False)


def test_opportunity_record_and_label(tmp_path: Path):
    db = str(tmp_path / "scalp.db")
    init_scalp_schema(db, principal=1000.0)
    epoch = time.time() - 90
    n = record_opportunity_cycle(
        db,
        rows=[
            {
                "symbol": "BTCUSDT",
                "mid": 100.0,
                "spread_pct": 0.0002,
                "rank_score": 0.4,
                "strategy_passed": False,
                "soft_reason": "NO_PULLBACK_RECOVERY",
                "all_signals": [{"setup_name": "vwap_ema_reclaim", "passed": False, "reject_reason": "NO_PULLBACK_RECOVERY"}],
                "rank_meta": {"setup_measurements": {"vwap_ema_reclaim": {"reclaim_strength": 0.1}}},
            }
        ],
        epoch=epoch,
    )
    assert n == 1

    class _Later:
        def read(self, symbol: str):
            snap = _Snap()
            snap.mid = 100.2
            return snap

    labeled = label_due_opportunities(db, _Later(), now_epoch=time.time(), cost_pct=0.0006)
    assert labeled >= 1
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT symbol,best_reject,plus_30s_net,plus_60s_net,horizon_labels_json FROM scalp_opportunity_snapshots"
    ).fetchone()
    conn.close()
    assert row[0] == "BTCUSDT"
    assert row[1] == "NO_PULLBACK_RECOVERY"
    assert row[2] is not None
    assert row[3] is not None
    assert "30" in (row[4] or "")
