"""Replay all nine SCALP strategy modules on historical 1m bars.

Does not enable disabled strategies in the live/paper runner.
Evaluates setup/opportunity structure across BTC/ETH/SOL/XRP equally.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.historical_forensic import _ohlcv_symbol, _parse_ts, load_ohlcv
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics
from backend.services.binance_scalp.strategies import ALL_STRATEGIES
from backend.services.binance_scalp.strategies.base import StrategyMarketContext

LOOKBACK = 30
FEE_RT = 0.0004  # taker+taker default
SLIP_RT = 0.0002
SPREAD = 0.0003
NOTIONAL = 50.0
MAX_HOLD_BARS = 5  # 5 minutes — original practical scalp horizon on 1m bars
STRUCTURAL_HOLD_BARS = 20  # hard-hold horizon (~20m)


def _deep_book(mid: float) -> tuple[list[list[float]], list[list[float]]]:
    bids = [[mid * (1.0 - SPREAD / 2.0 - i * 0.00005), 5_000.0] for i in range(8)]
    asks = [[mid * (1.0 + SPREAD / 2.0 + i * 0.00005), 5_000.0] for i in range(8)]
    return bids, asks


def _mom_from_bars(bars: list[dict[str, Any]]) -> MomentumDiagnostics:
    closes = [float(b["close"]) for b in bars]
    def ch(n: int) -> float:
        if len(closes) <= n or closes[-1 - n] <= 0:
            return 0.0
        return (closes[-1] - closes[-1 - n]) / closes[-1 - n]

    c15 = ch(1) * 0.35
    c30 = ch(1) * 0.70
    c60 = ch(1)
    last = bars[-1]
    rng = (float(last["high"]) - float(last["low"])) / float(last["close"] or 1)
    confirmed = c15 > 0 and c30 > 0 and c60 > 0
    return MomentumDiagnostics(
        mid_change_15s=c15,
        mid_change_30s=c30,
        mid_change_60s=c60,
        bid_change_15s=c15,
        bid_change_30s=c30,
        bid_change_60s=c60,
        last_n_ticks_up_count=sum(1 for i in range(1, min(6, len(closes))) if closes[-i] > closes[-i - 1]),
        sample_count=len(closes),
        history_sec=60.0 * min(len(closes), 5),
        recent_range_pct=rng,
        realized_volatility_pct=rng,
        momentum_confirmed=confirmed,
        flat_regime=abs(c60) < 0.0002,
    )


def _snapshot(symbol: str, mid: float, bars: list[dict[str, Any]]) -> MarketSnapshot:
    bids, asks = _deep_book(mid)
    last = bars[-1]
    direction = 1.0 if float(last["close"]) >= float(last["open"]) else -1.0
    imb = 0.14 * direction
    return MarketSnapshot(
        symbol=symbol,
        symbol_bus=symbol,
        best_bid=bids[0][0],
        best_ask=asks[0][0],
        mid=mid,
        spread_pct=SPREAD,
        bids=bids,
        asks=asks,
        redis_spread_pct=SPREAD,
        order_book_imbalance=imb,
        book_source="replay_synthetic",
        orderbook_age_sec=0.0,
    )


def _simulate_trade(
    future: list[dict[str, Any]],
    *,
    entry: float,
    target_pct: float,
    hold_bars: int,
    scratch_bars: int | None,
) -> dict[str, Any]:
    if not future:
        return {"completed": False}
    mfe = 0.0
    mae = 0.0
    exit_px = entry
    reason = "MAX_HOLD"
    used = 0
    for i, b in enumerate(future[:hold_bars], start=1):
        used = i
        hi = float(b["high"])
        lo = float(b["low"])
        cl = float(b["close"])
        mfe = max(mfe, (hi - entry) / entry if entry else 0.0)
        mae = min(mae, (lo - entry) / entry if entry else 0.0)
        # Conservative fill: target must trade through high.
        if hi >= entry * (1.0 + target_pct + FEE_RT + SLIP_RT + SPREAD / 2.0):
            exit_px = entry * (1.0 + target_pct + FEE_RT + SLIP_RT)
            reason = "NET_PROFIT_TARGET"
            break
        if scratch_bars is not None and i >= scratch_bars and mfe < target_pct * 0.40:
            exit_px = cl * (1.0 - SPREAD / 2.0)
            reason = "EARLY_SCRATCH"
            break
        exit_px = cl * (1.0 - SPREAD / 2.0)
    gross_pct = (exit_px - entry) / entry if entry else 0.0
    net_pct = gross_pct - FEE_RT - SLIP_RT
    net_usd = net_pct * NOTIONAL
    return {
        "completed": True,
        "exit_reason": reason,
        "hold_bars": used,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "gross_pnl": (gross_pct + FEE_RT + SLIP_RT) * NOTIONAL,
        "net_pnl": net_usd,
        "win": net_usd > 0,
        "cost_burden": (FEE_RT + SLIP_RT) * NOTIONAL,
    }


def _stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    if not n:
        return {
            "completed": 0,
            "win_rate": None,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "expectancy": None,
            "profit_factor": None,
            "avg_winner": None,
            "avg_loser": None,
            "mfe": None,
            "mae": None,
            "cost_burden": 0.0,
        }
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gross = sum(t["gross_pnl"] for t in trades)
    net = sum(t["net_pnl"] for t in trades)
    win_sum = sum(t["net_pnl"] for t in wins)
    loss_sum = abs(sum(t["net_pnl"] for t in losses))
    pf = (win_sum / loss_sum) if loss_sum > 0 else (None if not wins else 99.0)
    return {
        "completed": n,
        "win_rate": round(len(wins) / n, 4),
        "gross_pnl": round(gross, 4),
        "net_pnl": round(net, 4),
        "expectancy": round(net / n, 6),
        "profit_factor": round(pf, 4) if pf is not None else None,
        "avg_winner": round(win_sum / len(wins), 6) if wins else None,
        "avg_loser": round(sum(t["net_pnl"] for t in losses) / len(losses), 6) if losses else None,
        "mfe": round(sum(t["mfe_pct"] for t in trades) / n, 6),
        "mae": round(sum(t["mae_pct"] for t in trades) / n, 6),
        "cost_burden": round(sum(t["cost_burden"] for t in trades), 4),
    }


def replay_all_strategies(
    db_path: str,
    *,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"),
    step: int = 2,
) -> dict[str, Any]:
    """Evaluate every strategy module on stored 1m bars.

    `step` skips bars to keep runtime bounded while covering the book.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        bars_by_sym = load_ohlcv(conn)
    finally:
        conn.close()

    # Force paper-relaxed strategy branches (matches Ocean paper runtime).
    os.environ.setdefault("SCALP_PAPER_ENABLED", "true")
    config = ScalpConfig.from_env()
    econ = ScalpEconomics.from_env()
    target = float(econ.net_profit_target_pct)

    # Replay must see all nine modules even if runtime disables seven.
    results: dict[str, Any] = {}
    for strat in ALL_STRATEGIES:
        by_sym: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        opportunities = 0
        for sym in symbols:
            raw = bars_by_sym.get(_ohlcv_symbol(sym), [])
            if len(raw) < LOOKBACK + MAX_HOLD_BARS + 1:
                continue
            for i in range(LOOKBACK, len(raw) - MAX_HOLD_BARS, max(1, step)):
                window = raw[i - LOOKBACK : i + 1]
                mid = float(window[-1]["close"])
                if mid <= 0:
                    continue
                snap = _snapshot(sym, mid, window)
                mom = _mom_from_bars(window)
                ctx = StrategyMarketContext(
                    symbol=sym,
                    snap=snap,
                    mom=mom,
                    bars_1m=window,
                    econ=econ,
                    config=config,
                    notional_usd=NOTIONAL,
                )
                sig = strat.evaluate(ctx)
                if not sig.passed:
                    continue
                opportunities += 1
                future = raw[i + 1 : i + 1 + STRUCTURAL_HOLD_BARS]
                entry = float(snap.best_ask)
                current = _simulate_trade(
                    future,
                    entry=entry,
                    target_pct=target,
                    hold_bars=MAX_HOLD_BARS,
                    scratch_bars=3,
                )
                structural = _simulate_trade(
                    future,
                    entry=entry,
                    target_pct=target,
                    hold_bars=STRUCTURAL_HOLD_BARS,
                    scratch_bars=None,
                )
                if current.get("completed"):
                    current["symbol"] = sym
                    current["state"] = "up" if mom.mid_change_60s > 0 else "down_or_flat"
                    by_sym[sym].append(current)
                    by_state[current["state"]].append(current)
                # keep structural on the trade for later aggregation
                if structural.get("completed"):
                    current["structural_net"] = structural["net_pnl"]
                    current["structural_win"] = structural["win"]
                    current["structural_reason"] = structural["exit_reason"]

        all_trades = [t for rows in by_sym.values() for t in rows]
        results[strat.name] = {
            "opportunities": opportunities,
            "current_policy": _stats(all_trades),
            "structural_hold": _stats(
                [
                    {
                        **t,
                        "net_pnl": t.get("structural_net", t["net_pnl"]),
                        "win": t.get("structural_win", t["win"]),
                        "gross_pnl": t.get("structural_net", t["net_pnl"]) + t["cost_burden"],
                    }
                    for t in all_trades
                    if "structural_net" in t
                ]
            ),
            "btc": _stats(by_sym.get("BTCUSDT", [])),
            "eth": _stats(by_sym.get("ETHUSDT", [])),
            "sol": _stats(by_sym.get("SOLUSDT", [])),
            "xrp": _stats(by_sym.get("XRPUSDT", [])),
            "by_state": {k: _stats(v) for k, v in by_state.items()},
        }

    ranked = sorted(
        results.items(),
        key=lambda kv: (
            kv[1]["structural_hold"].get("expectancy") is not None,
            kv[1]["structural_hold"].get("expectancy") or -999,
            kv[1]["structural_hold"].get("win_rate") or 0,
        ),
        reverse=True,
    )
    return {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "symbols": list(symbols),
        "note": "Replay only. Disabled runtime strategies were NOT enabled in paper/live.",
        "ranked_by_structural_expectancy": [name for name, _ in ranked],
        "strategies": results,
    }
