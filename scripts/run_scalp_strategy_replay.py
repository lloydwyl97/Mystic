#!/usr/bin/env python3
"""Replay paper scalp strategies over recent Binance.US 1m klines (4–8h, all symbols)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("SCALP_LIVE", "false")
os.environ.setdefault("SCALP_CALIBRATION_MODE", "true")
os.environ.setdefault("SCALP_PAPER_ENABLED", "true")
os.environ.setdefault("SCALP_FEE_MODEL_VERIFIED", "true")
os.environ.setdefault("SCALP_PRODUCTS", "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")

from backend.services.binance_scalp.calibration_profiles import (  # noqa: E402
    apply_profile,
    economics_for_config,
)
from backend.services.binance_scalp.config import ScalpConfig  # noqa: E402
from backend.services.binance_scalp.exit_manager import (  # noqa: E402
    EXIT_MOMENTUM_FAILED,
    EXIT_NET_PROFIT_TARGET,
    EXIT_SETUP_INVALIDATED,
    DECISION_SELL,
    PositionTrack,
    evaluate_exit,
)
from backend.services.binance_scalp.market_reader import MarketSnapshot  # noqa: E402
from backend.services.binance_scalp.momentum_tracker import MomentumTracker  # noqa: E402
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter  # noqa: E402
from backend.services.binance_scalp.strategies import ALL_STRATEGIES, STRATEGY_NAMES, enabled_strategies  # noqa: E402

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
NOTIONAL = 25.0
STEP_SEC = 60
LOOKAHEAD_BARS = 8


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolated scalp strategy replay")
    p.add_argument(
        "--only-strategy",
        choices=STRATEGY_NAMES,
        default=None,
        help="Enable exactly one strategy (ignores SCALP_DISABLED_STRATEGIES env leakage)",
    )
    p.add_argument(
        "--profile",
        choices=("strict", "moderate", "fast"),
        default=os.getenv("SCALP_CALIBRATION_PROFILE", "moderate"),
    )
    p.add_argument("--hours", type=int, default=int(os.getenv("SCALP_REPLAY_HOURS", "168")))
    p.add_argument(
        "--spread-mult",
        type=float,
        default=float(os.getenv("SCALP_REPLAY_SPREAD_MULT", "1.0")),
        help="Multiply synthetic spread estimate for sensitivity tests",
    )
    return p.parse_args()


def _config_from_args(args: argparse.Namespace) -> ScalpConfig:
    base = ScalpConfig.from_env()
    disabled = base.disabled_strategies
    if args.only_strategy:
        disabled = frozenset(s for s in STRATEGY_NAMES if s != args.only_strategy)
    return replace(
        base,
        disabled_strategies=disabled,
        calibration_profile=args.profile,
        calibration_mode=True,
        scalp_live=False,
        scalp_paper_enabled=True,
    )


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "30", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            break
        rows = json.loads(proc.stdout)
        if not isinstance(rows, list) or not rows:
            break
        for r in rows:
            bars.append(
                {
                    "ts_ms": int(r[0]),
                    "epoch": int(r[0]) / 1000.0,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
            )
        last_ms = int(rows[-1][0])
        if last_ms <= cursor:
            break
        cursor = last_ms + 60_000
        time.sleep(0.08)
    return bars


def spread_est(bar: dict) -> float:
    c = bar["close"]
    if c <= 0:
        return 1.0
    return max((bar["high"] - bar["low"]) / c * 0.35, 0.00015)


def synthetic_book(close: float, spread_pct: float) -> tuple[float, float, float, list, list]:
    half = spread_pct / 2.0
    bid = close * (1.0 - half)
    ask = close * (1.0 + half)
    mid = (bid + ask) / 2.0
    tick = close * 0.00005
    qty = 2.0
    bids = [[bid - i * tick, qty] for i in range(25)]
    asks = [[ask + i * tick, qty] for i in range(25)]
    imb = sum(b[1] for b in bids[:10]) / max(sum(a[1] for a in asks[:10]), 1e-9)
    return bid, ask, mid, bids, asks


def make_snap(symbol: str, bar: dict, *, spread_mult: float = 1.0) -> MarketSnapshot:
    sp = spread_est(bar) * max(0.5, spread_mult)
    bid, ask, mid, bids, asks = synthetic_book(bar["close"], sp)
    imb = sum(b[1] for b in bids[:8]) / max(sum(a[1] for a in asks[:8]), 1e-9)
    return MarketSnapshot(
        symbol=symbol,
        symbol_bus=symbol,
        best_bid=bid,
        best_ask=ask,
        mid=mid,
        spread_pct=sp,
        bids=bids,
        asks=asks,
        redis_spread_pct=sp,
        order_book_imbalance=imb,
        book_source="replay_synthetic",
        orderbook_age_sec=0.5,
    )


def net_pct_at_bid(
    entry: float,
    bid: float,
    spread_pct: float,
    impact: float,
    econ,
) -> float:
    if entry <= 0:
        return 0.0
    gross = (bid - entry) / entry
    rt = econ.roundtrip_cost_pct(spread_pct, impact, impact * 0.5)
    return gross - rt


@dataclass
class OpenTrade:
    symbol: str
    setup_name: str
    entry_price: float
    entry_epoch: float
    impact_pct: float
    setup_context: dict
    track: PositionTrack


@dataclass
class StrategyStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    net_pnl_usd: float = 0.0
    profit_exits: int = 0
    setup_invalidated: int = 0
    momentum_failed: int = 0
    recovery_holds: int = 0
    hold_seconds: list[float] = field(default_factory=list)
    false_entries: int = 0


def replay_symbol(
    symbol: str,
    bars: list[dict],
    *,
    config: ScalpConfig,
    econ,
    router: ScalpStrategyRouter,
    momentum: MomentumTracker,
    spread_mult: float = 1.0,
) -> tuple[list[dict], dict[str, StrategyStats], int]:
    trades: list[dict] = []
    stats: dict[str, StrategyStats] = {
        s.name: StrategyStats() for s in enabled_strategies(config)
    }
    missed = 0
    open_pos: OpenTrade | None = None
    cooldown_until = 0.0

    for idx in range(20, len(bars) - LOOKAHEAD_BARS):
        bar = bars[idx]
        epoch = bar["epoch"]
        snap = make_snap(symbol, bar, spread_mult=spread_mult)

        for sub in range(4):
            t = epoch - 45 + sub * 15
            price = bar["open"] + (bar["close"] - bar["open"]) * (sub + 1) / 4
            sp = spread_est(bar)
            bid = price * (1.0 - sp / 2)
            mid = price
            momentum.record(symbol, t, bid, mid)
        momentum.record(symbol, epoch, snap.best_bid, snap.mid)

        window = bars[max(0, idx - 60) : idx + 1]
        kline_window = [
            {"high": b["high"], "low": b["low"], "close": b["close"], "volume": b["volume"]}
            for b in window
        ]

        if open_pos is not None:
            hold = epoch - open_pos.entry_epoch
            mom = momentum.diagnostics(symbol, epoch, snap.best_bid, snap.mid)
            net_pct = net_pct_at_bid(
                open_pos.entry_price,
                snap.best_bid,
                snap.spread_pct,
                open_pos.impact_pct,
                econ,
            )
            profit_hit = net_pct >= econ.net_profit_target_pct
            trigger = econ.stale_scalp_timeout_sec
            perform_review = hold >= trigger
            review = evaluate_exit(
                track=open_pos.track,
                snap=snap,
                mom=mom,
                econ=econ,
                config=config,
                trade_id="replay",
                hold_sec=hold,
                executable_net_pct=net_pct,
                profit_hit=profit_hit,
                exit_spread_ok=True,
                perform_review=perform_review,
            )
            open_pos.track = review.updated_track
            st = stats.get(open_pos.setup_name)
            if st and review.state in ("RECOVERY_HOLD", "HEALTHY_HOLD") and perform_review:
                st.recovery_holds += 1

            if review.decision == DECISION_SELL and review.exit_reason:
                qty = NOTIONAL / open_pos.entry_price
                pnl = net_pct * open_pos.entry_price * qty
                win = pnl > 0
                if st:
                    st.trades += 1
                    st.net_pnl_usd += pnl
                    st.hold_seconds.append(hold)
                    if win:
                        st.wins += 1
                    else:
                        st.losses += 1
                        st.false_entries += 1
                    if review.exit_reason == EXIT_NET_PROFIT_TARGET:
                        st.profit_exits += 1
                    elif review.exit_reason == EXIT_SETUP_INVALIDATED:
                        st.setup_invalidated += 1
                    elif review.exit_reason == EXIT_MOMENTUM_FAILED:
                        st.momentum_failed += 1
                trades.append(
                    {
                        "symbol": symbol,
                        "setup_name": open_pos.setup_name,
                        "entry_epoch": open_pos.entry_epoch,
                        "exit_epoch": epoch,
                        "hold_sec": hold,
                        "net_pct": net_pct,
                        "pnl_usd": pnl,
                        "exit_reason": review.exit_reason,
                        "win": win,
                    }
                )
                open_pos = None
                cooldown_until = epoch + 120
            continue

        if epoch < cooldown_until:
            continue

        best, _ = router.evaluate_symbol(
            symbol,
            epoch=epoch,
            notional_usd=NOTIONAL,
            snap=snap,
            bars=kline_window,
        )
        if best is None or not best.passed:
            chunk = bars[idx : idx + LOOKAHEAD_BARS]
            max_high = max(b["high"] for b in chunk)
            sp = snap.spread_pct
            potential = (max_high - snap.best_ask) / snap.best_ask - econ.roundtrip_cost_pct(sp, 0, 0)
            if potential >= econ.net_profit_target_pct:
                missed += 1
            continue

        track = PositionTrack(
            entry_price=best.limit_buy_price,
            state="OPEN",
            max_favorable_pct=0.0,
            max_adverse_pct=0.0,
            session_low_bid=best.limit_buy_price,
            stale_review_count=0,
            review_lows=(),
            setup_name=best.setup_name,
            setup_context=dict(best.setup_context),
        )
        open_pos = OpenTrade(
            symbol=symbol,
            setup_name=best.setup_name,
            entry_price=best.limit_buy_price,
            entry_epoch=epoch,
            impact_pct=best.impact_pct,
            setup_context=dict(best.setup_context),
            track=track,
        )

    if open_pos is not None:
        bar = bars[-LOOKAHEAD_BARS]
        snap = make_snap(symbol, bar, spread_mult=spread_mult)
        net_pct = net_pct_at_bid(
            open_pos.entry_price, snap.best_bid, snap.spread_pct, open_pos.impact_pct, econ
        )
        trades.append(
            {
                "symbol": symbol,
                "setup_name": open_pos.setup_name,
                "entry_epoch": open_pos.entry_epoch,
                "exit_epoch": bar["epoch"],
                "hold_sec": bar["epoch"] - open_pos.entry_epoch,
                "net_pct": net_pct,
                "pnl_usd": net_pct * NOTIONAL,
                "exit_reason": "REPLAY_END_OPEN",
                "win": net_pct > 0,
            }
        )

    return trades, stats, missed


def main() -> int:
    args = _parse_args()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    config = _config_from_args(args)
    assert not config.scalp_live
    econ = economics_for_config(config)
    momentum = MomentumTracker()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=type("R", (), {"read": lambda _s, sym: None})(),  # unused in replay
        momentum=momentum,
    )

    all_trades: list[dict] = []
    by_strategy_symbol: dict[str, dict[str, StrategyStats]] = {}
    missed_by_symbol: dict[str, int] = {}
    symbol_bars: dict[str, list[dict]] = {}

    for sym in SYMBOLS:
        bars = fetch_klines(sym, start_ms, end_ms)
        symbol_bars[sym] = bars
        trades, stats, missed = replay_symbol(
            sym,
            bars,
            config=config,
            econ=econ,
            router=router,
            momentum=momentum,
            spread_mult=args.spread_mult,
        )
        all_trades.extend(trades)
        by_strategy_symbol[sym] = stats
        missed_by_symbol[sym] = missed
        momentum._history.pop(sym, None)

    enabled = enabled_strategies(config)
    enabled_names = {s.name for s in enabled}
    agg: dict[str, StrategyStats] = {s.name: StrategyStats() for s in enabled}
    for sym_stats in by_strategy_symbol.values():
        for name, st in sym_stats.items():
            a = agg[name]
            a.trades += st.trades
            a.wins += st.wins
            a.losses += st.losses
            a.net_pnl_usd += st.net_pnl_usd
            a.profit_exits += st.profit_exits
            a.setup_invalidated += st.setup_invalidated
            a.momentum_failed += st.momentum_failed
            a.recovery_holds += st.recovery_holds
            a.false_entries += st.false_entries
            a.hold_seconds.extend(st.hold_seconds)

    best_by_symbol: dict[str, str] = {}
    for sym, sym_stats in by_strategy_symbol.items():
        ranked = sorted(
            sym_stats.items(),
            key=lambda x: (x[1].net_pnl_usd, x[1].wins - x[1].losses),
            reverse=True,
        )
        if ranked and ranked[0][1].trades > 0:
            best_by_symbol[sym] = ranked[0][0]
        else:
            best_by_symbol[sym] = "none"

    stale_exits = sum(1 for t in all_trades if t.get("exit_reason") == "STALE_SCALP_TIMEOUT")
    total_net = round(sum(t["pnl_usd"] for t in all_trades), 4)
    positive_strategies = [
        name
        for name, st in agg.items()
        if name in enabled_names and st.trades > 0 and st.net_pnl_usd > 0
    ]
    negative_enabled = [
        name
        for name, st in agg.items()
        if name in enabled_names and st.trades > 0 and st.net_pnl_usd < 0
    ]
    replay_pass = total_net >= 0 and not negative_enabled

    def _fmt_stats(st: StrategyStats) -> dict:
        avg_hold = statistics.mean(st.hold_seconds) if st.hold_seconds else 0.0
        return {
            "trades": st.trades,
            "wins": st.wins,
            "losses": st.losses,
            "net_pnl_usd": round(st.net_pnl_usd, 4),
            "profit_target_exits": st.profit_exits,
            "setup_invalidated_exits": st.setup_invalidated,
            "momentum_failed_exits": st.momentum_failed,
            "recovery_holds": st.recovery_holds,
            "avg_hold_sec": round(avg_hold, 1),
            "false_entries": st.false_entries,
        }

    report = {
        "replay_hours": args.hours,
        "symbols": list(SYMBOLS),
        "only_strategy": args.only_strategy,
        "calibration_profile": args.profile,
        "spread_mult": args.spread_mult,
        "scalp_live": config.scalp_live,
        "calibration_mode": config.calibration_mode,
        "replay_pass": replay_pass,
        "disabled_strategies": sorted(config.disabled_strategies),
        "enabled_strategies": sorted(enabled_names),
        "negative_enabled_strategies": negative_enabled,
        "positive_strategies": positive_strategies,
        "stale_timeout_exits": stale_exits,
        "missed_profitable_windows": missed_by_symbol,
        "by_strategy": {k: _fmt_stats(v) for k, v in agg.items()},
        "by_strategy_symbol": {
            sym: {k: _fmt_stats(v) for k, v in stats.items()}
            for sym, stats in by_strategy_symbol.items()
        },
        "best_strategy_by_symbol": best_by_symbol,
        "trades": all_trades,
        "total_trades": len(all_trades),
        "total_net_pnl_usd": total_net,
    }
    out_path = REPO / "scripts" / "scalp_strategy_replay_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if replay_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
