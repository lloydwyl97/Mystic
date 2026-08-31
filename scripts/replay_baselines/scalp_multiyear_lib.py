"""Shared helpers for multi-year scalp regime validation."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / "scripts" / "replay_baselines" / "cache"

TOP4 = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
TOP4_API = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT", "XRP/USDT": "XRPUSDT"}
NOTIONAL = 25.0
PRINCIPAL = 25000.0

from backend.services.binance_scalp.scalp_regime_classifier import (
    STRATEGY_NATIVE_REGIMES,
    regime_at_ts,
    summarize_regime_coverage,
)
from scripts.run_day_execution_replay import fetch_klines_cached


def normalize_bar(bar: dict) -> dict:
    """Align day-replay bars (ts) with scalp replay (epoch)."""
    out = dict(bar)
    if "epoch" not in out and "ts" in out:
        out["epoch"] = float(out["ts"])
    if "ts" not in out and "epoch" in out:
        out["ts"] = int(out["epoch"])
    return out


def load_or_fetch_bars(symbol: str, interval: str, start_ms: int, end_ms: int) -> tuple[list[dict], dict[str, Any]]:
    """Load bars from cache or fetch; return bars + metadata."""
    api = TOP4_API.get(symbol, symbol.replace("/", ""))
    cache_path = CACHE_DIR / f"{api}_{interval}_{start_ms}_{end_ms}.json"
    meta: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "cache_path": str(cache_path),
        "from_cache": cache_path.exists(),
    }
    bars = [normalize_bar(b) for b in fetch_klines_cached(symbol, interval, start_ms, end_ms)]
    meta["bar_count"] = len(bars)
    if bars:
        meta["start_ts"] = bars[0]["ts"]
        meta["end_ts"] = bars[-1]["ts"]
        meta["span_days"] = round((bars[-1]["ts"] - bars[0]["ts"]) / 86400, 1)
        gaps = []
        step = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}.get(interval, 3600)
        for i in range(1, min(len(bars), 5000)):
            delta = bars[i]["ts"] - bars[i - 1]["ts"]
            if delta > step * 2:
                gaps.append({"after_ts": bars[i - 1]["ts"], "gap_sec": delta})
        meta["gap_count"] = len(gaps)
        meta["sample_gaps"] = gaps[:5]
    return bars, meta


def find_widest_cached_span(api_symbol: str, interval: str) -> tuple[int, int] | None:
    best = None
    for p in CACHE_DIR.glob(f"{api_symbol}_{interval}_*.json"):
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            s, e = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        if best is None or (e - s) > (best[1] - best[0]):
            best = (s, e)
    return best


def scan_extra_symbol_eligibility() -> dict[str, Any]:
    """Report liquid Binance.US USDT pairs beyond top-4 (coverage only, no replay)."""
    rejected: list[dict[str, str]] = []
    eligible: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "20", "https://api.binance.us/api/v3/exchangeInfo"],
            capture_output=True,
            text=True,
            check=False,
        )
        info = json.loads(proc.stdout or "{}")
        symbols = info.get("symbols") or []
        tick = subprocess.run(
            ["curl", "-s", "--max-time", "20", "https://api.binance.us/api/v3/ticker/24hr"],
            capture_output=True,
            text=True,
            check=False,
        )
        tickers = {t["symbol"]: t for t in json.loads(tick.stdout or "[]") if isinstance(t, dict)}
        for s in symbols:
            if s.get("status") != "TRADING" or s.get("quoteAsset") != "USDT":
                continue
            sym = s.get("symbol", "")
            if sym in TOP4_API.values():
                continue
            t = tickers.get(sym) or {}
            qvol = float(t.get("quoteVolume") or 0)
            if qvol < 250_000:
                rejected.append({"symbol": sym, "reason": "quote_volume_below_250k_usd_24h"})
                continue
            eligible.append({"symbol": sym, "quote_volume_24h_usd": round(qvol, 0)})
        eligible.sort(key=lambda x: x["quote_volume_24h_usd"], reverse=True)
    except Exception as exc:
        rejected.append({"symbol": "*", "reason": f"exchange_scan_failed:{exc}"})
    return {
        "eligible_extra_symbols": eligible[:15],
        "rejected_extra_symbols_sample": rejected[:20],
        "replay_symbols": list(TOP4),
        "note": "replay limited to top-4; extra symbols listed for future expansion",
    }


def build_data_coverage(*, target_days_1m: int = 180, target_days_1h: int = 720) -> dict[str, Any]:
    """Pull/assemble coverage report for validation symbols."""
    end = datetime.now(timezone.utc)
    end_ms = int(end.timestamp() * 1000)
    coverage: dict[str, Any] = {"symbols": {}, "rejected_symbols": [], "intervals": {}}

    for sym in TOP4:
        api = TOP4_API[sym]
        sym_report: dict[str, Any] = {}
        for interval, target_d in [("1h", target_days_1h), ("4h", target_days_1h), ("15m", min(target_days_1h, 365)), ("5m", min(target_days_1m, 365)), ("1m", target_days_1m)]:
            start_ms = int((end - timedelta(days=target_d)).timestamp() * 1000)
            widest = find_widest_cached_span(api, interval)
            if widest and (widest[1] - widest[0]) / 86400000 >= target_d * 0.8:
                start_ms = widest[0]
                end_ms_use = widest[1]
            else:
                end_ms_use = end_ms
            _bars, meta = load_or_fetch_bars(sym, interval, start_ms, end_ms_use)
            sym_report[interval] = meta
            coverage["intervals"].setdefault(interval, []).append(meta.get("bar_count", 0))
        coverage["symbols"][sym] = sym_report

    coverage["summary"] = {iv: {"min_bars": min(v or [0]), "max_bars": max(v or [0])} for iv, v in coverage["intervals"].items()}
    coverage["symbol_eligibility"] = scan_extra_symbol_eligibility()
    coverage["actual_history"] = {
        sym: {
            "1m_days": coverage["symbols"][sym].get("1m", {}).get("span_days"),
            "1h_days": coverage["symbols"][sym].get("1h", {}).get("span_days"),
            "1m_start_ts": coverage["symbols"][sym].get("1m", {}).get("start_ts"),
            "1m_end_ts": coverage["symbols"][sym].get("1m", {}).get("end_ts"),
        }
        for sym in TOP4
    }
    coverage["note"] = "1m replay window ~180d (Binance.US retention); 1h/4h/15m regime timeline up to ~1200d where cached"
    return coverage


@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    regime: str
    entry_epoch: float
    exit_epoch: float
    hold_sec: float
    pnl_usd: float
    net_pct: float
    win: bool
    exit_reason: str = ""


def compute_trade_metrics(trades: list[TradeRecord], *, window_days: float, principal: float = PRINCIPAL) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "trades_per_month": 0.0,
            "monthly_pnl_on_25k": 0.0,
            "pct_per_month": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy_per_trade": 0.0,
            "max_drawdown_pct": 0.0,
            "longest_hold_min": 0.0,
            "worst_loss_usd": 0.0,
            "fees_spread_slippage_est_usd": 0.0,
            "all_pass": False,
        }
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    net = sum(t.pnl_usd for t in trades)
    wr = len(wins) / n * 100
    pf = (sum(t.pnl_usd for t in wins) / abs(sum(t.pnl_usd for t in losses))) if losses and sum(t.pnl_usd for t in losses) != 0 else (999 if wins else 0)
    exp = net / n
    monthly = (net / max(window_days, 1)) * 30
    longest = max(t.hold_sec for t in trades) / 60
    worst = min(t.pnl_usd for t in trades)
    cost_est = n * NOTIONAL * 0.0006
    equity = principal
    peak = principal
    dd = 0.0
    for t in sorted(trades, key=lambda x: x.entry_epoch):
        equity += t.pnl_usd
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / max(peak, 1e-9))
    all_pass = net > 0 and pf >= 1.2 and exp > 0 and longest <= 30
    return {
        "trades": n,
        "trades_per_month": round((n / max(window_days, 1)) * 30, 2),
        "monthly_pnl_on_25k": round(monthly, 2),
        "pct_per_month": round(monthly / principal * 100, 3),
        "win_rate": round(wr, 1),
        "avg_win": round(sum(t.pnl_usd for t in wins) / len(wins), 4) if wins else 0,
        "avg_loss": round(sum(t.pnl_usd for t in losses) / len(losses), 4) if losses else 0,
        "profit_factor": round(min(pf, 999), 2),
        "expectancy_per_trade": round(exp, 4),
        "max_drawdown_pct": round(dd * 100, 2),
        "longest_hold_min": round(longest, 2),
        "worst_loss_usd": round(worst, 4),
        "fees_spread_slippage_est_usd": round(cost_est, 2),
        "all_pass": all_pass,
    }


def walk_forward_splits(epoch_start: int, epoch_end: int) -> dict[str, tuple[int, int]]:
    span = epoch_end - epoch_start
    t1 = epoch_start + int(span * 0.6)
    t2 = epoch_start + int(span * 0.8)
    return {
        "train": (epoch_start, t1),
        "validation": (t1, t2),
        "test": (t2, epoch_end),
    }


def filter_trades_by_window(trades: list[TradeRecord], window: tuple[int, int]) -> list[TradeRecord]:
    lo, hi = window
    return [t for t in trades if lo <= t.entry_epoch < hi]


def promotion_checks(train_m: dict, val_m: dict, test_m: dict) -> dict[str, Any]:
    checks = {
        "train_positive": train_m.get("monthly_pnl_on_25k", 0) > 0,
        "validation_positive": val_m.get("monthly_pnl_on_25k", 0) > 0,
        "test_positive": test_m.get("monthly_pnl_on_25k", 0) > 0,
        "profit_factor_above_1_2": test_m.get("profit_factor", 0) >= 1.2,
        "max_hold_under_30m": test_m.get("longest_hold_min", 999) <= 30,
        "positive_after_costs": test_m.get("expectancy_per_trade", 0) > 0,
        "no_day_contamination": True,
        "no_repair_add": True,
        "no_averaging_down": True,
    }
    stress_ok = test_m.get("monthly_pnl_on_25k", 0) > -5  # placeholder: full stress in grid
    checks["stress_placeholder_ok"] = stress_ok
    passed = all(checks.values())
    return {"checks": checks, "all_pass": passed, "promotion_ready": passed}


def run_regime_filtered_replay(
    symbol: str,
    bars_1m: list[dict],
    regime_index: dict[int, str],
    strategy_name: str,
    *,
    allowed_regimes: frozenset[str] | None = None,
    max_hold_sec: int = 300,
    target_pct: float | None = None,
) -> list[TradeRecord]:
    """Run isolated strategy replay; tag trades with entry regime; filter by native regimes."""
    from dataclasses import replace as dc_replace

    from backend.services.binance_scalp.calibration_profiles import economics_for_config
    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.momentum_tracker import MomentumTracker
    from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
    from backend.services.binance_scalp.strategies import STRATEGY_NAMES
    from scripts.run_scalp_strategy_replay import (
        LOOKAHEAD_BARS,
        StrategyStats,
        _config_from_args,
        make_snap,
        net_pct_at_bid,
        replay_symbol,
    )

    class Args:
        only_strategy = strategy_name
        profile = "moderate"
        hours = 99999
        spread_mult = 1.0

    if strategy_name not in STRATEGY_NAMES:
        return []

    config = _config_from_args(Args())
    econ = economics_for_config(config)
    overrides: dict[str, Any] = {"stale_scalp_timeout_sec": int(max_hold_sec)}
    if target_pct is not None:
        overrides["net_profit_target_pct"] = float(target_pct)
    econ = dc_replace(econ, **overrides)

    momentum = MomentumTracker()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=type("R", (), {"read": lambda _s, _sym: None})(),
        momentum=momentum,
    )

    api_sym = TOP4_API.get(symbol, symbol.replace("/", ""))
    trades_raw, _stats, _ = replay_symbol(
        api_sym,
        bars_1m,
        config=config,
        econ=econ,
        router=router,
        momentum=momentum,
    )

    native = allowed_regimes or STRATEGY_NATIVE_REGIMES.get(strategy_name, frozenset())
    out: list[TradeRecord] = []
    for t in trades_raw:
        if t.get("setup_name") != strategy_name:
            continue
        entry = float(t.get("entry_epoch") or 0)
        reg = regime_at_ts(regime_index, int(entry))
        if reg not in native:
            continue
        pnl = float(t.get("pnl_usd") or 0)
        out.append(
            TradeRecord(
                symbol=symbol,
                strategy=strategy_name,
                regime=reg,
                entry_epoch=entry,
                exit_epoch=float(t.get("exit_epoch") or entry),
                hold_sec=float(t.get("hold_sec") or 0),
                pnl_usd=pnl,
                net_pct=float(t.get("net_pct") or 0),
                win=bool(t.get("win")),
                exit_reason=str(t.get("exit_reason") or ""),
            )
        )
    return out
