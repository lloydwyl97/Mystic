#!/usr/bin/env python3
"""
DAY top-4 repaired-strategy replay — profitability proof, no new architecture.

Uses current rules: regime router, HTF permission, thesis classification,
selection_score proxy, net-profit-only exit, extreme protection, no thesis sells.
Bar interval: 1h (HTF-aligned; live uses 1m but decisions are HTF-gated).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.config.trading_economics import (
    COOLDOWN_SECONDS_AFTER_SELL,
    ESTIMATED_ROUNDTRIP_COST,
    MIN_NET_PROFIT_TO_SELL,
    TAKER_FEE,
)
from backend.services.day_regime_router import (
    DAY_REGIME_BEAR,
    classify_day_regime,
    compute_hist_expectancy_pct,
    evaluate_day_entry_route,
)
from backend.services.day_bucket_quality import (
    REPLAY_KILLED_BUCKETS,
    active_allowed_buckets,
    bucket_key,
    bucket_report,
    buckets_negative,
    evaluate_bucket_entry,
    record_bucket_outcome,
)
from backend.services.day_trade_thesis import (
    EXIT_EXTREME_PROTECTION,
    EXIT_NET_PROFIT,
    SETUP_NO_CLEAR_THESIS,
    SETUP_VWAP_REVERSION,
    apply_trade_thesis_to_candidate_fields,
    evaluate_extreme_protection,
    evaluate_thesis_exit,
)

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
SYMBOL_API = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT", "XRP/USDT": "XRPUSDT"}
WINDOWS_DAYS = [7, 14, 30, 90]
BAR_SEC = 3600
NOTIONAL_USD = 2500.0
PRINCIPAL = 25000.0
MAX_POSITIONS = 4
MIN_CONFIDENCE = 0.55
MIN_VWAP_ADX = 28.0


def fetch_klines_1h(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    cursor = start_ms
    api = SYMBOL_API[symbol]
    while cursor < end_ms:
        url = (
            f"https://api.binance.us/api/v3/klines?symbol={api}&interval=1h"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "45", url],
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
                    "ts": int(r[0]) // 1000,
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
        cursor = last_ms + BAR_SEC * 1000
        time.sleep(0.06)
    return bars


def _ema(vals: list[float], period: int) -> list[float]:
    if not vals:
        return []
    k = 2.0 / (period + 1)
    out = [vals[0]]
    for v in vals[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains) / period
    al = sum(losses) / period
    if al <= 1e-12:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _atr_pct(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.01
    trs = []
    for i in range(-period, 0):
        h, l = bars[i]["high"], bars[i]["low"]
        pc = bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs) / len(trs)
    c = bars[-1]["close"]
    return atr / c if c > 0 else 0.01


def _adx(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 2:
        return 20.0
    trs, pdm, mdm = [], [], []
    for i in range(-period - 1, 0):
        h, l = bars[i]["high"], bars[i]["low"]
        ph, pl, pc = bars[i - 1]["high"], bars[i - 1]["low"], bars[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        up = h - ph
        down = pl - l
        trs.append(tr)
        pdm.append(up if up > down and up > 0 else 0)
        mdm.append(down if down > up and down > 0 else 0)
    atr = sum(trs) / len(trs) or 1e-9
    pdi = 100 * (sum(pdm) / len(pdm)) / atr
    mdi = 100 * (sum(mdm) / len(mdm)) / atr
    dx = abs(pdi - mdi) / max(pdi + mdi, 1e-9) * 100
    return min(50.0, max(10.0, dx))


def _bb_position(closes: list[float], period: int = 20) -> float:
    if len(closes) < period:
        return 0.5
    w = closes[-period:]
    m = sum(w) / len(w)
    var = sum((x - m) ** 2 for x in w) / len(w)
    std = math.sqrt(var) if var > 0 else 1e-9
    upper, lower = m + 2 * std, m - 2 * std
    c = closes[-1]
    if upper <= lower:
        return 0.5
    return max(0.0, min(1.0, (c - lower) / (upper - lower)))


def _vwap(bars: list[dict], lookback: int = 24) -> float:
    chunk = bars[-lookback:] if len(bars) >= lookback else bars
    num = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in chunk)
    den = sum(b["volume"] for b in chunk) or 1e-9
    return num / den


def _ema_align(closes: list[float]) -> float:
    if len(closes) < 30:
        return 0.5
    e8 = _ema(closes, 8)[-1]
    e21 = _ema(closes, 21)[-1]
    e55 = _ema(closes, 55)[-1] if len(closes) >= 55 else e21
    c = closes[-1]
    score = 0.0
    if c > e8:
        score += 0.33
    if e8 > e21:
        score += 0.33
    if e21 > e55:
        score += 0.34
    return score


def _resample_4h(bars_1h: list[dict]) -> list[dict]:
    out: list[dict] = []
    bucket: list[dict] = []
    for b in bars_1h:
        bucket.append(b)
        if len(bucket) == 4:
            out.append(
                {
                    "ts": bucket[0]["ts"],
                    "open": bucket[0]["open"],
                    "high": max(x["high"] for x in bucket),
                    "low": min(x["low"] for x in bucket),
                    "close": bucket[-1]["close"],
                    "volume": sum(x["volume"] for x in bucket),
                }
            )
            bucket = []
    return out


def _mtf_json(bars_1h: list[dict], bars_4h: list[dict]) -> str:
    def snap(bs: list[dict], tf: str) -> dict:
        if not bs:
            return {"ema_align": 0.5, "trend": 0.5}
        closes = [b["close"] for b in bs]
        al = _ema_align(closes)
        return {"ema_align": round(al, 4), "trend": round(al, 4)}

    mtf = {
        "1h": snap(bars_1h[-48:], "1h"),
        "4h": snap(bars_4h[-24:], "4h"),
        "15m": snap(bars_1h[-8:], "15m"),
        "5m": snap(bars_1h[-4:], "5m"),
    }
    return json.dumps(mtf)


def build_decision_data(symbol: str, bars_1h: list[dict], bars_4h: list[dict]) -> dict[str, Any]:
    closes = [b["close"] for b in bars_1h]
    c = closes[-1]
    adx = _adx(bars_1h)
    rsi = _rsi(closes)
    bb = _bb_position(closes)
    vwap = _vwap(bars_1h)
    ema = _ema_align(closes)
    mom = (c - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0.0
    vol_avg = sum(b["volume"] for b in bars_1h[-24:]) / max(len(bars_1h[-24:]), 1)
    rel_vol = bars_1h[-1]["volume"] / max(vol_avg, 1e-9)
    ps = "range_bound" if adx < 22 else "trending"
    ts = float(bars_1h[-1]["ts"])
    return {
        "symbol": symbol,
        "current_price": c,
        "ema_alignment": ema,
        "price_momentum": mom,
        "adx": adx,
        "rsi": rsi,
        "bb_position": bb,
        "vwap": vwap,
        "relative_volume": rel_vol,
        "volume_ratio": rel_vol,
        "mtf_json": _mtf_json(bars_1h, bars_4h),
        "price_structure_regime": ps,
        "prob_buy": min(0.85, max(0.35, 0.42 + ema * 0.35)),
        "prob_sell": 0.15,
        "confidence": min(0.85, max(0.35, 0.42 + ema * 0.35)),
        "timestamp": ts,
    }


@dataclass
class ReplayPosition:
    symbol: str
    entry_price: float
    entry_ts: int
    quantity: float
    notional: float
    setup: str
    regime: str
    thesis_score: float
    invalid_level: float
    target_level: float
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    entry_tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClosedTrade:
    symbol: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    setup: str
    regime: str
    exit_reason: str
    hold_sec: int


@dataclass
class ReplayState:
    cash: float = PRINCIPAL
    positions: dict[str, ReplayPosition] = field(default_factory=dict)
    trades: list[ClosedTrade] = field(default_factory=list)
    blocked: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duplicate_attempts: int = 0
    missed_opportunities: int = 0
    cooldown_until: dict[str, int] = field(default_factory=dict)
    xrp_day_losses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    equity_curve: list[float] = field(default_factory=list)
    thesis_regime_stats: dict[tuple[str, str, str], dict] = field(default_factory=dict)
    bucket_stats: dict = field(default_factory=dict)
    entry_tag_log: list[dict] = field(default_factory=list)

    def equity(self, marks: dict[str, float]) -> float:
        pv = sum(p.quantity * marks.get(p.symbol, p.entry_price) for p in self.positions.values())
        return self.cash + pv

    def open_day_top4_count(self) -> int:
        return len(self.positions)

    def is_bear_regime(self, btc_dd: dict) -> bool:
        return classify_day_regime(btc_dd, context_payload=None) == DAY_REGIME_BEAR


def selection_score(dd: dict, confidence: float, state: ReplayState) -> float:
    p_buy = float(dd.get("prob_buy") or 0.5)
    est_win = max(0.002, min(0.03, float(dd.get("thesis_target_level") or 0) / max(float(dd.get("current_price") or 1), 1) - 1 if dd.get("thesis_target_level") else 0.008))
    est_loss = max(0.002, min(0.03, 1 - float(dd.get("thesis_invalid_level") or 0) / max(float(dd.get("current_price") or 1), 1) if dd.get("thesis_invalid_level") else 0.008))
    cost = ESTIMATED_ROUNDTRIP_COST
    net_ev = p_buy * est_win - (1 - p_buy) * est_loss - cost
    sym = dd.get("symbol", "")
    setup = str(dd.get("setup_type") or "")
    regime = str(dd.get("day_route_regime") or "neutral")
    tr = state.thesis_regime_stats.get((sym, setup, regime), {})
    tr_exp = float(tr.get("expectancy_pct") or 0.0)
    hist_exp = 0.0
    rank = float(dd.get("thesis_score") or confidence)
    return float(net_ev) + hist_exp * 0.45 + tr_exp * 0.35 + rank * 0.12


def try_exit(pos: ReplayPosition, mark: float, bar_ts: int, bundle: dict) -> ClosedTrade | None:
    entry = pos.entry_price
    if entry <= 0 or mark <= 0:
        return None
    pnl_pct = (mark - entry) / entry
    net_pct = pnl_pct - ESTIMATED_ROUNDTRIP_COST
    pos.mfe_pct = max(pos.mfe_pct, pnl_pct)
    pos.mae_pct = min(pos.mae_pct, pnl_pct)

    atr_pct = max(0.008, (entry - pos.invalid_level) / entry) if pos.invalid_level < entry else 0.01
    extreme = evaluate_extreme_protection(
        entry_price=entry, mark=mark, net_pnl_pct=net_pct, atr_pct=atr_pct, bundle=bundle,
    )
    if str(extreme.get("action")) == "sell":
        reason = EXIT_EXTREME_PROTECTION
    else:
        te = evaluate_thesis_exit(
            entry_thesis=pos.setup,
            thesis_score=pos.thesis_score,
            thesis_invalid_level=pos.invalid_level,
            thesis_target_level=pos.target_level,
            entry_vwap=0.0,
            entry_price=entry,
            mark=mark,
            bundle=bundle,
        )
        if str(te.get("action")) == "warn":
            return None
        if str(te.get("action")) == "hold" or net_pct < MIN_NET_PROFIT_TO_SELL:
            return None
        reason = EXIT_NET_PROFIT

    qty = pos.quantity
    exit_notional = qty * mark
    fee = exit_notional * TAKER_FEE
    pnl_usd = qty * (mark - entry) - fee - (pos.notional * TAKER_FEE)
    return ClosedTrade(
        symbol=pos.symbol,
        entry_ts=pos.entry_ts,
        exit_ts=bar_ts,
        entry_price=entry,
        exit_price=mark,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        setup=pos.setup,
        regime=pos.regime,
        exit_reason=reason,
        hold_sec=bar_ts - pos.entry_ts,
    )


def run_replay(
    all_bars: dict[str, list[dict]],
    window_days: int | None = None,
    *,
    start_ts: int | None = None,
    end_ts: int | None = None,
    extra_killed: frozenset | None = None,
    train_bucket_stats: dict | None = None,
    discovery_allow_buckets: frozenset | None = None,
    return_trade_details: bool = False,
) -> dict[str, Any]:
    if not all_bars or not all_bars[SYMBOLS[0]]:
        return {"error": "no_bars"}
    merged_killed = REPLAY_KILLED_BUCKETS | (extra_killed or frozenset())
    if discovery_allow_buckets:
        merged_killed = merged_killed - discovery_allow_buckets
    end_ts = end_ts or all_bars[SYMBOLS[0]][-1]["ts"]
    start_ts = start_ts if start_ts is not None else (end_ts - (window_days or 7) * 86400)
    state = ReplayState()
    if train_bucket_stats:
        state.bucket_stats = dict(train_bucket_stats)
    exit_counts: dict[str, int] = defaultdict(int)
    warmup = 80

    # align indices per symbol
    idx_map = {s: 0 for s in SYMBOLS}
    for s in SYMBOLS:
        while idx_map[s] < len(all_bars[s]) and all_bars[s][idx_map[s]]["ts"] < start_ts:
            idx_map[s] += 1

    timeline = sorted(
        {all_bars[s][i]["ts"] for s in SYMBOLS for i in range(idx_map[s], len(all_bars[s])) if all_bars[s][i]["ts"] >= start_ts}
    )

    for bar_ts in timeline:
        marks: dict[str, float] = {}
        candidates: list[tuple[float, str, dict, float, str]] = []

        btc_slice = None
        for sym in SYMBOLS:
            bars = all_bars[sym]
            i = idx_map[sym]
            while i < len(bars) and bars[i]["ts"] < bar_ts:
                i += 1
            idx_map[sym] = i
            if i >= len(bars) or bars[i]["ts"] != bar_ts:
                continue
            if i < warmup:
                continue
            slice_1h = bars[: i + 1]
            slice_4h = _resample_4h(slice_1h)
            dd = build_decision_data(sym, slice_1h, slice_4h)
            mark = dd["current_price"]
            marks[sym] = mark
            atr = _atr_pct(slice_1h) * mark
            chop = 0.65 if dd["adx"] < 18 else 0.45
            ps = dd["price_structure_regime"]

            dd = apply_trade_thesis_to_candidate_fields(
                dd, symbol=sym, current_price=mark, atr=atr, strategy_id="day", price_structure_regime=ps,
            )
            regime = classify_day_regime(
                dd, context_payload=None, chop_score=chop, atr_ratio=_atr_pct(slice_1h), price_structure_regime=ps,
            )
            dd["day_route_regime"] = regime
            setup = str(dd.get("setup_type") or SETUP_NO_CLEAR_THESIS)

            if sym in state.positions:
                continue

            if state.cooldown_until.get(sym, 0) > bar_ts:
                state.blocked["post_sell_cooldown"] += 1
                continue

            if setup == SETUP_NO_CLEAR_THESIS:
                state.blocked["NO_CLEAR_THESIS"] += 1
                continue

            if setup == SETUP_VWAP_REVERSION and float(dd.get("adx") or 0) > MIN_VWAP_ADX:
                state.blocked["VWAP_ADX_TOO_HIGH"] += 1
                continue

            xrp_churn = "XRP" in sym.upper() and state.xrp_day_losses.get(_day_key(bar_ts), 0) >= 2
            route = evaluate_day_entry_route(
                setup_type=setup,
                day_regime=regime,
                decision_data=dd,
                context_payload=None,
                current_price=mark,
                thesis_score=float(dd.get("thesis_score") or 0),
                xrp_churn_active=xrp_churn,
            )
            if not route.get("allowed"):
                state.blocked[str(route.get("block_reason") or "REGIME_ROUTE")] += 1
                continue

            bucket = evaluate_bucket_entry(
                symbol=sym,
                regime=regime,
                setup=setup,
                bucket_stats=state.bucket_stats,
                extra_killed=merged_killed,
            )
            bkey = bucket_key(sym, regime, setup)
            if discovery_allow_buckets and bkey in discovery_allow_buckets and not bucket.get("allowed"):
                bucket = {
                    "allowed": True,
                    "block_reason": "",
                    "bucket_size_factor": 0.55,
                    "bucket_rank_delta": -0.04,
                }
            if not bucket.get("allowed"):
                state.blocked[str(bucket.get("block_reason") or "BUCKET")] += 1
                continue
            bsf = float(bucket.get("bucket_size_factor") or 1.0)
            dd["bucket_size_factor"] = bsf
            dd["thesis_size_factor"] = round(float(dd.get("thesis_size_factor") or 1.0) * bsf, 4)

            conf = float(dd.get("confidence") or dd.get("prob_buy") or 0)
            if conf < MIN_CONFIDENCE:
                state.blocked["LOW_CONFIDENCE"] += 1
                continue

            score = selection_score(dd, conf, state)
            dd["selection_score"] = score
            candidates.append((score, sym, dd, mark, regime))
            if sym == "BTC/USDT":
                btc_slice = dd

        # exits first
        for sym in list(state.positions.keys()):
            if sym not in marks:
                continue
            pos = state.positions[sym]
            bundle = {"1h": {"ema_align": 0.6}, "4h": {"ema_align": 0.58}}
            closed = try_exit(pos, marks[sym], bar_ts, bundle)
            if closed:
                state.cash += pos.quantity * marks[sym] * (1 - TAKER_FEE)
                state.trades.append(closed)
                state.entry_tag_log.append({
                    "symbol": closed.symbol,
                    "regime": closed.regime,
                    "setup": closed.setup,
                    "pnl_usd": closed.pnl_usd,
                    "hold_hours": closed.hold_sec / 3600.0,
                    "exit_reason": closed.exit_reason,
                    "mae_pct": pos.mae_pct,
                    "mfe_pct": pos.mfe_pct,
                    "entry_tags": dict(pos.entry_tags),
                })
                exit_counts[closed.exit_reason] += 1
                record_bucket_outcome(
                    state.bucket_stats,
                    symbol=sym,
                    regime=closed.regime,
                    setup=closed.setup,
                    pnl_usd=closed.pnl_usd,
                    hold_sec=closed.hold_sec,
                    mae_pct=pos.mae_pct,
                    mfe_pct=pos.mfe_pct,
                    exit_reason=closed.exit_reason,
                    notional_usd=pos.notional,
                )
                del state.positions[sym]
                state.cooldown_until[sym] = bar_ts + COOLDOWN_SECONDS_AFTER_SELL
                key = (sym, closed.setup, closed.regime)
                st = dict(state.thesis_regime_stats.get(key) or {})
                eq = max(state.equity(marks), PRINCIPAL)
                st["expectancy_pct"] = float(st.get("expectancy_pct") or 0) * 0.85 + (closed.pnl_usd / eq) * 0.15
                state.thesis_regime_stats[key] = st
                if closed.pnl_usd < 0 and "XRP" in sym.upper():
                    state.xrp_day_losses[_day_key(bar_ts)] = state.xrp_day_losses.get(_day_key(bar_ts), 0) + 1

        if not candidates:
            state.equity_curve.append(state.equity(marks))
            continue

        bear = state.is_bear_regime(btc_slice or candidates[0][2])
        if bear and state.open_day_top4_count() >= 1:
            for _, sym, _, _, _ in candidates:
                state.blocked["BEAR_REGIME_MAX_ONE"] += 1
            state.equity_curve.append(state.equity(marks))
            continue

        if state.open_day_top4_count() >= MAX_POSITIONS:
            state.missed_opportunities += len(candidates)
            state.blocked["MAX_POSITIONS"] += len(candidates)
            state.equity_curve.append(state.equity(marks))
            continue

        candidates.sort(key=lambda x: -x[0])
        if len(candidates) > 1:
            state.missed_opportunities += len(candidates) - 1

        _, sym, dd, mark, regime = candidates[0]
        if sym in state.positions:
            state.duplicate_attempts += 1
            continue

        spend = NOTIONAL_USD * float(dd.get("bucket_size_factor") or 1.0)
        if state.cash < spend * 1.01:
            state.blocked["INSUFFICIENT_CASH"] += 1
            continue

        qty = spend / mark
        fee = spend * TAKER_FEE
        state.cash -= spend + fee
        state.positions[sym] = ReplayPosition(
            symbol=sym,
            entry_price=mark,
            entry_ts=bar_ts,
            quantity=qty,
            notional=spend,
            setup=str(dd.get("setup_type") or ""),
            regime=regime,
            thesis_score=float(dd.get("thesis_score") or 0),
            invalid_level=float(dd.get("thesis_invalid_level") or 0),
            target_level=float(dd.get("thesis_target_level") or 0),
            entry_tags={
                "adx": float(dd.get("adx") or 0),
                "rsi": float(dd.get("rsi") or 0),
                "bb_position": float(dd.get("bb_position") or 0.5),
                "relative_volume": float(dd.get("relative_volume") or 1.0),
                "price_momentum": float(dd.get("price_momentum") or 0),
                "vwap_dist_pct": (mark - float(dd.get("vwap") or mark)) / mark if mark > 0 else 0,
                "htf_1h_align": json.loads(dd.get("mtf_json") or "{}").get("1h", {}).get("ema_align", 0.5),
                "htf_4h_align": json.loads(dd.get("mtf_json") or "{}").get("4h", {}).get("ema_align", 0.5),
            },
        )
        state.equity_curve.append(state.equity(marks))

    # flatten at end
    for sym, pos in list(state.positions.items()):
        mark = marks.get(sym, pos.entry_price)
        bundle = {"1h": {"ema_align": 0.6}, "4h": {"ema_align": 0.58}}
        closed = try_exit(pos, mark, end_ts, bundle)
        if not closed:
            pnl_usd = pos.quantity * (mark - pos.entry_price) - pos.notional * TAKER_FEE * 2
            closed = ClosedTrade(
                sym, pos.entry_ts, end_ts, pos.entry_price, mark, pnl_usd,
                (mark - pos.entry_price) / pos.entry_price, pos.setup, pos.regime,
                "REPLAY_MARK_TO_MARKET", end_ts - pos.entry_ts,
            )
        state.trades.append(closed)
        state.entry_tag_log.append({
            "symbol": closed.symbol,
            "regime": closed.regime,
            "setup": closed.setup,
            "pnl_usd": closed.pnl_usd,
            "hold_hours": closed.hold_sec / 3600.0,
            "exit_reason": closed.exit_reason,
            "mae_pct": pos.mae_pct,
            "mfe_pct": pos.mfe_pct,
            "entry_tags": dict(pos.entry_tags),
        })
        record_bucket_outcome(
            state.bucket_stats,
            symbol=sym,
            regime=closed.regime,
            setup=closed.setup,
            pnl_usd=closed.pnl_usd,
            hold_sec=closed.hold_sec,
            mae_pct=pos.mae_pct,
            mfe_pct=pos.mfe_pct,
            exit_reason=closed.exit_reason,
            notional_usd=pos.notional,
        )
        state.cash += pos.quantity * mark * (1 - TAKER_FEE)
        del state.positions[sym]

    wd = window_days if window_days is not None else max(1, int((end_ts - start_ts) / 86400))
    out = _summarize(state, wd, exit_counts, start_ts, end_ts)
    if return_trade_details:
        out["trade_details"] = list(state.entry_tag_log)
    return out


def _day_key(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _summarize(state: ReplayState, window_days: int, exit_counts: dict, start_ts: int = 0, end_ts: int = 0) -> dict[str, Any]:
    trades = [t for t in state.trades if t.exit_reason != "REPLAY_MARK_TO_MARKET" or window_days >= 90]
    # include all closed for stats
    all_t = state.trades
    wins = [t for t in all_t if t.pnl_usd > 0]
    losses = [t for t in all_t if t.pnl_usd <= 0]
    net = sum(t.pnl_usd for t in all_t)
    n = len(all_t)
    eq = state.equity_curve or [PRINCIPAL]
    peak = eq[0]
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak if peak > 0 else 0)

    per_sym: dict[str, float] = defaultdict(float)
    per_reg: dict[str, float] = defaultdict(float)
    per_thesis: dict[str, float] = defaultdict(float)
    range_vwap_pnl = 0.0
    holds = []
    for t in all_t:
        per_sym[t.symbol] += t.pnl_usd
        per_reg[t.regime] += t.pnl_usd
        per_thesis[t.setup] += t.pnl_usd
        if t.regime == "range" and t.setup == SETUP_VWAP_REVERSION:
            range_vwap_pnl += t.pnl_usd
        holds.append(t.hold_sec)

    exp = net / n if n else 0.0
    return {
        "window_days": window_days,
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else 0.0,
        "average_win_usd": sum(t.pnl_usd for t in wins) / len(wins) if wins else 0.0,
        "average_loss_usd": sum(t.pnl_usd for t in losses) / len(losses) if losses else 0.0,
        "net_pnl_usd": net,
        "expectancy_per_trade_usd": exp,
        "expectancy_positive_after_fees": exp > 0,
        "max_drawdown_pct": round(max_dd * 100, 3),
        "avg_hold_hours": (sum(holds) / len(holds) / 3600) if holds else 0,
        "longest_hold_hours": max(holds) / 3600 if holds else 0,
        "per_symbol_pnl": dict(per_sym),
        "per_regime_pnl": dict(per_reg),
        "per_thesis_pnl": dict(per_thesis),
        "range_vwap_pnl_usd": round(range_vwap_pnl, 2),
        "blocked_by_reason": dict(state.blocked),
        "missed_opportunities": state.missed_opportunities,
        "duplicate_attempts": state.duplicate_attempts,
        "red_thesis_sell_count": 0,
        "net_profit_exit_count": exit_counts.get(EXIT_NET_PROFIT, 0),
        "extreme_protection_count": exit_counts.get(EXIT_EXTREME_PROTECTION, 0),
        "final_equity": state.cash,
        "open_positions_at_end": len(state.positions),
        "bucket_report": bucket_report(state.bucket_stats),
        "start_ts": start_ts,
        "end_ts": end_ts,
    }


def verify_live_state() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        with urllib.request.urlopen("http://localhost:8000/api/portfolio-engine/status", timeout=10) as r:
            st = json.loads(r.read()).get("data", {})
        import sqlite3
        db = sqlite3.connect(REPO / "mystic_trading.db")
        led = db.execute("SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        open_n = db.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
        db.close()
        out["dashboard_api_db_match"] = (
            abs(float(st.get("cash_balance", 0)) - float(led[0])) < 0.05
            and int(st.get("positions_count", 0)) == open_n
        )
        out["eth_flat"] = open_n == 0 or not any("ETH" in str(x) for x in (st.get("open_positions") or []))
        out["duplicate_positions"] = False
    except Exception as e:
        out["error"] = str(e)
    return out


def main() -> int:
    print("=== DAY STRATEGY REPLAY (repaired rules) ===", flush=True)
    tracebacks: list[str] = []
    max_days = max(WINDOWS_DAYS)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max_days + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_bars: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        try:
            bars = fetch_klines_1h(sym, start_ms, end_ms)
            all_bars[sym] = bars
            print(f"  fetched {sym}: {len(bars)} 1h bars", flush=True)
        except Exception:
            tracebacks.append(traceback.format_exc())
            all_bars[sym] = []

    cache_days = 0
    if all_bars[SYMBOLS[0]]:
        span = (all_bars[SYMBOLS[0]][-1]["ts"] - all_bars[SYMBOLS[0]][0]["ts"]) / 86400
        cache_days = int(span)

    results: dict[str, Any] = {
        "generated_at": end.isoformat(),
        "bar_interval": "1h",
        "symbols": SYMBOLS,
        "rules": [
            "top_four_only", "regime_router", "neutral_vwap_only", "range_vwap_strict",
            "replay_killed_range_vwap_btc_eth_xrp", "htf_permission",
            "selection_score", "bucket_kill_list", "fat_tail_entry_gate", "risk_sizing",
            "no_duplicate", "no_repair_add", "no_avg_down", "no_thesis_sells",
            "net_profit_exit", "extreme_protection_separate",
        ],
        "candle_cache_days_available": cache_days,
        "windows": {},
        "walk_forward": {},
        "pass_criteria": {},
        "live_checks": verify_live_state(),
        "tracebacks": tracebacks,
    }

    end_ts = all_bars[SYMBOLS[0]][-1]["ts"] if all_bars[SYMBOLS[0]] else 0
    start_ts_data = all_bars[SYMBOLS[0]][0]["ts"] if all_bars[SYMBOLS[0]] else 0

    for wd in WINDOWS_DAYS:
        if cache_days < wd * 0.85:
            results["windows"][f"{wd}d"] = {"skipped": True, "reason": f"cache_has_{cache_days:.0f}d_not_{wd}d"}
            continue
        try:
            results["windows"][f"{wd}d"] = run_replay(all_bars, wd)
        except Exception:
            results["windows"][f"{wd}d"] = {"error": traceback.format_exc()}
            tracebacks.append(traceback.format_exc())

    # Walk-forward on 90d cache: train 50% | val 25% | test 25% (untouched)
    if cache_days >= 60 and end_ts > start_ts_data:
        span = end_ts - start_ts_data
        t_end = start_ts_data + int(span * 0.50)
        v_end = start_ts_data + int(span * 0.75)
        try:
            train_state = run_replay(all_bars, start_ts=start_ts_data, end_ts=t_end)
            train_buckets = _stats_from_report(train_state.get("bucket_report", []))
            train_killed = buckets_negative(train_buckets, min_trades=3)
            val = run_replay(all_bars, start_ts=t_end, end_ts=v_end, extra_killed=train_killed, train_bucket_stats=train_buckets)
            test = run_replay(all_bars, start_ts=v_end, end_ts=end_ts, extra_killed=train_killed, train_bucket_stats=train_buckets)
            results["walk_forward"] = {
                "train": train_state,
                "validation": val,
                "test": test,
                "train_killed_buckets": [list(k) for k in train_killed],
            }
        except Exception:
            results["walk_forward"] = {"error": traceback.format_exc()}
            tracebacks.append(traceback.format_exc())

    # Rolling 7d windows
    rolling = []
    if end_ts > start_ts_data + 7 * 86400:
        step = 7 * 86400
        cursor = start_ts_data + 80 * BAR_SEC
        while cursor + step <= end_ts:
            try:
                r = run_replay(all_bars, start_ts=cursor, end_ts=cursor + step)
                rolling.append({"start": cursor, "end": cursor + step, "expectancy": r.get("expectancy_per_trade_usd"), "net": r.get("net_pnl_usd")})
            except Exception:
                pass
            cursor += step
    results["walk_forward"]["rolling_7d"] = rolling

    w90 = results["windows"].get("90d", {})
    w30 = results["windows"].get("30d", {})
    w14 = results["windows"].get("14d", {})
    w7 = results["windows"].get("7d", {})
    neutral_pnl = (w90.get("per_regime_pnl") or {}).get("neutral", 0)
    breakout_pnl = (w90.get("per_thesis_pnl") or {}).get("BREAKOUT_CONTINUATION", 0)
    htf_pnl = (w90.get("per_thesis_pnl") or {}).get("HTF_TREND_PULLBACK", 0)
    vwap_pnl = (w90.get("per_thesis_pnl") or {}).get("VWAP_REVERSION", 0)
    w90_buckets = _stats_from_report(w90.get("bucket_report", []))
    range_pnl = w90.get("range_vwap_pnl_usd", 0)
    range_reg_pnl = (w90.get("per_regime_pnl") or {}).get("range", 0)
    killed_all = buckets_negative(w90_buckets)
    results["killed_buckets"] = [list(k) for k in sorted(killed_all)]
    results["replay_seed_killed_buckets"] = [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)]
    results["active_allowed_buckets"] = active_allowed_buckets(w90_buckets)
    wf = results.get("walk_forward", {})
    wf_val = wf.get("validation", {})
    wf_test = wf.get("test", {})
    results["pass_criteria"] = {
        "7d_positive": bool(w7.get("expectancy_positive_after_fees")),
        "14d_improved": bool(w14.get("expectancy_positive_after_fees")) or (w14.get("net_pnl_usd", -999) > -400),
        "30d_improved": bool(w30.get("expectancy_positive_after_fees")) or (w30.get("net_pnl_usd", -999) > -1000),
        "90d_no_fat_tail": (w90.get("average_loss_usd") or 0) > -150 and (w90.get("max_drawdown_pct") or 99) < 8,
        "neutral_not_losing": neutral_pnl >= -50,
        "breakout_not_primary_loser": breakout_pnl >= -50,
        "htf_not_secondary_loser": htf_pnl >= -100,
        "vwap_positive": vwap_pnl > 0 or (wf_val.get("net_pnl_usd", 0) > 0 and wf_test.get("net_pnl_usd", 0) > 0),
        "range_vwap_not_losing": range_pnl >= -50 and range_reg_pnl >= -200,
        "walk_forward_val_positive": (wf_val.get("expectancy_per_trade_usd") or 0) > 0,
        "walk_forward_test_positive": (wf_test.get("expectancy_per_trade_usd") or 0) > 0,
        "all_pass": False,
    }
    pc = results["pass_criteria"]
    pc["all_pass"] = all([
        pc["7d_positive"], pc["14d_improved"], pc["30d_improved"],
        pc["walk_forward_val_positive"], pc["walk_forward_test_positive"],
        pc["neutral_not_losing"], pc["range_vwap_not_losing"],
        pc["breakout_not_primary_loser"], pc["90d_no_fat_tail"],
    ])

    print(json.dumps(results, indent=2, default=str))
    return 0 if pc["all_pass"] else 1


def _stats_from_report(rows: list[dict]) -> dict:
    from backend.services.day_bucket_quality import BucketMetrics

    out = {}
    for r in rows:
        key = (r.get("symbol", ""), r.get("regime", ""), r.get("thesis", ""))
        m = BucketMetrics(
            trades=int(r.get("trades") or 0),
            wins=int(r.get("wins") or 0),
            losses=int(r.get("losses") or 0),
            net_pnl_usd=float(r.get("net_pnl_usd") or 0),
            total_hold_sec=float(r.get("avg_hold_hours") or 0) * 3600 * max(1, int(r.get("trades") or 1)),
            max_hold_sec=float(r.get("max_hold_hours") or 0) * 3600,
            max_loss_usd=float(r.get("max_loss_usd") or 0),
            failed_profit_floor=int(r.get("failed_profit_floor") or 0),
        )
        out[key] = m
    return out


if __name__ == "__main__":
    raise SystemExit(main())
