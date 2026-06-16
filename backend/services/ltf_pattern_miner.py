"""
Lower-timeframe pattern mining — discovers entries from bar data directly.

Does NOT use the DAY thesis engine for signal generation.
HTF (1h/4h) is permission context only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.config.trading_economics import (
    MIN_NET_PROFIT_TO_SELL,
    ORDERBOOK_HALF_SPREAD_ESTIMATE,
    SLIPPAGE_BUFFER,
    TAKER_FEE,
)

Bar = dict[str, Any]
PatternFn = Callable[[str, list[Bar], dict[str, list[Bar]], int], bool]

MAX_HOLD_SEC = 72 * 3600
MAX_DD_PCT = 8.0


@dataclass(frozen=True)
class Economics:
    taker_fee: float = TAKER_FEE
    half_spread: float = ORDERBOOK_HALF_SPREAD_ESTIMATE
    slippage: float = SLIPPAGE_BUFFER
    profit_floor: float = MIN_NET_PROFIT_TO_SELL
    spread_mult: float = 1.0

    def buy_fill(self, mid: float) -> float:
        ow = self.half_spread * self.spread_mult + self.slippage
        return mid * (1.0 + ow)

    def sell_fill(self, mid: float) -> float:
        ow = self.half_spread * self.spread_mult + self.slippage
        return mid * (1.0 - ow)

    def roundtrip_cost_pct(self) -> float:
        return 2 * self.taker_fee + 2 * self.half_spread * self.spread_mult + 2 * self.slippage


@dataclass
class MinedTrade:
    pattern_id: str
    symbol: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    hold_sec: int
    exit_reason: str
    mae_pct: float
    mfe_pct: float
    regime: str
    notional: float


@dataclass
class PatternSpec:
    pattern_id: str
    timeframe_min: int
    category: str  # day | scalp | regime
    signal_fn: Callable[[str, list[Bar], dict], bool]
    profit_target_pct: float = 0.006
    stop_atr_mult: float = 1.0
    time_stop_hours: float = 48.0
    notional_usd: float = 3750.0
    scalp: bool = False


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


def _atr_pct(bars: list[Bar], period: int = 14) -> float:
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


def _vwap(bars: list[Bar], lookback: int = 20) -> float:
    chunk = bars[-lookback:] if len(bars) >= lookback else bars
    num = sum((b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in chunk)
    den = sum(b["volume"] for b in chunk) or 1e-9
    return num / den


def _bb_width(closes: list[float], period: int = 20) -> tuple[float, float, float, float]:
    if len(closes) < period:
        c = closes[-1]
        return c, c, 0.5, 0.01
    w = closes[-period:]
    m = sum(w) / len(w)
    var = sum((x - m) ** 2 for x in w) / len(w)
    std = math.sqrt(var) if var > 0 else 1e-9
    upper, lower = m + 2 * std, m - 2 * std
    width = (upper - lower) / m if m > 0 else 0.01
    pos = (closes[-1] - lower) / (upper - lower) if upper > lower else 0.5
    return upper, lower, max(0.0, min(1.0, pos)), width


def resample_bars(bars: list[Bar], minutes: int) -> list[Bar]:
    if not bars:
        return []
    bucket_sec = minutes * 60
    out: list[Bar] = []
    cur: list[Bar] = []
    bucket_start = None
    for b in bars:
        ts = int(b["ts"])
        bs = ts - (ts % bucket_sec)
        if bucket_start is None:
            bucket_start = bs
        if bs != bucket_start and cur:
            out.append({
                "ts": bucket_start,
                "open": cur[0]["open"],
                "high": max(x["high"] for x in cur),
                "low": min(x["low"] for x in cur),
                "close": cur[-1]["close"],
                "volume": sum(x["volume"] for x in cur),
            })
            cur = []
            bucket_start = bs
        cur.append(b)
    if cur:
        out.append({
            "ts": bucket_start or int(cur[0]["ts"]),
            "open": cur[0]["open"],
            "high": max(x["high"] for x in cur),
            "low": min(x["low"] for x in cur),
            "close": cur[-1]["close"],
            "volume": sum(x["volume"] for x in cur),
        })
    return out


def bars_up_to(bars: list[Bar], ts: int) -> list[Bar]:
    lo, hi = 0, len(bars) - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid]["ts"] <= ts:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return bars[: idx + 1] if idx >= 0 else []


def htf_context(bars_1h: list[Bar], bars_4h: list[Bar], ts: int) -> dict[str, Any]:
    h1 = bars_up_to(bars_1h, ts)
    h4 = bars_up_to(bars_4h, ts)
    if len(h1) < 30:
        return {"allowed_long": False, "regime": "unknown"}
    closes = [b["close"] for b in h1]
    c = closes[-1]
    e8 = _ema(closes, 8)[-1]
    e21 = _ema(closes, 21)[-1]
    adx_proxy = abs(e8 - e21) / c if c > 0 else 0
    vol_avg = sum(b["volume"] for b in h1[-24:]) / max(len(h1[-24:]), 1)
    rel_vol = h1[-1]["volume"] / max(vol_avg, 1e-9)
    h4_closes = [b["close"] for b in h4] if len(h4) >= 10 else closes
    e21_4h = _ema(h4_closes, 21)[-1] if len(h4_closes) >= 21 else e21
    trending_up = c > e21 and e8 > e21
    trending_down = c < e21 and e8 < e21
    range_bound = adx_proxy < 0.008
    if trending_up:
        regime = "trending_up"
    elif trending_down:
        regime = "trending_down"
    elif range_bound:
        regime = "range"
    else:
        regime = "neutral"
    h4_bull = c > e21_4h
    allowed = h4_bull or regime in ("range", "neutral")
    return {
        "allowed_long": allowed,
        "regime": regime,
        "rel_vol": rel_vol,
        "adx_proxy": adx_proxy,
        "trending_up": trending_up,
        "rsi": _rsi(closes),
        "vwap_1h": _vwap(h1),
    }


def _ltf_slice(bars_by_tf: dict[str, list[Bar]], tf_min: int, ts: int) -> list[Bar]:
    if tf_min == 5:
        return bars_up_to(bars_by_tf.get("5m", []), ts)
    if tf_min == 15:
        b = bars_by_tf.get("15m")
        if b:
            return bars_up_to(b, ts)
        return bars_up_to(resample_bars(bars_by_tf.get("5m", []), 15), ts)
    if tf_min == 30:
        b = bars_by_tf.get("30m")
        if b:
            return bars_up_to(b, ts)
        return bars_up_to(resample_bars(bars_by_tf.get("5m", []), 30), ts)
    if tf_min == 1:
        return bars_up_to(bars_by_tf.get("1m", []), ts)
    return bars_up_to(bars_by_tf.get("5m", []), ts)


def _cross_above_vwap(ltf: list[Bar], lookback: int = 20) -> bool:
    if len(ltf) < lookback + 2:
        return False
    vwap = _vwap(ltf[:-1], lookback)
    prev = ltf[-2]["close"]
    cur = ltf[-1]["close"]
    return prev < vwap and cur > vwap and cur > ltf[-1]["open"]


def _detect_vwap_reclaim(sym: str, ltf: list[Bar], ctx: dict, lookback: int = 20) -> bool:
    if not ctx.get("allowed_long"):
        return False
    return _cross_above_vwap(ltf, lookback)


def _detect_pullback_reclaim(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if not ctx.get("allowed_long") or len(ltf) < 25:
        return False
    closes = [b["close"] for b in ltf]
    e8 = _ema(closes, 8)[-1]
    e21 = _ema(closes, 21)[-1]
    c = closes[-1]
    prev = closes[-2]
    touched = ltf[-2]["low"] <= e21 * 1.002
    reclaim = prev <= e21 and c > e8 > e21
    return touched and reclaim


def _detect_vol_compression_breakout(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if not ctx.get("allowed_long") or len(ltf) < 25:
        return False
    closes = [b["close"] for b in ltf]
    _, _, pos, width = _bb_width(closes)
    widths = []
    for k in range(10, 2, -1):
        if len(closes) >= period + k:
            _, _, _, w = _bb_width(closes[:-k])
            widths.append(w)
    squeeze = widths and width < min(widths) * 1.05
    breakout = pos > 0.85 and ltf[-1]["close"] > ltf[-1]["open"]
    return squeeze and breakout and ctx.get("rel_vol", 1) >= 1.0


def _detect_range_low_reclaim(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if not ctx.get("allowed_long") or len(ltf) < 22:
        return False
    closes = [b["close"] for b in ltf]
    _, lower, pos, _ = _bb_width(closes)
    vwap = _vwap(ltf)
    hammer = ltf[-1]["close"] > ltf[-1]["open"]
    at_low = pos < 0.20 or ltf[-1]["low"] <= lower * 1.002
    return at_low and hammer and ltf[-1]["close"] >= vwap * 0.998


def _detect_failed_breakdown(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 15:
        return False
    lows = [b["low"] for b in ltf[-12:-1]]
    if not lows:
        return False
    support = min(lows)
    broke = ltf[-2]["low"] < support * 0.998
    reclaimed = ltf[-1]["close"] > support and ltf[-1]["close"] > ltf[-2]["close"]
    return broke and reclaimed


def _detect_high_relvol_reversal(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 20 or ctx.get("rel_vol", 0) < 1.4:
        return False
    closes = [b["close"] for b in ltf]
    rsi = _rsi(closes)
    vwap = _vwap(ltf)
    c = closes[-1]
    reversal = c > vwap and rsi < 48 and ltf[-1]["close"] > ltf[-1]["open"]
    return reversal and ctx.get("allowed_long", False)


def _detect_trend_continuation_retest(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if not ctx.get("trending_up") or len(ltf) < 25:
        return False
    closes = [b["close"] for b in ltf]
    e21 = _ema(closes, 21)[-1]
    retest = ltf[-2]["low"] <= e21 * 1.003
    bounce = closes[-1] > e21 and ltf[-1]["close"] > ltf[-1]["open"]
    return retest and bounce


def _detect_momentum_continuation(sym: str, ltf: list[Bar], ctx: dict, min_rel: float = 1.2) -> bool:
    if sym not in ("ETH/USDT", "SOL/USDT") or len(ltf) < 15:
        return False
    if not ctx.get("trending_up") or ctx.get("rel_vol", 0) < min_rel:
        return False
    closes = [b["close"] for b in ltf]
    mom = (closes[-1] - closes[-4]) / closes[-4] if closes[-4] > 0 else 0
    return mom > 0.003 and ltf[-1]["close"] > ltf[-1]["open"]


def _detect_scalp_compression_breakout(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 30:
        return False
    closes = [b["close"] for b in ltf]
    _, _, pos, width = _bb_width(closes)
    prev_widths = []
    for k in range(8, 1, -1):
        if len(closes) >= 20 + k:
            _, _, _, w = _bb_width(closes[:-k])
            prev_widths.append(w)
    if not prev_widths:
        return False
    squeeze = width < min(prev_widths) * 1.08
    return squeeze and pos > 0.80 and ltf[-1]["volume"] > sum(b["volume"] for b in ltf[-6:-1]) / 5


def _detect_scalp_vwap_reclaim(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 15:
        return False
    return _cross_above_vwap(ltf, 15)


def _detect_scalp_range_bounce(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 20:
        return False
    closes = [b["close"] for b in ltf]
    _, lower, pos, _ = _bb_width(closes)
    return pos < 0.15 and ltf[-1]["close"] > lower and ltf[-1]["close"] > ltf[-1]["open"]


def _detect_scalp_volume_imbalance(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 10:
        return False
    b = ltf[-1]
    body = b["close"] - b["open"]
    rng = max(b["high"] - b["low"], 1e-9)
    vol_ratio = b["volume"] / max(sum(x["volume"] for x in ltf[-6:-1]) / 5, 1e-9)
    bullish = body > 0 and body / rng > 0.55 and vol_ratio > 1.5
    return bullish and b["close"] > _vwap(ltf)


def _detect_scalp_failed_breakdown(sym: str, ltf: list[Bar], ctx: dict) -> bool:
    if len(ltf) < 12:
        return False
    return _detect_failed_breakdown(sym, ltf, ctx)


def make_pattern_catalog() -> list[PatternSpec]:
    specs: list[PatternSpec] = []

    def add(pid: str, tf: int, cat: str, fn: Callable, **kw):
        specs.append(PatternSpec(pattern_id=pid, timeframe_min=tf, category=cat, signal_fn=fn, **kw))

    add("vwap_reclaim_15m", 15, "day", _detect_vwap_reclaim, profit_target_pct=0.006)
    add("vwap_reclaim_30m", 30, "day", _detect_vwap_reclaim, profit_target_pct=0.006)
    add("pullback_reclaim_5m", 5, "day", _detect_pullback_reclaim, profit_target_pct=0.006)
    add("pullback_reclaim_15m", 15, "day", _detect_pullback_reclaim, profit_target_pct=0.008)
    add("vol_compression_breakout", 15, "day", _detect_vol_compression_breakout, profit_target_pct=0.010)
    add("range_low_reclaim", 15, "day", _detect_range_low_reclaim, profit_target_pct=0.006)
    add("failed_breakdown_reversal", 15, "day", _detect_failed_breakdown, profit_target_pct=0.008)
    add("high_relvol_reversal", 15, "day", _detect_high_relvol_reversal, profit_target_pct=0.006)
    add("trend_continuation_retest", 15, "day", _detect_trend_continuation_retest, profit_target_pct=0.008)
    add("eth_momentum_continuation", 15, "day", _detect_momentum_continuation, profit_target_pct=0.008)

    for sym_tag in ("BTC", "ETH", "SOL", "XRP"):
        sym = f"{sym_tag}/USDT"

        def sym_vwap(s: str, ltf: list[Bar], ctx: dict, _sym=sym) -> bool:
            return s == _sym and _detect_vwap_reclaim(s, ltf, ctx)

        add(f"{sym_tag.lower()}_vwap_reclaim_15m", 15, "day", sym_vwap, profit_target_pct=0.006)

    # Scalp 1m
    add("scalp_compression_breakout_1m", 1, "scalp", _detect_scalp_compression_breakout,
        profit_target_pct=0.0035, stop_atr_mult=0.75, time_stop_hours=0.5, notional_usd=25.0, scalp=True)
    add("scalp_vwap_reclaim_1m", 1, "scalp", _detect_scalp_vwap_reclaim,
        profit_target_pct=0.0035, stop_atr_mult=0.5, time_stop_hours=0.75, notional_usd=25.0, scalp=True)
    add("scalp_range_bounce_1m", 1, "scalp", _detect_scalp_range_bounce,
        profit_target_pct=0.003, stop_atr_mult=0.5, time_stop_hours=0.5, notional_usd=25.0, scalp=True)
    add("scalp_volume_imbalance_1m", 1, "scalp", _detect_scalp_volume_imbalance,
        profit_target_pct=0.0035, stop_atr_mult=0.5, time_stop_hours=0.5, notional_usd=25.0, scalp=True)
    add("scalp_failed_breakdown_1m", 1, "scalp", _detect_scalp_failed_breakdown,
        profit_target_pct=0.004, stop_atr_mult=0.6, time_stop_hours=0.75, notional_usd=25.0, scalp=True)

    return specs


def simulate_exit(
    entry_ts: int,
    entry_price: float,
    notional: float,
    exec_bars: list[Bar],
    spec: PatternSpec,
    econ: Economics,
) -> tuple[int, float, str, float, float] | None:
    """Returns exit_ts, exit_mid, reason, mae_pct, mfe_pct."""
    atr = 0.01
    idx = 0
    while idx < len(exec_bars) and exec_bars[idx]["ts"] <= entry_ts:
        idx += 1
    if idx >= len(exec_bars):
        return None

    slice_for_atr = exec_bars[max(0, idx - 20): idx]
    if slice_for_atr:
        atr = _atr_pct(slice_for_atr)

    stop_pct = min(0.025, spec.stop_atr_mult * atr)
    target_net = spec.profit_target_pct
    time_stop = int(spec.time_stop_hours * 3600)
    qty = notional / entry_price
    mae = 0.0
    mfe = 0.0

    max_bars = 720 if spec.scalp else len(exec_bars)  # cap scalp exit scan ~12h of 1m bars

    for j in range(idx, min(len(exec_bars), idx + max_bars)):
        bar = exec_bars[j]
        hold = bar["ts"] - entry_ts
        mid = bar["close"]
        low = bar["low"]
        high = bar["high"]
        mae = min(mae, (low - entry_price) / entry_price)
        mfe = max(mfe, (high - entry_price) / entry_price)

        sell_mid = econ.sell_fill(mid)
        net_pct = (sell_mid - entry_price) / entry_price - econ.roundtrip_cost_pct()

        if net_pct >= target_net:
            return bar["ts"], mid, "NET_PROFIT_EXIT", mae, mfe
        if (entry_price - low) / entry_price >= stop_pct:
            return bar["ts"], mid, "VOLATILITY_STOP_EXIT", mae, mfe
        if hold >= time_stop:
            return bar["ts"], mid, "TIME_STOP_EXIT", mae, mfe
        if (low - entry_price) / entry_price <= -0.03:
            return bar["ts"], mid, "EXTREME_PROTECTION_EXIT", mae, mfe
        if hold >= MAX_HOLD_SEC:
            return bar["ts"], mid, "TIME_STOP_EXIT", mae, mfe

    last = exec_bars[-1]
    return last["ts"], last["close"], "REPLAY_END", mae, mfe


def mine_symbol_pattern(
    symbol: str,
    spec: PatternSpec,
    bars_by_tf: dict[str, list[Bar]],
    start_ts: int,
    end_ts: int,
    econ: Economics,
    cooldown_sec: int = 3600,
) -> list[MinedTrade]:
    trades: list[MinedTrade] = []
    ltf_all = _ltf_slice(bars_by_tf, spec.timeframe_min, end_ts)
    if len(ltf_all) < 40:
        return trades

    exec_bars = bars_by_tf.get("1m") if spec.scalp else bars_by_tf.get("5m", bars_by_tf.get("1m", []))
    if not exec_bars:
        return trades

    # Scan timeline: full LTF for day; 5m steps for 1m scalp (pattern checks last 1m window)
    if spec.timeframe_min == 1:
        scan_bars = bars_by_tf.get("5m") or resample_bars(ltf_all, 5)
        step = 3  # 15-minute scan steps for 1m scalp
    else:
        scan_bars = ltf_all
        step = 1

    htf_cache: dict[int, dict] = {}

    def ctx_at(ts: int) -> dict:
        hour = ts - (ts % 3600)
        if hour not in htf_cache:
            h1 = bars_up_to(bars_by_tf["1h"], ts)
            h4 = bars_up_to(bars_by_tf.get("4h", h1), ts)
            htf_cache[hour] = htf_context(h1, h4, ts)
        return htf_cache[hour]

    open_until = 0
    warmup = 40
    for i in range(warmup, len(scan_bars), step):
        bar = scan_bars[i]
        ts = int(bar["ts"])
        if ts < start_ts or ts > end_ts:
            continue
        if ts < open_until:
            continue
        if spec.timeframe_min == 1:
            ltf = _ltf_slice(bars_by_tf, 1, ts)
        else:
            ltf = bars_up_to(ltf_all, ts)
        if len(ltf) < 20:
            continue
        ctx = ctx_at(ts)
        if not spec.signal_fn(symbol, ltf, ctx):
            continue

        entry_mid = bar["close"]
        entry_fill = econ.buy_fill(entry_mid)
        sim = simulate_exit(ts, entry_fill, spec.notional_usd, exec_bars, spec, econ)
        if sim is None:
            continue
        exit_ts, exit_mid, reason, mae, mfe = sim
        exit_fill = econ.sell_fill(exit_mid)
        gross = spec.notional_usd * (exit_fill - entry_fill) / entry_fill
        fees = spec.notional_usd * econ.taker_fee * 2
        pnl = gross - fees
        pnl_pct = pnl / spec.notional_usd
        regime = ctx.get("regime", "unknown")

        trades.append(MinedTrade(
            pattern_id=spec.pattern_id,
            symbol=symbol,
            entry_ts=ts,
            exit_ts=exit_ts,
            entry_price=entry_fill,
            exit_price=exit_fill,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            hold_sec=exit_ts - ts,
            exit_reason=reason,
            mae_pct=mae,
            mfe_pct=mfe,
            regime=regime,
            notional=spec.notional_usd,
        ))
        open_until = exit_ts + cooldown_sec

    return trades


def aggregate_metrics(trades: list[MinedTrade], days: float, principal: float = 25000.0) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "trades_per_month": 0.0,
            "net_pnl_usd": 0.0,
            "monthly_pnl_usd": 0.0,
            "pct_per_month": 0.0,
            "expectancy_per_trade": 0.0,
            "win_rate_pct": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "longest_hold_hours": 0.0,
            "worst_mae_pct": 0.0,
            "avg_mfe_pct": 0.0,
        }
    n = len(trades)
    net = sum(t.pnl_usd for t in trades)
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    aw = sum(t.pnl_usd for t in wins) / max(len(wins), 1)
    al = sum(t.pnl_usd for t in losses) / max(len(losses), 1)
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 1e-9 else (999.0 if gw > 0 else 0.0)
    months = max(days / 30.0, 1e-9)
    monthly = net / months
    eq = peak = max_dd = 0.0
    for t in sorted(trades, key=lambda x: x.exit_ts):
        eq += t.pnl_usd
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    max_dd_pct = 100.0 * max_dd / principal
    return {
        "trades": n,
        "trades_per_month": round(n / months, 2),
        "net_pnl_usd": round(net, 2),
        "monthly_pnl_usd": round(monthly, 2),
        "pct_per_month": round(100.0 * monthly / principal, 4),
        "expectancy_per_trade": round(net / n, 4),
        "win_rate_pct": round(100.0 * len(wins) / n, 2),
        "avg_win_usd": round(aw, 2),
        "avg_loss_usd": round(al, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd_pct, 3),
        "longest_hold_hours": round(max(t.hold_sec for t in trades) / 3600.0, 2),
        "worst_mae_pct": round(min(t.mae_pct for t in trades) * 100, 4),
        "avg_mfe_pct": round(sum(t.mfe_pct for t in trades) / n * 100, 4),
    }


def walk_forward_split(
    trades: list[MinedTrade],
    split_ts: int,
) -> dict[str, Any]:
    train = [t for t in trades if t.entry_ts < split_ts]
    test = [t for t in trades if t.entry_ts >= split_ts]
    train_days = max((split_ts - (train[0].entry_ts if train else split_ts)) / 86400, 1)
    test_days = max((test[-1].entry_ts if test else split_ts) - split_ts, 1) / 86400 if test else 1
    tm = aggregate_metrics(train, train_days)
    te = aggregate_metrics(test, test_days)
    return {
        "train": tm,
        "test": te,
        "test_positive": te.get("expectancy_per_trade", 0) > 0 and te.get("net_pnl_usd", 0) > 0,
    }


def reject_candidate(
    metrics: dict[str, Any],
    wf: dict[str, Any],
    *,
    spread_ok: bool = True,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.get("net_pnl_usd", 0) <= 0:
        reasons.append("negative_net_pnl")
    if metrics.get("expectancy_per_trade", 0) <= 0:
        reasons.append("negative_expectancy")
    if not wf.get("test_positive"):
        reasons.append("walk_forward_test_fail")
    if metrics.get("longest_hold_hours", 0) > 72:
        reasons.append("fat_tail_hold")
    if metrics.get("max_drawdown_pct", 99) > MAX_DD_PCT:
        reasons.append("max_drawdown")
    if not spread_ok:
        reasons.append("spread_stress_fail")
    if metrics.get("trades", 0) < 3:
        reasons.append("insufficient_trades")
    return len(reasons) == 0, reasons


def regime_bucket_report(trades: list[MinedTrade], days: float) -> dict[str, Any]:
    buckets: dict[str, list[MinedTrade]] = {}
    for t in trades:
        buckets.setdefault(t.regime, []).append(t)
    return {k: aggregate_metrics(v, days) for k, v in buckets.items()}
