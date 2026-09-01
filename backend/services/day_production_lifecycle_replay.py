"""Production-faithful DAY lifecycle replay.

Reuses evaluate_engine_managed_exit + refresh_trailing_stop + get_coin_profile.
Does not invent a second exit ladder. Does not restore retired opinion gates.
"""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config.execution_cost_model import (
    ARM_B_MIN_PREDICTED_GROSS_PCT,
    DEFAULT_EAE_PCT,
    DEFAULT_EFE_PCT,
    LEGACY_BUY_VETO_FEE_PCT,
    LEGACY_BUY_VETO_SLIP_PCT,
    LEGACY_BUY_VETO_SPREAD_PCT,
    LEGACY_SELL_ROUNDTRIP_PCT,
    expected_exchange_commission_rt_pct,
    expected_slippage_rt_pct,
    expected_spread_pct,
    honest_all_in_rt_pct,
    named_cost_breakdown,
)
from backend.config.trading_economics import (
    COOLDOWN_SECONDS_AFTER_SELL,
    DAY_MAX_DEPLOYED_USD,
    DAY_MAX_OPEN_SLOTS,
    DAY_TARGET_NOTIONAL_PER_SLOT_USD,
)
from backend.services.day_controlled_exits import (
    evaluate_engine_managed_exit,
    refresh_trailing_stop,
)
from backend.services.day_direct_path_ev_authority import post_cost_economics_ev
from backend.services.portfolio_engine import get_coin_profile

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
BAR_DECISION_SEC = 900
HORIZON_PAD_SEC = 14 * 24 * 3600


_OCEAN_EXIT_ENV = {
    "DAY_PATH_AWARE_EXIT": "true",
    "DAY_STALL_EXIT_ENABLED": "false",
    "DAY_GIVEBACK_EXIT_ENABLED": "false",
    "DAY_GIVEBACK_ON_4H_HOLD": "true",
    "DAY_STALL_ON_4H_HOLD": "true",
    "ESTIMATED_ROUNDTRIP_COST": "0.0006",
}


@contextmanager
def production_exit_env() -> Iterator[None]:
    """Ocean DAY exit flags for the duration of a replay only. Restored after."""
    prior = {k: os.environ.get(k) for k in _OCEAN_EXIT_ENV}
    os.environ.update(_OCEAN_EXIT_ENV)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def apply_production_exit_env() -> None:
    """Compatibility no-op. Prefer production_exit_env() so tests stay isolated."""
    return None


@dataclass
class ReplayPos:
    symbol: str
    entry_price: float
    entry_time: float
    quantity: float
    notional: float
    stop_price: float = 0.0
    take_profit_1_price: float = 0.0
    trailing_stop_price: float = 0.0
    trail_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    entry_thesis: str = ""
    thesis_invalid_level: float = 0.0
    thesis_target_level: float = 0.0
    entry_vwap: float = 0.0
    day_route_regime_at_entry: str = ""
    max_hold_min: int = 360
    p_buy: float = 0.0
    setup: str = ""
    unlocked_band: bool = False


@dataclass
class ClosedTrade:
    symbol: str
    entry_epoch: float
    exit_epoch: float
    entry_price: float
    exit_price: float
    notional: float
    gross_pct: float
    commission_pct: float
    spread_pct: float
    slippage_pct: float
    net_pct: float
    net_usd: float
    exit_reason: str
    mfe_pct: float
    mae_pct: float
    p_buy: float
    setup: str
    unlocked_band: bool


def _api(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def parse_epoch(ts: Any) -> int | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        v = float(ts)
        return int(v / 1000.0) if v > 1e12 else int(v)
    try:
        return int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError):
        return None


def load_1m_bars(conn: sqlite3.Connection) -> dict[str, list[tuple[int, float, float, float, float]]]:
    out: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for sym in SYMBOLS:
        rows: list[tuple[Any, ...]] = []
        for name in (f"{sym[:-4]}-USDT", f"{sym[:-4]}/USDT", sym):
            rows = conn.execute(
                "SELECT ts, open, high, low, close FROM feature_ohlcv WHERE interval='1m' AND symbol=? ORDER BY ts ASC",
                (name,),
            ).fetchall()
            if rows:
                break
        bars: list[tuple[int, float, float, float, float]] = []
        for ts, o, h, low, c in rows:
            ep = parse_epoch(ts)
            if ep is None:
                continue
            bars.append((ep, float(o or 0), float(h or 0), float(low or 0), float(c or 0)))
        out[sym] = bars
    return out


def resample_4h(bars_1m: list[tuple[int, float, float, float, float]]) -> list[list[float]]:
    buckets: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for ep, o, h, low, c in bars_1m:
        open_sec = (ep // 14400) * 14400
        buckets[open_sec].append((ep, o, h, low, c))
    out: list[list[float]] = []
    for open_sec in sorted(buckets):
        chunk = buckets[open_sec]
        o = chunk[0][1]
        h = max(x[2] for x in chunk)
        low = min(x[3] for x in chunk)
        c = chunk[-1][4]
        if o > 0 and h > 0 and low > 0 and c > 0:
            out.append([float(open_sec), o, h, low, c, 0.0])
    return out


def fourh_bundle(rows: list[list[float]], now_epoch: float) -> dict[str, Any]:
    cutoff = now_epoch + 1
    kept = [r for r in rows if r[0] <= cutoff]
    return {"4h": kept[-8:]}


def load_inferences(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT symbol, ts_utc, prob_buy, prob_hold, prob_sell FROM ai_inference_log ORDER BY ts_utc ASC").fetchall()
    out: list[dict[str, Any]] = []
    for sym, ts, pb, ph, ps in rows:
        s = _api(sym)
        if s not in SYMBOLS:
            continue
        ep = parse_epoch(ts)
        if ep is None:
            continue
        out.append(
            {
                "symbol": s,
                "epoch": ep,
                "p_buy": float(pb or 0),
                "p_hold": float(ph or 0),
                "p_sell": float(ps or 0),
            }
        )
    return out


def decision_bars(inferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latest inference per symbol on each 15m production bar."""
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for inf in inferences:
        key = (inf["epoch"] // BAR_DECISION_SEC * BAR_DECISION_SEC, inf["symbol"])
        prev = latest.get(key)
        if prev is None or inf["epoch"] >= prev["epoch"]:
            latest[key] = inf
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (bar, _s), inf in latest.items():
        grouped[bar].append(inf)
    events: list[dict[str, Any]] = []
    for bar in sorted(grouped):
        events.append({"bar_epoch": bar, "inferences": grouped[bar]})
    return events


def _bar_index(bars: list[tuple[int, float, float, float, float]], epoch: int) -> int | None:
    lo, hi = 0, len(bars) - 1
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid][0] >= epoch:
            found = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return found


def executable_buy_price(mid: float, symbol: str) -> float:
    ow = expected_spread_pct(symbol) / 2.0 + expected_slippage_rt_pct() / 2.0
    return mid * (1.0 + ow)


def executable_sell_price(mid: float, symbol: str) -> float:
    ow = expected_spread_pct(symbol) / 2.0 + expected_slippage_rt_pct() / 2.0
    return mid * (1.0 - ow)


def veto_current(inf: dict[str, Any]) -> tuple[bool, float]:
    dd = {
        "expected_favorable_excursion": DEFAULT_EFE_PCT,
        "expected_adverse_excursion": DEFAULT_EAE_PCT,
        "prob_buy": inf["p_buy"],
        "prob_sell": inf["p_sell"],
        "prob_hold": inf["p_hold"],
        "estimated_fees_pct": LEGACY_BUY_VETO_FEE_PCT,
        "estimated_slippage_pct": LEGACY_BUY_VETO_SLIP_PCT,
        "spread_pct": LEGACY_BUY_VETO_SPREAD_PCT,
    }
    ev = post_cost_economics_ev(dd)
    if ev is None:
        return False, 0.0
    return ev > 0.0, float(ev)


def veto_honest_plus_quality_floor(inf: dict[str, Any]) -> tuple[bool, float]:
    br = named_cost_breakdown(inf["symbol"], p_buy=inf["p_buy"], p_sell=inf["p_sell"])
    if br.predicted_gross_trade_value < ARM_B_MIN_PREDICTED_GROSS_PCT:
        return False, br.predicted_net_trade_value
    return br.predicted_net_trade_value > 0.0, br.predicted_net_trade_value


def veto_honest_pure_net(inf: dict[str, Any]) -> tuple[bool, float]:
    br = named_cost_breakdown(inf["symbol"], p_buy=inf["p_buy"], p_sell=inf["p_sell"])
    return br.predicted_net_trade_value > 0.0, br.predicted_net_trade_value


def in_unlocked_band(p_buy: float) -> bool:
    return 0.0470 <= float(p_buy) < 0.18333


class LinearCalibrator:
    """p_buy -> realized net. Fit on an earlier chronological window only."""

    def __init__(self) -> None:
        self.a = 0.0
        self.b = 0.0
        self.fitted = False

    def fit(self, pairs: list[tuple[float, float]]) -> None:
        if len(pairs) < 8:
            self.fitted = False
            return
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = float(len(xs))
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 1e-12:
            self.fitted = False
            return
        self.b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den
        self.a = my - self.b * mx
        self.fitted = True

    def predict(self, p_buy: float) -> float:
        return self.a + self.b * float(p_buy)


def make_calibrated_veto(cal: LinearCalibrator) -> Callable[[dict[str, Any]], tuple[bool, float]]:
    def _veto(inf: dict[str, Any]) -> tuple[bool, float]:
        if not cal.fitted:
            return veto_honest_pure_net(inf)
        pred = cal.predict(inf["p_buy"])
        return pred > 0.0, pred

    return _veto


def _advance_position(
    pos: ReplayPos,
    bars: list[tuple[int, float, float, float, float]],
    fourh: list[list[float]],
    start_epoch: int,
    end_epoch: int,
    sell_cost: float,
) -> ClosedTrade | None:
    idx = _bar_index(bars, start_epoch)
    if idx is None:
        return None
    profile = get_coin_profile(pos.symbol)
    for j in range(idx, len(bars)):
        ep, _o, high, low, close = bars[j]
        if ep > end_epoch:
            break
        pos.highest_price = max(pos.highest_price, high)
        if pos.lowest_price <= 0 or low < pos.lowest_price:
            pos.lowest_price = low
        refresh_trailing_stop(pos, high, profile)
        hold_min = max(0.0, (ep - pos.entry_time) / 60.0)
        net_pnl = (close - pos.entry_price) / pos.entry_price - sell_cost
        bundle = fourh_bundle(fourh, float(ep))
        decision = evaluate_engine_managed_exit(
            position=pos,
            current_price=close,
            net_pnl_pct=net_pnl,
            hold_minutes=hold_min,
            coin_profile=profile,
            bundle=bundle,
            bar_low=low,
            now_epoch=float(ep),
        )
        if str(decision.get("action") or "") != "sell":
            continue
        reason = str(decision.get("reason") or "")
        urgent = any(x in reason.upper() for x in ("STOP", "FLOOR", "EXTREME", "TRAIL", "STRUCTURE"))
        exit_mid = low if urgent and low > 0 else close
        exit_px = executable_sell_price(exit_mid, pos.symbol)
        return _close(pos, ep, exit_px, reason, sell_cost)
    return None


def _close(pos: ReplayPos, exit_epoch: float, exit_px: float, reason: str, sell_cost: float) -> ClosedTrade:
    gross = (exit_px - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
    comm = expected_exchange_commission_rt_pct()
    spread = expected_spread_pct(pos.symbol)
    slip = expected_slippage_rt_pct()
    net = gross - comm - spread - slip
    mfe = (pos.highest_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
    mae = (pos.lowest_price - pos.entry_price) / pos.entry_price if pos.entry_price and pos.lowest_price else 0.0
    return ClosedTrade(
        symbol=pos.symbol,
        entry_epoch=pos.entry_time,
        exit_epoch=exit_epoch,
        entry_price=pos.entry_price,
        exit_price=exit_px,
        notional=pos.notional,
        gross_pct=gross,
        commission_pct=comm,
        spread_pct=spread,
        slippage_pct=slip,
        net_pct=net,
        net_usd=net * pos.notional,
        exit_reason=reason,
        mfe_pct=mfe,
        mae_pct=mae,
        p_buy=pos.p_buy,
        setup=pos.setup,
        unlocked_band=pos.unlocked_band,
    )


def summarize(trades: list[ClosedTrade], *, accepted: int, rejected: int, span_sec: float) -> dict[str, Any]:
    n = len(trades)
    empty = {
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "trades": 0,
        "win_rate_pct": 0.0,
        "gross_bps": 0.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "net_bps": 0.0,
        "expectancy_bps": 0.0,
        "profit_factor": None,
        "avg_win_bps": 0.0,
        "avg_loss_bps": 0.0,
        "mfe_bps": 0.0,
        "mae_bps": 0.0,
        "capture_ratio": None,
        "exit_reason_waterfall": {},
        "max_drawdown_bps": 0.0,
        "capital_utilization": 0.0,
        "by_symbol": {},
        "by_setup": {},
    }
    if n == 0:
        return empty
    nets = [t.net_pct for t in trades]
    wins = [t for t in trades if t.net_pct > 0]
    losses = [t for t in trades if t.net_pct <= 0]
    gp = sum(t.net_pct for t in wins)
    gl = abs(sum(t.net_pct for t in losses))
    peak = 0.0
    eq = 0.0
    dd = 0.0
    for t in sorted(trades, key=lambda r: r.exit_epoch):
        eq += t.net_pct
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    exposure = sum(max(0.0, t.exit_epoch - t.entry_epoch) for t in trades)
    waterfall: dict[str, int] = defaultdict(int)
    for t in trades:
        waterfall[t.exit_reason] += 1
    by_sym: dict[str, dict[str, Any]] = {}
    for s in SYMBOLS:
        st = [t for t in trades if t.symbol == s]
        if not st:
            continue
        by_sym[s] = {
            "trades": len(st),
            "win_rate_pct": round(100.0 * sum(1 for t in st if t.net_pct > 0) / len(st), 2),
            "net_bps": round(sum(t.net_pct for t in st) / len(st) * 1e4, 3),
        }
    by_setup: dict[str, dict[str, Any]] = {}
    setups = {t.setup or "unknown" for t in trades}
    for setup in setups:
        st = [t for t in trades if (t.setup or "unknown") == setup]
        by_setup[setup] = {
            "trades": len(st),
            "net_bps": round(sum(t.net_pct for t in st) / len(st) * 1e4, 3),
        }
    mean_mfe = sum(t.mfe_pct for t in trades) / n
    mean_gross = sum(t.gross_pct for t in trades) / n
    return {
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "trades": n,
        "win_rate_pct": round(100.0 * len(wins) / n, 2),
        "gross_bps": round(mean_gross * 1e4, 3),
        "commission_bps": round(sum(t.commission_pct for t in trades) / n * 1e4, 3),
        "spread_bps": round(sum(t.spread_pct for t in trades) / n * 1e4, 3),
        "slippage_bps": round(sum(t.slippage_pct for t in trades) / n * 1e4, 3),
        "net_bps": round(sum(nets) / n * 1e4, 3),
        "expectancy_bps": round(sum(nets) / n * 1e4, 3),
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "avg_win_bps": round((gp / len(wins)) * 1e4, 3) if wins else 0.0,
        "avg_loss_bps": round((sum(t.net_pct for t in losses) / len(losses)) * 1e4, 3) if losses else 0.0,
        "mfe_bps": round(mean_mfe * 1e4, 3),
        "mae_bps": round(sum(t.mae_pct for t in trades) / n * 1e4, 3),
        "capture_ratio": round(mean_gross / mean_mfe, 4) if mean_mfe > 1e-12 else None,
        "exit_reason_waterfall": dict(waterfall),
        "max_drawdown_bps": round(dd * 1e4, 2),
        "capital_utilization": round(exposure / max(1.0, span_sec * DAY_MAX_OPEN_SLOTS), 4),
        "by_symbol": by_sym,
        "by_setup": by_setup,
        "total_net_usd": round(sum(t.net_usd for t in trades), 4),
    }


def run_arm(
    *,
    name: str,
    events: list[dict[str, Any]],
    bars: dict[str, list[tuple[int, float, float, float, float]]],
    fourh: dict[str, list[list[float]]],
    admit: Callable[[dict[str, Any]], tuple[bool, float]],
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    sell_cost: float = LEGACY_SELL_ROUNDTRIP_PCT,
) -> tuple[list[ClosedTrade], int, int]:
    with production_exit_env():
        return _run_arm_body(
            name=name,
            events=events,
            bars=bars,
            fourh=fourh,
            admit=admit,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            sell_cost=sell_cost,
        )


def _run_arm_body(
    *,
    name: str,
    events: list[dict[str, Any]],
    bars: dict[str, list[tuple[int, float, float, float, float]]],
    fourh: dict[str, list[list[float]]],
    admit: Callable[[dict[str, Any]], tuple[bool, float]],
    start_epoch: int | None,
    end_epoch: int | None,
    sell_cost: float,
) -> tuple[list[ClosedTrade], int, int]:
    open_pos: dict[str, ReplayPos] = {}
    cooldown: dict[str, float] = defaultdict(float)
    last_adv: dict[str, int] = {}
    closed: list[ClosedTrade] = []
    accepted = 0
    rejected = 0
    deployed = 0.0

    def _flush_until(until: int) -> None:
        nonlocal deployed
        for sym, pos in list(open_pos.items()):
            start = int(last_adv.get(sym, pos.entry_time)) + 1
            tr = _advance_position(pos, bars.get(sym, []), fourh.get(sym, []), start, until, sell_cost)
            last_adv[sym] = until
            if tr is None:
                continue
            closed.append(tr)
            deployed = max(0.0, deployed - pos.notional)
            cooldown[sym] = tr.exit_epoch + COOLDOWN_SECONDS_AFTER_SELL
            del open_pos[sym]

    for ev in events:
        bar = int(ev["bar_epoch"])
        if start_epoch is not None and bar < start_epoch:
            continue
        if end_epoch is not None and bar >= end_epoch:
            break
        _flush_until(bar)
        ranked = sorted(ev["inferences"], key=lambda r: r["p_buy"], reverse=True)
        for inf in ranked:
            ok, _ev = admit(inf)
            if not ok:
                rejected += 1
                continue
            accepted += 1
            sym = inf["symbol"]
            if sym in open_pos:
                continue
            if bar < cooldown[sym]:
                continue
            if len(open_pos) >= DAY_MAX_OPEN_SLOTS:
                continue
            slot = float(DAY_TARGET_NOTIONAL_PER_SLOT_USD)
            if deployed + slot > float(DAY_MAX_DEPLOYED_USD) + 1e-9:
                remain = float(DAY_MAX_DEPLOYED_USD) - deployed
                if remain < slot * 0.25:
                    continue
                slot = remain
            bset = bars.get(sym, [])
            bidx = _bar_index(bset, bar)
            if bidx is None:
                continue
            mid = bset[bidx][4]
            if mid <= 0:
                continue
            fill = executable_buy_price(mid, sym)
            profile = get_coin_profile(sym)
            pos = ReplayPos(
                symbol=sym,
                entry_price=fill,
                entry_time=float(bar),
                quantity=slot / fill,
                notional=slot,
                stop_price=fill * (1.0 - float(profile["sl"])),
                take_profit_1_price=fill * (1.0 + float(profile["tp"])),
                trail_pct=float(profile["trail"]),
                highest_price=fill,
                lowest_price=fill,
                max_hold_min=int(profile["max_hold_min"]),
                p_buy=inf["p_buy"],
                setup="path_net",
                unlocked_band=in_unlocked_band(inf["p_buy"]),
            )
            open_pos[sym] = pos
            last_adv[sym] = bar
            deployed += slot
    last_bar = int(events[-1]["bar_epoch"]) if events else 0
    flush_to = (end_epoch - 1) if end_epoch else last_bar + HORIZON_PAD_SEC
    _flush_until(int(flush_to))
    return closed, accepted, rejected


def fold_bounds(events: list[dict[str, Any]], n_folds: int = 3) -> list[tuple[int, int]]:
    if not events:
        return []
    times = [int(e["bar_epoch"]) for e in events]
    lo, hi = times[0], times[-1] + BAR_DECISION_SEC
    width = max(BAR_DECISION_SEC, (hi - lo) // n_folds)
    out: list[tuple[int, int]] = []
    for i in range(n_folds):
        a = lo + i * width
        b = hi if i == n_folds - 1 else lo + (i + 1) * width
        out.append((a, b))
    return out


def run_all_arms(
    conn: sqlite3.Connection,
    *,
    calibrator_train_frac: float = 0.5,
) -> dict[str, Any]:
    bars = load_1m_bars(conn)
    fourh = {s: resample_4h(bars[s]) for s in SYMBOLS}
    inferences = load_inferences(conn)
    events = decision_bars(inferences)
    if not events:
        return {"error": "no_decision_events", "inferences": len(inferences)}
    span = float(events[-1]["bar_epoch"] - events[0]["bar_epoch"] + BAR_DECISION_SEC)
    t0 = events[0]["bar_epoch"]
    t1 = events[-1]["bar_epoch"] + BAR_DECISION_SEC
    split = t0 + int((t1 - t0) * calibrator_train_frac)

    train_closed, _, _ = run_arm(
        name="calib_train",
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_honest_pure_net,
        start_epoch=t0,
        end_epoch=split,
    )
    cal = LinearCalibrator()
    cal.fit([(t.p_buy, t.net_pct) for t in train_closed])

    arms = {
        "A_current_production": veto_current,
        "B_honest_same_quality_floor": veto_honest_plus_quality_floor,
        "C_honest_pure_net_ev": veto_honest_pure_net,
        "D_calibrated_net_ev": make_calibrated_veto(cal),
    }
    report: dict[str, Any] = {
        "inferences": len(inferences),
        "decision_bars": len(events),
        "span_sec": span,
        "honest_cost_bps": {s: round(honest_all_in_rt_pct(s) * 1e4, 3) for s in SYMBOLS},
        "calibrator": {"fitted": cal.fitted, "a": cal.a, "b": cal.b, "train_trades": len(train_closed), "train_end_epoch": split},
        "arms": {},
        "unlocked_band": {},
        "folds": {},
    }
    folds = fold_bounds(events)
    for name, admit in arms.items():
        closed, acc, rej = run_arm(name=name, events=events, bars=bars, fourh=fourh, admit=admit)
        summary = summarize(closed, accepted=acc, rejected=rej, span_sec=span)
        summary["arm"] = name
        report["arms"][name] = summary
        unlocked = [t for t in closed if t.unlocked_band]
        report["unlocked_band"][name] = summarize(unlocked, accepted=sum(1 for t in closed if t.unlocked_band), rejected=0, span_sec=span)
        fold_rows = []
        for i, (a, b) in enumerate(folds):
            fc, fa, fr = run_arm(name=f"{name}_fold{i}", events=events, bars=bars, fourh=fourh, admit=admit, start_epoch=a, end_epoch=b)
            row = summarize(fc, accepted=fa, rejected=fr, span_sec=float(b - a))
            row["fold"] = i
            row["start_epoch"] = a
            row["end_epoch"] = b
            fold_rows.append(row)
        report["folds"][name] = fold_rows
    return report
