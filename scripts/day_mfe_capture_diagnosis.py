#!/usr/bin/env python3
"""Parts 1-5: MFE-capture attribution, path-net entry, exit-capture, evidence, scaling.

Reuses the production-faithful replay infrastructure. No live changes.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("DAY_PATH_AWARE_EXIT", "true")
os.environ.setdefault("DAY_STALL_EXIT_ENABLED", "false")
os.environ.setdefault("DAY_GIVEBACK_EXIT_ENABLED", "false")
os.environ.setdefault("DAY_GIVEBACK_ON_4H_HOLD", "true")
os.environ.setdefault("DAY_STALL_ON_4H_HOLD", "true")
os.environ.setdefault("ESTIMATED_ROUNDTRIP_COST", "0.0006")

from backend.config.execution_cost_model import (
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
    evaluate_pre_buy_exit_consistency,
    refresh_trailing_stop,
)
from backend.services.day_production_lifecycle_replay import (
    BAR_DECISION_SEC,
    HORIZON_PAD_SEC,
    SYMBOLS,
    ClosedTrade,
    ReplayPos,
    _bar_index,
    _close,
    decision_bars,
    executable_buy_price,
    executable_sell_price,
    fold_bounds,
    fourh_bundle,
    in_unlocked_band,
    load_1m_bars,
    load_inferences,
    production_exit_env,
    resample_4h,
    veto_current,
)
from backend.services.portfolio_engine import get_coin_profile

# ── Extended ClosedTrade with path detail ──────────────────────────────


@dataclass
class DetailedTrade:
    trade: ClosedTrade
    trail_ever_active: bool = False
    be_ever_active: bool = False
    time_to_mfe_min: float = 0.0
    time_mfe_to_exit_min: float = 0.0
    mfe_epoch: float = 0.0
    was_net_profitable_ever: bool = False
    max_net_pct: float = 0.0
    final_selection_score: float = 0.0
    market_role: str = "unknown"
    volatility_bucket: str = "unknown"
    entry_hour: int = -1
    holding_min: float = 0.0
    bars_held: int = 0
    structure_break_subtype: str = ""


# ── Replay engine with path instrumentation ────────────────────────────


def _advance_position_instrumented(
    pos: ReplayPos,
    bars: list[tuple[int, float, float, float, float]],
    fourh: list[list[float]],
    start_epoch: int,
    end_epoch: int,
    sell_cost: float,
) -> DetailedTrade | None:
    idx = _bar_index(bars, start_epoch)
    if idx is None:
        return None
    profile = get_coin_profile(pos.symbol)
    trail_active = False
    be_active = False
    mfe_epoch = pos.entry_time
    max_net = 0.0
    was_profitable = False
    bars_counted = 0

    for j in range(idx, len(bars)):
        ep, _o, high, low, close = bars[j]
        if ep > end_epoch:
            break
        bars_counted += 1
        prev_highest = pos.highest_price
        pos.highest_price = max(pos.highest_price, high)
        if pos.lowest_price <= 0 or low < pos.lowest_price:
            pos.lowest_price = low
        if pos.highest_price > prev_highest:
            mfe_epoch = float(ep)

        old_trail = float(pos.trailing_stop_price)
        refresh_trailing_stop(pos, high, profile)
        new_trail = float(pos.trailing_stop_price)
        if new_trail > 0 and new_trail > old_trail + 1e-12:
            trail_active = True
        if new_trail > pos.entry_price + 1e-12:
            be_active = True

        hold_min = max(0.0, (ep - pos.entry_time) / 60.0)
        current_net = (close - pos.entry_price) / pos.entry_price - sell_cost
        max_net = max(max_net, current_net)
        if current_net > 0:
            was_profitable = True

        bundle = fourh_bundle(fourh, float(ep), bars)
        decision = evaluate_engine_managed_exit(
            position=pos,
            current_price=close,
            net_pnl_pct=current_net,
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
        ct = _close(pos, ep, exit_px, reason, sell_cost)
        dt = DetailedTrade(
            trade=ct,
            trail_ever_active=trail_active,
            be_ever_active=be_active,
            time_to_mfe_min=max(0.0, (mfe_epoch - pos.entry_time) / 60.0),
            time_mfe_to_exit_min=max(0.0, (ep - mfe_epoch) / 60.0),
            mfe_epoch=mfe_epoch,
            was_net_profitable_ever=was_profitable,
            max_net_pct=max_net,
            holding_min=hold_min,
            bars_held=bars_counted,
        )
        # classify structure break subtype
        if "STRUCTURE" in reason:
            if ct.mfe_pct * 1e4 < 5.0:
                dt.structure_break_subtype = "never_developed"
            elif was_profitable and ct.gross_pct < 0:
                dt.structure_break_subtype = "profitable_then_reversed"
            elif be_active and ct.net_pct < 0:
                dt.structure_break_subtype = "be_active_still_lost"
            elif trail_active and ct.net_pct < 0:
                dt.structure_break_subtype = "trail_active_lost"
            elif not trail_active and not be_active:
                dt.structure_break_subtype = "no_protection_ever"
            else:
                dt.structure_break_subtype = "other"
        return dt
    return None


def run_arm_detailed(
    *,
    events: list[dict[str, Any]],
    bars: dict[str, list[tuple[int, float, float, float, float]]],
    fourh: dict[str, list[list[float]]],
    admit: Callable[[dict[str, Any]], tuple[bool, float]],
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    sell_cost: float = LEGACY_SELL_ROUNDTRIP_PCT,
    exit_override: Callable | None = None,
) -> tuple[list[DetailedTrade], int, int]:
    with production_exit_env():
        return _run_arm_detailed_body(
            events=events,
            bars=bars,
            fourh=fourh,
            admit=admit,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            sell_cost=sell_cost,
            exit_override=exit_override,
        )


def _run_arm_detailed_body(
    *,
    events: list[dict[str, Any]],
    bars: dict[str, list[tuple[int, float, float, float, float]]],
    fourh: dict[str, list[list[float]]],
    admit: Callable[[dict[str, Any]], tuple[bool, float]],
    start_epoch: int | None,
    end_epoch: int | None,
    sell_cost: float,
    exit_override: Callable | None = None,
) -> tuple[list[DetailedTrade], int, int]:
    open_pos: dict[str, ReplayPos] = {}
    cooldown: dict[str, float] = defaultdict(float)
    last_adv: dict[str, int] = {}
    closed: list[DetailedTrade] = []
    accepted = 0
    rejected = 0
    deployed = 0.0

    def _flush_until(until: int) -> None:
        nonlocal deployed
        for sym, pos in list(open_pos.items()):
            start = int(last_adv.get(sym, pos.entry_time)) + 1
            if exit_override is not None:
                tr = _advance_with_override(pos, bars.get(sym, []), fourh.get(sym, []), start, until, sell_cost, exit_override)
            else:
                tr = _advance_position_instrumented(pos, bars.get(sym, []), fourh.get(sym, []), start, until, sell_cost)
            last_adv[sym] = until
            if tr is None:
                continue
            closed.append(tr)
            deployed = max(0.0, deployed - pos.notional)
            cooldown[sym] = tr.trade.exit_epoch + COOLDOWN_SECONDS_AFTER_SELL
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
            prebuy = evaluate_pre_buy_exit_consistency(
                setup="HTF_TREND_PULLBACK",
                entry_price=fill,
                stop_price=fill * (1.0 - float(profile["sl"])),
                thesis_invalid_level=0.0,
                thesis_target_level=fill * (1.0 + float(profile["tp"])),
                entry_vwap=fill,
                entry_ts=float(bar),
                coin_profile=profile,
                bundle=fourh_bundle(fourh.get(sym, []), float(bar), bset),
                bar_ts=float(bar),
            )
            if not prebuy.get("allowed"):
                continue
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


# ── Part 3: Modified exit logic ────────────────────────────────────────


def _advance_with_override(
    pos: ReplayPos,
    bars: list[tuple[int, float, float, float, float]],
    fourh: list[list[float]],
    start_epoch: int,
    end_epoch: int,
    sell_cost: float,
    exit_fn: Callable,
) -> DetailedTrade | None:
    """Like _advance_position_instrumented but uses exit_fn for decisions."""
    idx = _bar_index(bars, start_epoch)
    if idx is None:
        return None
    profile = get_coin_profile(pos.symbol)
    trail_active = False
    be_active = False
    mfe_epoch = pos.entry_time
    max_net = 0.0
    was_profitable = False
    bars_counted = 0

    for j in range(idx, len(bars)):
        ep, _o, high, low, close = bars[j]
        if ep > end_epoch:
            break
        bars_counted += 1
        prev_highest = pos.highest_price
        pos.highest_price = max(pos.highest_price, high)
        if pos.lowest_price <= 0 or low < pos.lowest_price:
            pos.lowest_price = low
        if pos.highest_price > prev_highest:
            mfe_epoch = float(ep)

        old_trail = float(pos.trailing_stop_price)
        refresh_trailing_stop(pos, high, profile)
        new_trail = float(pos.trailing_stop_price)
        if new_trail > 0 and new_trail > old_trail + 1e-12:
            trail_active = True
        if new_trail > pos.entry_price + 1e-12:
            be_active = True

        hold_min = max(0.0, (ep - pos.entry_time) / 60.0)
        current_net = (close - pos.entry_price) / pos.entry_price - sell_cost

        max_net = max(max_net, current_net)
        if current_net > 0:
            was_profitable = True

        bundle = fourh_bundle(fourh, float(ep), bars)
        decision = exit_fn(
            position=pos,
            current_price=close,
            net_pnl_pct=current_net,
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
        ct = _close(pos, ep, exit_px, reason, sell_cost)
        dt = DetailedTrade(
            trade=ct,
            trail_ever_active=trail_active,
            be_ever_active=be_active,
            time_to_mfe_min=max(0.0, (mfe_epoch - pos.entry_time) / 60.0),
            time_mfe_to_exit_min=max(0.0, (ep - mfe_epoch) / 60.0),
            mfe_epoch=mfe_epoch,
            was_net_profitable_ever=was_profitable,
            max_net_pct=max_net,
            holding_min=hold_min,
            bars_held=bars_counted,
        )
        if "STRUCTURE" in reason:
            if ct.mfe_pct * 1e4 < 5.0:
                dt.structure_break_subtype = "never_developed"
            elif was_profitable and ct.gross_pct < 0:
                dt.structure_break_subtype = "profitable_then_reversed"
            elif be_active and ct.net_pct < 0:
                dt.structure_break_subtype = "be_active_still_lost"
            elif trail_active and ct.net_pct < 0:
                dt.structure_break_subtype = "trail_active_lost"
            elif not trail_active and not be_active:
                dt.structure_break_subtype = "no_protection_ever"
            else:
                dt.structure_break_subtype = "other"
        return dt
    return None


# ── Group statistics ───────────────────────────────────────────────────


def group_stats(dts: list[DetailedTrade]) -> dict[str, Any]:
    if not dts:
        return {"trades": 0}
    trades = [d.trade for d in dts]
    n = len(trades)
    wins = [t for t in trades if t.net_pct > 0]
    losses = [t for t in trades if t.net_pct <= 0]
    gp = sum(t.net_pct for t in wins)
    gl = abs(sum(t.net_pct for t in losses))
    mean_gross = sum(t.gross_pct for t in trades) / n
    mean_net = sum(t.net_pct for t in trades) / n
    mean_mfe = sum(t.mfe_pct for t in trades) / n
    mean_mae = sum(t.mae_pct for t in trades) / n
    peak = eq = dd = 0.0
    for t in sorted(trades, key=lambda r: r.exit_epoch):
        eq += t.net_pct
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    holds = [max(0.0, (t.exit_epoch - t.entry_epoch) / 60.0) for t in trades]
    waterfall: dict[str, int] = defaultdict(int)
    for t in trades:
        waterfall[t.exit_reason] += 1
    pnl_usd = sum(t.net_usd for t in trades)
    capture = round(mean_gross / mean_mfe, 4) if abs(mean_mfe) > 1e-12 else None
    return {
        "trades": n,
        "win_rate_pct": round(100.0 * len(wins) / n, 2),
        "gross_bps": round(mean_gross * 1e4, 2),
        "net_bps": round(mean_net * 1e4, 2),
        "profit_factor": round(gp / gl, 4) if gl > 1e-12 else None,
        "mfe_bps": round(mean_mfe * 1e4, 2),
        "mae_bps": round(mean_mae * 1e4, 2),
        "capture": capture,
        "median_hold_min": round(sorted(holds)[n // 2], 1),
        "max_dd_bps": round(dd * 1e4, 1),
        "exit_reasons": dict(waterfall),
        "pnl_usd": round(pnl_usd, 2),
    }


# ── Bucketing helpers ──────────────────────────────────────────────────


def p_buy_decile(p: float) -> str:
    d = min(9, int(p * 10))
    return f"d{d}"


def mfe_bucket(mfe_bps: float) -> str:
    if mfe_bps < 5:
        return "<5bp"
    if mfe_bps < 15:
        return "5-15bp"
    if mfe_bps < 30:
        return "15-30bp"
    if mfe_bps < 50:
        return "30-50bp"
    return ">=50bp"


def hold_bucket(hold_min: float) -> str:
    if hold_min < 30:
        return "<30m"
    if hold_min < 60:
        return "30-60m"
    if hold_min < 120:
        return "60-120m"
    if hold_min < 240:
        return "120-240m"
    return ">=240m"


def time_to_mfe_bucket(t_min: float) -> str:
    if t_min < 15:
        return "<15m"
    if t_min < 60:
        return "15-60m"
    if t_min < 120:
        return "60-120m"
    return ">=120m"


def entry_hour(epoch: float) -> int:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).hour


def vol_bucket_from_bars(bars: list[tuple[int, float, float, float, float]], epoch: int) -> str:
    idx = _bar_index(bars, epoch)
    if idx is None or idx < 20:
        return "unknown"
    window = bars[max(0, idx - 20) : idx]
    rets = []
    for i in range(1, len(window)):
        if window[i - 1][4] > 0:
            rets.append(abs((window[i][4] - window[i - 1][4]) / window[i - 1][4]))
    if not rets:
        return "unknown"
    avg_vol = sum(rets) / len(rets)
    if avg_vol < 0.001:
        return "low"
    if avg_vol < 0.003:
        return "mid"
    return "high"


# ── Modified exit functions for Part 3 ─────────────────────────────────


def exit_earlier_protection(**kw):
    """Lower break-even trigger; tighter trail activation after modest MFE."""
    prev_trigger = os.environ.get("DAY_BREAK_EVEN_TRIGGER_PCT")
    prev_offset = os.environ.get("DAY_BREAK_EVEN_OFFSET_PCT")
    os.environ["DAY_BREAK_EVEN_TRIGGER_PCT"] = "0.0015"
    os.environ["DAY_BREAK_EVEN_OFFSET_PCT"] = "0.0003"
    try:
        return evaluate_engine_managed_exit(**kw)
    finally:
        if prev_trigger is None:
            os.environ.pop("DAY_BREAK_EVEN_TRIGGER_PCT", None)
        else:
            os.environ["DAY_BREAK_EVEN_TRIGGER_PCT"] = prev_trigger
        if prev_offset is None:
            os.environ.pop("DAY_BREAK_EVEN_OFFSET_PCT", None)
        else:
            os.environ["DAY_BREAK_EVEN_OFFSET_PCT"] = prev_offset


def exit_with_giveback(**kw):
    """Enable giveback exit on 4H hold."""
    prev_gb = os.environ.get("DAY_GIVEBACK_EXIT_ENABLED")
    prev_stall = os.environ.get("DAY_STALL_EXIT_ENABLED")
    os.environ["DAY_GIVEBACK_EXIT_ENABLED"] = "true"
    os.environ["DAY_STALL_EXIT_ENABLED"] = "true"
    os.environ["DAY_GIVEBACK_MIN_MFE_PCT"] = "0.0015"
    os.environ["DAY_GIVEBACK_TRIGGER_PNL_PCT"] = "-0.0010"
    os.environ["DAY_STALL_MIN_HOLD_MIN"] = "90"
    os.environ["DAY_STALL_MAX_MFE_PCT"] = "0.0030"
    try:
        return evaluate_engine_managed_exit(**kw)
    finally:
        if prev_gb is None:
            os.environ.pop("DAY_GIVEBACK_EXIT_ENABLED", None)
        else:
            os.environ["DAY_GIVEBACK_EXIT_ENABLED"] = prev_gb
        if prev_stall is None:
            os.environ.pop("DAY_STALL_EXIT_ENABLED", None)
        else:
            os.environ["DAY_STALL_EXIT_ENABLED"] = prev_stall
        os.environ.pop("DAY_GIVEBACK_MIN_MFE_PCT", None)
        os.environ.pop("DAY_GIVEBACK_TRIGGER_PNL_PCT", None)
        os.environ.pop("DAY_STALL_MIN_HOLD_MIN", None)
        os.environ.pop("DAY_STALL_MAX_MFE_PCT", None)


def exit_combined_improved(**kw):
    """Earlier BE + giveback + stall in one package."""
    saves = {}
    changes = {
        "DAY_BREAK_EVEN_TRIGGER_PCT": "0.0015",
        "DAY_BREAK_EVEN_OFFSET_PCT": "0.0003",
        "DAY_GIVEBACK_EXIT_ENABLED": "true",
        "DAY_STALL_EXIT_ENABLED": "true",
        "DAY_GIVEBACK_MIN_MFE_PCT": "0.0015",
        "DAY_GIVEBACK_TRIGGER_PNL_PCT": "-0.0010",
        "DAY_STALL_MIN_HOLD_MIN": "90",
        "DAY_STALL_MAX_MFE_PCT": "0.0030",
    }
    for k, v in changes.items():
        saves[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        return evaluate_engine_managed_exit(**kw)
    finally:
        for k, v in saves.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── Part 2: simple realized-net regression ─────────────────────────────


class RidgeRealizedNet:
    """Ridge regression on entry features -> realized net bps."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: list[float] = []
        self.bias = 0.0
        self.fitted = False

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        n = len(X)
        if n < 10 or not X[0]:
            self.fitted = False
            return
        d = len(X[0])
        means_x = [sum(row[j] for row in X) / n for j in range(d)]
        mean_y = sum(y) / n
        Xc = [[row[j] - means_x[j] for j in range(d)] for row in X]
        yc = [yi - mean_y for yi in y]
        XtX = [[sum(Xc[i][j] * Xc[i][k] for i in range(n)) for k in range(d)] for j in range(d)]
        for j in range(d):
            XtX[j][j] += self.alpha * n
        Xty = [sum(Xc[i][j] * yc[i] for i in range(n)) for j in range(d)]
        w = self._solve(XtX, Xty, d)
        if w is None:
            self.fitted = False
            return
        self.weights = w
        self.bias = mean_y - sum(w[j] * means_x[j] for j in range(d))
        self.fitted = True

    def _solve(self, A: list[list[float]], b: list[float], d: int) -> list[float] | None:
        aug = [[*A[i][:], b[i]] for i in range(d)]
        for col in range(d):
            pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-12:
                return None
            aug[col], aug[pivot] = aug[pivot], aug[col]
            for row in range(d):
                if row == col:
                    continue
                f = aug[row][col] / aug[col][col]
                for j in range(d + 1):
                    aug[row][j] -= f * aug[col][j]
        return [aug[i][d] / aug[i][i] for i in range(d)]

    def predict(self, x: list[float]) -> float:
        if not self.fitted:
            return 0.0
        return self.bias + sum(self.weights[j] * x[j] for j in range(len(self.weights)))


class TreeNode:
    """Minimal decision stump for realized net bps."""

    def __init__(self):
        self.feature = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None

    def predict(self, x: list[float]) -> float:
        if x[self.feature] <= self.threshold:
            return self.left.predict(x) if self.left else self.left_val
        return self.right.predict(x) if self.right else self.right_val


def build_tree(X: list[list[float]], y: list[float], depth: int = 0, max_depth: int = 4, min_leaf: int = 15) -> TreeNode:
    node = TreeNode()
    mean_y = sum(y) / len(y) if y else 0.0
    node.left_val = mean_y
    node.right_val = mean_y
    if depth >= max_depth or len(y) < min_leaf * 2:
        return node
    best_loss = float("inf")
    d = len(X[0]) if X else 0
    for f in range(d):
        vals = sorted({row[f] for row in X})
        thresholds = [(vals[i] + vals[i + 1]) / 2.0 for i in range(min(len(vals) - 1, 20))]
        for thr in thresholds:
            left_y = [y[i] for i in range(len(y)) if X[i][f] <= thr]
            right_y = [y[i] for i in range(len(y)) if X[i][f] > thr]
            if len(left_y) < min_leaf or len(right_y) < min_leaf:
                continue
            ml = sum(left_y) / len(left_y)
            mr = sum(right_y) / len(right_y)
            loss = sum((v - ml) ** 2 for v in left_y) + sum((v - mr) ** 2 for v in right_y)
            if loss < best_loss:
                best_loss = loss
                node.feature = f
                node.threshold = thr
                node.left_val = ml
                node.right_val = mr
    if best_loss == float("inf"):
        return node
    left_idx = [i for i in range(len(y)) if X[i][node.feature] <= node.threshold]
    right_idx = [i for i in range(len(y)) if X[i][node.feature] > node.threshold]
    if len(left_idx) >= min_leaf * 2:
        node.left = build_tree([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1, max_depth, min_leaf)
    if len(right_idx) >= min_leaf * 2:
        node.right = build_tree([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1, max_depth, min_leaf)
    return node


def extract_features(inf: dict[str, Any], bars: dict[str, list], epoch: int) -> list[float]:
    sym = inf["symbol"]
    p_buy = inf["p_buy"]
    p_sell = inf["p_sell"]
    p_hold = inf["p_hold"]
    sym_bars = bars.get(sym, [])
    bidx = _bar_index(sym_bars, epoch)
    ret_5 = ret_20 = vol_20 = range_ratio = 0.0
    if bidx is not None and bidx >= 20:
        c_now = sym_bars[bidx][4]
        c_5 = sym_bars[max(0, bidx - 5)][4]
        c_20 = sym_bars[max(0, bidx - 20)][4]
        if c_5 > 0:
            ret_5 = (c_now - c_5) / c_5
        if c_20 > 0:
            ret_20 = (c_now - c_20) / c_20
        rets = []
        for k in range(max(0, bidx - 20), bidx):
            if sym_bars[k][4] > 0 and k > 0:
                rets.append((sym_bars[k][4] - sym_bars[k - 1][4]) / sym_bars[k - 1][4])
        if rets:
            vol_20 = (sum(r**2 for r in rets) / len(rets)) ** 0.5
        h20 = max(sym_bars[k][2] for k in range(max(0, bidx - 20), bidx + 1))
        l20 = min(sym_bars[k][3] for k in range(max(0, bidx - 20), bidx + 1))
        if l20 > 0:
            range_ratio = (h20 - l20) / l20
    sym_idx = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2, "XRPUSDT": 3}.get(sym, 0)
    hour = datetime.fromtimestamp(epoch, tz=timezone.utc).hour
    return [p_buy, p_sell, p_hold, ret_5, ret_20, vol_20, range_ratio, float(sym_idx), float(hour)]


FEATURE_NAMES = ["p_buy", "p_sell", "p_hold", "ret_5m", "ret_20m", "vol_20m", "range_ratio", "sym_idx", "hour"]


# ── Part 2 entry models ───────────────────────────────────────────────


def admit_by_model(model, bars, threshold=0.0):
    def _admit(inf):
        ep = inf.get("_epoch", inf.get("epoch", 0))
        feats = extract_features(inf, bars, ep)
        pred = model.predict(feats)
        return pred > threshold, pred

    return _admit


def admit_downside_aware(model_net, model_mae, bars, penalty_weight=0.5, threshold=0.0):
    def _admit(inf):
        ep = inf.get("_epoch", inf.get("epoch", 0))
        feats = extract_features(inf, bars, ep)
        pred_net = model_net.predict(feats)
        pred_mae = model_mae.predict(feats)
        score = pred_net - penalty_weight * abs(pred_mae)
        return score > threshold, score

    return _admit


# ── Main ───────────────────────────────────────────────────────────────


def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "mystic_trading.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)

    t0_load = time.time()
    print("Loading data...", flush=True)
    bars = load_1m_bars(conn)
    fourh = {s: resample_4h(bars[s]) for s in SYMBOLS}
    inferences = load_inferences(conn)
    events = decision_bars(inferences)
    print(f"  {len(inferences)} inferences, {len(events)} decision bars, {sum(len(b) for b in bars.values())} 1m bars loaded in {time.time() - t0_load:.1f}s", flush=True)

    if not events:
        print("ERROR: no decision events found")
        return

    folds = fold_bounds(events, n_folds=3)
    all_epochs = [e["bar_epoch"] for e in events]
    t_start, t_end = all_epochs[0], all_epochs[-1] + BAR_DECISION_SEC

    results: dict[str, Any] = {}

    # ================================================================
    # PART 1: Full attribution on Arm A
    # ================================================================
    print("\n=== PART 1: Arm A Attribution ===", flush=True)
    t1 = time.time()
    arm_a_dts, _acc_a, _rej_a = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_current,
        sell_cost=LEGACY_SELL_ROUNDTRIP_PCT,
    )
    print(f"  Arm A: {len(arm_a_dts)} trades in {time.time() - t1:.1f}s", flush=True)

    # Add entry hour and volatility
    for dt in arm_a_dts:
        dt.entry_hour = entry_hour(dt.trade.entry_epoch)
        dt.volatility_bucket = vol_bucket_from_bars(bars.get(dt.trade.symbol, []), int(dt.trade.entry_epoch))

    # By symbol
    by_sym = {}
    for s in SYMBOLS:
        sub = [d for d in arm_a_dts if d.trade.symbol == s]
        by_sym[s] = group_stats(sub)
    results["by_symbol"] = by_sym

    # By fold
    by_fold = {}
    for i, (a, b) in enumerate(folds):
        sub = [d for d in arm_a_dts if a <= d.trade.entry_epoch < b]
        by_fold[f"fold_{i}"] = group_stats(sub)
    results["by_fold"] = by_fold

    # By p_buy decile
    by_decile = {}
    for d_label in [f"d{i}" for i in range(10)]:
        sub = [d for d in arm_a_dts if p_buy_decile(d.trade.p_buy) == d_label]
        if sub:
            by_decile[d_label] = group_stats(sub)
    results["by_p_buy_decile"] = by_decile

    # By MFE bucket
    by_mfe = {}
    for dt in arm_a_dts:
        b_label = mfe_bucket(dt.trade.mfe_pct * 1e4)
        by_mfe.setdefault(b_label, []).append(dt)
    results["by_mfe_bucket"] = {k: group_stats(v) for k, v in by_mfe.items()}

    # By hold bucket
    by_hold = {}
    for dt in arm_a_dts:
        b_label = hold_bucket(dt.holding_min)
        by_hold.setdefault(b_label, []).append(dt)
    results["by_hold_bucket"] = {k: group_stats(v) for k, v in by_hold.items()}

    # By entry hour
    by_hour = {}
    for dt in arm_a_dts:
        h = str(dt.entry_hour)
        by_hour.setdefault(h, []).append(dt)
    results["by_entry_hour"] = {k: group_stats(v) for k, v in sorted(by_hour.items())}

    # By volatility bucket
    by_vol = {}
    for dt in arm_a_dts:
        by_vol.setdefault(dt.volatility_bucket, []).append(dt)
    results["by_vol_bucket"] = {k: group_stats(v) for k, v in by_vol.items()}

    # By trail activation
    trail_yes = [d for d in arm_a_dts if d.trail_ever_active]
    trail_no = [d for d in arm_a_dts if not d.trail_ever_active]
    results["by_trail_active"] = {"yes": group_stats(trail_yes), "no": group_stats(trail_no)}

    # By BE activation
    be_yes = [d for d in arm_a_dts if d.be_ever_active]
    be_no = [d for d in arm_a_dts if not d.be_ever_active]
    results["by_be_active"] = {"yes": group_stats(be_yes), "no": group_stats(be_no)}

    # By time-to-MFE bucket
    by_ttm = {}
    for dt in arm_a_dts:
        b_label = time_to_mfe_bucket(dt.time_to_mfe_min)
        by_ttm.setdefault(b_label, []).append(dt)
    results["by_time_to_mfe"] = {k: group_stats(v) for k, v in by_ttm.items()}

    # Structure break subtype analysis
    structure_breaks = [d for d in arm_a_dts if "STRUCTURE" in d.trade.exit_reason]
    sb_subtypes: dict[str, list] = {}
    for d in structure_breaks:
        sb_subtypes.setdefault(d.structure_break_subtype, []).append(d)
    results["structure_break_subtypes"] = {k: group_stats(v) for k, v in sb_subtypes.items()}
    results["structure_break_total"] = group_stats(structure_breaks)

    # 5 pathologies
    pathology_counts = {
        "never_enough_mfe_to_cover_costs": len([d for d in structure_breaks if d.trade.mfe_pct * 1e4 < honest_all_in_rt_pct(d.trade.symbol) * 1e4]),
        "became_net_profitable_then_reversed": len([d for d in structure_breaks if d.was_net_profitable_ever and d.trade.net_pct < 0]),
        "be_activated_still_lost": len([d for d in structure_breaks if d.be_ever_active and d.trade.net_pct < 0]),
        "trail_qualified_lost": len([d for d in structure_breaks if d.trail_ever_active and d.trade.net_pct < 0]),
        "no_protection_ever_activated": len([d for d in structure_breaks if not d.trail_ever_active and not d.be_ever_active]),
    }
    results["structure_break_pathologies"] = pathology_counts

    # Overall Arm A summary
    results["arm_a_overall"] = group_stats(arm_a_dts)

    print(json.dumps(results, indent=2, default=str), flush=True)

    # ================================================================
    # PART 2: Path-net entry models (train on fold 0+1, test on fold 2)
    # ================================================================
    print("\n=== PART 2: Entry Model Comparison ===", flush=True)
    train_end = folds[1][1] if len(folds) > 1 else folds[0][1]

    # Build training dataset from arm A trades in train folds
    train_dts = [d for d in arm_a_dts if d.trade.entry_epoch < train_end]
    test_dts = [d for d in arm_a_dts if d.trade.entry_epoch >= train_end]
    print(f"  Train: {len(train_dts)} trades, Test: {len(test_dts)} trades", flush=True)

    # We need to map back to inferences to get full feature set
    # Build training features from ALL inferences in train period (accepted + rejected)
    # to properly train on the full candidate set
    train_infs_by_epoch: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        bar = ev["bar_epoch"]
        if bar >= train_end:
            break
        for inf in ev["inferences"]:
            inf["_epoch"] = bar
        train_infs_by_epoch[bar] = ev["inferences"]

    # Run full lifecycle on ALL train candidates to get realized returns
    # For candidates that don't pass veto, we simulate what would have happened
    def admit_all(inf):
        return True, inf["p_buy"]

    all_train_dts, _, _ = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=admit_all,
        start_epoch=folds[0][0],
        end_epoch=train_end,
        sell_cost=LEGACY_SELL_ROUNDTRIP_PCT,
    )

    # Build (features, realized_net_bps) pairs
    trade_by_key: dict[tuple[str, int], DetailedTrade] = {}
    for dt in all_train_dts:
        bar = int(dt.trade.entry_epoch) // BAR_DECISION_SEC * BAR_DECISION_SEC
        trade_by_key[(dt.trade.symbol, bar)] = dt

    X_train: list[list[float]] = []
    y_net_train: list[float] = []
    y_mae_train: list[float] = []
    for ev in events:
        bar = ev["bar_epoch"]
        if bar >= train_end:
            break
        for inf in ev["inferences"]:
            key = (inf["symbol"], bar)
            if key in trade_by_key:
                dt = trade_by_key[key]
                feats = extract_features(inf, bars, bar)
                X_train.append(feats)
                y_net_train.append(dt.trade.net_pct * 1e4)
                y_mae_train.append(dt.trade.mae_pct * 1e4)

    print(f"  Training samples: {len(X_train)}", flush=True)

    # Train ridge
    ridge = RidgeRealizedNet(alpha=1.0)
    ridge.fit(X_train, y_net_train)
    print(f"  Ridge fitted: {ridge.fitted}", flush=True)
    if ridge.fitted:
        print(f"  Ridge weights: {dict(zip(FEATURE_NAMES, [round(w, 4) for w in ridge.weights], strict=False))}", flush=True)

    # Train tree
    tree = build_tree(X_train, y_net_train, max_depth=4, min_leaf=20) if len(X_train) >= 40 else None
    print(f"  Tree built: {tree is not None}", flush=True)

    # Train MAE model for downside-aware
    ridge_mae = RidgeRealizedNet(alpha=1.0)
    ridge_mae.fit(X_train, y_mae_train)

    # Test each entry model on fold 2
    entry_models: dict[str, Any] = {}

    # A) Current p_buy (baseline)
    test_a_dts, _acc, _rej = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_current,
        start_epoch=train_end,
        end_epoch=t_end,
    )
    entry_models["A_current_veto"] = group_stats(test_a_dts)
    print(f"  A_current_veto on test fold: {len(test_a_dts)} trades", flush=True)

    # B) Ridge net-EV
    if ridge.fitted:
        for ev in events:
            for inf in ev["inferences"]:
                inf["_epoch"] = ev["bar_epoch"]
        test_b_dts, _, _ = run_arm_detailed(
            events=events,
            bars=bars,
            fourh=fourh,
            admit=admit_by_model(ridge, bars, threshold=0.0),
            start_epoch=train_end,
            end_epoch=t_end,
        )
        entry_models["B_ridge_net_ev"] = group_stats(test_b_dts)
        print(f"  B_ridge_net_ev on test fold: {len(test_b_dts)} trades", flush=True)

    # C) Tree net-EV
    if tree is not None:
        test_c_dts, _, _ = run_arm_detailed(
            events=events,
            bars=bars,
            fourh=fourh,
            admit=admit_by_model(tree, bars, threshold=0.0),
            start_epoch=train_end,
            end_epoch=t_end,
        )
        entry_models["C_tree_net_ev"] = group_stats(test_c_dts)
        print(f"  C_tree_net_ev on test fold: {len(test_c_dts)} trades", flush=True)

    # D) Downside-aware
    if ridge.fitted and ridge_mae.fitted:
        test_d_dts, _, _ = run_arm_detailed(
            events=events,
            bars=bars,
            fourh=fourh,
            admit=admit_downside_aware(ridge, ridge_mae, bars, penalty_weight=0.5, threshold=0.0),
            start_epoch=train_end,
            end_epoch=t_end,
        )
        entry_models["D_downside_aware"] = group_stats(test_d_dts)
        print(f"  D_downside_aware on test fold: {len(test_d_dts)} trades", flush=True)

    # E) Final selection score (p_buy - p_sell) as ranking, same veto
    def veto_selection_score(inf):
        score = inf["p_buy"] - inf["p_sell"]
        return score > 0.10, score

    test_e_dts, _, _ = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_selection_score,
        start_epoch=train_end,
        end_epoch=t_end,
    )
    entry_models["E_selection_score"] = group_stats(test_e_dts)
    print(f"  E_selection_score on test fold: {len(test_e_dts)} trades", flush=True)

    results["part2_entry_models"] = entry_models

    # ================================================================
    # PART 3: Exit-capture replay
    # ================================================================
    print("\n=== PART 3: Exit-Capture Alternatives ===", flush=True)
    exit_arms: dict[str, Any] = {}

    # Control: exact current production
    print("  Running: production exits (control)...", flush=True)
    ctrl_dts = arm_a_dts  # already have full run
    exit_arms["control_production"] = group_stats(ctrl_dts)

    # Test a) Earlier protection (lower BE trigger)
    print("  Running: earlier protection...", flush=True)
    earlier_dts, _, _ = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_current,
        exit_override=exit_earlier_protection,
    )
    exit_arms["earlier_protection"] = group_stats(earlier_dts)

    # Test b) Giveback + stall enabled
    print("  Running: giveback + stall...", flush=True)
    gb_dts, _, _ = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_current,
        exit_override=exit_with_giveback,
    )
    exit_arms["giveback_stall"] = group_stats(gb_dts)

    # Test c) Combined
    print("  Running: combined exit...", flush=True)
    comb_dts, _, _ = run_arm_detailed(
        events=events,
        bars=bars,
        fourh=fourh,
        admit=veto_current,
        exit_override=exit_combined_improved,
    )
    exit_arms["combined_exit"] = group_stats(comb_dts)

    # Fold-level validation for each exit variant
    exit_fold_results: dict[str, list] = {}
    for arm_name, exit_fn in [("control_production", None), ("earlier_protection", exit_earlier_protection), ("giveback_stall", exit_with_giveback), ("combined_exit", exit_combined_improved)]:
        fold_rows = []
        for i, (a, b) in enumerate(folds):
            fc, _, _ = run_arm_detailed(
                events=events,
                bars=bars,
                fourh=fourh,
                admit=veto_current,
                start_epoch=a,
                end_epoch=b,
                exit_override=exit_fn,
            )
            row = group_stats(fc)
            row["fold"] = i
            fold_rows.append(row)
        exit_fold_results[arm_name] = fold_rows

    results["part3_exit_arms"] = exit_arms
    results["part3_exit_folds"] = exit_fold_results

    # ================================================================
    # PART 3 + PART 2 combined: best entry + best exit
    # ================================================================
    print("\n=== Combined Entry + Exit ===", flush=True)
    combined_results: dict[str, Any] = {}

    # Entry-only change (best model from Part 2)
    best_entry_model_name = None
    best_entry_net = -float("inf")
    for name, stats in entry_models.items():
        if stats.get("trades", 0) > 0 and stats.get("net_bps", -999) > best_entry_net:
            best_entry_net = stats["net_bps"]
            best_entry_model_name = name

    combined_results["entry_only_best"] = best_entry_model_name
    combined_results["entry_only_stats"] = entry_models.get(best_entry_model_name or "A_current_veto", {})

    # Exit-only change (best from Part 3)
    best_exit_name = None
    best_exit_net = -float("inf")
    for name, stats in exit_arms.items():
        if stats.get("trades", 0) > 0 and stats.get("net_bps", -999) > best_exit_net:
            best_exit_net = stats["net_bps"]
            best_exit_name = name

    combined_results["exit_only_best"] = best_exit_name
    combined_results["exit_only_stats"] = exit_arms.get(best_exit_name or "control_production", {})

    # Combined entry + exit change
    if best_entry_model_name and best_entry_model_name != "A_current_veto":
        if best_exit_name and best_exit_name != "control_production":
            exit_fn_map = {
                "earlier_protection": exit_earlier_protection,
                "giveback_stall": exit_with_giveback,
                "combined_exit": exit_combined_improved,
            }
            admit_fn_map = {
                "B_ridge_net_ev": admit_by_model(ridge, bars, threshold=0.0) if ridge.fitted else veto_current,
                "C_tree_net_ev": admit_by_model(tree, bars, threshold=0.0) if tree else veto_current,
                "D_downside_aware": admit_downside_aware(ridge, ridge_mae, bars) if ridge.fitted and ridge_mae.fitted else veto_current,
                "E_selection_score": veto_selection_score,
            }
            best_admit = admit_fn_map.get(best_entry_model_name, veto_current)
            best_exit_fn = exit_fn_map.get(best_exit_name)
            if best_exit_fn:
                for ev in events:
                    for inf in ev["inferences"]:
                        inf["_epoch"] = ev["bar_epoch"]
                combo_dts, _, _ = run_arm_detailed(
                    events=events,
                    bars=bars,
                    fourh=fourh,
                    admit=best_admit,
                    exit_override=best_exit_fn,
                )
                combined_results["combined_entry_exit"] = group_stats(combo_dts)
                # Fold validation
                combo_folds = []
                for i, (a, b) in enumerate(folds):
                    fc, _, _ = run_arm_detailed(
                        events=events,
                        bars=bars,
                        fourh=fourh,
                        admit=best_admit,
                        start_epoch=a,
                        end_epoch=b,
                        exit_override=best_exit_fn,
                    )
                    row = group_stats(fc)
                    row["fold"] = i
                    combo_folds.append(row)
                combined_results["combined_folds"] = combo_folds

    results["part3_combined"] = combined_results

    # ================================================================
    # PART 5: Scaling illustration
    # ================================================================
    print("\n=== PART 5: Scaling Illustration ===", flush=True)
    span_days = (t_end - t_start) / 86400.0
    arm_a_stats = results["arm_a_overall"]
    arm_a_trades = arm_a_stats["trades"]
    net_bps = arm_a_stats.get("net_bps", 0.0)

    # Use the best result (may still be negative)
    best_net_bps = net_bps
    best_source = "current_production"
    for name, stats in exit_arms.items():
        if stats.get("net_bps", -999) > best_net_bps and stats.get("trades", 0) > 20:
            best_net_bps = stats["net_bps"]
            best_source = f"exit:{name}"
    for name, stats in entry_models.items():
        if stats.get("net_bps", -999) > best_net_bps and stats.get("trades", 0) > 20:
            best_net_bps = stats["net_bps"]
            best_source = f"entry:{name}"
    if "combined_entry_exit" in combined_results:
        cs = combined_results["combined_entry_exit"]
        if cs.get("net_bps", -999) > best_net_bps and cs.get("trades", 0) > 20:
            best_net_bps = cs["net_bps"]
            best_source = "combined"

    trades_per_day = arm_a_trades / max(1.0, span_days)
    scaling = {}
    for slot_usd in [float(DAY_TARGET_NOTIONAL_PER_SLOT_USD), 625.0, 2500.0, 4000.0]:
        daily_pnl = trades_per_day * (best_net_bps / 1e4) * slot_usd
        monthly_pnl = daily_pnl * 30
        yearly_pnl = daily_pnl * 365
        scaling[f"${int(slot_usd)}/slot"] = {
            "daily_pnl": round(daily_pnl, 2),
            "monthly_pnl": round(monthly_pnl, 2),
            "yearly_pnl": round(yearly_pnl, 2),
            "net_bps": best_net_bps,
            "source": best_source,
        }
    results["part5_scaling"] = scaling
    results["span_days"] = round(span_days, 1)
    results["trades_per_day"] = round(trades_per_day, 2)

    # ================================================================
    # Output
    # ================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str), flush=True)

    conn.close()


if __name__ == "__main__":
    main()
