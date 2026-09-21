"""Production-faithful DAY counterfactual labels, oracle, and root-cause.

Reuses evaluate_engine_managed_exit / replay lifecycle helpers. Ranking losers
are labeled as alternatives and never counted as actual fills.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from backend.config.execution_cost_model import (
    expected_exchange_commission_rt_pct,
    expected_slippage_rt_pct,
    expected_spread_pct,
)
from backend.config.trading_economics import DAY_TARGET_NOTIONAL_PER_SLOT_USD
from backend.services.day_asof_4h import FourHAsOfTracker
from backend.services.day_grouped_decision_ledger import (
    COINS,
    HOLD_SYMBOL,
    CandidateRow,
    DecisionGroup,
)
from backend.services.day_production_lifecycle_replay import (
    HORIZON_PAD_SEC,
    ReplayPos,
    _advance_position,
    _bar_index,
    executable_buy_price,
    get_coin_profile,
    resample_4h,
)

MARKOUT_SEC = (15 * 60, 30 * 60, 60 * 60, 120 * 60, 240 * 60)
ELIGIBLE_ISOLATED = frozenset({"ranking_loser", "selected_execute", "partial_fill", "hold"})
SAFETY_INELIGIBLE = frozenset({"stale_invalid", "spread_impact_ineligible"})
CONSTRAINT_INELIGIBLE = frozenset({"symbol_open", "no_slot_or_capital", "blocked_after_ranking"})
UNKNOWN_CLASSES = frozenset({"no_order_match", "terminal_fill_failure"})
FROZEN_LOCKED = {
    "champion": {"net_usd": -75.70, "net_bps": -15.1, "profit_factor": 0.43},
    "calibrated_score": {"net_usd": -321.30, "profit_factor": 0.25},
    "pooled_ridge": {"net_usd": -315.30, "profit_factor": 0.28},
    "grouped_ranker": "deferred",
    "promote": None,
    "note": "Frozen from prior locked slice. Not re-searched.",
}


@dataclass
class CandidateLabel:
    decision_group_id: str
    symbol: str
    outcome_class: str
    labeled: bool
    label_kind: str
    entry_epoch: float | None
    exit_epoch: float | None
    entry_price: float | None
    exit_price: float | None
    fill_status: str
    filled_qty: float
    latency_sec: float
    gross_markout_bps: dict[str, float | None]
    mfe_bps: float | None
    mae_bps: float | None
    time_to_mfe_sec: float | None
    time_to_mae_sec: float | None
    production_exit_gross_bps: float | None
    commission_bps: float
    spread_bps: float
    slippage_bps: float
    net_bps: float
    net_usd: float
    hold_sec: float
    capital_hours: float
    capture_ratio: float | None
    exit_reason: str
    favorable_first: bool | None
    missing_reason: str = ""
    interval_start: float | None = None
    interval_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroupAttribution:
    decision_group_id: str
    selected_symbol: str
    selected_action: str
    labels: dict[str, CandidateLabel]
    best_eligible_symbol: str
    best_eligible_net_bps: float
    regret_vs_best_bps: float
    regret_vs_hold_bps: float
    opportunity: bool
    selected_positive: bool
    root_cause: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["labels"] = {k: v if isinstance(v, dict) else v.to_dict() for k, v in self.labels.items()}
        return payload


def isolated_eligible(row: CandidateRow) -> bool:
    if row.symbol == HOLD_SYMBOL:
        return True
    if row.outcome_class in SAFETY_INELIGIBLE or row.outcome_class in CONSTRAINT_INELIGIBLE:
        return False
    return row.outcome_class in ELIGIBLE_ISOLATED


def hold_label(group_id: str) -> CandidateLabel:
    return CandidateLabel(
        decision_group_id=group_id,
        symbol=HOLD_SYMBOL,
        outcome_class="hold",
        labeled=True,
        label_kind="hold_zero",
        entry_epoch=None,
        exit_epoch=None,
        entry_price=None,
        exit_price=None,
        fill_status="hold",
        filled_qty=0.0,
        latency_sec=0.0,
        gross_markout_bps={str(s): 0.0 for s in MARKOUT_SEC},
        mfe_bps=0.0,
        mae_bps=0.0,
        time_to_mfe_sec=0.0,
        time_to_mae_sec=0.0,
        production_exit_gross_bps=0.0,
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        net_bps=0.0,
        net_usd=0.0,
        hold_sec=0.0,
        capital_hours=0.0,
        capture_ratio=None,
        exit_reason="HOLD",
        favorable_first=None,
        interval_start=None,
        interval_end=None,
    )


def _markouts(bars: list[tuple[int, float, ...]], entry_epoch: int, entry_px: float) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if entry_px <= 0:
        return {str(s): None for s in MARKOUT_SEC}
    for horizon in MARKOUT_SEC:
        target = entry_epoch + horizon
        idx = _bar_index(bars, target)
        if idx is None:
            out[str(horizon)] = None
            continue
        close = float(bars[idx][4])
        out[str(horizon)] = ((close - entry_px) / entry_px) * 1e4
    return out


def _time_to_mae(bars: list[tuple[int, float, ...]], start: int, end: int, entry_px: float) -> float:
    idx = _bar_index(bars, start)
    if idx is None or entry_px <= 0:
        return 0.0
    trough = entry_px
    t_mae = 0.0
    for j in range(idx, len(bars)):
        ep, low = int(bars[j][0]), float(bars[j][3])
        if ep > end:
            break
        if low < trough:
            trough = low
            t_mae = max(0.0, float(ep) - start)
    return t_mae


def _unlabeled(group_id: str, row: CandidateRow, reason: str) -> CandidateLabel:
    return CandidateLabel(
        decision_group_id=group_id,
        symbol=row.symbol,
        outcome_class=row.outcome_class,
        labeled=False,
        label_kind="unlabeled",
        entry_epoch=None,
        exit_epoch=None,
        entry_price=None,
        exit_price=None,
        fill_status="unlabeled",
        filled_qty=0.0,
        latency_sec=0.0,
        gross_markout_bps={str(s): None for s in MARKOUT_SEC},
        mfe_bps=None,
        mae_bps=None,
        time_to_mfe_sec=None,
        time_to_mae_sec=None,
        production_exit_gross_bps=None,
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        net_bps=0.0,
        net_usd=0.0,
        hold_sec=0.0,
        capital_hours=0.0,
        capture_ratio=None,
        exit_reason="",
        favorable_first=None,
        missing_reason=reason,
    )


def _label_from_actual(row: CandidateRow, series: list[tuple[int, float, ...]], fill: float, sell: dict[str, Any]) -> CandidateLabel:
    exit_px = float(sell.get("price") or 0.0)
    exit_ep = float(sell.get("epoch") or 0.0)
    if fill <= 0 or exit_px <= 0 or exit_ep <= 0:
        return _unlabeled(row.decision_group_id, row, "actual_sell_incomplete")
    comm = expected_exchange_commission_rt_pct()
    spread = expected_spread_pct(row.symbol)
    slip = expected_slippage_rt_pct()
    if sell.get("pnl_pct_net") not in (None, ""):
        net_pct = float(sell["pnl_pct_net"])
        if abs(net_pct) > 2:
            net_pct = net_pct / 100.0
    else:
        net_pct = (exit_px - fill) / fill - comm - spread - slip
    gross = (exit_px - fill) / fill
    hold_sec = float(sell.get("hold_sec") or max(0.0, exit_ep - float(row.epoch)))
    idx = _bar_index(series, int(row.epoch))
    peak = fill
    trough = fill
    t_mfe = 0.0
    t_mae = 0.0
    if idx is not None:
        for j in range(idx, len(series)):
            ep, high, low = int(series[j][0]), float(series[j][2]), float(series[j][3])
            if ep > exit_ep:
                break
            if high > peak:
                peak = high
                t_mfe = max(0.0, float(ep) - float(row.epoch))
            if low < trough:
                trough = low
                t_mae = max(0.0, float(ep) - float(row.epoch))
    mfe = (peak - fill) / fill if fill else 0.0
    mae = (trough - fill) / fill if fill else 0.0
    capture = (net_pct / mfe) if mfe > 1e-12 else None
    notional = float(row.proposed_notional or DAY_TARGET_NOTIONAL_PER_SLOT_USD)
    return CandidateLabel(
        decision_group_id=row.decision_group_id,
        symbol=row.symbol,
        outcome_class=row.outcome_class,
        labeled=True,
        label_kind="actual_production_exit",
        entry_epoch=float(row.epoch),
        exit_epoch=exit_ep,
        entry_price=fill,
        exit_price=exit_px,
        fill_status="actual_fill" if row.outcome_class == "selected_execute" else "partial_fill",
        filled_qty=float(row.extras.get("fill_qty") or 0.0),
        latency_sec=0.0,
        gross_markout_bps=_markouts(series, int(row.epoch), fill),
        mfe_bps=mfe * 1e4,
        mae_bps=mae * 1e4,
        time_to_mfe_sec=t_mfe,
        time_to_mae_sec=t_mae,
        production_exit_gross_bps=gross * 1e4,
        commission_bps=comm * 1e4,
        spread_bps=spread * 1e4,
        slippage_bps=slip * 1e4,
        net_bps=net_pct * 1e4,
        net_usd=net_pct * notional,
        hold_sec=hold_sec,
        capital_hours=hold_sec / 3600.0,
        capture_ratio=capture,
        exit_reason=str(sell.get("exit_reason") or "PRODUCTION_SELL"),
        favorable_first=t_mfe > 0 and (t_mae == 0 or t_mfe <= t_mae),
        interval_start=float(row.epoch),
        interval_end=exit_ep,
    )


def simulate_isolated(
    row: CandidateRow,
    bars: dict[str, list[tuple[int, float, ...]]],
    fourh: dict[str, list[list[float]]],
    *,
    notional: float = DAY_TARGET_NOTIONAL_PER_SLOT_USD,
) -> CandidateLabel:
    if row.symbol == HOLD_SYMBOL:
        return hold_label(row.decision_group_id)
    if row.outcome_class in ("terminal_fill_failure", "fill_failed"):
        lab = _unlabeled(row.decision_group_id, row, "fill_failed")
        lab.labeled = True
        lab.label_kind = "execution_failure"
        lab.fill_status = "failed"
        lab.net_bps = 0.0
        lab.net_usd = 0.0
        return lab
    if not isolated_eligible(row):
        return _unlabeled(row.decision_group_id, row, f"not_isolated_eligible:{row.outcome_class}")
    series = bars.get(row.symbol) or []
    idx = _bar_index(series, row.epoch)
    if idx is None:
        return _unlabeled(row.decision_group_id, row, "missing_1m_bar")
    mid = float(row.mid or series[idx][4] or 0.0)
    if mid <= 0:
        return _unlabeled(row.decision_group_id, row, "nonpositive_mid")
    ask = float(row.best_ask) if row.best_ask not in (None, 0, 0.0) else None
    recorded = row.extras.get("fill_price")
    if recorded not in (None, "", 0, 0.0):
        fill = float(recorded)
    else:
        fill = float(ask) if ask and ask > 0 else executable_buy_price(mid, row.symbol)
    actual = row.extras.get("sell") if row.outcome_class in ("selected_execute", "partial_fill") else None
    if isinstance(actual, dict) and actual.get("price") and actual.get("epoch"):
        return _label_from_actual(row, series, fill, actual)
    profile = get_coin_profile(row.symbol)
    pos = ReplayPos(
        symbol=row.symbol,
        entry_price=fill,
        entry_time=float(row.epoch),
        quantity=float(notional) / fill,
        notional=float(notional),
        stop_price=fill * (1.0 - float(profile["sl"])),
        take_profit_1_price=fill * (1.0 + float(profile["tp"])),
        trail_pct=float(profile["trail"]),
        highest_price=fill,
        lowest_price=fill,
        max_hold_min=int(profile["max_hold_min"]),
        p_buy=float(row.p_buy or 0.0),
        setup="HTF_TREND_PULLBACK",
    )
    sell_cost = expected_exchange_commission_rt_pct() / 2.0 + expected_spread_pct(row.symbol) / 2.0 + expected_slippage_rt_pct() / 2.0
    tracker = FourHAsOfTracker(fourh.get(row.symbol, []), series)
    end_epoch = int(row.epoch + HORIZON_PAD_SEC)
    closed = _advance_position(
        pos,
        series,
        fourh.get(row.symbol, []),
        int(row.epoch) + 1,
        end_epoch,
        sell_cost,
        tracker=tracker,
    )
    if closed is None:
        return _unlabeled(row.decision_group_id, row, "horizon_unclosed")
    capture = None
    if closed.mfe_pct > 1e-12:
        capture = closed.net_pct / closed.mfe_pct
    fav = None
    if closed.time_to_mfe_sec > 0 or closed.hold_sec > 0:
        fav = closed.time_to_mfe_sec > 0 and (closed.mae_pct >= 0 or closed.time_to_mfe_sec <= _time_to_mae(series, int(row.epoch), int(closed.exit_epoch), fill))
    fill_qty = float(row.extras.get("fill_qty") or pos.quantity) if row.outcome_class in ("selected_execute", "partial_fill") else pos.quantity
    fill_status = "actual_fill" if row.outcome_class == "selected_execute" else "counterfactual"
    if row.outcome_class == "partial_fill":
        fill_status = "partial_fill"
    return CandidateLabel(
        decision_group_id=row.decision_group_id,
        symbol=row.symbol,
        outcome_class=row.outcome_class,
        labeled=True,
        label_kind="actual_fill" if fill_status == "actual_fill" else "counterfactual_isolated",
        entry_epoch=closed.entry_epoch,
        exit_epoch=closed.exit_epoch,
        entry_price=closed.entry_price,
        exit_price=closed.exit_price,
        fill_status=fill_status,
        filled_qty=float(fill_qty or 0.0),
        latency_sec=0.0,
        gross_markout_bps=_markouts(series, int(row.epoch), fill),
        mfe_bps=closed.mfe_pct * 1e4,
        mae_bps=closed.mae_pct * 1e4,
        time_to_mfe_sec=closed.time_to_mfe_sec,
        time_to_mae_sec=_time_to_mae(series, int(row.epoch), int(closed.exit_epoch), fill),
        production_exit_gross_bps=closed.gross_pct * 1e4,
        commission_bps=closed.commission_pct * 1e4,
        spread_bps=closed.spread_pct * 1e4,
        slippage_bps=closed.slippage_pct * 1e4,
        net_bps=closed.net_pct * 1e4,
        net_usd=closed.net_usd,
        hold_sec=closed.hold_sec,
        capital_hours=(closed.hold_sec / 3600.0) if closed.hold_sec else 0.0,
        capture_ratio=capture,
        exit_reason=closed.exit_reason,
        favorable_first=fav,
        interval_start=closed.entry_epoch,
        interval_end=closed.exit_epoch,
    )


def label_groups(
    groups: list[DecisionGroup],
    bars: dict[str, list[tuple[int, float, ...]]],
) -> list[GroupAttribution]:
    fourh = {s: resample_4h(bars.get(s, [])) for s in COINS}
    out: list[GroupAttribution] = []
    for group in groups:
        labels: dict[str, CandidateLabel] = {}
        for row in group.candidates:
            labels[row.symbol] = simulate_isolated(row, bars, fourh)
        eligible = [lab for lab in labels.values() if lab.labeled and isolated_eligible(next(c for c in group.candidates if c.symbol == lab.symbol))]
        if not eligible:
            best_sym, best_net = HOLD_SYMBOL, 0.0
        else:
            best = max(eligible, key=lambda lab: (lab.net_bps, 0 if lab.symbol == HOLD_SYMBOL else 1))
            best_sym, best_net = best.symbol, best.net_bps
        selected = labels.get(group.selected_symbol) if group.selected_symbol else labels[HOLD_SYMBOL]
        if selected is None or not selected.labeled:
            selected = labels[HOLD_SYMBOL]
        selected_net = selected.net_bps if selected.labeled else 0.0
        opportunity = any(lab.symbol != HOLD_SYMBOL and lab.labeled and lab.net_bps > 0 for lab in eligible)
        selected_positive = bool(selected.symbol != HOLD_SYMBOL and selected.labeled and selected.net_bps > 0)
        root = _root_cause(group, labels, opportunity, selected_positive, best_sym, selected)
        out.append(
            GroupAttribution(
                decision_group_id=group.decision_group_id,
                selected_symbol=group.selected_symbol or HOLD_SYMBOL,
                selected_action=group.selected_action,
                labels=labels,
                best_eligible_symbol=best_sym,
                best_eligible_net_bps=best_net,
                regret_vs_best_bps=best_net - selected_net,
                regret_vs_hold_bps=0.0 - selected_net,
                opportunity=opportunity,
                selected_positive=selected_positive,
                root_cause=root,
            )
        )
    return out


def _root_cause(
    group: DecisionGroup,
    labels: dict[str, CandidateLabel],
    opportunity: bool,
    selected_positive: bool,
    best_sym: str,
    selected: CandidateLabel,
) -> str:
    selected_row = next((c for c in group.candidates if c.live_selected and c.symbol in COINS), None)
    if selected_row is None:
        return "hold_or_no_selection" if not opportunity else "ranking_error_hold_missed_positive"
    if selected.outcome_class == "terminal_fill_failure":
        return "terminal_fill_failure"
    if selected.outcome_class == "blocked_after_ranking":
        return "blocked_after_ranking"
    if selected.outcome_class == "no_order_match":
        return "unknown_missing_order_evidence"
    if not opportunity:
        if selected.labeled and selected.mfe_bps is not None and selected.mfe_bps <= (selected.commission_bps + selected.spread_bps + selected.slippage_bps):
            return "no_long_opportunity_mfe_below_cost"
        return "no_long_opportunity"
    if not selected_positive and best_sym != selected.symbol:
        return "ranking_error"
    if selected.labeled and selected.mfe_bps is not None and selected.mfe_bps > (selected.commission_bps + selected.spread_bps + selected.slippage_bps) and selected.net_bps <= 0:
        return "exit_failed_to_capture"
    if selected.labeled and selected.production_exit_gross_bps is not None and selected.production_exit_gross_bps > 0 and selected.net_bps <= 0:
        return "costs_erased_gross"
    if selected_positive:
        return "selected_positive"
    return "other"


def summarize_attribution(rows: list[GroupAttribution]) -> dict[str, Any]:
    n = len(rows)
    empty = {
        "groups": n,
        "opportunity_availability": 0.0,
        "selection_quality": None,
        "capture_quality": None,
        "execution_cost_bps": 0.0,
        "oracle_net_bps": 0.0,
        "oracle_net_usd": 0.0,
        "production_net_bps": 0.0,
        "production_net_usd": 0.0,
        "mean_regret_vs_best_bps": 0.0,
        "mean_regret_vs_hold_bps": 0.0,
        "root_causes": {},
        "bought_with_no_positive": 0,
        "ranked_wrong_when_opportunity": 0,
        "positive_mfe_negative_exit": 0,
        "gross_positive_net_negative": 0,
        "hindsight_positive_paths": False,
        "oracle_claim": "positive_ex_post_paths_only",
    }
    if n == 0:
        return empty
    opp = [r for r in rows if r.opportunity]
    selected_on_opp = [r for r in opp if r.selected_positive]
    capture_rows = []
    exec_cost = []
    oracle_nets = []
    prod_nets = []
    causes: dict[str, int] = defaultdict(int)
    bought_none = 0
    ranked_wrong = 0
    mfe_miss = 0
    cost_erase = 0
    for row in rows:
        causes[row.root_cause] += 1
        best = row.labels.get(row.best_eligible_symbol)
        sel = row.labels.get(row.selected_symbol) or row.labels.get(HOLD_SYMBOL)
        if best and best.labeled:
            oracle_nets.append(best.net_bps)
        if sel and sel.labeled:
            prod_nets.append(sel.net_bps)
            exec_cost.append(sel.commission_bps + sel.spread_bps + sel.slippage_bps)
            if sel.symbol != HOLD_SYMBOL and sel.mfe_bps is not None and sel.mfe_bps > 0 and sel.capture_ratio is not None:
                capture_rows.append(sel.capture_ratio)
            if sel.symbol != HOLD_SYMBOL and not row.opportunity:
                bought_none += 1
            if row.opportunity and not row.selected_positive and row.best_eligible_symbol != row.selected_symbol:
                ranked_wrong += 1
            if sel.symbol != HOLD_SYMBOL and sel.mfe_bps is not None and sel.mfe_bps > (sel.commission_bps + sel.spread_bps + sel.slippage_bps) and sel.net_bps <= 0:
                mfe_miss += 1
            if sel.production_exit_gross_bps is not None and sel.production_exit_gross_bps > 0 and sel.net_bps <= 0:
                cost_erase += 1
    oracle_mean = sum(oracle_nets) / len(oracle_nets) if oracle_nets else 0.0
    return {
        "groups": n,
        "opportunity_availability": len(opp) / n,
        "selection_quality": (len(selected_on_opp) / len(opp)) if opp else None,
        "capture_quality": (sum(capture_rows) / len(capture_rows)) if capture_rows else None,
        "execution_cost_bps": (sum(exec_cost) / len(exec_cost)) if exec_cost else 0.0,
        "oracle_net_bps": oracle_mean,
        "oracle_net_usd": sum((rows[i].labels[rows[i].best_eligible_symbol].net_usd) for i in range(n) if rows[i].best_eligible_symbol in rows[i].labels),
        "production_net_bps": (sum(prod_nets) / len(prod_nets)) if prod_nets else 0.0,
        "production_net_usd": sum((r.labels.get(r.selected_symbol) or r.labels[HOLD_SYMBOL]).net_usd for r in rows if (r.labels.get(r.selected_symbol) or r.labels.get(HOLD_SYMBOL))),
        "mean_regret_vs_best_bps": sum(r.regret_vs_best_bps for r in rows) / n,
        "mean_regret_vs_hold_bps": sum(r.regret_vs_hold_bps for r in rows) / n,
        "root_causes": dict(causes),
        "bought_with_no_positive": bought_none,
        "ranked_wrong_when_opportunity": ranked_wrong,
        "positive_mfe_negative_exit": mfe_miss,
        "gross_positive_net_negative": cost_erase,
        "hindsight_positive_paths": oracle_mean > 0.0,
        "oracle_claim": ("Positive ex-post candidate paths under the simulator. The oracle does not establish predictability at decision time."),
        "exclusive_waterfall": exclusive_group_waterfall(rows),
        "capture_definitions": capture_definitions(rows),
        "cost_report": cost_report(rows),
    }


EXCLUSIVE_BUCKETS = (
    "positive_selected_best",
    "positive_selected_inferior_positive",
    "positive_selected_negative",
    "positive_selected_hold",
    "no_positive_bought",
    "no_positive_held",
    "unknown_missing_labels",
)


def _group_unknown(attr: GroupAttribution) -> bool:
    sel = attr.labels.get(attr.selected_symbol) or attr.labels.get(HOLD_SYMBOL)
    if sel is None:
        return True
    if sel.outcome_class in UNKNOWN_CLASSES:
        return True
    if attr.selected_symbol in COINS and not sel.labeled and sel.missing_reason:
        return True
    labeled_coins = [lab for lab in attr.labels.values() if lab.symbol in COINS and lab.labeled]
    return bool(not labeled_coins and attr.selected_symbol in COINS)


def exclusive_group_waterfall(rows: list[GroupAttribution]) -> dict[str, Any]:
    counts = dict.fromkeys(EXCLUSIVE_BUCKETS, 0)
    for attr in rows:
        if _group_unknown(attr):
            counts["unknown_missing_labels"] += 1
            continue
        bought = bool(attr.selected_symbol in COINS)
        if attr.opportunity:
            if bought and attr.selected_symbol == attr.best_eligible_symbol:
                counts["positive_selected_best"] += 1
            elif bought and attr.selected_positive and attr.selected_symbol != attr.best_eligible_symbol:
                counts["positive_selected_inferior_positive"] += 1
            elif bought and not attr.selected_positive:
                counts["positive_selected_negative"] += 1
            else:
                counts["positive_selected_hold"] += 1
        elif bought:
            counts["no_positive_bought"] += 1
        else:
            counts["no_positive_held"] += 1
    n = len(rows)
    if sum(counts.values()) != n:
        counts["integrity_error"] = n - sum(counts.values())
    known = n - counts["unknown_missing_labels"]
    opp_n = counts["positive_selected_best"] + counts["positive_selected_inferior_positive"] + counts["positive_selected_negative"] + counts["positive_selected_hold"]
    return {
        "counts": counts,
        "sum": sum(counts.values()),
        "groups": n,
        "mutually_exclusive": True,
        "opportunity_availability": {
            "pct": (opp_n / known) if known else None,
            "numerator": opp_n,
            "denominator": known,
            "denominator_note": "groups with sufficient labels; unknown excluded",
        },
        "selection_quality": {
            "pct": (counts["positive_selected_best"] / opp_n) if opp_n else None,
            "numerator": counts["positive_selected_best"],
            "denominator": opp_n,
            "denominator_note": "groups with a positive eligible long",
        },
    }


def capture_definitions(rows: list[GroupAttribution]) -> dict[str, Any]:
    nets = []
    mfes = []
    ratios_net = []
    ratios_gross = []
    for attr in rows:
        sel = attr.labels.get(attr.selected_symbol)
        if sel is None or sel.symbol == HOLD_SYMBOL or not sel.labeled:
            continue
        if sel.mfe_bps is None or sel.mfe_bps <= 0:
            continue
        nets.append(sel.net_bps)
        mfes.append(sel.mfe_bps)
        ratios_net.append(sel.net_bps / sel.mfe_bps)
        if sel.production_exit_gross_bps is not None:
            ratios_gross.append(sel.production_exit_gross_bps / sel.mfe_bps)
    return {
        "net_capture_mean_of_ratios": (sum(ratios_net) / len(ratios_net)) if ratios_net else None,
        "gross_capture_mean_of_ratios": (sum(ratios_gross) / len(ratios_gross)) if ratios_gross else None,
        "net_capture_aggregate": (sum(nets) / sum(mfes)) if mfes and sum(mfes) else None,
        "n": len(ratios_net),
        "unit": "ratio (not percent)",
        "definition": "mean of (exit_bps / MFE_bps) on selected filled/labeled coins with MFE>0; also aggregate sum(exit)/sum(MFE)",
    }


def _total_cost_usd(closed: list[CandidateLabel], all_in: list[float]) -> float:
    total = 0.0
    for lab, bps in zip(closed, all_in, strict=True):
        if lab.net_bps:
            notional = abs(lab.net_usd / (lab.net_bps / 1e4))
        else:
            notional = float(DAY_TARGET_NOTIONAL_PER_SLOT_USD)
        total += (bps / 1e4) * notional
    return total


def cost_report(rows: list[GroupAttribution]) -> dict[str, Any]:
    closed = []
    for attr in rows:
        sel = attr.labels.get(attr.selected_symbol)
        if sel is None or sel.symbol == HOLD_SYMBOL or not sel.labeled:
            continue
        closed.append(sel)
    n_closed = len(closed)
    n_groups = len(rows)
    comm = [lab.commission_bps for lab in closed]
    spread = [lab.spread_bps for lab in closed]
    slip = [lab.slippage_bps for lab in closed]
    all_in = [lab.commission_bps + lab.spread_bps + lab.slippage_bps for lab in closed]
    per_group_all_in = []
    for attr in rows:
        sel = attr.labels.get(attr.selected_symbol) if attr.selected_symbol in COINS else None
        if sel and sel.labeled:
            per_group_all_in.append(sel.commission_bps + sel.spread_bps + sel.slippage_bps)
        else:
            per_group_all_in.append(0.0)
    return {
        "commission_bps_per_filled_side_model": (sum(comm) / n_closed / 2.0) if n_closed else None,
        "round_trip_commission_bps_per_closed_trade": (sum(comm) / n_closed) if n_closed else None,
        "entry_spread_slippage_bps_model_half": (sum(spread) / n_closed / 2.0 + sum(slip) / n_closed / 2.0) if n_closed else None,
        "exit_spread_slippage_bps_model_half": (sum(spread) / n_closed / 2.0 + sum(slip) / n_closed / 2.0) if n_closed else None,
        "all_in_round_trip_bps_per_trade": (sum(all_in) / n_closed) if n_closed else None,
        "all_in_bps_averaged_per_decision_group_including_hold": (sum(per_group_all_in) / n_groups) if n_groups else None,
        "all_in_bps_per_group_note": "HOLD and unlabeled selected groups contribute 0. This is not round-trip trade cost.",
        "total_cost_usd": _total_cost_usd(closed, all_in),
        "closed_trades": n_closed,
        "groups": n_groups,
        "accepted_audit_rt_commission_bps": [3.87, 4.00],
        "accepted_audit_commission_plus_exit_slip_bps": 4.6,
        "accepted_audit_all_in_with_spread_bps": [6.0, 7.6],
    }


def null_oracle_distribution(
    groups: list[DecisionGroup],
    attributions: list[GroupAttribution],
    *,
    n_perm: int = 200,
    block: int = 8,
    seed: int = 7,
) -> dict[str, Any]:
    """Block-permute group label vectors to disconnect outcomes from as-of features."""
    import random

    rng = random.Random(seed)
    observed = portfolio_replay_from_policy(groups, attributions, picker=oracle_picker)
    by_id = {a.decision_group_id: a for a in attributions}
    vectors = []
    for group in groups:
        attr = by_id[group.decision_group_id]
        vectors.append({sym: attr.labels[sym] for sym in attr.labels})
    null_usd = []
    null_opp = []
    for _ in range(n_perm):
        shuffled = list(vectors)
        for i in range(0, len(shuffled), block):
            chunk = shuffled[i : i + block]
            rng.shuffle(chunk)
            shuffled[i : i + block] = chunk
        perm_attrs = []
        for group, labs in zip(groups, shuffled, strict=True):
            src = by_id[group.decision_group_id]
            clone = GroupAttribution(
                decision_group_id=group.decision_group_id,
                selected_symbol=src.selected_symbol,
                selected_action=src.selected_action,
                labels=labs,
                best_eligible_symbol=max(labs.values(), key=lambda lab: (lab.net_bps if lab.labeled else -1e18, 0 if lab.symbol == HOLD_SYMBOL else 1)).symbol,
                best_eligible_net_bps=max((lab.net_bps for lab in labs.values() if lab.labeled), default=0.0),
                regret_vs_best_bps=0.0,
                regret_vs_hold_bps=0.0,
                opportunity=any(lab.symbol != HOLD_SYMBOL and lab.labeled and lab.net_bps > 0 for lab in labs.values()),
                selected_positive=False,
                root_cause="null",
            )
            perm_attrs.append(clone)
        port = portfolio_replay_from_policy(groups, perm_attrs, picker=oracle_picker)
        null_usd.append(port["net_usd"])
        null_opp.append(sum(1 for a in perm_attrs if a.opportunity) / len(perm_attrs) if perm_attrs else 0.0)
    null_usd_sorted = sorted(null_usd)
    obs = observed["net_usd"]
    rank = sum(1 for x in null_usd_sorted if x <= obs)
    return {
        "observed_oracle": observed,
        "null_mean_usd": sum(null_usd) / len(null_usd) if null_usd else None,
        "null_p05_usd": null_usd_sorted[int(0.05 * (len(null_usd_sorted) - 1))] if null_usd_sorted else None,
        "null_p50_usd": null_usd_sorted[len(null_usd_sorted) // 2] if null_usd_sorted else None,
        "null_p95_usd": null_usd_sorted[int(0.95 * (len(null_usd_sorted) - 1))] if null_usd_sorted else None,
        "observed_percentile": (100.0 * rank / len(null_usd)) if null_usd else None,
        "opportunity_rate_observed": sum(1 for a in attributions if a.opportunity) / len(attributions) if attributions else None,
        "opportunity_rate_null_mean": sum(null_opp) / len(null_opp) if null_opp else None,
        "n_perm": n_perm,
        "block": block,
        "predecision_incremental_information": "not_established_by_oracle",
        "note": "Hindsight best-of-four remains a label-only rule under permutation. A pre-decision challenger on a locked test is required to claim predictability.",
    }


def portfolio_replay_from_policy(
    groups: list[DecisionGroup],
    attributions: list[GroupAttribution],
    *,
    picker,
) -> dict[str, Any]:
    """Chronological one-position-per-symbol / four-slot replay of a picker."""
    by_id = {a.decision_group_id: a for a in attributions}
    open_until: dict[str, float] = {}
    closed_nets: list[float] = []
    closed_usd: list[float] = []
    peak = 0.0
    eq = 0.0
    dd = 0.0
    n_trades = 0
    for group in groups:
        attr = by_id.get(group.decision_group_id)
        if attr is None:
            continue
        now = float(group.epoch)
        open_until = {s: t for s, t in open_until.items() if t > now}
        choice = picker(group, attr, set(open_until))
        if choice == HOLD_SYMBOL or choice in open_until or len(open_until) >= 4:
            continue
        lab = attr.labels.get(choice)
        if lab is None or not lab.labeled or lab.symbol == HOLD_SYMBOL or lab.interval_end is None:
            continue
        n_trades += 1
        open_until[choice] = float(lab.interval_end)
        closed_nets.append(lab.net_bps)
        closed_usd.append(lab.net_usd)
        eq += lab.net_bps
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    wins = [x for x in closed_nets if x > 0]
    losses = [x for x in closed_nets if x <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    return {
        "trades": n_trades,
        "net_bps": (sum(closed_nets) / len(closed_nets)) if closed_nets else 0.0,
        "net_usd": sum(closed_usd),
        "profit_factor": (gp / gl) if gl > 0 else (None if not wins else float("inf")),
        "max_drawdown_bps": dd,
        "hold_count": sum(1 for g in groups if picker(g, by_id[g.decision_group_id], set()) == HOLD_SYMBOL) if groups else 0,
    }


def champion_picker(group: DecisionGroup, attr: GroupAttribution, _open: set[str]) -> str:
    if group.selected_symbol in COINS:
        return group.selected_symbol
    return HOLD_SYMBOL


def oracle_picker(_group: DecisionGroup, attr: GroupAttribution, open_set: set[str]) -> str:
    ranked = sorted(
        (lab for lab in attr.labels.values() if lab.labeled and lab.symbol != HOLD_SYMBOL and lab.symbol not in open_set),
        key=lambda lab: lab.net_bps,
        reverse=True,
    )
    for lab in ranked:
        if lab.net_bps > 0:
            return lab.symbol
    return HOLD_SYMBOL
