"""Live open-scalp lifecycle fields for dashboard / positions API."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.exit_manager import (
    _max_hold_hard_sec,
    _review_trigger_sec,
    _scratch_min_hold_sec,
    evaluate_exit,
    track_from_row,
)
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight


def _stale_review_due(
    *,
    hold_sec: float,
    stale_timeout_sec: float,
    stale_review_count: int,
    last_review_ts: Any,
    now_epoch: float,
    review_interval_sec: int,
) -> bool:
    if hold_sec < stale_timeout_sec:
        return False
    if stale_review_count == 0:
        return True
    last_review_epoch = 0.0
    try:
        if last_review_ts:
            last_review_epoch = datetime.fromisoformat(str(last_review_ts).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        last_review_epoch = 0.0
    return (now_epoch - last_review_epoch) >= review_interval_sec


def _next_exit_trigger(
    *,
    hold_sec: float,
    profit_hit: bool,
    review: Any,
    econ,
) -> str:
    if profit_hit:
        return "NET_PROFIT_TARGET (ready)"
    hard = _max_hold_hard_sec(econ)
    trigger = _review_trigger_sec(econ)
    scratch_min = _scratch_min_hold_sec()
    if review and getattr(review, "decision", None) == "SELL" and review.exit_reason:
        return str(review.exit_reason)
    if hold_sec >= hard:
        return "MAX_HOLD_HARD_LIMIT"
    if hold_sec >= trigger:
        return "exit_review (momentum/setup/scratch)"
    if hold_sec >= scratch_min:
        return "EARLY_SCRATCH watch"
    return "profit_target_or_review"


def enrich_open_scalp_positions(
    open_rows: list[dict[str, Any]],
    *,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    if not open_rows:
        return []
    config = get_scalp_config()
    econ = economics_for_config(config)
    reader = ScalpMarketReader(config)
    tracker = MomentumTracker()

    now = time.time()
    out: list[dict[str, Any]] = []
    for row in open_rows:
        enriched = dict(row)
        sym = str(row.get("symbol") or "")
        entry = float(row.get("entry_price") or 0.0)
        qty = float(row.get("quantity") or 0.0)
        hold_sec = float(row.get("hold_seconds") or 0.0)
        pos_diag: dict = {}
        raw_diag = row.get("diagnostics_json")
        if raw_diag and isinstance(raw_diag, str):
            try:
                pos_diag = json.loads(raw_diag)
            except json.JSONDecodeError:
                pos_diag = {}
        elif isinstance(raw_diag, dict):
            pos_diag = raw_diag

        setup = str(pos_diag.get("setup_name") or pos_diag.get("scalp_setup") or row.get("setup") or "")
        enriched["setup"] = setup or None
        enriched["regime"] = str(pos_diag.get("micro_regime") or "")

        target_pct = econ.net_profit_target_pct
        setup_sig = pos_diag.get("setup_signal") or {}
        entry_buy_impact = float(setup_sig.get("impact_pct") or 0.0)
        target_pct = float(setup_sig.get("required_target_pct") or target_pct)

        snap = reader.read(sym)
        if snap is None:
            enriched.update(
                {
                    "executable_net_pnl_usd": None,
                    "executable_net_pct": None,
                    "target_gap_pct": None,
                    "lifecycle_state": row.get("state") or "OPEN",
                    "lifecycle_reason": row.get("last_state_reason") or "no_market_data",
                    "next_exit_trigger": "market_data_unavailable",
                }
            )
            out.append(enriched)
            continue

        tracker.record(sym, now, snap.best_bid, snap.mid)
        mom = tracker.diagnostics(sym, now, snap.best_bid, snap.mid)
        pf = run_scalp_preflight(
            snap,
            econ,
            config,
            side="SELL",
            entry_price=entry,
            entry_buy_impact_pct=entry_buy_impact,
            quantity=qty,
            check_paper_enabled=False,
        )
        exit_price = pf.expected_avg_fill if pf.expected_avg_fill > 0 else pf.limit_sell_price
        net_pct = pf.expected_net_edge_pct
        net_usd = (exit_price - entry) * qty - (
            exit_price * qty * econ.taker_fee_pct + exit_price * qty * econ.slippage_buffer_pct + entry * qty * econ.taker_fee_pct + entry * qty * econ.slippage_buffer_pct
        )
        profit_hit = net_pct >= target_pct
        exit_spread_ok = pf.reject_reason != "SPREAD_TOO_WIDE"
        track = track_from_row(row, pos_diag)
        review_interval = int(os.getenv("SCALP_REVIEW_INTERVAL_SEC", "30"))
        perform_review = _stale_review_due(
            hold_sec=hold_sec,
            stale_timeout_sec=econ.stale_scalp_timeout_sec,
            stale_review_count=track.stale_review_count,
            last_review_ts=row.get("last_review_ts"),
            now_epoch=now,
            review_interval_sec=review_interval,
        )
        review = evaluate_exit(
            track=track,
            snap=snap,
            mom=mom,
            econ=econ,
            config=config,
            trade_id=str(row.get("trade_id") or ""),
            hold_sec=hold_sec,
            executable_net_pct=net_pct,
            profit_hit=profit_hit,
            exit_spread_ok=exit_spread_ok,
            perform_review=perform_review,
        )

        enriched.update(
            {
                "executable_net_pnl_usd": round(net_usd, 4),
                "executable_net_pct": round(net_pct, 6),
                "target_pct": round(target_pct, 6),
                "target_gap_pct": round(max(0.0, target_pct - net_pct), 6),
                "max_favorable_pct": round(review.diagnostics.get("max_favorable_pct") or 0.0, 6),
                "lifecycle_state": review.state,
                "lifecycle_reason": review.reason,
                "next_exit_trigger": _next_exit_trigger(
                    hold_sec=hold_sec,
                    profit_hit=profit_hit,
                    review=review,
                    econ=econ,
                ),
                "exit_decision_preview": review.decision,
                "exit_reason_preview": review.exit_reason,
            }
        )
        out.append(enriched)
    return out


__all__ = ["enrich_open_scalp_positions"]
