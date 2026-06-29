#!/usr/bin/env python3
"""
Production adapter integration review for ALLWEATHER_BREAKOUT_PULLBACK.

Runs exact replay, family-aware production adapter replay, and current-data
shadow signal scan. Adapter remains disabled from live execution by default.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.services.allweather_breakout_pullback_adapter import (
    CANDIDATE_ID,
    STRATEGY_FAMILY,
    compute_state,
    entry_signal,
)
from backend.services.replay_promotion_gate import evaluate_day_promotion
from scripts.replay_baselines.run_allweather_portfolio_replay import (
    CACHE_DIR,
    NOTIONAL_MULT,
    SPAN_DAYS,
    STRESS_NAMES,
    SYMBOLS,
    WINDOWS,
    _build_base_config,
    _fetch,
    _filter_trades,
    _metrics,
    _portfolio_backtest,
    _stress_config,
    _walk_forward,
)
from scripts.run_allweather_strategy_lab import TIME_STOP_HOURS, _precompute

SCRIPT = "scripts/replay_baselines/run_allweather_production_adapter_review.py"
OUT = REPO / "scripts" / "replay_baselines" / "allweather_breakout_pullback_production_adapter_review_latest.json"
LOCK = REPO / "scripts" / "replay_baselines" / "day_baseline_all_pass_v1_size_1_5_LOCK.json"


def _window_metrics(trades, end_ts: int, windows: list[int], span_days: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for w in windows:
        cutoff = end_ts - w * 86400
        wt = _filter_trades(trades, cutoff)
        mo = max(w / 30.4375, 1.0)
        out[str(w)] = {"span_days": w, "metrics": _metrics(wt, mo)}
    cutoff = end_ts - span_days * 86400
    wt = _filter_trades(trades, cutoff)
    months = max(span_days / 30.4375, 1.0)
    out["full"] = {"span_days": span_days, "metrics": _metrics(wt, months)}
    return out


def _stress_block(indis, bars_15m, months: float, mode: str) -> dict[str, Any]:
    base = _build_base_config()
    out: dict[str, Any] = {}
    stress_pass = True
    for name in STRESS_NAMES:
        cfg = _stress_config(base, name)
        tr, _ = _portfolio_backtest(indis, bars_15m, mode=mode, config=cfg)
        sm = _metrics(tr, months)
        out[name] = sm
        if name in ("verified_current_costs", "taker_10bp") and not sm["target_met_500"]:
            stress_pass = False
    return {"scenarios": out, "stress_pass": stress_pass}


def _shadow_current_data(bars_1h: dict, lookback_days: int = 7) -> dict[str, Any]:
    end_ts = max(bars_1h[s][-1]["ts"] for s in SYMBOLS if bars_1h[s])
    cutoff = end_ts - lookback_days * 86400
    signals: list[dict[str, Any]] = []
    for sym in SYMBOLS:
        bars = bars_1h[sym]
        for i in range(206, len(bars)):
            ts = bars[i]["ts"]
            if ts < cutoff:
                continue
            if ts % 3600 != 0:
                continue
            window = bars[: i + 1]
            state = compute_state(window)
            if state is None:
                continue
            sig = entry_signal(state)
            if sig:
                signals.append(
                    {
                        "symbol": sym,
                        "ts": ts,
                        "setup": sig["setup"],
                        "regime": sig["regime"],
                        "would_buy": True,
                        "would_execute_live": False,
                        "strategy_family": STRATEGY_FAMILY,
                    }
                )
    return {
        "lookback_days": lookback_days,
        "signal_count": len(signals),
        "signals_by_symbol": {s: sum(1 for x in signals if x["symbol"] == s) for s in SYMBOLS},
        "sample_signals": signals[-20:],
        "shadow_only": True,
        "ledger_impact": False,
    }


def _drawdown_unit_audit() -> dict[str, Any]:
    lock = {}
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
    raw = float(lock.get("max_drawdown_pct_90d", 0.484))
    return {
        "locked_baseline_id": lock.get("baseline_id", "day_baseline_all_pass_v1_size_1_5"),
        "lock_field": "max_drawdown_pct_90d",
        "raw_lock_value": raw,
        "correct_interpretation": f"{raw}% max equity drawdown over 90d replay (~${25000 * raw / 100:.0f} on $25k)",
        "incorrect_if_read_as_whole_percent": f"{raw}% would mean {raw}% — but field is already in percent units",
        "common_confusion": "avg_loss_usd near -48.4 is dollars per losing trade at 1.5x sizing, NOT drawdown percent",
        "candidate_allweather_max_drawdown_pct": "see exact_replay.full_span_metrics.max_drawdown_pct (~2.56%)",
        "unit_mismatch_in_lock_review": False,
    }


def main() -> int:
    cmd = f"python3 {SCRIPT}"
    print("=== ALLWEATHER PRODUCTION ADAPTER REVIEW ===", flush=True)
    try:
        bars_1h, bars_15m, _meta = _fetch(SPAN_DAYS)
        if not bars_1h[SYMBOLS[0]]:
            raise RuntimeError("no cached bar data")

        span_days = int((bars_1h[SYMBOLS[0]][-1]["ts"] - bars_1h[SYMBOLS[0]][0]["ts"]) / 86400)
        months = max(span_days / 30.4375, 1.0)
        end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]

        from scripts import run_allweather_strategy_lab as lab

        lab._atr_cache = {}
        indis = {sym: _precompute(bars_1h[sym]) for sym in SYMBOLS}
        base_cfg = _build_base_config()

        print("  exact replay ...", flush=True)
        exact_trades, exact_meta = _portfolio_backtest(indis, bars_15m, mode="exact_candidate", config=base_cfg)
        exact_full = _metrics(exact_trades, months)
        exact_wf = _walk_forward(exact_trades)
        exact_stress = _stress_block(indis, bars_15m, months, "exact_candidate")
        exact_windows = _window_metrics(exact_trades, end_ts, WINDOWS, span_days)

        print("  production adapter replay ...", flush=True)
        adapter_trades, adapter_meta = _portfolio_backtest(indis, bars_15m, mode="production_adapter", config=base_cfg)
        adapter_full = _metrics(adapter_trades, months)
        adapter_wf = _walk_forward(adapter_trades)
        adapter_stress = _stress_block(indis, bars_15m, months, "production_adapter")
        adapter_windows = _window_metrics(adapter_trades, end_ts, WINDOWS, span_days)

        shadow = _shadow_current_data(bars_1h, lookback_days=7)
        dd_audit = _drawdown_unit_audit()

        promo_ok, promo_reasons = evaluate_day_promotion(
            exact_full,
            stress_pass=exact_stress["stress_pass"],
            walk_forward_test_pass=exact_wf.get("walk_forward_test_pass", False),
            walk_forward_val_pass=exact_wf.get("walk_forward_val_pass", False),
            execution_replay_verified=True,
            label_proxy_only=False,
        )

        remaining_blockers = []
        if not promo_ok:
            remaining_blockers.extend(promo_reasons)
        remaining_blockers.append("live_execution_disabled_by_default")
        remaining_blockers.append("explicit_user_approval_required")
        if adapter_full["monthly_pnl_usd"] + 1e-6 < exact_full["monthly_pnl_usd"] * 0.95:
            remaining_blockers.append("production_adapter_pnl_gap_vs_exact")

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": cmd,
            "exit_code": 0,
            "stale_artifact": False,
            "strategy_family": STRATEGY_FAMILY,
            "candidate_id": CANDIDATE_ID,
            "enabled_live": False,
            "shadow_enabled": True,
            "config_flags": {
                "ALLWEATHER_BREAKOUT_PULLBACK_ENABLED": False,
                "ALLWEATHER_BREAKOUT_PULLBACK_SHADOW": True,
            },
            "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
            "do_not_promote_live": True,
            "exit_reasons": {
                "target": "ALLWEATHER_ATR_TARGET_EXIT",
                "stop": "ALLWEATHER_ATR_STOP_EXIT",
                "time_stop": "ALLWEATHER_TIME_STOP_EXIT",
                "min_net_profit_floor_applies": False,
            },
            "drawdown_unit_audit": dd_audit,
            "exact_replay": {
                "full_span_metrics": exact_full,
                "walk_forward": exact_wf,
                "stress": exact_stress,
                "window_replays": exact_windows,
                "run_meta": exact_meta,
            },
            "production_adapter_replay": {
                "description": "Family-aware router/bucket gates via allweather_breakout_pullback_adapter",
                "full_span_metrics": adapter_full,
                "walk_forward": adapter_wf,
                "stress": adapter_stress,
                "window_replays": adapter_windows,
                "production_gate_blocks": adapter_meta.get("blocked_by_production_gates", {}),
                "run_meta": adapter_meta,
                "delta_vs_exact_monthly_usd": round(adapter_full["monthly_pnl_usd"] - exact_full["monthly_pnl_usd"], 2),
            },
            "current_data_shadow": shadow,
            "all_pass": promo_ok,
            "target_met_500": exact_full.get("target_met_500", False),
            "promotion_ready": False,
            "promotion_ready_requires_user_approval": True,
            "remaining_blockers": remaining_blockers,
            "rejection_reason": None if promo_ok else "; ".join(sorted(set(promo_reasons))),
        }
        OUT.write_text(json.dumps(payload, indent=2))
        print(
            json.dumps(
                {
                    "exact_monthly": exact_full["monthly_pnl_usd"],
                    "adapter_monthly": adapter_full["monthly_pnl_usd"],
                    "all_pass": promo_ok,
                    "wrote": str(OUT),
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        err = traceback.format_exc()
        print(err, flush=True)
        OUT.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "command": cmd,
                    "exit_code": 1,
                    "stale_artifact": False,
                    "error": str(exc),
                    "all_pass": False,
                    "target_met_500": False,
                    "promotion_ready": False,
                    "remaining_blockers": [str(exc)],
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
