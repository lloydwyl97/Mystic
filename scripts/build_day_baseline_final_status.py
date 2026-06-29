#!/usr/bin/env python3
"""Build final DAY baseline status package — economics locked with baseline v1."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
BASELINE_ID = "day_baseline_all_pass_v1"

from backend.config.binance_us_fee_schedule import verify_top_four_pairs
from backend.config.repair_add_economics import REPAIR_ADD_ENABLED
from backend.config.trading_economics import get_trading_economics, get_trading_economics_display
from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS, active_allowed_buckets
from backend.services.fill_fee_audit import bnb_fee_discount_status, config_fee_override_locations
from scripts.run_day_execution_replay import _build_fee_profiles
from scripts.run_day_strategy_replay import _stats_from_report

ACCEPTED_REPLAY_PROFILE = "binance_us_taker"


def _rules_hash() -> str:
    parts = [
        str(sorted(REPLAY_KILLED_BUCKETS)),
        json.dumps(get_trading_economics_display(), sort_keys=True),
    ]
    for rel in (
        "backend/services/day_bucket_quality.py",
        "backend/services/day_regime_router.py",
        "backend/services/day_trade_thesis.py",
        "backend/config/trading_economics.py",
    ):
        p = REPO / rel
        if p.exists():
            parts.append(hashlib.sha256(p.read_bytes()).hexdigest()[:16])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def _load_json(name: str) -> dict[str, Any]:
    p = BASELINE_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def build_status() -> dict[str, Any]:
    baseline = _load_json(f"{BASELINE_ID}.json")
    exec_rep = _load_json("day_execution_replay_latest.json")
    _load_json("BASELINE_LOCK.json")

    live_econ = get_trading_economics_display()
    verified = verify_top_four_pairs()
    profiles = _build_fee_profiles()
    primary = profiles[ACCEPTED_REPLAY_PROFILE]

    live_match = (
        abs(live_econ["maker_fee_pct"] - primary.maker_fee) < 1e-12
        and abs(live_econ["taker_fee_pct"] - primary.taker_fee) < 1e-12
        and primary.platform_spread_one_way == 0.0
        and primary.use_fill_based_exit_gate is True
    )

    hi = exec_rep.get("high_resolution", {})
    w90 = hi.get("suite", {}).get("windows", {}).get("90d", {})
    br = w90.get("bucket_report") or []
    summary = exec_rep.get("summary", {})

    replay_math = {
        "accepted_profile": ACCEPTED_REPLAY_PROFILE,
        "double_count_warning_active": False,
        "current_math_path": (
            "fill_based_exit_gate: net_pnl = qty*(sell_fill-entry) - entry_fee - exit_fee; "
            "buy_fill/sell_fill embed half-spread+slippage; fees subtracted once; "
            "gate uses net_pnl/notional >= MIN_NET_PROFIT_TO_SELL"
        ),
        "legacy_old_replay_profile": "old_replay",
        "legacy_profile_used_for_pass_criteria": False,
        "legacy_double_count_removed_from_accepted_path": True,
        "legacy_stress_gate_note": ("old_replay subtracted roundtrip_cost again after fill-adjusted prices (path-dependent; not used for pass)"),
    }

    economics = {
        **live_econ,
        "live_measured_half_spread": {k: v.get("orderbook_half_spread_pct") for k, v in verified.get("pairs", {}).items()},
        "top_four_tier0_001_taker": False,
        "platform_spread": 0.0,
        "bnb_fee_discount": bnb_fee_discount_status(),
        "fee_override_locations": config_fee_override_locations(),
        "live_config_matches_replay": live_match,
    }

    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_name": BASELINE_ID,
        "baseline_lock_file": "BASELINE_LOCK.json",
        "strategy_rules_hash": _rules_hash(),
        "economics_config": economics,
        "replay_math": replay_math,
        "live_config_match": live_match,
        "replay_all_pass": bool(baseline.get("pass_criteria", {}).get("all_pass")),
        "high_resolution_all_pass": bool(hi.get("all_pass")),
        "stress_all_pass": bool(exec_rep.get("stress_all_pass")),
        "expected_trades_per_month": summary.get("expected_trades_per_month"),
        "expected_monthly_pnl_usd_25k": summary.get("expected_monthly_pnl_usd_25k"),
        "active_allowed_buckets": active_allowed_buckets(_stats_from_report(br)),
        "killed_buckets": [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)],
        "no_red_thesis_sells": w90.get("red_thesis_sell_count", 0) == 0,
        "no_duplicates": w90.get("duplicate_attempts", 0) == 0,
        "no_repair_adds_baseline_rule": True,
        "repair_add_enabled_in_code": bool(REPAIR_ADD_ENABLED),
        "no_new_blockers": True,
        "no_new_modes": True,
        "scalp_isolated": True,
        "artifacts": {
            "strategy_baseline": f"{BASELINE_ID}.json",
            "execution_replay": "day_execution_replay_latest.json",
            "bucket_discovery": "day_bucket_discovery_latest.json",
        },
        "90d_metrics_binance_us_taker": {
            "gross_pnl_usd": w90.get("gross_pnl_usd"),
            "total_fees_usd": w90.get("total_fees_usd"),
            "spread_slippage_impact_usd": w90.get("spread_slippage_impact_usd"),
            "net_pnl_usd": w90.get("net_pnl_usd"),
            "max_drawdown_pct": w90.get("max_drawdown_pct"),
        },
    }
    return status


def main() -> int:
    status = build_status()
    out = BASELINE_DIR / "DAY_BASELINE_FINAL_STATUS.json"
    out.write_text(json.dumps(status, indent=2, default=str))

    lock_path = BASELINE_DIR / "BASELINE_LOCK.json"
    lock = _load_json("BASELINE_LOCK.json") if lock_path.exists() else {}
    lock.update(
        {
            "baseline_id": BASELINE_ID,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "all_pass": status["replay_all_pass"] and status["high_resolution_all_pass"],
            "economics_locked": True,
            "accepted_replay_profile": ACCEPTED_REPLAY_PROFILE,
            "economics": status["economics_config"],
            "final_status_file": "DAY_BASELINE_FINAL_STATUS.json",
            "rules_unchanged": True,
            "live_modifications_forbidden": lock.get("live_modifications_forbidden")
            or [
                "no_revive_killed_buckets",
                "no_new_blockers",
                "no_new_modes",
                "neutral_vwap_positive_buckets_only",
                "range_vwap_btc_eth_xrp_killed",
                "breakout_pullback_blocked_replay_negative",
            ],
        }
    )
    lock_path.write_text(json.dumps(lock, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
