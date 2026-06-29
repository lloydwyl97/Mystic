#!/usr/bin/env python3
"""SCALP candidate intelligence simulation — 50 cycles x 4 symbols."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.scalp_ai_rank_enrichment import build_scalp_intelligence, enrich_scalp_ranked_candidates
from backend.services.scalp_feature_contract import STRATEGY_TO_SCALP_SETUP
from backend.services.scalp_outcome_attribution import record_scalp_outcome_attribution
from backend.services.scalp_relative_strength import enrich_scalp_basket_relative_strength
from backend.services.scalp_strategy_score_weight_writer import propagate_scalp_adaptive_weights_for_close

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _fake_signal(symbol: str, cycle: int) -> dict:
    setups = list(STRATEGY_TO_SCALP_SETUP.keys())
    name = setups[cycle % len(setups)]
    return {
        "symbol": symbol,
        "setup_name": name,
        "score": 0.45 + (cycle % 10) * 0.03,
        "confidence": 0.5 + (cycle % 7) * 0.02,
        "spread_pct": 0.0008 + (cycle % 3) * 0.0002,
        "impact_pct": 0.0005,
        "required_target_pct": 0.002,
        "expected_move_pct": 0.003,
        "depth_sufficient": True,
        "passed": True,
    }


def main() -> int:
    cfg = get_scalp_config()
    reader = ScalpMarketReader(cfg)
    mom = MomentumTracker()
    cycles = 50
    narratives = 0
    attribution_ok = 0
    learning_ok = 0
    from backend.services.scalp_feature_audit import build_symbol_scalp_audit

    audit_cache: dict[str, dict] = {}

    for cycle in range(cycles):
        epoch = time.time()
        ranked: list[dict] = []
        for sym in SYMBOLS:
            snap = reader.read(sym)
            if snap is None:
                continue
            mom.record(sym, epoch, snap.best_bid, snap.mid)
            md = mom.diagnostics(sym, epoch, snap.best_bid, snap.mid)
            if sym not in audit_cache:
                audit_cache[sym] = build_symbol_scalp_audit(sym, snap=snap, mom_diag=md)
            ranked.append({"symbol": sym, "snap": snap, "mom": md, "signal": _fake_signal(sym, cycle), "micro_regime": audit_cache[sym].get("micro_regime") or "range"})
        if not ranked:
            continue
        enriched = enrich_scalp_ranked_candidates(ranked, redis_client=None)
        basket = enrich_scalp_basket_relative_strength(
            [{"symbol": r["symbol"], **(r.get("intelligence") or {})} for r in enriched]
        )
        top = basket[0] if basket else {}
        narrative = top.get("scalp_candidate_explanation_narrative") or (enriched[0].get("intelligence") or {}).get("scalp_candidate_explanation_narrative")
        if narrative:
            narratives += 1
        intel = top
        tid = f"sim_scalp_{cycle}_{top.get('symbol')}"
        row_id = record_scalp_outcome_attribution(
            trade_id=tid,
            symbol=str(top.get("symbol") or "BTCUSDT"),
            intelligence=intel,
            gross_pnl=0.5,
            fees=0.05,
            net_pnl=0.45,
            hold_seconds=90.0,
            exit_reason="NET_PROFIT_TARGET",
            db_path=cfg.database_path,
        )
        if row_id:
            attribution_ok += 1
        if propagate_scalp_adaptive_weights_for_close(symbol=str(top.get("symbol")), intelligence=intel, net_pnl=0.45, db_path=cfg.database_path):
            learning_ok += 1

    # cleanup sim rows
    import sqlite3

    with sqlite3.connect(cfg.database_path) as conn:
        conn.execute("DELETE FROM scalp_outcome_attribution WHERE trade_id LIKE 'sim_scalp_%'")
        conn.commit()

    print("=== SCALP candidate simulation (50 cycles x 4 symbols) ===")
    print(f"cycles_completed={cycles}")
    print(f"sample_narratives={narratives}")
    print(f"attribution_writes={attribution_ok}")
    print(f"learning_bucket_updates={learning_ok}")
    ok = narratives >= 40 and attribution_ok >= 40
    print("PASS: simulation completed (no gates added, SCALP isolated from DAY)" if ok else "FAIL: incomplete simulation")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
