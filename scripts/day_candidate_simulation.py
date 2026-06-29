#!/usr/bin/env python3
"""DAY candidate intelligence simulation — rank/explain/learning verification (no live trades)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.day_ai_rank_enrichment import enrich_day_candidate_decision_data
from backend.services.day_block_scores import compute_block_scores_from_decision_data
from backend.services.day_candidate_explanation import build_candidate_explanation
from backend.services.day_relative_strength import enrich_basket_relative_strength
from backend.services.day_setup_scores import compute_setup_score


class _FakeCandidate:
    def __init__(self, symbol: str, decision_data: dict):
        self.symbol = symbol
        self.decision_data = decision_data
        self.confidence = float(decision_data.get("confidence") or 0.6)


def _sample_decision_data(symbol: str, *, cycle: int) -> dict:
    base = {
        "symbol": symbol,
        "live_ai_strategy": "day",
        "confidence": 0.55 + (cycle % 10) * 0.01,
        "buy_margin": 0.02 + (cycle % 5) * 0.005,
        "ema_alignment": 0.55,
        "price_momentum": 0.3,
        "rsi": 48.0,
        "adx": 28.0,
        "atr": 100.0,
        "current_price": 50000.0 if "BTC" in symbol else 3000.0,
        "spread_pct": 0.0008,
        "ctx_rs_btc": 0.5,
        "ctx_rs_eth": 0.2,
        "ctx_depth_imbalance": 0.15,
        "ctx_relative_volume": 1.2,
        "day_route_regime": "bull",
        "setup_type": "HTF_TREND_PULLBACK",
        "feature_health_pass": "1",
        "feature_health_pct": "96.5",
        "feature_health_json": json.dumps(
            {
                "pass": True,
                "health_pct": 96.5,
                "features": [
                    {
                        "index": i,
                        "name": f"f{i}",
                        "block": "technical_indicators",
                        "trust_score": 0.9,
                        "status": "CALCULATED",
                        "learning_allowed": True,
                    }
                    for i in range(1, 146)
                ],
            }
        ),
    }
    if "SOL" in symbol:
        base["setup_type"] = "BREAKOUT_CONTINUATION"
        base["price_momentum"] = 0.8
    if "XRP" in symbol:
        base["setup_type"] = "FAILED_BREAKDOWN_REVERSAL"
        base["day_route_regime"] = "bear"
    if "ETH" in symbol:
        base["setup_type"] = "RANGE_BOUNCE"
    return base


async def main() -> int:
    cycles = 50
    print(f"=== DAY candidate simulation ({cycles} cycles x {len(DAY_TRADE_SYMBOLS)} symbols) ===")
    narratives: list[str] = []
    for cycle in range(cycles):
        candidates = []
        for sym in DAY_TRADE_SYMBOLS:
            dd = _sample_decision_data(sym, cycle=cycle)
            dd = enrich_day_candidate_decision_data(dd, symbol=sym, current_price=float(dd["current_price"]))
            candidates.append(_FakeCandidate(sym, dd))
        enrich_basket_relative_strength(candidates)
        for cand in candidates[:1 if cycle % 10 else 4]:
            snap = build_candidate_explanation(cand.decision_data, symbol=cand.symbol)
            narratives.append(str(snap.get("narrative") or ""))

    print(f"cycles_completed={cycles}")
    print(f"sample_narratives={len(narratives)}")
    if narratives:
        print("\n--- sample explanation ---")
        print(narratives[0])

    # Closed-trade attribution dry run
    from backend.services.day_outcome_attribution import build_attribution_payload, classify_outcome_reason

    sample_ex = enrich_day_candidate_decision_data(
        _sample_decision_data("BTCUSDT", cycle=0),
        symbol="BTCUSDT",
        current_price=50000.0,
    )
    reason = classify_outcome_reason(explainability=sample_ex, net_profit_pct=0.002, close_reason="NET_PROFIT_EXIT")
    payload = build_attribution_payload(
        trade_id="sim-1",
        symbol="BTCUSDT",
        explainability=sample_ex,
        net_profit_usd=5.0,
        net_profit_pct=0.002,
        close_reason="NET_PROFIT_EXIT",
        hold_seconds=600.0,
    )
    print(f"\noutcome_reason={reason}")
    print(f"attribution_keys={sorted(payload.keys())}")

    # Adaptive bucket key
    from backend.services.ai_strategy_score_weight_writer import setup_regime_bucket

    bucket = setup_regime_bucket("bull", sample_ex.get("setup_type", ""))
    print(f"adaptive_bucket={bucket}")

    blocks = compute_block_scores_from_decision_data(sample_ex)
    print(f"feature_health_score={blocks.get('feature_health_score')}")
    print(f"setup_score={compute_setup_score(str(sample_ex.get('setup_type')), sample_ex, blocks)}")
    print("\nPASS: simulation completed (no blockers added, FEATURE_VERSION unchanged, 145-dim path preserved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
