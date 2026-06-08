#!/usr/bin/env python3
"""Pre-soak entry gate diagnostics — BTC/ETH projected edge vs required gross move."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config  # noqa: E402
from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from backend.services.binance_scalp.market_reader import ScalpMarketReader  # noqa: E402
from backend.services.binance_scalp.momentum_tracker import MomentumTracker  # noqa: E402
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight  # noqa: E402


def _warm_momentum(reader: ScalpMarketReader, tracker: MomentumTracker, symbols: tuple[str, ...]) -> None:
    for _ in range(8):
        now = time.time()
        for sym in symbols:
            snap = reader.read(sym)
            if snap:
                tracker.record(sym, now, snap.best_bid, snap.mid)
        time.sleep(5)


def _evaluate_symbol(
    sym: str,
    reader: ScalpMarketReader,
    tracker: MomentumTracker,
    econ: ScalpEconomics,
    config,
) -> dict:
    snap = reader.read(sym)
    if snap is None:
        return {"symbol": sym, "error": "NO_MARKET_DATA"}
    now = time.time()
    tracker.record(sym, now, snap.best_bid, snap.mid)
    mom = tracker.diagnostics(sym, now, snap.best_bid, snap.mid)
    pf = run_scalp_preflight(
        snap,
        econ,
        config,
        side="BUY",
        check_paper_enabled=False,
        momentum=mom,
        apply_entry_gate=True,
    )
    reach = pf.reachability or {}
    return {
        "symbol": sym,
        "best_bid": snap.best_bid,
        "best_ask": snap.best_ask,
        "spread_pct": snap.spread_pct,
        "projected_edge_pct": reach.get("projected_edge_pct"),
        "required_gross_move_pct": reach.get("required_gross_move_pct"),
        "projected_surplus_pct": reach.get("projected_surplus_pct"),
        "projected_edge_minus_required": reach.get("projected_edge_minus_required"),
        "min_projected_surplus_pct": reach.get("min_projected_surplus_pct"),
        "momentum": mom.as_dict(),
        "momentum_pass": mom.momentum_confirmed,
        "preflight_pass": pf.passed,
        "reject_reason": pf.reject_reason or None,
        "entry_decision": "ENTER" if pf.passed else "REJECT",
        "reachability": reach,
    }


def main() -> int:
    config = get_scalp_config()
    econ = ScalpEconomics.from_env()
    reader = ScalpMarketReader(config)
    tracker = MomentumTracker()
    symbols = config.products

    _warm_momentum(reader, tracker, symbols)
    results = [_evaluate_symbol(sym, reader, tracker, econ, config) for sym in symbols]

    btc = next((r for r in results if r.get("symbol") == "BTCUSDT"), {})
    eth = next((r for r in results if r.get("symbol") == "ETHUSDT"), {})
    stronger = None
    if btc.get("momentum") and eth.get("momentum"):
        b15 = float(btc["momentum"].get("bid_change_15s", 0))
        e15 = float(eth["momentum"].get("bid_change_15s", 0))
        if e15 > b15 + 1e-8:
            stronger = "ETHUSDT"
        elif b15 > e15 + 1e-8:
            stronger = "BTCUSDT"

    selected = None
    for r in results:
        if r.get("entry_decision") == "ENTER":
            selected = r["symbol"]
            break

    out = {
        "economics": econ.as_dict(),
        "computed_entry_required_formula": (
            "net_profit_target + roundtrip_cost(spread,impact) + entry_edge_buffer"
        ),
        "products": results,
        "stronger_momentum_15s": stronger,
        "btc_priority_would_select": selected or (
            "NONE"
            if not any(r.get("entry_decision") == "ENTER" for r in results)
            else next(
                r["symbol"]
                for r in results
                if r.get("entry_decision") == "ENTER"
            )
        ),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
