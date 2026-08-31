#!/usr/bin/env python3
"""Diagnostic only. Not wired into live DAY or SCALP authority.

Checks whether a predicted path-EV > HOLD(0) would survive a live
Binance.US top-of-book / depth test. Does not buy. Does not change
ranking, exits, or models.

Usage (Ocean or VM, read-only):
  PYTHONPATH=/home/mystic/mystic python3 scripts/path_ev_book_depth_check.py \\
      --symbol SOLUSDT --predicted-ev 0.00016
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOLD_EV = 0.0


def assumed_cost_pct(engine: str) -> float:
    if engine == "day":
        from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST

        return float(ESTIMATED_ROUNDTRIP_COST)
    return 0.0006


def book_metrics(symbol: str) -> dict[str, Any]:
    from backend.services.binance_scalp.market_reader import fetch_depth_sync, symbol_bus

    sym = symbol_bus(symbol)
    bids, asks = fetch_depth_sync(sym, limit=20)
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    spread_pct = ((best_ask - best_bid) / mid) if mid > 0 else None
    bid_qty = sum(float(q) for _, q in bids[:5])
    ask_qty = sum(float(q) for _, q in asks[:5])
    imb = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else None
    return {
        "symbol": sym,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_pct": spread_pct,
        "bid_qty_top5": bid_qty,
        "ask_qty_top5": ask_qty,
        "imbalance_top5": imb,
        "levels_bid": len(bids),
        "levels_ask": len(asks),
    }


def evaluate_depth_vs_path_ev(
    *,
    predicted_ev: float,
    book: dict[str, Any],
    engine: str,
    min_top5_qty: float = 0.0,
) -> dict[str, Any]:
    """Return whether path-EV > HOLD would still stand after book costs.

    Fail-closed: missing book or empty book is not a BUY.
    Does not replace HOLD. Does not execute.
    """
    cost = assumed_cost_pct(engine)
    spread = book.get("spread_pct")
    ev_beats_hold = float(predicted_ev) > HOLD_EV
    book_ok = bool(book.get("best_bid") and book.get("best_ask") and book.get("mid"))
    spread_ok = spread is not None and float(spread) <= cost
    depth_ok = True
    if min_top5_qty > 0:
        depth_ok = float(book.get("ask_qty_top5") or 0) >= min_top5_qty
    # Live one-way spread already consumes the assumed round-trip haircut
    # when spread > assumed_cost. Residual EV after live spread:
    residual = float(predicted_ev) - (float(spread) - cost) if spread is not None else None
    residual_beats_hold = residual is not None and residual > HOLD_EV
    allowed = bool(ev_beats_hold and book_ok and spread_ok and depth_ok and residual_beats_hold)
    return {
        "engine": engine,
        "predicted_ev": float(predicted_ev),
        "assumed_cost_pct": cost,
        "live_spread_pct": spread,
        "residual_ev_after_live_spread": residual,
        "ev_beats_hold": ev_beats_hold,
        "book_ok": book_ok,
        "spread_ok": spread_ok,
        "depth_ok": depth_ok,
        "would_allow_buy": allowed,
        "blocked_reason": (
            None
            if allowed
            else (
                "EV_NOT_ABOVE_HOLD" if not ev_beats_hold else "NO_BOOK" if not book_ok else "SPREAD_GT_ASSUMED_COST" if not spread_ok else "DEPTH_TOO_THIN" if not depth_ok else "RESIDUAL_EV_LE_HOLD"
            )
        ),
        "book": book,
        "live_wired": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Read-only path-EV vs live book check. Not a live gate.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--predicted-ev", type=float, required=True)
    p.add_argument("--engine", choices=("day", "scalp"), default="scalp")
    p.add_argument("--min-top5-qty", type=float, default=0.0)
    args = p.parse_args()
    book = book_metrics(args.symbol)
    out = evaluate_depth_vs_path_ev(
        predicted_ev=args.predicted_ev,
        book=book,
        engine=args.engine,
        min_top5_qty=args.min_top5_qty,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
