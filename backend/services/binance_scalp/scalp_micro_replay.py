"""Deterministic SCALP microstructure replay — no future leakage."""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.l2_book import LocalL2Book
from backend.services.binance_scalp.scalp_markout import compute_markout_point
from backend.services.binance_scalp.scalp_micro_ev import multi_horizon_ev
from backend.services.microstructure_engine import compute_features, record_agg_trade, record_snapshot


def replay_events(events: list[dict[str, Any]], *, symbol: str = "BTCUSDT") -> dict[str, Any]:
    """Replay time-ordered events. Each event may use only data at or before ts."""
    book = LocalL2Book(symbol=symbol)
    ranks: list[dict[str, Any]] = []
    books: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: float(e.get("ts") or 0.0)):
        ts = float(ev["ts"])
        kind = ev.get("kind")
        if kind == "snapshot":
            book.apply_snapshot(ev["bids"], ev["asks"], ev.get("last_update_id"), ts)
            record_snapshot(symbol, ev["bids"], ev["asks"], ts=ts)
        elif kind == "diff":
            book.apply_diff(ev["bids"], ev["asks"], int(ev["U"]), int(ev["u"]), ts)
        elif kind == "trade":
            record_agg_trade(symbol, float(ev["qty"]), bool(ev["is_buyer_maker"]), ts=ts)
        feats = compute_features(symbol) if kind in {"snapshot", "trade"} else {}
        if feats:
            evs = multi_horizon_ev(feats)
            ranks.append({"ts": ts, "feats": feats, "ev": evs, "book": book.as_dict()})
        books.append(book.as_dict())
    return {"book": book.as_dict(), "steps": ranks, "n_events": len(events)}


def replay_four_coin_rank(per_symbol_feats: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    """Rank all four coins from decision-time features only (select_v2: EV_10s)."""
    from backend.services.binance_scalp.scalp_micro_rank import repaired_primary_score

    scored = []
    for sym, feats in per_symbol_feats.items():
        ev = multi_horizon_ev(feats)
        scored.append((sym, float(repaired_primary_score(feats, ev))))
    scored.sort(key=lambda x: -x[1])
    return scored


__all__ = ["compute_markout_point", "replay_events", "replay_four_coin_rank"]
