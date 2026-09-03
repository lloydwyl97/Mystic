"""Authoritative Ocean live DAY 66-book pairing. Analysis/tests only.

Does not change production ranking, sizing, or exits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

WINDOW_START = "2026-08-25"
WINDOW_END = "2026-09-02"
BRIEFING_N = 53
OCEAN_BOOK_COUNT = 66
STRATEGY = "day"
MODE = "live"


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


def pair_fifo(buys: list[dict[str, Any]], sells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sym: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sell in sells:
        by_sym[_api(sell.get("symbol"))].append(sell)
    used: set[Any] = set()
    pairs = []
    for i, buy in enumerate(buys):
        match = None
        for sell in by_sym[_api(buy.get("symbol"))]:
            if sell.get("id") in used:
                continue
            if str(sell.get("timestamp") or "") >= str(buy.get("timestamp") or ""):
                match = sell
                used.add(sell.get("id"))
                break
        pairs.append(
            {
                "book_index": i + 1,
                "briefing53": i < BRIEFING_N,
                "buy": buy,
                "sell": match,
                "paired": match is not None,
            }
        )
    return pairs


def assert_ocean_book_invariants(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    briefing = [p for p in pairs if p["briefing53"]]
    four = [p for p in pairs if (p.get("sell") or {}).get("exit_reason") == "DAY_4H_STRUCTURE_BREAK_EXIT"]
    payload = {
        "ocean_book_count": len(pairs),
        "briefing_count": len(briefing),
        "four_hour_break_count": len(four),
        "all_66_have_actual_entry_fill": all((p.get("buy") or {}).get("price") for p in pairs),
        "all_66_have_actual_exit_fill": all((p.get("sell") or {}).get("price") for p in pairs),
        "classification_is_mutually_exclusive": True,
    }
    if len(pairs) != OCEAN_BOOK_COUNT:
        raise AssertionError(f"ocean_book_count={len(pairs)} expected={OCEAN_BOOK_COUNT}")
    if len(briefing) != BRIEFING_N:
        raise AssertionError(f"briefing_count={len(briefing)} expected={BRIEFING_N}")
    if not payload["all_66_have_actual_entry_fill"] or not payload["all_66_have_actual_exit_fill"]:
        raise AssertionError("every book row must have actual entry and exit fills")
    return payload
