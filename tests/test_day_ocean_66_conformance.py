"""53/53 and 66/66 Ocean live DAY book conformance."""

from backend.services.day_ocean_live_book import (
    BRIEFING_N,
    OCEAN_BOOK_COUNT,
    assert_ocean_book_invariants,
    pair_fifo,
)


def _buys(n: int = 66):
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    out = []
    for i in range(n):
        day = 25 + (i // 12)
        out.append(
            {
                "id": i + 1,
                "timestamp": f"2026-08-{day:02d}T00:{i % 60:02d}:00+00:00",
                "symbol": symbols[i % 4],
                "price": 100.0 + i,
                "quantity": 1.0,
            }
        )
    return out


def _sells_for(buys):
    out = []
    for i, buy in enumerate(buys):
        reason = "DAY_4H_STRUCTURE_BREAK_EXIT" if i < 43 else ("TRAILING_STOP_EXIT" if i < 61 else "NET_PROFIT_EXIT")
        out.append(
            {
                "id": 1000 + i,
                "timestamp": buy["timestamp"].replace("T00:", "T04:"),
                "symbol": buy["symbol"],
                "price": buy["price"] + 0.1,
                "exit_reason": reason,
            }
        )
    return out


def test_ocean_book_count_and_briefing_53():
    buys = _buys(66)
    pairs = pair_fifo(buys, _sells_for(buys))
    payload = assert_ocean_book_invariants(pairs)
    assert payload["ocean_book_count"] == 66
    assert payload["briefing_count"] == 53
    assert payload["four_hour_break_count"] == 43
    assert payload["all_66_have_actual_entry_fill"] is True
    assert payload["all_66_have_actual_exit_fill"] is True
    assert all(p["briefing53"] for p in pairs[:BRIEFING_N])
    assert not any(p["briefing53"] for p in pairs[BRIEFING_N:])
    assert OCEAN_BOOK_COUNT == 66


def test_fifo_keeps_symbol_order():
    buys = [
        {"id": 1, "timestamp": "2026-08-25T00:00:00+00:00", "symbol": "ETH/USDT", "price": 1},
        {"id": 2, "timestamp": "2026-08-25T01:00:00+00:00", "symbol": "ETH/USDT", "price": 2},
    ]
    sells = [
        {"id": 10, "timestamp": "2026-08-25T00:30:00+00:00", "symbol": "ETHUSDT", "price": 1.1, "exit_reason": "TRAILING_STOP_EXIT"},
        {"id": 11, "timestamp": "2026-08-25T02:00:00+00:00", "symbol": "ETHUSDT", "price": 2.1, "exit_reason": "NET_PROFIT_EXIT"},
    ]
    pairs = pair_fifo(buys, sells)
    assert pairs[0]["sell"]["id"] == 10
    assert pairs[1]["sell"]["id"] == 11
