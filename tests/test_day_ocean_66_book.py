from backend.services.day_ocean_66_book import BRIEFING_N, pair_book, summarize_book


def test_ocean_book_pairs_fifo_and_keeps_briefing_53():
    buys = [
        {
            "id": i,
            "timestamp": f"2026-08-25T00:0{i}:00+00:00",
            "symbol": "ETHUSDT",
            "decision_id": f"d{i}",
            "order_id": "",
            "notional": 50.0,
            "fees_paid": 0.01,
            "slippage_cost": 0.005,
        }
        for i in range(3)
    ]
    sells = [
        {
            "id": 100 + i,
            "timestamp": f"2026-08-25T01:0{i}:00+00:00",
            "symbol": "ETHUSDT",
            "pnl": -0.1 * (i + 1),
            "exit_reason": "DAY_4H_STRUCTURE_BREAK_EXIT",
            "hold_sec": 3600,
            "fees_paid": 0.01,
            "slippage_cost": 0.002,
        }
        for i in range(3)
    ]
    pairs = pair_book(buys, sells)
    assert len(pairs) == 3
    assert all(p["paired"] for p in pairs)
    assert pairs[0]["briefing53"] is True
    summary = summarize_book(pairs)
    assert summary["fills"] == 3
    assert summary["briefing_subset"] == BRIEFING_N
    assert abs(summary["actual_pnl_usd"] + 0.6) < 1e-9
    assert summary["oracle_not_used_as_edge"] is True
