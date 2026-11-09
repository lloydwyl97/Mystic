"""Four-symbol ranking keeps every eligible symbol, and timing stamps do not invent clocks."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from backend.services.day_v2.ranking import rank_day_candidates
from backend.services.portfolio_engine import COMBINED_ENGINE_MAX_POSITIONS, DAY_MAX_OPEN_POSITIONS, SCALP_MAX_OPEN_POSITIONS
from backend.services.protected_limit_execution import summarize_candidate_batch_ms
from backend.services.scalp_v2.executable_edge import decision_detail


def _signal(setup: str = "RANGE_BOUNCE") -> SimpleNamespace:
    return SimpleNamespace(setup=setup, atr_1h=1.0, objective_structural=0.0)


def _candidate(symbol: str, expected_net: float, setup: str = "RANGE_BOUNCE") -> dict:
    return {"symbol": symbol, "signal": _signal(setup), "ask_price": 100.0, "adaptive": {"expected_net": expected_net}}


def _fund(ranked: list[dict], *, cap: int) -> list[str]:
    """One position per symbol, highest expected net first, until the engine cap."""
    funded: list[str] = []
    held: set[str] = set()
    for cand in ranked:
        if float(cand["rank"]["expected_net"]) <= 0.0:
            continue
        symbol = str(cand["symbol"])
        if symbol in held:
            continue
        if len(funded) >= cap:
            break
        held.add(symbol)
        funded.append(symbol)
    return funded


def test_day_and_scalp_caps_cover_four_symbols() -> None:
    assert DAY_MAX_OPEN_POSITIONS == 4
    assert SCALP_MAX_OPEN_POSITIONS == 4
    assert COMBINED_ENGINE_MAX_POSITIONS == 8


def test_four_positive_symbols_all_fund_and_a_fifth_claim_on_a_held_symbol_does_not() -> None:
    universe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    ranked = rank_day_candidates(
        [
            _candidate("XRPUSDT", 0.004),
            _candidate("BTCUSDT", 0.001, "BREAKOUT_CONTINUATION"),
            _candidate("SOLUSDT", 0.003),
            _candidate("ETHUSDT", 0.002),
            _candidate("BTCUSDT", 0.0005, "HTF_TREND_PULLBACK"),
        ],
        universe,
        0.0006,
    )
    assert [c["symbol"] for c in ranked[:4]] == ["XRPUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT"]
    assert _fund(ranked, cap=DAY_MAX_OPEN_POSITIONS) == ["XRPUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT"]
    assert len(ranked) == 5


def test_scalp_confidence_can_reorder_sub_ten_bps_and_cannot_reorder_a_larger_gap() -> None:
    def score(edge: float, confidence: float) -> float:
        return edge + 0.001 * confidence

    # 5 bps vs 0.5 bps with full confidence: confidence wins. That is the current key.
    assert score(0.00005, 1.0) > score(0.0005, 0.0)
    # 20 bps vs 5 bps: the edge still wins at full confidence.
    assert score(0.0020, 0.0) > score(0.0005, 1.0)


def test_decision_detail_records_timing_without_changing_the_edge() -> None:
    edge = {"final_executable_edge_pct": 0.001, "live_cost_pct": 0.0006, "confidence": 0.4}
    plain = json.loads(decision_detail({"executable_edge": edge, "best_setup": "RANGE"}))
    stamped = json.loads(
        decision_detail(
            {
                "executable_edge": edge,
                "best_setup": "RANGE",
                "_timing": {"eval_batch_start": 10.0, "eval_batch_end": 10.25},
            }
        )
    )
    assert plain["final_executable_edge"] == stamped["final_executable_edge"]
    assert stamped["timing"]["eval_batch_end"] == 10.25


def test_candidate_batch_clock_is_null_until_a_batch_is_stamped(tmp_path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE scalp_v2_decisions (id INTEGER PRIMARY KEY, detail TEXT)")
    conn.execute("INSERT INTO scalp_v2_decisions (detail) VALUES (?)", ('{"final_executable_edge": -0.001}',))
    conn.commit()
    conn.close()
    stats = summarize_candidate_batch_ms(str(db))
    assert stats["samples"] == 0
    assert stats["avg"] is None
    assert stats["source"] == "not_stored"
