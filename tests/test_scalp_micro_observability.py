"""Peer-EV and size stamps are observability only — no ranking or sizing change."""

from __future__ import annotations

import inspect
import json

from backend.services.binance_scalp.scalp_candidate_ranking import pick_best_global_candidate
from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size
from backend.services.binance_scalp.scalp_micro_ev import heuristic_horizon_ev
from backend.services.binance_scalp.scalp_micro_observability import (
    PEER_SYMBOLS,
    build_peer_micro_snapshot,
    size_diagnostics,
)
from backend.services.binance_scalp.scalp_opportunity_dataset import record_opportunity_cycle
from backend.services.microstructure_engine import _TAPE_OVERLAY_KEYS, _enrich_micro_scores, get_microstructure_ranking_delta


def _row(symbol: str, *, rank: float, ev10: float, eligible: bool = True, passed: bool = False) -> dict:
    return {
        "symbol": symbol,
        "rank_score": rank,
        "entry_eligible": eligible,
        "strategy_passed": passed,
        "soft_reason": None if passed else "NOT_NEAR_SUPPORT",
        "static_rank_score": rank - 0.01,
        "microstructure_adjustment": 0.002,
        "learned_adjustment": 0.0,
        "EV_1s": ev10 * 0.4,
        "EV_5s": ev10 * 0.8,
        "EV_10s": ev10,
        "EV_30s": ev10 * 0.9,
        "EV_60s": ev10 * 0.7,
        "expected_net_ev": 0.0004 if eligible else -0.0001,
    }


def _four_rows() -> list[dict]:
    return [
        _row("BTCUSDT", rank=1.80, ev10=-0.00012, passed=False),
        _row("ETHUSDT", rank=1.70, ev10=-0.00020, passed=False),
        _row("SOLUSDT", rank=1.90, ev10=-0.00008, passed=False),
        _row("XRPUSDT", rank=1.60, ev10=-0.00030, passed=False),
    ]


def test_peer_snapshot_does_not_alter_sort_order():
    rows = _four_rows()
    before = [r["symbol"] for r in rows]
    snap = build_peer_micro_snapshot(rows, open_symbols=set(), max_open=4, open_count=0)
    after = [r["symbol"] for r in rows]
    assert before == after
    assert [r["rank_score"] for r in rows] == [1.80, 1.70, 1.90, 1.60]
    assert snap["peers"]["SOLUSDT"]["rank_position"] == 1
    assert snap["peers"]["BTCUSDT"]["rank_position"] == 2


def test_peer_snapshot_does_not_affect_eligibility_or_soft_rank():
    rows = _four_rows()
    elig_before = [bool(r["entry_eligible"]) for r in rows]
    passed_before = [bool(r["strategy_passed"]) for r in rows]
    snap = build_peer_micro_snapshot(rows, selected_symbol="SOLUSDT")
    assert [bool(r["entry_eligible"]) for r in rows] == elig_before
    assert [bool(r["strategy_passed"]) for r in rows] == passed_before
    assert snap["peers"]["SOLUSDT"]["soft_rank"] is True
    assert snap["peers"]["SOLUSDT"]["strategy_passed"] is False
    assert all(snap["peers"][s]["entry_eligible"] for s in PEER_SYMBOLS)


def test_peer_snapshot_keeps_all_four_coins():
    rows = _four_rows()[:3]
    snap = build_peer_micro_snapshot(rows, open_symbols={"ETHUSDT"}, max_open=4, open_count=1, selected_symbol="SOLUSDT")
    assert set(snap["peers"]) >= set(PEER_SYMBOLS)
    assert snap["peers"]["ETHUSDT"]["already_open"] is True
    assert snap["peers"]["ETHUSDT"]["available"] is False
    assert snap["peers"]["XRPUSDT"]["hard_block"] == "NOT_IN_CYCLE"


def test_pick_unchanged_when_peer_snapshot_attached():
    from types import SimpleNamespace

    rows = _four_rows()
    for r in rows:
        r["signal"] = SimpleNamespace(
            passed=False,
            spread_pct=0.0002,
            expected_move_pct=0.0035,
            impact_pct=0.0,
            confidence=0.6,
            setup_context={"soft_rank_entry": True},
        )
        r["rank_meta"] = {"soft_reason": "NOT_NEAR_SUPPORT", "regime_native": True, "reachability_surplus": 0.001}
    before = pick_best_global_candidate([dict(r) for r in rows])
    snap = build_peer_micro_snapshot(rows, selected_symbol=None)
    for r in rows:
        r["peer_micro_snapshot"] = snap
    after = pick_best_global_candidate(rows)
    assert (before is None) == (after is None)
    if before is None:
        assert after is None
    else:
        assert before["symbol"] == after["symbol"]
        assert before["rank_score"] == after["rank_score"]
        assert after.get("entry_eligible") is True
        assert after["signal"].passed is False


def test_size_diagnostics_equal_existing_inputs_qty_unchanged():
    kwargs = dict(
        base_cap=50.0,
        free_cash=1000.0,
        min_notional=5.0,
        strategy_passed=False,
        arm_penalty_mult=1.0,
        mtf_penalty_mult=1.0,
        micro_quality_mult=1.0,
        calibration_mult=1.0,
        spread_pct=0.0002,
        impact_pct=0.0001,
        realized_volatility_pct=0.001,
    )
    a = compute_scalp_position_size(**kwargs)
    b = compute_scalp_position_size(**kwargs)
    assert a.notional == b.notional
    limit = 1.47
    qty = a.notional / limit
    diag = size_diagnostics(
        a,
        base_notional=50.0,
        qty=qty,
        strategy_passed=False,
        microstructure_size_factor=1.0,
        learning_size_multiplier=1.0,
        soft_rank_multiplier=0.35,
        calibration_mult=1.0,
        arm_penalty_mult=1.0,
        mtf_penalty_mult=1.0,
    )
    after_qty = a.notional / limit
    assert after_qty == qty
    assert diag["capped_final_notional"] == a.notional
    assert diag["actual_selected_qty"] == qty
    assert diag["confidence_factor"] == a.confidence_factor
    assert diag["final_combined_multiplier"] == a.combined_multiplier
    assert diag["soft_rank_multiplier"] == 0.35
    assert a.notional == b.notional


def test_opportunity_cycle_persists_peer_json(tmp_path):
    db = str(tmp_path / "opp.db")
    rows = _four_rows()
    snap = build_peer_micro_snapshot(rows, selected_symbol="SOLUSDT")
    for r in rows:
        r["peer_micro_snapshot"] = snap
        r["snap"] = type("S", (), {"mid": 1.0, "spread_pct": 0.0002})()
        r["rank_meta"] = {"impact_pct": 0.0, "regime": "range", "setup_measurements": {}, "feature_vector": []}
        r["all_signals"] = []
        r["best_setup"] = "range_bounce_scalp"
    written = record_opportunity_cycle(db, rows=rows, epoch=1.0)
    assert written == 4
    import sqlite3

    conn = sqlite3.connect(db)
    raw = conn.execute("SELECT peer_micro_json FROM scalp_opportunity_snapshots WHERE symbol='SOLUSDT'").fetchone()[0]
    conn.close()
    stored = json.loads(raw)
    assert stored["selected_symbol"] == "SOLUSDT"
    assert stored["peers"]["BTCUSDT"]["EV_10s"] == -0.00012
    assert stored["peers"]["SOLUSDT"]["rank_position"] == 1


def test_absorption_semantics_long_scalp_no_sign_inversion():
    enrich_src = inspect.getsource(_enrich_micro_scores)
    assert 'out["bid_absorption_score"] = round(sell_abs, 4)' in enrich_src
    assert 'out["ask_absorption_score"] = round(buy_abs, 4)' in enrich_src
    delta_src = inspect.getsource(get_microstructure_ranking_delta)
    assert 'bid_absorption_score") or 0.0) - float(feats.get("ask_absorption_score")' in delta_src
    ev_src = inspect.getsource(heuristic_horizon_ev)
    assert 'bid_absorption_score") - _f(f, "ask_absorption_score")' in ev_src
    assert "bid_absorption_score" in _TAPE_OVERLAY_KEYS
    assert "ask_absorption_score" in _TAPE_OVERLAY_KEYS
    base = {"ofi_5s": 0.0, "agg_flow_imbalance_5s": 0.0, "microprice_pressure": 0.0, "obi_l5": 0.0, "adverse_selection_score": 0.2, "spread_pct": 0.0002}
    bid_sup = heuristic_horizon_ev({**base, "bid_absorption_score": 0.8, "ask_absorption_score": 0.0}, 10)
    ask_abs = heuristic_horizon_ev({**base, "bid_absorption_score": 0.0, "ask_absorption_score": 0.8}, 10)
    assert bid_sup > ask_abs
