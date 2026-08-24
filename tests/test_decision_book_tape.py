"""Decision book/tape collector: persist only, never changes authority."""

from __future__ import annotations

import json

import backend.services.decision_book_tape as dbt
from backend.services.binance_scalp.scalp_candidate_ranking import (
    HOLD_ACTION_EV,
    pick_best_global_candidate,
)
from backend.services.decision_book_tape import (
    record_day_decision,
    record_rows,
    record_scalp_cycle,
    snapshot_book,
)


def _tmp_db(monkeypatch, tmp_path):
    path = str(tmp_path / "decision_book_tape.db")
    monkeypatch.setattr(dbt, "DATABASE_PATH", path)
    monkeypatch.setattr(dbt, "_TABLE_READY", False)
    monkeypatch.setattr(dbt, "_LAST_QUIET", {})
    return path


def test_snapshot_without_redis_is_missing_not_rest():
    book = snapshot_book("SOLUSDT", redis_client=None)
    assert book["book_source"] in ("missing", "redis_orderbook", "redis_depth_cache", "redis_microstructure")
    assert "would_allow_buy" not in book


def test_record_day_does_not_raise_and_does_not_change_action(monkeypatch, tmp_path):
    _tmp_db(monkeypatch, tmp_path)
    prov = {
        "selected_action": "HOLD",
        "selection_reason": "HOLD_WINS",
        "buy_ev": -0.0002,
        "hold_ev": 0.0,
        "model_version": "day_path_net_v1",
        "prediction_timestamp": "2026-08-16T21:00:00+00:00",
    }
    n = record_day_decision(symbol="ETHUSDT", provenance=prov, redis_client=None)
    assert n >= 0
    assert prov["selected_action"] == "HOLD"


def test_record_scalp_hold_does_not_flip_pick(monkeypatch, tmp_path):
    _tmp_db(monkeypatch, tmp_path)
    ranked = [
        {"symbol": "BTCUSDT", "entry_eligible": True, "expected_net_ev": -0.0001, "rank_score": 1.0, "forward_net_model_version": "scalp_path_net_v1"},
        {"symbol": "ETHUSDT", "entry_eligible": True, "expected_net_ev": -0.0002, "rank_score": 0.9, "forward_net_model_version": "scalp_path_net_v1"},
        {"symbol": "SOLUSDT", "entry_eligible": True, "expected_net_ev": -0.0003, "rank_score": 0.8, "forward_net_model_version": "scalp_path_net_v1"},
        {"symbol": "XRPUSDT", "entry_eligible": True, "expected_net_ev": -0.0004, "rank_score": 0.7, "forward_net_model_version": "scalp_path_net_v1"},
    ]
    before = pick_best_global_candidate(ranked)
    n = record_scalp_cycle(ranked=ranked, chosen=before, redis_client=None, hold_ev=HOLD_ACTION_EV)
    after = pick_best_global_candidate(ranked)
    assert before is not None
    assert before["symbol"] == "BTCUSDT"
    assert after is not None
    assert after["symbol"] == "BTCUSDT"
    assert n >= 0


def test_broken_insert_is_fail_open(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr("backend.services.decision_book_tape._ensure_table", boom)
    assert record_rows([{"engine": "day", "symbol": "BTCUSDT", "buy_ev": 0.001, "selected_action": "BUY"}]) == 0


def test_snapshot_uses_depth_cache_not_rest():
    class FakeRedis:
        def hgetall(self, key):
            return {}

        def get(self, key):
            if key == "scalp:depth_cache:SOLUSDT":
                return json.dumps(
                    {
                        "fetched_at": 1.0,
                        "bids": [[100.0, 2.0], [99.9, 3.0]],
                        "asks": [[100.1, 1.5], [100.2, 4.0]],
                    }
                )
            return None

    book = snapshot_book("SOLUSDT", redis_client=FakeRedis())
    assert book["book_source"] == "redis_depth_cache"
    assert book["best_bid"] == 100.0
    assert book["best_ask"] == 100.1
    assert book["bid_qty_top5"] == 5.0
    assert book["ask_qty_top5"] == 5.5
