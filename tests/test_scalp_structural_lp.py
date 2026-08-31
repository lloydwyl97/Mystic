from __future__ import annotations

from types import SimpleNamespace

from backend.services.binance_scalp.structural_lp import (
    FILL_ASK,
    FILL_BID,
    TIMEOUT,
    BookPrint,
    RestingQuote,
    is_structural_position,
    step,
    through_fill,
)
from backend.services.binance_scalp.structural_thesis import (
    prediction_circuit_breaker_applies,
    ranking_eval_permitted,
    status_fields,
    structural_lp_executable,
)


def _book(bid: float, ask: float, *, epoch: float = 10.0) -> BookPrint:
    return BookPrint(symbol="BTCUSDT", best_bid=bid, best_ask=ask, epoch=epoch)


def test_touch_is_not_a_fill():
    q = RestingQuote("BTCUSDT", "BID", 100.0, 0.01, 1.0)
    assert through_fill(q, _book(100.0, 100.10)) is False
    q2 = RestingQuote("BTCUSDT", "ASK", 100.10, 0.01, 1.0)
    assert through_fill(q2, _book(100.0, 100.10)) is False


def test_bid_fills_only_when_ask_walks_through():
    q = RestingQuote("BTCUSDT", "BID", 100.0, 0.01, 1.0)
    assert through_fill(q, _book(99.90, 99.95)) is True
    nxt, actions = step(q, _book(99.90, 99.95), inventory_qty=0.0, lot_qty=0.01, hold_sec=1.0, max_hold_sec=600.0)
    kinds = [a.kind for a in actions]
    assert "buy" in kinds
    assert any(a.reason == FILL_BID for a in actions)
    assert nxt is not None and nxt.side == "ASK"


def test_ask_fills_only_when_bid_walks_through():
    q = RestingQuote("BTCUSDT", "ASK", 100.10, 0.01, 1.0)
    nxt, actions = step(q, _book(100.12, 100.20), inventory_qty=0.01, lot_qty=0.01, hold_sec=1.0, max_hold_sec=600.0)
    assert any(a.kind == "sell" and a.reason == FILL_ASK for a in actions)
    assert nxt is not None and nxt.side == "BID"


def test_timeout_is_taker_flatten_not_prediction_exit():
    q = RestingQuote("BTCUSDT", "ASK", 100.10, 0.01, 1.0)
    nxt, actions = step(q, _book(100.0, 100.10), inventory_qty=0.01, lot_qty=0.01, hold_sec=601.0, max_hold_sec=600.0)
    assert any(a.kind == "timeout_sell" and a.reason == TIMEOUT for a in actions)
    assert nxt is not None and nxt.side == "BID"


def test_mid_move_without_through_does_not_fill():
    q = RestingQuote("BTCUSDT", "BID", 100.0, 0.01, 1.0)
    _, actions = step(q, _book(99.99, 100.05), inventory_qty=0.0, lot_qty=0.01, hold_sec=1.0, max_hold_sec=600.0)
    assert not any(a.kind == "buy" for a in actions)


def test_breaker_and_ranking_off_for_structural():
    cfg = SimpleNamespace(
        scalp_thesis="structural",
        legacy_prediction_entries=False,
        scalp_paper_enabled=True,
        scalp_live=False,
    )
    assert prediction_circuit_breaker_applies(cfg) is False
    assert ranking_eval_permitted(cfg) is False
    assert structural_lp_executable(cfg) is True
    fields = status_fields(cfg)
    assert fields["structural_arb_executable"] is False
    assert fields["prediction_circuit_breaker_applies"] is False


def test_live_blocks_structural_lp():
    cfg = SimpleNamespace(scalp_thesis="structural", scalp_paper_enabled=True, scalp_live=True)
    assert structural_lp_executable(cfg) is False


def test_structural_position_tag():
    assert is_structural_position({"strategy_id": "structural_lp", "diagnostics_json": "{}"})
    assert is_structural_position({"strategy_id": "vwap_ema_reclaim", "diagnostics_json": '{"setup_name": "structural_lp"}'})
    assert not is_structural_position({"strategy_id": "vwap_ema_reclaim", "diagnostics_json": "{}"})
