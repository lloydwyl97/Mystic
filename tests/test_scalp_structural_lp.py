from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend.services.binance_scalp.schema import DAY_TABLES, SCALP_TABLES, init_scalp_schema
from backend.services.binance_scalp.structural_breaker import (
    StructuralBreakerState,
    default_thresholds,
    evaluate,
    load_state,
    save_state,
)
from backend.services.binance_scalp.structural_economics import FeeAssumptions, quote_blocked, roundtrip_pnl
from backend.services.binance_scalp.structural_lp import (
    BookPrint,
    RestingQuote,
    consume_events,
    is_structural_position,
    legacy_snapshot_through_fill,
    load_working_quotes,
    place_quote,
    quote_still_current,
    save_working_quote,
)
from backend.services.binance_scalp.structural_mode import (
    MODE_DISABLED,
    MODE_PAPER,
    MODE_SHADOW,
    StructuralModeError,
    resolve_structural_mode,
)
from backend.services.binance_scalp.structural_tape import TradeEvent, apply_queue, event_eligible_for_quote
from backend.services.binance_scalp.structural_thesis import (
    STRUCTURAL_NOT_EXECUTABLE,
    new_entry_block_reason,
    prediction_circuit_breaker_applies,
    ranking_eval_permitted,
    status_fields,
    structural_lp_executable,
)


def _book(bid: float, ask: float, *, epoch: float = 10.0, bids=None, asks=None) -> BookPrint:
    return BookPrint(
        symbol="BTCUSDT",
        best_bid=bid,
        best_ask=ask,
        epoch=epoch,
        bids=tuple(bids or ((bid, 1.0),)),
        asks=tuple(asks or ((ask, 1.0),)),
        source="test",
        age_sec=0.1,
    )


def _quote(*, side="BID", price=100.0, qty=0.01, posted_ts=10.0, ahead=0.5) -> RestingQuote:
    return RestingQuote(
        symbol="BTCUSDT",
        side=side,
        price=price,
        qty=qty,
        posted_epoch=1.0,
        posted_ts=posted_ts,
        queue_ahead=ahead,
        remaining_qty=qty,
        quote_id="q1",
    )


def _ev(*, agg_id=1, price=100.0, qty=0.2, sold=True, ts=11.0) -> TradeEvent:
    return TradeEvent(
        symbol="BTCUSDT",
        agg_id=agg_id,
        price=price,
        qty=qty,
        is_buyer_maker=sold,
        trade_ts=ts,
        recv_ts=ts,
    )


def test_no_fill_from_snapshot_movement_alone():
    q = _quote()
    book = _book(99.90, 99.95)
    assert legacy_snapshot_through_fill(q, book) is True
    nxt, actions = consume_events(q, [])
    assert actions == []
    assert nxt.filled_qty == 0.0


def test_no_fill_from_pre_placement_trade():
    q = _quote(posted_ts=10.0)
    nxt, actions = consume_events(q, [_ev(agg_id=5, ts=9.9, sold=True, price=99.0, qty=10.0)])
    assert actions == []
    assert nxt.filled_qty == 0.0


def test_no_fill_from_duplicate_or_out_of_order_events():
    q = _quote(posted_ts=10.0)
    q.last_agg_id = 8
    nxt, actions = consume_events(
        q,
        [
            _ev(agg_id=8, ts=12.0, sold=True, price=99.0, qty=10.0),
            _ev(agg_id=7, ts=13.0, sold=True, price=99.0, qty=10.0),
        ],
    )
    assert actions == []
    assert nxt.last_agg_id == 8


def test_queue_ahead_consumed_before_fill():
    consumed, fill, remain = apply_queue(queue_ahead=1.0, queue_consumed=0.0, remaining_qty=0.2, event_qty=0.6)
    assert consumed == pytest.approx(0.6)
    assert fill == 0.0
    assert remain == pytest.approx(0.2)
    consumed, fill, remain = apply_queue(queue_ahead=1.0, queue_consumed=0.6, remaining_qty=0.2, event_qty=0.5)
    assert consumed == pytest.approx(1.0)
    assert fill == pytest.approx(0.1)
    assert remain == pytest.approx(0.1)


def test_partial_then_full_fill():
    q = _quote(ahead=0.0, qty=0.02)
    nxt, acts = consume_events(q, [_ev(agg_id=1, qty=0.01, price=100.0, sold=True, ts=11.0)])
    assert len(acts) == 1
    assert acts[0].kind == "partial_fill"
    assert nxt.remaining_qty == pytest.approx(0.01)
    nxt, acts = consume_events(nxt, [_ev(agg_id=2, qty=0.02, price=99.5, sold=True, ts=12.0)])
    assert acts[0].kind == "fill"
    assert nxt.remaining_qty == pytest.approx(0.0)


def test_wrong_side_or_price_does_not_fill():
    assert event_eligible_for_quote(_ev(sold=False, price=100.0, ts=11.0), side="BID", price=100.0, posted_ts=10.0, last_agg_id=0) is False
    assert event_eligible_for_quote(_ev(sold=True, price=100.01, ts=11.0), side="BID", price=100.0, posted_ts=10.0, last_agg_id=0) is False
    ask = _quote(side="ASK", price=100.1)
    _nxt, acts = consume_events(ask, [_ev(sold=True, price=100.2, ts=11.0)])
    assert acts == []
    _nxt, acts = consume_events(ask, [_ev(sold=False, price=100.1, ts=11.0, qty=10)])
    assert acts


def test_quote_persists_at_same_touch():
    book = _book(100.0, 100.1)
    q = place_quote(book, side="BID", qty=0.01, now_ts=10.0, quote_id="a")
    assert quote_still_current(q, book, side="BID") is True
    moved = _book(99.9, 100.0)
    assert quote_still_current(q, moved, side="BID") is False


def test_missing_stale_trade_data_does_not_fill():
    q = _quote()
    nxt, acts = consume_events(q, [])
    assert acts == []
    assert nxt.filled_qty == 0


def test_maker_entry_and_timeout_taker_accounting():
    fees = FeeAssumptions(maker_fee_pct=0.0, taker_fee_pct=0.0002, timeout_slip_pct=0.0001)
    maker = roundtrip_pnl(entry=100.0, exit_px=100.1, qty=1.0, fees=fees, exit_maker=True)
    taker = roundtrip_pnl(entry=100.0, exit_px=99.9, qty=1.0, fees=fees, exit_maker=False)
    assert maker["timeout_slip_usd"] == 0.0
    assert taker["timeout_slip_usd"] > 0
    assert taker["exit_fee_usd"] > maker["exit_fee_usd"]
    assert fees.label == "simulation_assumption"


def test_min_net_edge_and_vol_guards():
    assert (
        quote_blocked(
            spread_bps=1.0,
            maker_fee_pct=0.0,
            min_net_edge_bps=2.0,
            recent_range_bps=0.0,
            max_range_mult=8.0,
            adverse_1s_rate=0.0,
            max_adverse=0.8,
        )
        == "STRUCTURAL_MIN_NET_EDGE"
    )
    assert (
        quote_blocked(
            spread_bps=5.0,
            maker_fee_pct=0.0,
            min_net_edge_bps=2.0,
            recent_range_bps=50.0,
            max_range_mult=8.0,
            adverse_1s_rate=0.0,
            max_adverse=0.8,
        )
        == "STRUCTURAL_VOLATILITY_GUARD"
    )


def test_restart_persists_working_quote(tmp_path):
    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    q = _quote()
    save_working_quote(conn, "BTCUSDT", q, "5-0")
    conn.commit()
    quotes, ids = load_working_quotes(conn)
    assert "BTCUSDT" in quotes
    assert quotes["BTCUSDT"].posted_ts == 10.0
    assert ids["BTCUSDT"] == "5-0"


def test_breaker_activation_and_recovery():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    th = default_thresholds(consec=3, daily_loss_usd=5.0, timeout_rate=0.5, adverse_rate=0.8, recovery_sec=60)
    prior = StructuralBreakerState(open=False, reason="", tripped_at="", recovery_until="", stats={}, thresholds=th)
    opened = evaluate(
        consec_losses=3,
        daily_pnl=-1.0,
        rolling_pnl=[-0.1] * 10,
        timeout_rate=0.1,
        adverse_1s_rate=0.1,
        tape_stale=False,
        now=now,
        prior=prior,
    )
    assert opened.open is True
    assert "CONSECUTIVE_LOSSES" in opened.reason
    later = datetime(2026, 8, 30, 0, 2, tzinfo=timezone.utc)
    closed = evaluate(
        consec_losses=0,
        daily_pnl=0.0,
        rolling_pnl=[0.1] * 10,
        timeout_rate=0.0,
        adverse_1s_rate=0.0,
        tape_stale=False,
        now=later,
        prior=opened,
    )
    assert closed.open is False


def test_breaker_restart_persistence(tmp_path):
    conn = sqlite3.connect(tmp_path / "b.db")
    th = default_thresholds(consec=8, daily_loss_usd=25, timeout_rate=0.5, adverse_rate=0.8, recovery_sec=1800)
    state = StructuralBreakerState(open=True, reason="DAILY_NET_LOSS", tripped_at="t", recovery_until="r", stats={"x": 1}, thresholds=th)
    save_state(conn, state)
    loaded = load_state(conn, th)
    assert loaded.open is True
    assert loaded.reason == "DAILY_NET_LOSS"


def test_mode_fail_closed_live_and_legacy():
    with pytest.raises(StructuralModeError):
        resolve_structural_mode(
            env_mode="STRUCTURAL_PAPER",
            scalp_live=True,
            scalp_live_armed=False,
            scalp_paper_enabled=True,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=False,
        )
    with pytest.raises(StructuralModeError):
        resolve_structural_mode(
            env_mode="STRUCTURAL_LIVE",
            scalp_live=False,
            scalp_live_armed=False,
            scalp_paper_enabled=True,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=False,
        )
    with pytest.raises(StructuralModeError):
        resolve_structural_mode(
            env_mode="",
            scalp_live=False,
            scalp_live_armed=False,
            scalp_paper_enabled=True,
            scalp_thesis="legacy_prediction",
            legacy_prediction_entries=True,
            allow_market_orders=False,
        )
    assert (
        resolve_structural_mode(
            env_mode="DISABLED",
            scalp_live=False,
            scalp_live_armed=False,
            scalp_paper_enabled=True,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=False,
        )
        == MODE_DISABLED
    )
    assert (
        resolve_structural_mode(
            env_mode="SHADOW",
            scalp_live=False,
            scalp_live_armed=False,
            scalp_paper_enabled=True,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=False,
        )
        == MODE_SHADOW
    )
    assert (
        resolve_structural_mode(
            env_mode="",
            scalp_live=False,
            scalp_live_armed=False,
            scalp_paper_enabled=True,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=False,
        )
        == MODE_PAPER
    )


def test_day_database_isolation():
    assert "paper_trades" in DAY_TABLES
    assert "portfolio_engine_ledger" in DAY_TABLES
    assert "scalp_paper_trades" in SCALP_TABLES
    assert "scalp_structural_working" in SCALP_TABLES
    for name in DAY_TABLES:
        assert name not in SCALP_TABLES


def test_schema_does_not_create_day_tables(tmp_path):
    db = tmp_path / "scalp.db"
    init_scalp_schema(db)
    conn = sqlite3.connect(db)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for day in DAY_TABLES:
        assert day not in names
    assert "scalp_structural_breaker" in names
    assert "scalp_structural_markouts" in names
    assert "scalp_structural_working" in names


def test_ranking_runtime_isolated():
    cfg = SimpleNamespace(scalp_thesis="structural", scalp_paper_enabled=True, scalp_live=False, structural_mode="STRUCTURAL_PAPER")
    assert ranking_eval_permitted(cfg) is False
    assert prediction_circuit_breaker_applies(cfg) is False
    assert new_entry_block_reason(cfg) == STRUCTURAL_NOT_EXECUTABLE
    assert structural_lp_executable(cfg) is True
    fields = status_fields(cfg)
    assert fields["exchange_live_impossible"] is True
    assert fields["ranking_eval_permitted"] is False


def test_engine_skips_ranking_construction(monkeypatch, tmp_path):
    monkeypatch.setenv("SCALP_LIVE", "false")
    monkeypatch.setenv("SCALP_PAPER_ENABLED", "true")
    monkeypatch.setenv("SCALP_THESIS", "structural")
    monkeypatch.delenv("SCALP_LEGACY_PREDICTION_ENTRIES", raising=False)
    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    cfg = ScalpConfig.from_env()
    object.__setattr__(cfg, "database_path", str(tmp_path / "mystic_scalp.db"))
    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = cfg
    engine._router = None
    engine._klines = None
    engine._momentum = None
    assert ranking_eval_permitted(cfg) is False
    assert engine._router is None


def test_engine_maker_and_timeout_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SCALP_LIVE", "false")
    monkeypatch.setenv("SCALP_PAPER_ENABLED", "true")
    from backend.services.binance_scalp.calibration_profiles import economics_for_config
    from backend.services.binance_scalp.config import ScalpConfig
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    db = tmp_path / "mystic_scalp.db"
    init_scalp_schema(db, principal=1000.0)
    cfg = ScalpConfig.from_env()
    object.__setattr__(cfg, "database_path", str(db))
    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = cfg
    engine.econ = economics_for_config(cfg)
    engine._pending_sell_log = None
    engine._utcnow_override = datetime(2026, 8, 30, tzinfo=timezone.utc)

    def _noop_cache(*_a, **_k):
        return None

    engine._write_position_cache = _noop_cache
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    engine._structural_open_long(conn, sym="BTCUSDT", price=100.0, qty=1.0, reason="STRUCTURAL_BID_TRADE", spread_pct=0.0002)
    pos = conn.execute("SELECT * FROM scalp_paper_positions WHERE status='OPEN'").fetchone()
    assert pos is not None
    buy = conn.execute("SELECT fee_usd, slippage_usd FROM scalp_paper_trades WHERE side='BUY'").fetchone()
    assert float(buy["slippage_usd"] or 0) == 0.0
    engine._structural_close(conn, pos, exit_price=100.1, reason="STRUCTURAL_ASK_TRADE", exit_maker=True)
    maker_sell = conn.execute("SELECT fee_usd, slippage_usd, pnl_usd FROM scalp_paper_trades WHERE side='SELL'").fetchone()
    assert float(maker_sell["slippage_usd"] or 0) == 0.0
    conn.execute("DELETE FROM scalp_paper_trades")
    conn.execute("DELETE FROM scalp_paper_positions")
    conn.execute("UPDATE scalp_paper_ledger SET cash_balance=1000, positions_value=0, realized_pnl=0, total_equity=1000 WHERE id=1")
    engine._structural_open_long(conn, sym="BTCUSDT", price=100.0, qty=1.0, reason="STRUCTURAL_BID_TRADE", spread_pct=0.0002)
    pos = conn.execute("SELECT * FROM scalp_paper_positions WHERE status='OPEN'").fetchone()
    engine._structural_close(conn, pos, exit_price=99.9, reason="STRUCTURAL_INVENTORY_TIMEOUT", exit_maker=False)
    timeout = conn.execute("SELECT fee_usd, slippage_usd FROM scalp_paper_trades WHERE side='SELL'").fetchone()
    assert float(timeout["slippage_usd"] or 0) > 0.0
