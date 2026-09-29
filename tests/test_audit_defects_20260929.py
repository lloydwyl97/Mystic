"""Regression tests A-AB for the 2026-09-29 audit defects.

Learning A-H, dust I-M, external balance N-P, sell aggregation Q-U,
exit labels V-Y, model Z/AA, setup propagation AB.
"""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services import audit_defect_repair as rep
from backend.services import engine_strategy_dust as esd
from backend.services.learning_provenance import (
    capture_close_provenance,
    close_is_dust,
    learning_provenance,
)
from backend.services.portfolio_engine import ExitType, OpenPosition, PortfolioEngine, Sleeve, strategy_exit_type
from tests.test_scalp_v2_exit_reason_persistence import _run_sell
from tests.test_scalp_v2_learning_exit_labels import _learning_row
from tests.test_sell_cash_credit import _init_test_db

# ------------------------------------------------------------------ learning


def _pos(status="ACTIVE", engine="SCALP_V2", tid="scalp_v2_XRPUSDT_1"):
    return SimpleNamespace(engine_id=engine, trade_id=tid, status=status, quantity=10.0, scalp_opportunity_id="", entry_thesis="")


def test_a_capture_marks_active_lot_as_strategy_close():
    prov = capture_close_provenance(_pos(), exit_trigger="SCALP_V2_NET_PROFIT", sell_qty=9.9)
    assert prov["pre_close_status"] == "ACTIVE"
    assert prov["is_strategy_close"] is True
    assert prov["original_trade_id"] == "scalp_v2_XRPUSDT_1"
    assert prov["sell_qty"] == 9.9


def test_b_residual_dust_after_real_sell_is_not_a_dust_close():
    pos = _pos()
    pos._close_provenance = capture_close_provenance(pos, exit_trigger="SCALP_V2_NET_PROFIT", sell_qty=9.9)
    pos.status = "DUST_PENDING"
    assert close_is_dust(pos, "NET_PROFIT_EXIT") is False


def test_c_genuine_dust_and_writeoff_stay_dust():
    pos = _pos(status="DUST_PENDING")
    pos._close_provenance = capture_close_provenance(pos, exit_trigger="DUST_CLEANUP", sell_qty=0.1)
    assert close_is_dust(pos, "NET_PROFIT_EXIT") is True
    assert close_is_dust(_pos(), "DUST_WRITEOFF") is True
    assert close_is_dust(_pos(status="DUST_PENDING"), "TRAILING_STOP_EXIT") is True


def test_d_learning_row_carries_provenance_and_strategy_label(tmp_path):
    db = str(tmp_path / "p.db")
    sqlite3.connect(db).close()
    pos = _pos()
    pos._close_provenance = capture_close_provenance(pos, exit_trigger="SCALP_V2_NET_PROFIT", sell_qty=9.9)
    pos._close_provenance["residual_qty"] = 0.1
    pos.status = "DUST_PENDING"
    prov = learning_provenance(db, pos, "NET_PROFIT_EXIT")
    assert prov["is_dust"] is False
    assert prov["label_strategy"] == "scalp"
    assert prov["pre_close_status"] == "ACTIVE"
    assert prov["residual_qty"] == 0.1
    assert prov["is_strategy_close"] is True


def test_d2_stale_provenance_from_another_lot_is_ignored():
    pos = _pos(status="DUST_PENDING")
    pos._close_provenance = capture_close_provenance(_pos(tid="other"), exit_trigger="X", sell_qty=1.0)
    assert close_is_dust(pos, "NET_PROFIT_EXIT") is True


def _fill(qty, px, ts, oid="1"):
    return {"id": 7, "executed_qty": qty, "avg_fill_price": px, "event_ts_exchange": ts, "exchange_order_id": oid}


def test_e_real_close_needs_executable_venue_sell_near_exit():
    t = 1790684238.6
    ev = rep.real_close_evidence(t, [_fill(59.7, 1.52, t - 3.0, "506876898")])
    assert ev and ev["exchange_order_id"] == "506876898" and ev["notional"] > 90
    assert rep.real_close_evidence(t, [_fill(0.08, 1.52, t - 3.0)]) is None
    assert rep.real_close_evidence(t, [_fill(59.7, 1.52, t - 3600.0)]) is None
    assert rep.real_close_evidence(t, []) is None


def test_f_repair_changes_metadata_only():
    extra = {"engine_id": "DAY_V2", "trade_id": "mystic_XRP/USDT_1", "strategy": "day", "is_dust": True, "label_strategy": "dust", "setup": "HTF_TREND_PULLBACK"}
    ev = {"exchange_order_id": "9", "fill_row_id": 3, "notional": 90.7, "sell_qty": 59.7}
    out = rep.repaired_learning_extra(extra, ev, residual_qty=0.08804)
    assert out["is_dust"] is False and out["label_strategy"] == "day"
    assert out["pre_close_status"] == "ACTIVE" and out["residual_qty"] == 0.08804
    assert out["setup"] == "HTF_TREND_PULLBACK"
    assert set(extra) <= set(out)
    assert not {"net_profit_usd", "exit_price", "fees_paid", "exit_timestamp", "quantity"} & set(out)


def _tlo_db(tmp_path):
    from backend.services.ai_learning_ingestion import _ensure_scalp_outcomes_table

    db = str(tmp_path / "t.db")
    with sqlite3.connect(db) as c:
        c.execute(
            """CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, symbol TEXT, entry_timestamp REAL, exit_timestamp REAL,
               entry_price REAL, exit_price REAL, quantity REAL, fees_paid REAL, slippage_cost REAL, net_profit_usd REAL,
               net_profit_pct REAL, hold_seconds REAL, close_reason TEXT, extra_json TEXT)"""
        )
        c.execute(
            "INSERT INTO trade_learning_outcomes VALUES (1,'XRP/USDT',1,2,1.5,1.52,10,0.01,0,0.19,0.012,60,'NET_PROFIT_EXIT',?)",
            (json.dumps({"engine_id": "SCALP_V2", "is_dust": True, "setup": "VWAP"}),),
        )
    _ensure_scalp_outcomes_table(db)
    return db


def test_g_scalp_ingest_skips_mislabeled_row_until_repaired(tmp_path):
    from backend.services.ai_learning_ingestion import ingest_scalp_outcomes

    db = _tlo_db(tmp_path)
    ingest_scalp_outcomes(db)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM scalp_learning_outcomes").fetchone()[0] == 0
        extra = rep.repaired_learning_extra(
            json.loads(c.execute("SELECT extra_json FROM trade_learning_outcomes").fetchone()[0]), {"exchange_order_id": "1", "fill_row_id": 1, "notional": 15, "sell_qty": 10}, residual_qty=0
        )
        c.execute("UPDATE trade_learning_outcomes SET extra_json=?", (json.dumps(extra),))
    ingest_scalp_outcomes(db)
    ingest_scalp_outcomes(db)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM scalp_learning_outcomes WHERE source_id=1").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_h_scalp_real_close_with_residual_learns_as_scalp(tmp_path):
    run = await _run_sell(tmp_path, exit_type=ExitType.STRATEGY, trigger="SCALP_V2_NET_PROFIT", engine_id="SCALP_V2")
    assert run.result is not None
    row = _learning_row(run.db_path)
    assert row["extra"]["is_dust"] is False
    assert row["extra"]["label_strategy"] == "scalp"
    assert row["extra"]["pre_close_status"] == "ACTIVE"


# ---------------------------------------------------------------------- dust


def _positions_db(tmp_path):
    db = str(tmp_path / "d.db")
    with sqlite3.connect(db) as c:
        c.execute(
            """CREATE TABLE portfolio_engine_positions (engine_id TEXT, symbol TEXT, trade_id TEXT, quantity REAL, entry_price REAL,
               status TEXT, entry_time REAL, entry_order_id TEXT, PRIMARY KEY (engine_id, symbol))"""
        )
    return db


def test_i_reentry_preserves_engine_dust(tmp_path):
    db = _positions_db(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO portfolio_engine_positions VALUES ('SCALP_V2','XRP/USDT','old',0.09474,1.5,'DUST_PENDING',1,'o1')")
        kept = esd.preserve_overwritten_dust(c, engine_id="SCALP_V2", symbol="XRP/USDT", new_trade_id="new")
        assert kept == {"engine_id": "SCALP_V2", "symbol": "XRP/USDT", "source_trade_id": "old", "quantity": 0.09474}
        assert esd.held_quantity(c, "XRP/USDT") == pytest.approx(0.09474)
        esd.preserve_overwritten_dust(c, engine_id="SCALP_V2", symbol="XRP/USDT", new_trade_id="new")
        assert len(esd.held_lots(c, "XRP/USDT")) == 1


def test_i2_retired_held_dust_leaves_no_remaining_on_its_buy(tmp_path):
    db = _positions_db(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, trade_id TEXT, side TEXT, remaining_position REAL)")
        c.execute("INSERT INTO paper_trades VALUES (1,'old','BUY',0.0009166), (2,'other','BUY',0.5)")
        c.execute("INSERT INTO portfolio_engine_positions VALUES ('SCALP_V2','SOL/USDT','old',0.0009166,200,'DUST_PENDING',1,'o1')")
        esd.preserve_overwritten_dust(c, engine_id="SCALP_V2", symbol="SOL/USDT", new_trade_id="new")
        assert esd.retire_held_dust(c, "old", event_class=esd.EVENT_CONVERSION, venue_ref="2468948737")
        assert c.execute("SELECT remaining_position FROM paper_trades ORDER BY id").fetchall() == [(0,), (0.5,)]
        assert not esd.retire_held_dust(c, "old", event_class=esd.EVENT_CONVERSION)


def test_j_active_row_or_same_trade_is_not_preserved(tmp_path):
    db = _positions_db(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO portfolio_engine_positions VALUES ('DAY_V2','BTC/USDT','a',0.001,60000,'ACTIVE',1,'o')")
        assert esd.preserve_overwritten_dust(c, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id="b") is None
        c.execute("UPDATE portfolio_engine_positions SET status='DUST_PENDING'")
        assert esd.preserve_overwritten_dust(c, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id="a") is None
        assert esd.preserve_overwritten_dust(c, engine_id="SCALP_V2", symbol="BTC/USDT", new_trade_id="b") is None


def test_k_identity_accounts_for_every_bucket():
    ident = esd.symbol_inventory_identity(
        exchange_qty=34.755,
        active={"SCALP_V2": 32.29354},
        lot_dust={"DAY_V2": 0.08804},
        held_dust={"SCALP_V2": 0.09474},
        protected=2.27868,
    )
    assert ident["residue"] == pytest.approx(0.0, abs=1e-9)


def test_l_engine_dust_is_never_protected_inventory():
    ident = esd.symbol_inventory_identity(exchange_qty=34.755, active={"SCALP_V2": 32.29354}, lot_dust={"DAY_V2": 0.08804}, held_dust={}, protected=2.37342)
    assert ident["residue"] == pytest.approx(0.0, abs=1e-9)
    fixed = esd.symbol_inventory_identity(exchange_qty=34.755, active={"SCALP_V2": 32.29354}, lot_dust={"DAY_V2": 0.08804}, held_dust={"SCALP_V2": 0.09474}, protected=2.37342 - 0.09474)
    assert fixed["residue"] == pytest.approx(0.0, abs=1e-9)
    assert fixed["protected"] == pytest.approx(2.27868)


@pytest.mark.asyncio
async def test_m_engine_persist_keeps_dust_and_one_slot(tmp_path):
    db = tmp_path / "e.db"
    _init_test_db(db, cash=1000.0)
    engine = PortfolioEngine(db_path=str(db), principal=1000.0, test_mode=True)
    await engine.initialize_from_db()

    def lot(tid, qty, status):
        p = OpenPosition(
            symbol="XRP/USDT",
            quantity=qty,
            entry_price=1.5,
            entry_time=time.time(),
            trade_id=tid,
            stop_price=0.0,
            take_profit_1_price=0.0,
            take_profit_2_price=0.0,
            sleeve=Sleeve.ACTIVE.value,
            engine_id="SCALP_V2",
        )
        p.status = status
        return p

    await engine._persist_position_to_sqlite(lot("old", 0.09474, "DUST_PENDING"))
    await engine._persist_position_to_sqlite(lot("new", 32.29354, "ACTIVE"))
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT trade_id, status FROM portfolio_engine_positions WHERE engine_id='SCALP_V2' AND symbol='XRP/USDT'").fetchall()
    assert rows == [("new", "ACTIVE")]
    assert engine._held_engine_dust_qty("XRP/USDT") == pytest.approx(0.09474)


# --------------------------------------------------------- external balance

DUST_LOG = [
    {
        "operateTime": 1790688254000,
        "userAssetDribbletDetails": [
            {"fromAsset": "BTC", "amount": "0.00003823", "tranId": 2468948736, "transferedAmount": "2.1", "serviceChargeAmount": "0.1"},
            {"fromAsset": "SOL", "amount": "0.0045836", "tranId": 2468948737, "transferedAmount": "0.9", "serviceChargeAmount": "0.05"},
        ],
    }
]


def test_n_conversion_matches_asset_after_entry_only():
    m = esd.match_dust_conversion(DUST_LOG, asset="BTC", after_epoch=1790679671.0)
    assert m["tran_id"] == "2468948736" and m["amount"] == pytest.approx(0.00003823)
    assert esd.match_dust_conversion(DUST_LOG, asset="BTC", after_epoch=1790688300.0) is None
    assert esd.match_dust_conversion(DUST_LOG, asset="ETH", after_epoch=0) is None


def test_o_external_event_never_a_manual_sell(tmp_path):
    with sqlite3.connect(str(tmp_path / "x.db")) as c:
        for bad in ("MANUAL_EXIT", "HUMAN_MANUAL_SELL", "MANUAL"):
            with pytest.raises(ValueError):
                esd.record_external_balance_event(c, symbol="BTC/USDT", quantity=1e-5, event_class=bad, source="t")
        assert esd.record_external_balance_event(c, symbol="BTC/USDT", quantity=1e-5, event_class=esd.EVENT_CONVERSION, source="t", source_trade_id="a", venue_ref="1")
        assert not esd.record_external_balance_event(c, symbol="BTC/USDT", quantity=1e-5, event_class=esd.EVENT_CONVERSION, source="t", source_trade_id="a", venue_ref="1")


async def _phantom_engine(tmp_path, rows):
    db = tmp_path / "ph.db"
    _init_test_db(db, cash=1000.0)
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status, strategy_id) "
            "VALUES ('mystic_BTC/USDT_1790679671164', 'r', 'live', 'BTC/USDT', 'BUY', 0.00076985, 60000, 9.85e-06, datetime('now'), 'executed', 'day')"
        )
    engine = PortfolioEngine(db_path=str(db), principal=1000.0, test_mode=True)
    await engine.initialize_from_db()
    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=9.85e-06,
        entry_price=60000.0,
        entry_time=1790679671.0,
        trade_id="mystic_BTC/USDT_1790679671164",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        sleeve=Sleeve.ACTIVE.value,
        engine_id="DAY_V2",
    )
    pos.status = "DUST_PENDING"
    engine.open_positions["DAY_V2::BTC/USDT"] = pos
    engine._venue_dust_conversions = AsyncMock(return_value=rows)
    engine._delete_position_from_sqlite = AsyncMock()
    engine._persist_ledger_to_sqlite = AsyncMock()
    engine._record_learning_outcome = AsyncMock()
    engine._record_position_close_ledger = AsyncMock()
    engine.record_sell_cooldown = MagicMock()
    return engine, pos, str(db)


def _events(db):
    with sqlite3.connect(db) as c:
        esd.ensure_schema(c)
        return c.execute("SELECT event_class, venue_ref, quantity FROM external_balance_events").fetchall()


@pytest.mark.asyncio
async def test_p_phantom_dust_retired_as_conversion_without_pnl_or_learning(tmp_path):
    engine, pos, db = await _phantom_engine(tmp_path, DUST_LOG)
    assert await engine._retire_phantom_strategy_dust("BTC/USDT", pos, source="test") is True
    assert "DAY_V2::BTC/USDT" not in engine.open_positions
    assert _events(db) == [(esd.EVENT_CONVERSION, "2468948736", 9.85e-06)]
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT remaining_position FROM paper_trades WHERE trade_id=?", (pos.trade_id,)).fetchone()[0] == 0
    engine._record_learning_outcome.assert_not_awaited()
    engine._record_position_close_ledger.assert_not_awaited()
    engine.record_sell_cooldown.assert_not_called()


@pytest.mark.asyncio
async def test_p2_unproven_removal_waits_then_is_unattributed(tmp_path):
    engine, pos, db = await _phantom_engine(tmp_path, [])
    assert await engine._retire_phantom_strategy_dust("BTC/USDT", pos, source="test") is True
    assert "DAY_V2::BTC/USDT" in engine.open_positions and _events(db) == []
    engine._phantom_dust_first_absent[pos.trade_id] = time.time() - 4000
    assert await engine._retire_phantom_strategy_dust("BTC/USDT", pos, source="test") is True
    assert [e[0] for e in _events(db)] == [esd.EVENT_UNATTRIBUTED]
    engine._record_learning_outcome.assert_not_awaited()


# ---------------------------------------------------------- sell aggregation


def _chunk(oid, qty, px, fee, fills=True):
    o = {"id": oid, "filled": qty, "average": px, "cost": qty * px, "status": "closed", "fee": {"cost": fee, "currency": "USDT"}, "info": {"orderId": oid}}
    if fills:
        o["info"]["fills"] = [{"qty": str(qty), "price": str(px), "commission": str(fee), "commissionAsset": "USDT", "tradeId": f"t{oid}"}]
    return o


def test_q_combined_order_sums_cost_fee_and_keeps_all_ids():
    from backend.services.day_mandatory_exit_execution import _combine_orders

    out = _combine_orders([_chunk("506876828", 45.4, 1.5202, 0.0138), _chunk("506876898", 14.3, 1.5203, 0.00434749)], 59.7)
    assert out["filled"] == pytest.approx(59.7)
    assert out["cost"] == pytest.approx(45.4 * 1.5202 + 14.3 * 1.5203)
    assert out["average"] == pytest.approx(out["cost"] / 59.7)
    assert out["_mystic_order_ids"] == ["506876828", "506876898"]
    assert len(out["info"]["fills"]) == 2
    assert out["fee"]["cost"] == pytest.approx(0.0138 + 0.00434749)


def test_r_chunk_without_fills_contributes_synthetic_fill():
    from backend.services.day_mandatory_exit_execution import _combine_orders

    out = _combine_orders([_chunk("1", 1.0, 10.0, 0.002, fills=False), _chunk("2", 1.0, 10.2, 0.00204)], 2.0)
    assert len(out["info"]["fills"]) == 2
    assert out["fee"]["cost"] == pytest.approx(0.00404)


def test_s_economics_read_every_chunk_fee():
    from backend.services.day_mandatory_exit_execution import _combine_orders
    from backend.services.live_fill_economics import extract_live_commission

    out = _combine_orders([_chunk("1", 1.0, 10.0, 0.002), _chunk("2", 3.0, 10.0, 0.006)], 4.0)
    assert extract_live_commission(out, symbol="SOL/USDT", fill_price=10.0).usd == pytest.approx(0.008)


@pytest.mark.asyncio
async def test_t_verify_does_not_replace_combined_fill_with_last_chunk(tmp_path):
    db = tmp_path / "v.db"
    _init_test_db(db, cash=1000.0)
    engine = PortfolioEngine(db_path=str(db), principal=1000.0, test_mode=True)
    engine._live_service = MagicMock()
    engine._attach_venue_trades = AsyncMock(side_effect=AssertionError("must not re-fetch"))
    order = {"id": "2", "filled": 4.0, "_mystic_order_ids": ["1", "2"]}
    assert await engine._verify_order_fill(order, "SOLUSDT", "sell") is order


def test_u_historical_chunk_repair_is_exact_or_nothing():
    row = {"executed_qty": 59.7, "avg_fill_price": (45.4 * 1.5202 + 14.3 * 1.5203) / 59.7, "exchange_order_id": "506876898", "fee_amount": 0.00434749}
    trades = [
        {"side": "sell", "order": "506876828", "amount": 45.4, "price": 1.5202, "cost": 45.4 * 1.5202, "fee": {"cost": 0.0138, "currency": "USDT"}, "id": "a"},
        {"side": "sell", "order": "506876898", "amount": 14.3, "price": 1.5203, "cost": 14.3 * 1.5203, "fee": {"cost": 0.00434749, "currency": "USDT"}, "id": "b"},
    ]
    v = rep.verify_chunk_trades(row, trades)
    assert v and v["chunks"] == 2
    assert rep.fee_correction(row["fee_amount"], v) == pytest.approx(0.0138)
    assert rep.verify_chunk_trades(row, trades[:1]) is None
    assert rep.verify_chunk_trades({**row, "exchange_order_id": "999"}, trades) is None
    assert rep.fee_correction(0.0, {"fees": {"BNB": 0.001}}) is None


# ---------------------------------------------------------------- exit labels

DAY_FIVE = [
    ("DAY_V2_CATASTROPHIC_PROTECTION", "STOP_LOSS_EXIT"),
    ("DAY_V2_STRUCTURAL_INVALIDATION", "THESIS_INVALIDATION_EXIT"),
    ("DAY_V2_WINNER_PROTECTION", "TRAILING_STOP_EXIT"),
    ("DAY_V2_OBJECTIVE_COMPLETE", "NET_PROFIT_EXIT"),
    ("DAY_V2_TIME_EXPIRATION", "TIME_STOP_EXIT"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger", "expected"), [*DAY_FIVE, ("SCALP_V2_TIME_STOP", "TIME_STOP_EXIT")])
async def test_v_strategy_exit_carries_real_reason_through_every_layer(tmp_path, trigger, expected):
    engine_id = "SCALP_V2" if trigger.startswith("SCALP") else "DAY_V2"
    run = await _run_sell(tmp_path, exit_type=ExitType.STRATEGY, trigger=trigger, engine_id=engine_id)
    assert run.result is not None
    assert run.gate.await_args.kwargs["exit_trigger"] != "MANUAL_EXIT"
    assert run.idempotency.call_args.args[3] != "MANUAL_EXIT"
    assert run.engine._record_position_close_ledger.await_args.kwargs["close_reason"] == expected
    assert _learning_row(run.db_path)["close_reason"] == expected


@pytest.mark.parametrize(("trigger", "_exp"), [*DAY_FIVE, ("SCALP_V2_NET_PROFIT", "")])
def test_w_known_trigger_maps_to_strategy_exit_type(trigger, _exp):
    assert strategy_exit_type(trigger) == ExitType.STRATEGY
    assert strategy_exit_type("DAY_V2_EXIT") == ExitType.MANUAL


def test_x_forced_strategy_exit_is_not_emergency_but_catastrophic_stays(tmp_path):
    engine = PortfolioEngine(db_path=str(tmp_path / "x.db"), principal=1000.0, test_mode=True)
    assert engine._is_emergency_sell(ExitType.STRATEGY, "DAY_V2_OBJECTIVE_COMPLETE", force_sell=True) is False
    assert engine._is_emergency_sell(ExitType.MANUAL, "MANUAL_EXIT", force_sell=True) is True
    assert engine._is_emergency_sell(ExitType.STRATEGY, "EXTREME_PROTECTION_CRASH", force_sell=True) is True


def test_y_forced_strategy_exit_is_a_mandatory_flatten():
    from backend.services.day_mandatory_exit_execution import is_mandatory_day_flatten

    assert is_mandatory_day_flatten("NET_PROFIT_EXIT", force_sell=True, exit_type_name="STRATEGY") is True
    assert is_mandatory_day_flatten("TIME_STOP_EXIT", force_sell=True, exit_type_name="STRATEGY") is True


@pytest.mark.asyncio
async def test_y2_strategy_forced_fill_log_is_not_emergency_bypass(tmp_path, caplog):
    caplog.set_level("INFO")
    await _run_sell(tmp_path, exit_type=ExitType.STRATEGY, trigger="DAY_V2_OBJECTIVE_COMPLETE", engine_id="DAY_V2")
    assert "emergency bypass" not in caplog.text


# --------------------------------------------------------------------- model


def test_z_no_new_labels_gives_identical_boundary():
    from backend.ai_training_pipeline import training_label_boundary

    rows = [{"id": 5}, {"id": 9}]
    selfs = [{"symbol": "BTCUSDT", "label_anchor_4h_open_ms": 100}, {"symbol": "BTCUSDT", "label_anchor_4h_open_ms": 100}]
    b = training_label_boundary(rows, selfs)
    assert b == (9, 2, 100, 1)
    assert training_label_boundary(list(rows), list(selfs)) == b
    assert training_label_boundary([*rows, {"id": 10}], selfs) != b
    assert training_label_boundary(rows, [*selfs, {"symbol": "ETHUSDT", "label_anchor_4h_open_ms": 100}]) != b


def test_aa_promotion_holdout_rows_are_excluded_from_training():
    from backend import ai_training_pipeline as p

    rows = [{"id": i} for i in range(1, 11)]
    with patch("backend.services.ai_model_promotion_holdout.holdout_window", return_value={"ids": [8, 9, 10], "n": 3}):
        kept, windows = p._exclude_promotion_holdout(rows, "day", 145, 5)
    assert {r["id"] for r in kept} == set(range(1, 8))
    assert all(set(w["ids"]).isdisjoint(r["id"] for r in kept) for w in windows.values())


# --------------------------------------------------------------------- setup


@pytest.mark.asyncio
async def test_ab_day_sell_row_inherits_setup_from_buy_row(tmp_path):
    import tests.test_scalp_v2_exit_reason_persistence as harness

    seed = harness._seed_btc_position

    def seed_with_setup(db_path, **kw):
        seed(db_path, **kw)
        with sqlite3.connect(db_path) as c:
            c.execute("UPDATE paper_trades SET explainability_json=? WHERE side='BUY'", (json.dumps({"setup_type_canonical": "RANGE_BOUNCE", "day_route_regime": "range"}),))

    with patch.object(harness, "_seed_btc_position", seed_with_setup):
        run = await _run_sell(tmp_path, exit_type=ExitType.STRATEGY, trigger="DAY_V2_OBJECTIVE_COMPLETE", engine_id="DAY_V2")
    with sqlite3.connect(run.db_path) as c:
        sell = json.loads(c.execute("SELECT explainability_json FROM paper_trades WHERE side='SELL'").fetchone()[0] or "{}")
    assert sell["setup_type_canonical"] == "RANGE_BOUNCE"
    assert sell["day_route_regime"] == "range"


def test_ab2_label_plan_backfills_day_sell_from_buy(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("rad", Path(__file__).resolve().parents[1] / "scripts" / "repair_audit_defects_20260929.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    db = str(tmp_path / "s.db")
    with sqlite3.connect(db) as c:
        c.executescript(
            """
            CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, trade_id TEXT, engine_id TEXT, symbol TEXT, side TEXT, timestamp TEXT,
                order_id TEXT, explainability_json TEXT, scalp_opportunity_id TEXT, decision_id TEXT, exit_reason TEXT);
            CREATE TABLE live_exchange_fills (id INTEGER PRIMARY KEY, exchange_order_id TEXT, mystic_trade_id TEXT);
            CREATE TABLE position_close_ledger (id INTEGER PRIMARY KEY, symbol TEXT, close_reason TEXT, sell_trade_id TEXT, closed_at TEXT);
            CREATE TABLE day_outcome_attribution (id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, closed_at_utc TEXT);
            CREATE TABLE scalp_v2_opportunities (id INTEGER PRIMARY KEY, opportunity_id TEXT, setup_family TEXT);
            INSERT INTO paper_trades VALUES (2653,'mystic_ETH/USDT_1','DAY_V2','ETH/USDT','BUY','t','b1','{"setup_type_canonical":"RANGE_BOUNCE"}',NULL,'d',NULL);
            INSERT INTO paper_trades VALUES (2657,'mystic_sell_ETH/USDT_2','DAY_V2','ETH/USDT','SELL','t','s1','{}',NULL,'d','NET_PROFIT_EXIT');
            INSERT INTO live_exchange_fills VALUES (1,'s1','mystic_ETH/USDT_1');
            INSERT INTO position_close_ledger VALUES (1,'ETH/USDT','MANUAL_EXIT','mystic_sell_ETH/USDT_2','t');
            INSERT INTO day_outcome_attribution VALUES (1,'scalp_v2_XRPUSDT_1','XRP/USDT','t'), (2,'mystic_ETH/USDT_1','ETH/USDT','t');
            """
        )
        plan = mod.plan_label_repairs(c)
        assert plan["setup"] == [
            {"id": 2657, "engine_id": "DAY_V2", "symbol": "ETH/USDT", "timestamp": "t", "buy_trade_id": "mystic_ETH/USDT_1", "setup": "RANGE_BOUNCE", "source": "buy_row_explainability"}
        ]
        assert plan["close_ledger"][0]["after"] == "NET_PROFIT_EXIT"
        assert [r["id"] for r in plan["scalp_in_day_attribution"]] == [1]
        mod.apply_label_repairs(c, plan)
        ex = json.loads(c.execute("SELECT explainability_json FROM paper_trades WHERE id=2657").fetchone()[0])
        assert ex["setup_type_canonical"] == "RANGE_BOUNCE"
        assert c.execute("SELECT close_reason FROM position_close_ledger").fetchone()[0] == "NET_PROFIT_EXIT"
        assert [tuple(r) for r in c.execute("SELECT trade_id FROM day_outcome_attribution")] == [("mystic_ETH/USDT_1",)]
        assert c.execute("SELECT COUNT(*) FROM ownership_repair_backup").fetchone()[0] == 3
