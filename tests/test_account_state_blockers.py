"""Account-state, reservation, candle-boundary, and authority proofs."""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

from backend.config.canonical_candle_intervals import stale_after_sec
from backend.config.day_entry_execution import (
    ENTRY_AUTHORITY_DAY_V2_CONFIRMED,
    ENTRY_AUTHORITY_SCALP_V2_CONFIRMED,
    ENTRY_AUTHORITY_SCALP_V2_LIVE,
    ENTRY_AUTHORITY_TRAILING_BUY,
    accepted_live_buy_authority,
)
from backend.services.candle_contract import candle_contract_matrix
from backend.services.day_entry_reservations import create_reservation, release_orphan_reservations, unreserved_cash
from backend.services.day_v2.candle_wait import claim_bar, claim_result, evaluate_candle_gate, note_pending
from backend.services.day_v2.cycle_gate import required_15m_open
from backend.services.live_fifo_performance import strategy_performance_status
from backend.services.protected_external_inventory import (
    PROTECTED_EXTERNAL_INVENTORY,
    exit_route,
    reclassify_unowned_imports,
    strategy_sell_quantity,
)
from backend.services.scalp_v2.opportunity import SCALP_V2_OPP_EXPIRY_SEC, arm_opportunity, reap_expired_armed
from backend.services.two_engine_claim import claim_symbol, held_slot_count


def test_protected_inventory_does_not_consume_a_slot(tmp_path):
    protected = SimpleNamespace(status=PROTECTED_EXTERNAL_INVENTORY, engine_id=PROTECTED_EXTERNAL_INVENTORY, quantity=0.425)
    strategy = SimpleNamespace(status="ACTIVE", engine_id="SCALP_V2", quantity=0.1)
    assert held_slot_count({"SOL/USDT": protected, "BTC/USDT": strategy}) == 1
    ok, reason, _rid = claim_symbol(tmp_path / "claim.db", "SOL/USDT", "DAY_V2", "dec-sol", 25.0, positions={"SOL/USDT": protected})
    assert ok, reason


def test_strategy_sell_uses_only_its_own_quantity():
    assert strategy_sell_quantity(0.01, 0.435) == 0.01
    assert strategy_sell_quantity(0.0, 22.5) == 0.0
    assert exit_route(PROTECTED_EXTERNAL_INVENTORY) == "NONE"
    assert exit_route("LEGACY_DAY_LIVE") == "FAIL_CLOSED"
    assert exit_route("LEGACY_EXIT_ONLY") == "FAIL_CLOSED"
    assert exit_route("SOMETHING_ELSE") == "FAIL_CLOSED"
    assert exit_route("DAY_V2") == "DAY_V2"
    assert exit_route("SCALP_V2") == "SCALP_V2"


def test_reconcile_import_is_protected_and_survives_reload(tmp_path):
    db = tmp_path / "acct.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE portfolio_engine_positions (
            symbol TEXT PRIMARY KEY, quantity REAL, entry_price REAL, trade_id TEXT, entry_order_id TEXT, engine_id TEXT
        )
        """
    )
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('SOL/USDT', 0.425, 117.2, 'reconcile_import_SOL_USDT_1', '', 'LEGACY_EXIT_ONLY')")
    conn.commit()
    moved = reclassify_unowned_imports(conn)
    conn.commit()
    assert moved == ["SOL/USDT"]
    assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] == 0
    row = conn.execute("SELECT quantity, classification FROM protected_external_inventory").fetchone()
    assert row[0] == 0.425
    assert row[1] == PROTECTED_EXTERNAL_INVENTORY
    assert reclassify_unowned_imports(conn) == []
    conn.close()
    again = sqlite3.connect(db)
    kept = again.execute("SELECT classification, quantity FROM protected_external_inventory").fetchone()
    again.close()
    assert kept == (PROTECTED_EXTERNAL_INVENTORY, 0.425)


def test_reserved_import_keeps_scalp_identity(tmp_path):
    db = tmp_path / "owned.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE portfolio_engine_positions (
            symbol TEXT PRIMARY KEY, quantity REAL, entry_price REAL, trade_id TEXT,
            entry_order_id TEXT, engine_id TEXT, scalp_opportunity_id TEXT, entry_decision_id TEXT
        )
        """
    )
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('ETH/USDT', 0.0059, 2689.81, 'reconcile_import_ETH_USDT_1', '', 'LEGACY_DAY_LIVE', '', '')")
    conn.commit()
    conn.close()
    from backend.services.day_entry_reservations import create_reservation

    create_reservation(db, decision_id="802320234d01fa1b", symbol="ETH/USDT", notional_usd=47.88, sleeve="SCALP_V2", reservation_id="res_eth")
    conn = sqlite3.connect(db)
    moved = reclassify_unowned_imports(conn)
    conn.commit()
    engine, opp, order_id = conn.execute("SELECT engine_id, scalp_opportunity_id, entry_order_id FROM portfolio_engine_positions").fetchone()
    protected = conn.execute("SELECT COUNT(*) FROM protected_external_inventory").fetchone()[0]
    conn.close()
    assert moved == []
    assert engine == "SCALP_V2"
    assert opp == "802320234d01fa1b"
    assert order_id == "1597036548"
    assert protected == 0


def test_orphan_reservation_releases_once(tmp_path):
    db = tmp_path / "res.db"
    ok, reason, rid = create_reservation(db, decision_id="dec-xrp", symbol="XRP/USDT", notional_usd=43.5, sleeve="SCALP_V2")
    assert ok, reason
    first = release_orphan_reservations(db)
    second = release_orphan_reservations(db)
    assert [row["reservation_id"] for row in first] == [rid]
    assert first[0]["state"] == "RELEASED"
    assert second == []
    status = sqlite3.connect(db).execute("SELECT status FROM day_entry_reservations").fetchone()[0]
    assert status == "RELEASED"
    assert unreserved_cash(123.70, 0.0, 0.0) == 123.70


def test_one_actionable_scalp_opportunity_and_expired_does_not_block(tmp_path):
    db = tmp_path / "opp.db"
    _first, blocked_first = arm_opportunity(db, "XRP/USDT", "SCALP", 1.50)
    _second, blocked_second = arm_opportunity(db, "XRP/USDT", "SCALP", 1.20)
    assert blocked_first is False
    assert blocked_second is True
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE scalp_v2_opportunities SET created_at=? WHERE state='ARMED'",
        (time.time() - SCALP_V2_OPP_EXPIRY_SEC - 5,),
    )
    conn.commit()
    conn.close()
    released: list[str] = []
    reap_expired_armed(db, release_reservation=lambda rid, _sym: released.append(rid))
    successor, blocked_successor = arm_opportunity(db, "XRP/USDT", "SCALP", 1.20)
    assert blocked_successor is False
    assert successor
    conn = sqlite3.connect(db)
    states = [row[0] for row in conn.execute("SELECT state FROM scalp_v2_opportunities ORDER BY id")]
    conn.close()
    assert states[0] == "EXPIRED"
    assert states[-1] == "ARMED"
    assert sum(1 for state in states if state == "ARMED") == 1
    conn = sqlite3.connect(db)
    conn.execute("UPDATE scalp_v2_opportunities SET state='EXPIRED' WHERE state='ARMED'")
    conn.commit()
    conn.close()
    reloaded = sqlite3.connect(db).execute("SELECT COUNT(*) FROM scalp_v2_opportunities WHERE state='ARMED'").fetchone()[0]
    assert reloaded == 0


def test_missing_candle_stays_pending_then_one_claim_and_one_timeout(tmp_path):
    db = tmp_path / "day.db"
    now = 1_800_000.0
    required = required_15m_open(now)
    pending = evaluate_candle_gate(
        completed_bar_count=40,
        minimum_bars=32,
        latest_open=required - 900,
        required_open=required,
        executable_price=100.0,
        book_age_sec=1.0,
        book_stale_sec=30.0,
        now=now,
    )
    assert pending["action"] == "pending"
    assert pending["reason"] == "PENDING_CANDLE"
    note_pending(db, "BTCUSDT", required, now=now)
    ready = evaluate_candle_gate(
        completed_bar_count=40,
        minimum_bars=32,
        latest_open=required,
        required_open=required,
        executable_price=100.0,
        book_age_sec=1.0,
        book_stale_sec=30.0,
        now=now + 30,
    )
    assert ready["action"] == "proceed"
    assert claim_bar(db, "BTCUSDT", required, "PROCEEDING") is True
    assert claim_bar(db, "BTCUSDT", required, "PROCEEDING") is False
    assert claim_result(db, "BTCUSDT", required) == "PROCEEDING"
    timeout_at = required + 900 + stale_after_sec("15m") + 1
    timed = evaluate_candle_gate(
        completed_bar_count=40,
        minimum_bars=32,
        latest_open=required - 900,
        required_open=required,
        executable_price=100.0,
        book_age_sec=1.0,
        book_stale_sec=30.0,
        now=timeout_at,
    )
    assert timed["reason"] == "MISSING_COMPLETED_CANDLE_TIMEOUT"
    assert claim_bar(db, "ETHUSDT", required, timed["reason"]) is True
    assert claim_bar(db, "ETHUSDT", required, timed["reason"]) is False


def test_timeframe_ages_differ_and_missing_book_is_absent(tmp_path):
    db = tmp_path / "bars.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    close_at = 1_780_000_000
    fifteen_open = close_at - 900
    four_open = close_at - 14400
    conn.execute(
        "INSERT INTO feature_ohlcv VALUES ('BTC-USDT','15m', ?, 1, 1, 1, 1, 10)",
        (fifteen_open,),
    )
    conn.execute(
        "INSERT INTO feature_ohlcv VALUES ('BTC-USDT','4h', ?, 1, 1, 1, 1, 10)",
        (four_open,),
    )
    conn.commit()
    conn.close()
    matrix = candle_contract_matrix(str(db), now=close_at, books={})
    by_tf = {cell["timeframe"]: cell for cell in matrix["cells"] if cell["symbol"] == "BTC-USDT"}
    assert by_tf["15m"]["age_sec"] == 900
    assert by_tf["4h"]["age_sec"] == 14400
    assert by_tf["15m"]["age_sec"] != by_tf["4h"]["age_sec"]
    assert by_tf["book"]["source"] == "absent"
    assert by_tf["book"]["bid"] is None
    assert by_tf["book"]["ask"] is None
    assert by_tf["book"]["stale"] is True


def test_only_two_buy_authorities_are_accepted():
    assert accepted_live_buy_authority(ENTRY_AUTHORITY_SCALP_V2_CONFIRMED)
    assert accepted_live_buy_authority(ENTRY_AUTHORITY_DAY_V2_CONFIRMED)
    for rejected in (
        ENTRY_AUTHORITY_SCALP_V2_LIVE,
        ENTRY_AUTHORITY_TRAILING_BUY,
        "SCALP_V2_LIVE_ENTRY",
        "process_bar_candidates",
        "LEGACY_DAY_LIVE",
        "",
    ):
        assert not accepted_live_buy_authority(rejected)


def test_imported_order_does_not_rewrite_an_older_paper_lot():
    from backend.services.protected_external_inventory import should_align_paper_remaining

    assert should_align_paper_remaining("reconcile_import_ETH_USDT_1", "1597036548", False) is False
    assert should_align_paper_remaining("mystic_ETH/USDT_1790172572088", "1597036548", False) is False
    assert should_align_paper_remaining("mystic_ETH/USDT_1790172572088", "1594719537", True) is True


def test_unmatched_sells_make_lifetime_performance_unknown():
    status = strategy_performance_status({"exchange_sell_qty": 10.0, "matched_qty": 4.0, "unmatched_sell_qty": 6.0})
    assert status["lifetime_strategy_performance_known"] is False
    assert status["performance_statement"] == "lifetime strategy performance is unknown"
    assert status["coverage_qty"] == 0.4
