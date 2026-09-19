"""The engine's own persist paths must carry the exchange order id.

The identity table is only half the repair. These tests pin the two SQL
statements that write FIFO trade rows, because those were the statements that
dropped the identifier: the BUY insert did not list ``order_id`` at all, and
neither SELL variant did, so ``paper_trades.order_id`` was NULL on every live
row ever written.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from pathlib import Path

from backend.database_schema import initialize_paper_trading_schema
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, Sleeve

ENGINE = Path(__file__).resolve().parents[1] / "backend" / "services" / "portfolio_engine.py"
SOURCE = ENGINE.read_text()


def _insert_statements() -> list[str]:
    """Every INSERT INTO paper_trades statement in the engine, column list only."""
    return re.findall(r"INSERT INTO paper_trades\s*\((.*?)\)\s*VALUES", SOURCE, re.DOTALL)


def _pairs() -> list[tuple[str, str]]:
    return re.findall(r"INSERT INTO paper_trades\s*\((.*?)\)\s*VALUES\s*\((.*?)\)", SOURCE, re.DOTALL)


def test_engine_has_the_expected_number_of_paper_trades_inserts():
    """Guard the scan below: a new insert path must be reviewed for identifiers.

    This was 6. The dust-cleanup SELL insert was removed so leftover dust
    cannot create a completed live trade or invented realized P&L.
    """
    assert len(_insert_statements()) == 5


def test_every_insert_that_can_write_a_live_row_persists_order_id():
    """Only the dust write-off may omit the identifier, because it places no order."""
    missing = []
    for columns, values in _pairs():
        if "'paper'" in values:  # repair-add is paper by construction
            continue
        if "DUST_WRITEOFF" in values:  # no exchange order exists to identify
            continue
        if "order_id" not in columns:
            missing.append(" ".join(columns.split())[:120])
    assert missing == [], f"live-capable insert does not persist order_id: {missing}"


def test_dust_writeoff_is_the_only_identifier_free_exit():
    identifier_free = [v for c, v in _pairs() if "order_id" not in c and "'paper'" not in v]
    assert identifier_free == []


def test_column_and_placeholder_counts_match_on_every_insert():
    """A bind-arity mistake here would abort a settled trade's commit."""
    for columns, values in _pairs():
        n_cols = len([c for c in columns.split(",") if c.strip()])
        parts = [v.strip() for v in values.split(",") if v.strip()]
        n_placeholders = sum(1 for p in parts if p == "?")
        n_literals = len(parts) - n_placeholders
        assert n_cols == n_placeholders + n_literals, f"arity mismatch: {n_cols} columns vs {n_placeholders}?+{n_literals} literals"


def test_manual_flatten_row_persists_any_venue_id_it_was_given():
    """The no-fill-packet fallback no longer discards a reported trade id."""
    assert 'str(fill.get("trade_id") or "") or None' in SOURCE


def test_recovered_close_still_requires_an_exchange_order_id():
    """The packet path refuses to write without one; unchanged by this repair."""
    assert '"reason": "missing_exchange_order_id"' in SOURCE
    assert "exchange_sell_order_id=exchange_order_id" in SOURCE


def test_buy_commit_binds_the_live_order_id():
    """The atomic OPEN must bind the venue order id, not a placeholder."""
    assert 'str((live_order_buy or {}).get("id") or "") or None' in SOURCE


def test_both_sell_variants_bind_the_live_order_id():
    occurrences = SOURCE.count('str((live_order_sell or {}).get("id") or "") or None')
    # One per SELL insert variant (with and without the net-P&L columns).
    assert occurrences == 2


def test_identity_is_recorded_for_both_sides():
    assert "LIVE_BUY_IDENTITY_RECORD_FAILED" in SOURCE
    assert "LIVE_SELL_IDENTITY_RECORD_FAILED" in SOURCE
    assert SOURCE.count("record_fill") >= 2


def test_identity_write_cannot_abort_a_settled_trade():
    """Both record sites must be inside a try/except, never bare."""
    for marker in ("LIVE_BUY_IDENTITY_RECORD_FAILED", "LIVE_SELL_IDENTITY_RECORD_FAILED"):
        idx = SOURCE.index(marker)
        window = SOURCE[max(0, idx - 2000) : idx]
        assert "try:" in window
        assert "record_fill" in window


def test_dust_writeoff_is_excluded_from_identity_records():
    """A dust write-off places no order, so it has no venue identity."""
    idx = SOURCE.index("LIVE_SELL_IDENTITY_RECORD_FAILED")
    window = SOURCE[max(0, idx - 2000) : idx]
    assert "if live_order_sell and not dust_writeoff:" in window


def test_identity_precedes_the_commit_on_both_sides():
    """Identity is durable before the local commit can fail.

    A commit failure after a confirmed fill previously left no trace of the
    venue order at all, which is what allowed a second buy of the same intent.
    """
    buy_identity = SOURCE.index("LIVE_BUY_IDENTITY_RECORD_FAILED")
    buy_commit = SOURCE.index("_commit_atomic_day_open_sync,")
    assert buy_identity < buy_commit

    sell_identity = SOURCE.index("LIVE_SELL_IDENTITY_RECORD_FAILED")
    sell_commit = SOURCE.index("_sync_fifo_sell)")
    assert sell_identity < sell_commit


def test_intent_stamp_still_carries_order_and_fill_ids():
    """The intent side of the chain is unchanged by this repair."""
    assert "mark_order_accepted as _mark_accepted" in SOURCE
    assert 'order_id=str(live_order_buy.get("id") or "")' in SOURCE
    assert 'fill_id=str(live_order_buy.get("fill_id") or "")' in SOURCE


def test_paper_trades_has_an_order_id_column_to_write_into(tmp_path):
    """The column already existed and was simply never populated."""
    db = str(tmp_path / "pt.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (trade_id TEXT, mode TEXT, order_id TEXT)")
        conn.execute("INSERT INTO paper_trades VALUES ('t1','live','26417215')")
        conn.commit()
        row = conn.execute("SELECT order_id FROM paper_trades WHERE trade_id='t1'").fetchone()
    assert row[0] == "26417215"


def _engine(tmp_path, cash: float = 10_000.0) -> PortfolioEngine:
    db = tmp_path / "engine.db"
    eng = PortfolioEngine(db_path=str(db), principal=cash, test_mode=True)
    eng._ensure_db_schema()
    initialize_paper_trading_schema(str(db))
    eng.cash_balance = cash
    eng._available_balance = cash
    eng._positions_value = 0.0
    eng._total_equity = cash
    eng._realized_pnl = 0.0
    eng._unrealized_pnl = 0.0
    asyncio.run(eng._persist_ledger_to_sqlite())
    return eng


def _position(symbol: str = "BTC/USDT", qty: float = 0.00021, price: float = 95012.34) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time(),
        trade_id="mystic_buy_BTCUSDT_1788600000",
        stop_price=price * 0.97,
        take_profit_1_price=price * 1.02,
        take_profit_2_price=price * 1.05,
        highest_price=price,
        lowest_price=price,
        atr_at_entry=1.0,
        entry_bar_timestamp=0,
        confidence_at_entry=0.5,
        entry_fee=0.02,
        sleeve=Sleeve.ACTIVE.value,
        original_position_cost=qty * price,
    )


def _bind(trade_id: str, symbol: str, qty: float, price: float, order_id):
    ts = "2026-09-14T00:00:00+00:00"
    return (
        trade_id,
        "test-run",
        "live",
        symbol,
        qty,
        price,
        qty,
        price * 0.97,
        price * 1.02,
        1.0,
        0,
        0.5,
        0.02,
        0.01,
        ts,
        "{}",
        "{}",
        Sleeve.ACTIVE.value,
        ts,
        None,
        "day",
        "{}",
        order_id,
    )


def test_atomic_open_writes_the_order_id_into_the_trade_row(tmp_path):
    """End-to-end: the real commit lands the venue order id on the FIFO row."""
    eng = _engine(tmp_path)
    pos = _position()
    eng._commit_atomic_day_open_sync(
        trade_bind=_bind(pos.trade_id, pos.symbol, pos.quantity, pos.entry_price, "26417215"),
        position=pos,
        cash_balance=eng.cash_balance - (pos.quantity * pos.entry_price),
        positions_value=pos.quantity * pos.entry_price,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_equity=eng.cash_balance,
        pre_ledger={"cash_balance": eng.cash_balance, "positions_value": 0.0, "total_equity": eng.cash_balance},
        fee=0.02,
        slippage_cost=0.01,
        quantity=pos.quantity,
        fill_price=pos.entry_price,
        symbol=pos.symbol,
        trade_id=pos.trade_id,
        entry_reason="identifier-persistence-test",
        sleeve=Sleeve.ACTIVE.value,
    )
    with sqlite3.connect(str(tmp_path / "engine.db")) as conn:
        row = conn.execute(
            "SELECT order_id, mode, side, symbol FROM paper_trades WHERE trade_id = ?",
            (pos.trade_id,),
        ).fetchone()
    assert row is not None, "atomic OPEN wrote no trade row"
    assert row[0] == "26417215"
    assert row[1] == "live"
    assert row[2] == "BUY"


def test_atomic_open_accepts_a_paper_open_with_no_order_id(tmp_path):
    """A paper open has no venue order, and that must not break the commit."""
    eng = _engine(tmp_path)
    pos = _position()
    eng._commit_atomic_day_open_sync(
        trade_bind=_bind(pos.trade_id, pos.symbol, pos.quantity, pos.entry_price, None),
        position=pos,
        cash_balance=eng.cash_balance - (pos.quantity * pos.entry_price),
        positions_value=pos.quantity * pos.entry_price,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        total_equity=eng.cash_balance,
        pre_ledger={"cash_balance": eng.cash_balance, "positions_value": 0.0, "total_equity": eng.cash_balance},
        fee=0.02,
        slippage_cost=0.01,
        quantity=pos.quantity,
        fill_price=pos.entry_price,
        symbol=pos.symbol,
        trade_id=pos.trade_id,
        entry_reason="identifier-persistence-test",
        sleeve=Sleeve.ACTIVE.value,
    )
    with sqlite3.connect(str(tmp_path / "engine.db")) as conn:
        row = conn.execute("SELECT order_id FROM paper_trades WHERE trade_id = ?", (pos.trade_id,)).fetchone()
    assert row is not None
    assert row[0] is None
