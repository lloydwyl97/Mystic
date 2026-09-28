"""Engine lot ownership: a lot holds and sells only what its own fills prove.

Ocean 2026-09-28: a 0.00000985 BTC DAY dust lot absorbed protected inventory
through full-exchange reconcile paths, was restored to ACTIVE at 0.00048866,
and sold 0.00046 BTC it never owned under a generic MANUAL_EXIT (row 2642).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services import live_order_identity as loi
from backend.services.engine_lot_ownership import (
    UNMATCHED_OVERSELL_REASON,
    capped_lot_quantity,
    fill_owned_quantity_at,
    is_generic_manual_strategy_exit,
)
from backend.services.portfolio_engine import ExitType, OpenPosition, PortfolioEngine, Sleeve, make_position_key
from backend.services.protected_external_inventory import list_protected, record_protected, shrink_to_exchange

DAY = "DAY_V2"
SCALP = "SCALP_V2"


def _fill(db: str, trade_id: str, symbol: str, side: str, qty: float, price: float, *, fee: float = 0.0, fee_asset: str = "", oid: str = "") -> None:
    loi.ensure_schema(db)
    loi.record_fill(
        db,
        loi.OrderIdentity(
            symbol=symbol.replace("/", ""),
            side=side,
            exchange_order_id=oid or f"{trade_id}-{side}-{qty}",
            executed_qty=qty,
            avg_fill_price=price,
            cost_quote=qty * price,
            fee_amount=fee,
            fee_asset=fee_asset,
            mystic_trade_id=trade_id,
            order_status="FILLED",
        ),
    )


def _lot(symbol: str, engine_id: str, *, qty: float, price: float = 100.0, status: str = "ACTIVE", trade_id: str = "") -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time() - 600,
        trade_id=trade_id or f"t-{engine_id}-{symbol.replace('/', '')}",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=price,
        lowest_price=price,
        status=status,
        engine_id=engine_id,
        original_position_cost=qty * price,
    )


def _engine(tmp_path: Path, balances: dict[str, float] | None = None) -> PortfolioEngine:
    eng = PortfolioEngine(db_path=str(tmp_path / "own.db"), principal=300.0, test_mode=True)
    eng._ensure_db_schema()
    loi.ensure_schema(eng.db_path)
    eng._live_execution_enabled = True
    bal = balances or {}
    eng._live_service = SimpleNamespace(
        get_balance=AsyncMock(return_value={"status": "success", "balance": {"total": dict(bal), "free": dict(bal)}, "fetched_at": time.time()}),
        get_market_price=AsyncMock(return_value={"price": 83_900.0}),
        order_fence_active=lambda *_a, **_k: False,
    )
    eng._ensure_symbol_constraints = AsyncMock()
    eng._persist_position_to_sqlite = AsyncMock()
    eng._symbol_constraints["BTC/USDT"] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 1.0}
    eng._position_mark_prices["BTC/USDT"] = 83_900.0
    return eng


# The Ocean BTC DAY lot: 0.00074 bought, 0.00000015 BTC fee, 0.00073 sold.
BTC_DAY_TID = "mystic_BTC/USDT_1790567174345"
BTC_DAY_OWNED = 0.00074 - 0.00000015 - 0.00073


def _seed_btc_day_fills(db: str) -> None:
    _fill(db, BTC_DAY_TID, "BTC/USDT", "BUY", 0.00074, 83_246.53, fee=0.00000015, fee_asset="BTC")
    _fill(db, BTC_DAY_TID, "BTC/USDT", "SELL", 0.00073, 82_557.0)


# --------------------------------------------------------------- helpers
def test_fill_owned_quantity_nets_buy_fee_and_sells(tmp_path):
    db = str(tmp_path / "f.db")
    _seed_btc_day_fills(db)
    assert float(fill_owned_quantity_at(db, BTC_DAY_TID, "BTC/USDT")) == pytest.approx(BTC_DAY_OWNED, abs=1e-12)
    assert fill_owned_quantity_at(db, "unknown", "BTC/USDT") is None


def test_capped_quantity_never_grows_without_fill_proof():
    assert capped_lot_quantity(proposed=0.5, booked=0.1, owned=None) == pytest.approx(0.1)
    assert capped_lot_quantity(proposed=0.5, booked=0.1, owned=0.3) == pytest.approx(0.3)
    assert capped_lot_quantity(proposed=0.05, booked=0.1, owned=0.3) == pytest.approx(0.05)


# ------------------------------------------------------ A-D dust absorption
async def test_a_dust_lot_does_not_absorb_protected_inventory(tmp_path, monkeypatch):
    import backend.services.protected_external_inventory as pei

    eng = _engine(tmp_path, {"BTC": 0.00048866})
    _seed_btc_day_fills(eng.db_path)
    monkeypatch.setattr(pei, "protected_quantity", lambda *_a, **_k: 0.0)
    eng._dust_check = lambda _s, qty, price: (qty * price < 5.0, 0, 0, 0)
    lot = _lot("BTC/USDT", DAY, qty=BTC_DAY_OWNED, price=83_246.53, status="DUST_PENDING", trade_id=BTC_DAY_TID)
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = lot
    await eng.run_dust_reconciliation({"BTC/USDT": 83_900.0})
    assert lot.quantity == pytest.approx(BTC_DAY_OWNED, abs=1e-12)
    assert lot.status == "DUST_PENDING"


async def test_b_dust_lot_does_not_absorb_sibling_quantity(tmp_path, monkeypatch):
    import backend.services.protected_external_inventory as pei

    eng = _engine(tmp_path, {"BTC": 0.00100985})
    _seed_btc_day_fills(eng.db_path)
    _fill(eng.db_path, "scalp_btc", "BTC/USDT", "BUY", 0.001, 83_000.0)
    monkeypatch.setattr(pei, "protected_quantity", lambda *_a, **_k: 0.0)
    eng._dust_check = lambda _s, qty, price: (qty * price < 5.0, 0, 0, 0)
    dust = _lot("BTC/USDT", DAY, qty=BTC_DAY_OWNED, status="DUST_PENDING", trade_id=BTC_DAY_TID)
    sibling = _lot("BTC/USDT", SCALP, qty=0.001, trade_id="scalp_btc")
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = dust
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = sibling
    await eng.run_dust_reconciliation({"BTC/USDT": 83_900.0})
    assert dust.quantity == pytest.approx(BTC_DAY_OWNED, abs=1e-12)
    assert sibling.quantity == pytest.approx(0.001)


async def test_c_importer_all_dust_branch_keeps_lot_at_its_fills(tmp_path):
    eng = _engine(tmp_path, {"BTC": 0.00048866})
    _seed_btc_day_fills(eng.db_path)
    lot = _lot("BTC/USDT", DAY, qty=BTC_DAY_OWNED, status="DUST_PENDING", trade_id=BTC_DAY_TID)
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = lot
    await eng._import_missing_exchange_positions({"BTC": 0.00048866})
    assert lot.quantity == pytest.approx(BTC_DAY_OWNED, abs=1e-12)


async def test_d_inflated_lot_is_shrunk_to_its_fills(tmp_path):
    eng = _engine(tmp_path)
    _seed_btc_day_fills(eng.db_path)
    lot = _lot("BTC/USDT", DAY, qty=0.00048866, status="ACTIVE", trade_id=BTC_DAY_TID)
    await eng._enforce_lot_ownership([lot])
    assert lot.quantity == pytest.approx(BTC_DAY_OWNED, abs=1e-12)
    eng._persist_position_to_sqlite.assert_awaited()


# ------------------------------------------------- E-H reconcile / protected
async def test_e_reconcile_never_grows_a_lot_from_the_exchange_total(tmp_path):
    eng = _engine(tmp_path, {"BTC": 0.0015})
    _fill(eng.db_path, "day_btc", "BTC/USDT", "BUY", 0.001, 83_000.0)
    lot = _lot("BTC/USDT", DAY, qty=0.001, price=83_000.0, trade_id="day_btc")
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = lot
    await eng.run_live_reconcile({"BTC": 0.0015}, free_balances={"BTC": 0.0015})
    assert lot.quantity == pytest.approx(0.001)


async def test_f_remainder_above_proven_lots_becomes_protected(tmp_path):
    eng = _engine(tmp_path)
    lot = _lot("BTC/USDT", DAY, qty=BTC_DAY_OWNED, status="DUST_PENDING", trade_id=BTC_DAY_TID)
    await eng._sync_protected_remainder("BTC/USDT", 0.00048866, [lot], 0.00001)
    rows = {r["symbol"]: float(r["quantity"]) for r in list_protected(eng.db_path)}
    assert rows["BTC/USDT"] == pytest.approx(0.00048866 - BTC_DAY_OWNED, abs=1e-12)


def _protected_db(tmp_path: Path, qty: float) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / "p.db"))
    conn.row_factory = sqlite3.Row
    record_protected(conn, "BTC/USDT", qty, 83_000.0, source_trade_id="ext")
    conn.commit()
    return conn


def test_g_protected_row_persists_while_unexplained_quantity_remains(tmp_path):
    conn = _protected_db(tmp_path, 0.00046885)
    shrink_to_exchange(conn, {"BTC": 0.00046885 + 0.001}, {"BTC/USDT": 0.001}, min_notional=1.0)
    row = conn.execute("SELECT quantity FROM protected_external_inventory WHERE symbol='BTC/USDT'").fetchone()
    assert float(row[0]) == pytest.approx(0.00046885)


def test_h_protected_row_shrinks_only_with_the_unexplained_quantity(tmp_path):
    conn = _protected_db(tmp_path, 0.0004)
    shrink_to_exchange(conn, {"BTC": 0.001 + 0.0002}, {"BTC/USDT": 0.001}, min_notional=1.0)
    row = conn.execute("SELECT quantity FROM protected_external_inventory WHERE symbol='BTC/USDT'").fetchone()
    assert float(row[0]) == pytest.approx(0.0002)


# ------------------------------------------------------------ I-N sell caps
QTY = 0.001
ENTRY = 60_000.0
SELL = 60_300.0


async def _live_sell(
    tmp_path: Path,
    *,
    engine_id: str,
    exit_type: ExitType,
    trigger: str,
    booked: float = QTY,
    owned_fills: tuple[float, float] = (QTY, 0.0),
    free: float = 0.01,
    status: str = "ACTIVE",
    force_sell: bool = True,
) -> SimpleNamespace:
    from tests.test_sell_cash_credit import _allowed_sell_eval, _init_test_db, _seed_btc_position

    db_path = tmp_path / "sell.db"
    trade_id = f"{engine_id.lower()}_BTCUSDT_1"
    _init_test_db(db_path, cash=1_000.0)
    _seed_btc_position(db_path, trade_id=trade_id, qty=booked, entry=ENTRY, cash_after_buy=940.0)
    bought, sold = owned_fills
    _fill(str(db_path), trade_id, "BTC/USDT", "BUY", bought, ENTRY)
    if sold:
        _fill(str(db_path), trade_id, "BTC/USDT", "SELL", sold, ENTRY)

    engine = PortfolioEngine(db_path=str(db_path), principal=1_000.0, test_mode=True)
    await engine.initialize_from_db()
    engine.open_positions.clear()
    engine.open_positions["BTC/USDT"] = OpenPosition(
        symbol="BTC/USDT",
        quantity=booked,
        entry_price=ENTRY,
        entry_time=time.time() - 600,
        trade_id=trade_id,
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=SELL,
        lowest_price=ENTRY,
        sleeve=Sleeve.ACTIVE.value,
        engine_id=engine_id,
        status=status,
    )
    engine._positions_initialized = True
    engine._live_execution_enabled = True
    engine._live_service = SimpleNamespace(
        get_balance=AsyncMock(return_value={"status": "success", "balance": {"free": {"BTC": free}, "total": {"BTC": free}}}),
        get_open_orders=AsyncMock(return_value={"status": "success", "orders": {}}),
        order_fence_active=lambda *_a, **_k: False,
    )
    engine._exit_in_progress = set()
    engine._paper_service = MagicMock(paper_run_id="test-run")
    engine._check_kill_switch_sell = MagicMock(return_value=(True, ""))
    engine._record_reject = AsyncMock()
    engine._persist_position_to_sqlite = AsyncMock()
    engine._get_loss_hold_until = AsyncMock(return_value=None)
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
    engine._validate_invariants = AsyncMock(return_value=True)
    engine._record_audit = AsyncMock()
    engine._entry_ensure_constraints = AsyncMock()
    engine._ensure_symbol_constraints = AsyncMock(return_value=None)
    engine._symbol_constraints["BTC/USDT"] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 1.0}
    engine._normalize_order_amount = MagicMock(side_effect=lambda *a, **k: (k.get("raw_qty", a[1] if len(a) > 1 else 0), "ok", 0.0))
    engine._dust_check = MagicMock(side_effect=lambda _s, q, p: (False, q, "", q * p))
    engine._floor_to_step = lambda q, s: (int(q / s + 1e-9) * s) if s else q

    venue = AsyncMock(return_value=None)
    preflight = MagicMock(passed=True, expected_avg_fill=SELL, protected_limit_price=SELL)
    preflight.to_audit_dict = MagicMock(return_value={"passed": True})
    gate = AsyncMock(return_value=_allowed_sell_eval(mark_price=SELL, avg_entry_price=ENTRY))
    with (
        patch.object(engine, "_evaluate_sell_profitability", gate),
        patch("backend.services.protected_limit_execution.execute_protected_limit_live", venue),
        patch("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=preflight)),
        patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True),
        patch("backend.config.live_test_mode.can_place_live_orders_sync", return_value=(True, "")),
        patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=engine._paper_service),
    ):
        result = await engine.execute_sell_fifo("BTC/USDT", booked, SELL, exit_type, trigger, force_sell=force_sell)
    sent = [float(c.kwargs.get("quantity", c.args[1] if len(c.args) > 1 else 0)) for c in venue.await_args_list]
    return SimpleNamespace(engine=engine, result=result, sent=sent, venue=venue)


def _assert_capped(run: SimpleNamespace, owned: float) -> None:
    assert all(q <= owned + 1e-12 for q in run.sent), run.sent
    assert sum(run.sent) <= owned * max(1, len(run.sent)) + 1e-12


@pytest.mark.parametrize(
    ("label", "engine_id", "exit_type", "trigger", "force"),
    [
        ("I_day_normal", DAY, ExitType.TAKE_PROFIT_1, "NET_PROFIT_EXIT", False),
        ("J_scalp_normal", SCALP, ExitType.TAKE_PROFIT_1, "NET_PROFIT_EXIT", False),
        ("K_day_mandatory", DAY, ExitType.MANUAL, "DAY_V2_CATASTROPHIC_PROTECTION", True),
        ("L_scalp_mandatory", SCALP, ExitType.MANUAL, "SCALP_V2_TIME_STOP", True),
        ("M_day_residual_resume", DAY, ExitType.MANUAL, "DAY_V2_WINNER_PROTECTION", True),
    ],
)
async def test_i_to_m_every_strategy_sell_is_capped_to_fill_ownership(tmp_path, label, engine_id, exit_type, trigger, force):
    """The lot is booked at 0.001 but its fills prove only 0.0004; the exchange holds 0.01."""
    run = await _live_sell(tmp_path, engine_id=engine_id, exit_type=exit_type, trigger=trigger, owned_fills=(0.001, 0.0006), force_sell=force)
    assert run.sent, f"{label}: expected a capped venue order"
    _assert_capped(run, 0.0004)


async def test_n_dust_exit_cannot_sell_protected_quantity(tmp_path):
    """Row 2642: an inflated dust lot with a large free balance sells only its own dust."""
    run = await _live_sell(
        tmp_path,
        engine_id=DAY,
        exit_type=ExitType.MANUAL,
        trigger="DAY_V2_WINNER_PROTECTION",
        booked=0.00046,
        owned_fills=(0.00074, 0.00073),
        free=0.00048866,
    )
    _assert_capped(run, 0.00001)


# ---------------------------------------------------------- O restart
async def test_o_restart_restore_is_capped_to_fill_ownership(tmp_path, monkeypatch):
    import backend.services.protected_external_inventory as pei

    eng = _engine(tmp_path)
    eng._scalp_v2_write_position_row = MagicMock()
    _fill(eng.db_path, "scalp_xrp", "XRP/USDT", "BUY", 26.29474, 1.5046)
    _fill(eng.db_path, "scalp_xrp", "XRP/USDT", "SELL", 20.0, 1.51)
    lot = {"trade_id": "scalp_xrp", "remaining": 26.29474, "price": 1.5046, "fee": 0.0, "order_id": "1", "atr": 0.0, "opportunity_id": "", "decision_id": "", "entry_time": time.time()}
    monkeypatch.setattr(pei, "unsold_scalp_v2_lot", lambda *_a, **_k: lot)
    restored = await eng._restore_unsold_scalp_lot("XRP/USDT", 28.5, 1.5, 1.0)
    assert restored == pytest.approx(6.29474)


# ------------------------------------------------------- P sibling engine
async def test_p_enforcing_one_engine_leaves_the_sibling_unchanged(tmp_path):
    eng = _engine(tmp_path, {"BTC": 0.00048866})
    _seed_btc_day_fills(eng.db_path)
    _fill(eng.db_path, "scalp_btc", "BTC/USDT", "BUY", 0.00000988, 83_900.0)
    day = _lot("BTC/USDT", DAY, qty=0.00046, trade_id=BTC_DAY_TID)
    scalp = _lot("BTC/USDT", SCALP, qty=0.00000988, status="DUST_PENDING", trade_id="scalp_btc")
    await eng._enforce_lot_ownership([day, scalp])
    assert day.quantity == pytest.approx(BTC_DAY_OWNED, abs=1e-12)
    assert scalp.quantity == pytest.approx(0.00000988, abs=1e-12)


# ------------------------------------------------ Q-R no MANUAL_EXIT persisted
def test_generic_manual_is_a_strategy_violation_only_for_strategy_engines():
    assert is_generic_manual_strategy_exit(DAY, "MANUAL_EXIT")
    assert is_generic_manual_strategy_exit(SCALP, "MANUAL")
    assert not is_generic_manual_strategy_exit(DAY, "TRAILING_STOP_EXIT")
    assert not is_generic_manual_strategy_exit("", "MANUAL_EXIT")


@pytest.mark.parametrize("engine_id", [DAY, SCALP])
async def test_q_r_generic_manual_exit_never_reaches_the_venue(tmp_path, engine_id):
    run = await _live_sell(tmp_path, engine_id=engine_id, exit_type=ExitType.MANUAL, trigger="MANUAL_EXIT")
    assert run.result is None
    assert run.venue.await_count == 0
    reasons = [c.args[2] for c in run.engine._record_reject.await_args_list]
    assert "MANUAL_EXIT_INVARIANT_VIOLATION" in reasons
    with sqlite3.connect(run.engine.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL' AND exit_reason='MANUAL_EXIT'").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("trigger", "label"),
    [("DAY_V2_WINNER_PROTECTION", "TRAILING_STOP_EXIT"), ("SCALP_V2_TIME_STOP", "TIME_STOP_EXIT")],
)
def test_q_r_residual_keeps_the_strategy_trigger(trigger, label):
    from backend.services.day_mandatory_exit_execution import mark_exit_residual_pending
    from backend.services.ownership_repair import strategy_label_for

    pos = SimpleNamespace(status="ACTIVE", exit_residual_reason="", _learning_raw_exit_reason=trigger)
    mark_exit_residual_pending(pos, "MANUAL_EXIT")
    assert pos.exit_residual_reason == trigger
    assert strategy_label_for(pos.exit_residual_reason) == label


# ------------------------------------------ S-T MANUAL_UNMATCHED exclusions
def _trades_db(tmp_path: Path) -> sqlite3.Connection:
    from backend.services.live_close_integrity import ensure_live_close_tables

    conn = sqlite3.connect(str(tmp_path / "t.db"))
    conn.execute(
        """CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL,
            pnl REAL, pnl_usd_net REAL, exit_type TEXT, exit_reason TEXT, order_id TEXT, mode TEXT,
            status TEXT, timestamp TEXT, is_synthetic INTEGER DEFAULT 0, counts_toward_realized INTEGER DEFAULT 1,
            explainability_json TEXT)"""
    )
    rows = [
        (2641, "s1", "TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", "1845996201", 0.0128, "{}"),
        (2642, "s2", "MANUAL", "MANUAL_EXIT", "1845996508", 0.3091, json.dumps({"raw_exit_reason": "MANUAL_EXIT"})),
        (2409, "s3", "MANUAL", "MANUAL_EXIT", "496757876", -0.2731, json.dumps({"raw_exit_reason": "DAY_V2_WINNER_PROTECTION"})),
    ]
    for rid, tid, et, er, oid, pnl, ex in rows:
        conn.execute(
            "INSERT INTO paper_trades (id, trade_id, symbol, side, quantity, price, pnl, pnl_usd_net, exit_type, exit_reason, order_id, mode, status, timestamp, explainability_json) "
            "VALUES (?,?,'BTC/USDT','SELL',0.00046,83935.55,?,?,?,?,?,'live','executed','2026-09-28T18:08:33+00:00',?)",
            (rid, tid, pnl, pnl, et, er, oid, ex),
        )
    ensure_live_close_tables(conn)
    conn.execute("CREATE TABLE day_outcome_attribution (id INTEGER PRIMARY KEY, trade_id TEXT, net_pnl_after_fees REAL)")
    conn.execute("INSERT INTO day_outcome_attribution VALUES (2431, 'mystic_BTC/USDT_1790567174345', 0.3091)")
    conn.execute(
        "CREATE TABLE day_outcome_bandit_arms (arm_key TEXT PRIMARY KEY, alpha REAL, beta REAL, wins INTEGER, losses INTEGER, total_pnl REAL, last_pnl REAL, last_exit_reason TEXT, n_obs INTEGER)"
    )
    conn.execute("INSERT INTO day_outcome_bandit_arms VALUES ('BTC/USDT|RANGE_BOUNCE|range', 10.98, 19.2011414, 6, 14, -1.7168941, 0.3090673, 'MANUAL_EXIT', 20)")
    conn.commit()
    return conn


def test_s_manual_unmatched_is_excluded_from_strategy_pnl_with_economics_intact(tmp_path):
    from backend.services.live_close_integrity import strategy_scorecard_sql_predicate
    from backend.services.ownership_repair import reclassify_unmatched_sell

    conn = _trades_db(tmp_path)
    before = conn.execute("SELECT quantity, price, pnl, order_id, timestamp FROM paper_trades WHERE id=2642").fetchone()
    assert reclassify_unmatched_sell(conn, 2642, UNMATCHED_OVERSELL_REASON)
    after = conn.execute("SELECT quantity, price, pnl, order_id, timestamp, counts_toward_realized, exit_type, exit_reason FROM paper_trades WHERE id=2642").fetchone()
    assert tuple(after[:5]) == tuple(before)
    assert after[5:] == (0, "MANUAL_UNMATCHED", UNMATCHED_OVERSELL_REASON)
    ids = [r[0] for r in conn.execute(f"SELECT id FROM paper_trades t WHERE UPPER(t.side)='SELL' AND {strategy_scorecard_sql_predicate('t')} ORDER BY id")]
    assert 2642 not in ids and 2641 in ids
    klass = conn.execute("SELECT accounting_class FROM live_close_classifications WHERE trade_id='s2'").fetchone()[0]
    assert klass == "MANUAL_UNMATCHED"
    assert conn.execute("SELECT COUNT(*) FROM ownership_repair_backup WHERE row_key='id=2642'").fetchone()[0] == 1


def test_t_manual_unmatched_is_excluded_from_learning(tmp_path):
    from backend.services.ownership_repair import exclude_learning_rows, revert_bandit_observation

    conn = _trades_db(tmp_path)
    assert exclude_learning_rows(conn, "day_outcome_attribution", "id", [2431], UNMATCHED_OVERSELL_REASON) == 1
    assert conn.execute("SELECT COUNT(*) FROM day_outcome_attribution").fetchone()[0] == 0
    w = 1.0 + 0.3090673 / 12.0
    assert revert_bandit_observation(conn, "BTC/USDT|RANGE_BOUNCE|range", win=False, weight_now=w, pnl=0.3090673, restore_last=(0.0128124, "TRAILING_STOP_EXIT"), reason=UNMATCHED_OVERSELL_REASON)
    arm = conn.execute("SELECT beta, losses, n_obs, total_pnl, last_exit_reason FROM day_outcome_bandit_arms").fetchone()
    assert arm[0] == pytest.approx(18.175, abs=1e-3)
    assert arm[1:3] == (13, 19)
    assert arm[3] == pytest.approx(-2.0259614)
    assert arm[4] == "TRAILING_STOP_EXIT"
    assert conn.execute("SELECT COUNT(*) FROM ownership_repair_backup").fetchone()[0] == 2


def test_true_strategy_exit_is_relabeled_with_its_real_reason(tmp_path):
    from backend.services.ownership_repair import relabel_generic_manual_exit

    conn = _trades_db(tmp_path)
    assert relabel_generic_manual_exit(conn, 2409) == "TRAILING_STOP_EXIT"
    assert relabel_generic_manual_exit(conn, 2642) is None
    row = tuple(conn.execute("SELECT exit_reason, exit_type, pnl FROM paper_trades WHERE id=2409").fetchone())
    assert row == ("TRAILING_STOP_EXIT", "TRAILING_STOP_EXIT", -0.2731)


# ------------------------------------------------------- U-V restoration
async def test_u_inflated_dust_lot_is_restored_from_fills(tmp_path):
    eng = _engine(tmp_path, {"ETH": 0.00028882})
    _fill(eng.db_path, "day_eth", "ETH/USDT", "BUY", 0.0198, 2684.02, fee=0.0000198, fee_asset="ETH")
    _fill(eng.db_path, "day_eth", "ETH/USDT", "SELL", 0.0197, 2700.0)
    lot = _lot("ETH/USDT", DAY, qty=0.00028882, status="DUST_PENDING", trade_id="day_eth")
    await eng._enforce_lot_ownership([lot])
    assert lot.quantity == pytest.approx(0.0198 - 0.0000198 - 0.0197, abs=1e-12)
    assert lot.dust_qty_canonical == pytest.approx(lot.quantity)


async def test_v_xrp_remainder_with_no_lot_stays_protected(tmp_path):
    eng = _engine(tmp_path, {"XRP": 28.57342})
    eng._symbol_constraints["XRP/USDT"] = {"qty_step": 0.1, "min_qty": 0.1, "min_notional": 1.0}
    eng._live_service.get_market_price = AsyncMock(return_value={"price": 1.4954})
    eng._restore_unsold_scalp_lot = AsyncMock(return_value=0.0)
    eng._dust_check = lambda _s, qty, price: (False, qty, "", qty * price)
    await eng._import_missing_exchange_positions({"XRP": 28.57342}, {"XRP": 28.57342})
    assert not eng._symbol_lots("XRP/USDT")
    rows = {r["symbol"]: float(r["quantity"]) for r in list_protected(eng.db_path)}
    assert rows["XRP/USDT"] == pytest.approx(28.5)
