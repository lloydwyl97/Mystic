"""Tests for the two mechanical accounting defects from the 4-day production audit (2026-11-21).

Covers:
1. Vanished DUST_PENDING retirement (quantity=0, dust_qty_canonical=0 before cleanup, no re-persist,
   positions_value=0, equity=cash, EXTERNAL_BALANCE_CONVERSION vs DUST_WRITEOFF, no bad_trade=true).
2. Strategy dust provenance (SCALP_V2 9.6e-6 BTC & DAY_V2 ETH dust preserved across single-fill and
   multi-fill exits and reconciliation; never re-imported as LEGACY_DAY_LIVE / dust_exchange:*;
   venue conversion retires cleanly).
3. Learning sold_quantity provenance (full close, partial close, multi-fill 0.00197 & 0.00203 BTC
   exits, scalp_learning_outcomes ingestion, deterministic post-227e036 backfill, TLO 12736 cleanliness).
4. Stale protected_preexisting_dust cap after venue conversion so sellable steps are never stranded.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.database_schema import initialize_paper_trading_schema
from backend.services.ai_learning_ingestion import ingest_scalp_outcomes
from backend.services.engine_strategy_dust import EVENT_CONVERSION, held_lots
from backend.services.learning_quantity_backfill import (
    POST_227E036_SHA,
    backfill_post_227e036_learning_quantities,
    verify_and_clean_venue_converted_dust_learning_row,
)
from backend.services.live_exchange_equity import (
    load_protected_preexisting_dust,
    stamp_protected_preexisting_dust,
)
from backend.services.portfolio_engine import (
    ExitType,
    OpenPosition,
    PortfolioEngine,
    Sleeve,
    make_position_key,
)
from backend.services.trade_learning_writer import _ensure_table

BTC_CONVERSION_LOG = [
    {
        "operateTime": 1795286013000,  # 2026-11-21T18:33:33Z
        "totalTransferedAmount": "1.64627362",
        "totalServiceChargeAmount": "0.08664599",
        "tranId": 2483316880,
        "userAssetDribbletDetails": [
            {
                "tranId": 2483316880,
                "fromAsset": "BTC",
                "amount": "0.0000288",
                "transferedAmount": "1.64627362",
                "serviceChargeAmount": "0.08664599",
                "operateTime": 1795286013000,
                "toAsset": "USDT",
            }
        ],
    }
]


async def _make_live_engine(tmp_path: Path, *, cash: float = 990.55080448, prices: dict[str, float] | None = None) -> PortfolioEngine:
    db_path = tmp_path / "mystic_audit.db"
    os.environ["MYSTIC_RECONCILE_STATE_FILE"] = str(tmp_path / "reconcile.json")
    initialize_paper_trading_schema(str(db_path))
    eng = PortfolioEngine(db_path=str(db_path), principal=1000.0, test_mode=True)
    eng._ensure_db_schema()
    await eng.initialize_from_db()
    eng._live_execution_enabled = True
    eng.cash_balance = cash
    eng._available_balance = cash
    px_map = prices or {"BTC/USDT": 60379.715, "ETH/USDT": 1800.0, "SOL/USDT": 75.0, "XRP/USDT": 1.32}
    eng._live_service = SimpleNamespace(
        get_balance=AsyncMock(
            return_value={
                "status": "success",
                "balance": {"total": {"USDT": cash}, "free": {"USDT": cash}},
                "fetched_at": time.time(),
            }
        ),
        get_market_price=AsyncMock(side_effect=lambda sym: {"price": px_map.get(sym, 60379.715)}),
        order_fence_active=lambda *_a, **_k: False,
    )
    eng._symbol_constraints.update(
        {
            "BTC/USDT": {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 5.0},
            "ETH/USDT": {"qty_step": 0.0001, "min_qty": 0.0001, "min_notional": 5.0},
            "SOL/USDT": {"qty_step": 0.001, "min_qty": 0.001, "min_notional": 5.0},
            "XRP/USDT": {"qty_step": 0.1, "min_qty": 0.1, "min_notional": 5.0},
        }
    )
    eng._ensure_symbol_constraints = AsyncMock()
    orig_mtm = eng._recompute_positions_values

    async def _mtm(override=None, allow_network_mtm=True):
        return await orig_mtm(override or px_map, allow_network_mtm=False)

    eng._recompute_positions_values = _mtm
    return eng


@pytest.mark.asyncio
async def test_defect1_phantom_dust_exchange_retired_as_external_conversion_no_repersist_no_bad_trade(tmp_path: Path):
    """Exact reproduction of production dust_exchange:BTCUSDT (1.919e-5 BTC, tran_id=2483316880)."""
    cash = 990.55080448
    eng = await _make_live_engine(tmp_path, cash=cash)
    eng._venue_dust_conversions = AsyncMock(return_value=BTC_CONVERSION_LOG)

    phantom = OpenPosition(
        symbol="BTC/USDT",
        quantity=1.919e-05,
        entry_price=58024.96,
        entry_time=1795253381.8019865,
        trade_id="dust_exchange:BTCUSDT",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=58024.96,
        lowest_price=58024.96,
        status="DUST_PENDING",
        engine_id="LEGACY_DAY_LIVE",
        dust_qty_canonical=1.919e-05,
        dust_detected_at=1795253381.8019865,
        quantity_exact="0.00001919",
    )
    eng.open_positions[make_position_key("LEGACY_DAY_LIVE", "BTC/USDT")] = phantom
    await eng._persist_position_to_sqlite(phantom)
    await eng._recompute_positions_values()
    await eng._persist_ledger_to_sqlite()

    # Pre-repair overstated state (~$991.7095 vs $990.5508)
    assert eng._positions_value == pytest.approx(1.15868673085, rel=1e-6)
    assert eng._total_equity == pytest.approx(991.70949121085, rel=1e-9)

    flat_balances = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "XRP": 0.0, "USDT": cash}
    await eng.run_live_reconcile(flat_balances, free_balances=dict(flat_balances))

    # Row must zero quantity/dust_qty_canonical and vanish from memory, SQLite, and positions_value
    assert phantom.quantity == 0.0
    assert phantom.dust_qty_canonical == 0.0
    assert eng.open_positions == {}
    assert eng._positions_value == pytest.approx(0.0)
    assert eng.cash_balance == pytest.approx(cash)
    assert eng._total_equity == pytest.approx(cash)

    with sqlite3.connect(eng.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] == 0
        led = conn.execute("SELECT cash_balance, positions_value, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        assert led[0] == pytest.approx(cash)
        assert led[1] == pytest.approx(0.0)
        assert led[2] == pytest.approx(cash)

        events = conn.execute("SELECT event_class, venue_ref, quantity FROM external_balance_events").fetchall()
        assert events == [(EVENT_CONVERSION, "2483316880", pytest.approx(1.919e-05))]

        # Zero DUST_WRITEOFF rows, zero bad_trade=true learning rows
        assert conn.execute("SELECT COUNT(*) FROM position_close_ledger WHERE close_reason='DUST_WRITEOFF'").fetchone()[0] == 0
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_learning_outcomes'").fetchone():
            assert conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes").fetchone()[0] == 0

    # Subsequent two-minute reconcile passes are completely quiet no-ops
    for _ in range(3):
        await eng.run_live_reconcile(flat_balances, free_balances=dict(flat_balances))
    assert eng.open_positions == {}
    assert eng._positions_value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_run_dust_reconciliation_zero_canonical_qty_removes_without_repersisting(tmp_path: Path):
    eng = await _make_live_engine(tmp_path, cash=990.55)
    eng._venue_dust_conversions = AsyncMock(return_value=[])
    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=1.919e-05,
        entry_price=58024.96,
        entry_time=1795253381.8,
        trade_id="dust_exchange:BTCUSDT",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        status="DUST_PENDING",
        engine_id="LEGACY_DAY_LIVE",
        dust_qty_canonical=1.919e-05,
    )
    eng.open_positions[make_position_key("LEGACY_DAY_LIVE", "BTC/USDT")] = pos
    await eng._persist_position_to_sqlite(pos)

    await eng.run_dust_reconciliation({"BTC/USDT": 60000.0})
    assert pos.quantity == 0.0
    assert pos.dust_qty_canonical == 0.0
    assert eng.open_positions == {}
    with sqlite3.connect(eng.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0] == 0


def _seed_live_fill(db_path: str, *, tid: str, sym: str, side: str, oid: str, qty: float, px: float, fee: float, fee_asset: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_exchange_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT,
                symbol TEXT,
                side TEXT,
                exchange_order_id TEXT,
                executed_qty REAL,
                avg_fill_price REAL,
                fee_amount REAL,
                fee_asset TEXT,
                mystic_trade_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO live_exchange_fills (venue, symbol, side, exchange_order_id, executed_qty, avg_fill_price, fee_amount, fee_asset, mystic_trade_id) VALUES (?,?,?,?,?,?,?,?,?)",
            ("binanceus", sym, side, oid, qty, px, fee, fee_asset, tid),
        )
        conn.commit()


@pytest.mark.asyncio
async def test_section_13_and_14_scalp_dust_reimport_and_venue_conversion(tmp_path: Path):
    """Section 13 & 14: SCALP creates 9.6e-6 BTC residual -> reconcile keeps SCALP_V2 provenance
    (never dust_exchange / LEGACY_DAY_LIVE) -> Binance converts -> retires cleanly."""
    cash_pre = 988.90453086
    eng = await _make_live_engine(tmp_path, cash=cash_pre)
    tid = "scalp_v2_BTCUSDT_1795252798012"

    # Seed immutable fill provenance: bought 0.00198 BTC with 4e-7 BTC fee -> sold 0.00197 -> 9.6e-6 residual
    _seed_live_fill(eng.db_path, tid=tid, sym="BTC/USDT", side="BUY", oid="1914046770", qty=0.00198, px=57814.01, fee=4e-7, fee_asset="BTC")
    _seed_live_fill(eng.db_path, tid=tid, sym="BTC/USDT", side="SELL", oid="1914053506", qty=0.00062, px=57906.19, fee=0.00718, fee_asset="USDT")
    _seed_live_fill(eng.db_path, tid=tid, sym="BTC/USDT", side="SELL", oid="1914053798", qty=0.00135, px=57919.68, fee=0.01564, fee_asset="USDT")
    with sqlite3.connect(eng.db_path) as conn:
        pt_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
        if "engine_id" not in pt_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN engine_id TEXT")
        conn.execute(
            "INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status, strategy_id, engine_id) "
            "VALUES (?, 'live', 'live', 'BTC/USDT', 'BUY', 0.0019796, 57814.01, 9.6e-06, '2026-11-21T09:19:58+00:00', 'executed', 'binance_scalp_paper', 'SCALP_V2')",
            (tid,),
        )
        conn.commit()

    # Step 1 (Section 13): Reconcile sees 9.6e-6 BTC on Binance even with open_positions empty.
    bal_with_dust = {"BTC": 9.6e-06, "ETH": 0.0, "SOL": 0.0, "XRP": 0.0, "USDT": cash_pre}
    await eng.run_live_reconcile(bal_with_dust, free_balances=dict(bal_with_dust))

    # Must NOT create a dust_exchange / LEGACY_DAY_LIVE open position; must stay SCALP_V2 held dust!
    assert all(p.engine_id != "LEGACY_DAY_LIVE" and not str(p.trade_id).startswith("dust_exchange:") for p in eng.open_positions.values())
    held = held_lots(eng.db_path, "BTC/USDT")
    assert len(held) == 1
    assert held[0]["engine_id"] == "SCALP_V2"
    assert held[0]["source_trade_id"] == tid
    assert held[0]["quantity"] == pytest.approx(9.6e-06)

    # Step 2 (Section 14): Binance converts 9.6e-6 BTC to USDT (cash already reflects proceeds).
    cash_post = 990.55080448
    eng.cash_balance = cash_post
    eng._available_balance = cash_post
    eng._venue_dust_conversions = AsyncMock(return_value=BTC_CONVERSION_LOG)
    bal_flat = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "XRP": 0.0, "USDT": cash_post}
    await eng.run_live_reconcile(bal_flat, free_balances=dict(bal_flat))

    assert held_lots(eng.db_path, "BTC/USDT") == []
    assert eng.open_positions == {}
    assert eng._positions_value == pytest.approx(0.0)
    assert eng.cash_balance == pytest.approx(cash_post)
    assert eng._total_equity == pytest.approx(cash_post)

    with sqlite3.connect(eng.db_path) as conn:
        ev = conn.execute("SELECT engine_id, source_trade_id, event_class, venue_ref FROM external_balance_events WHERE symbol='BTC/USDT'").fetchall()
        assert ev == [("SCALP_V2", tid, EVENT_CONVERSION, "2483316880")]
        assert conn.execute("SELECT COUNT(*) FROM position_close_ledger WHERE close_reason='DUST_WRITEOFF'").fetchone()[0] == 0
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='trade_learning_outcomes'").fetchone():
            assert conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_day_v2_dust_provenance_preserved_never_legacy_day_live(tmp_path: Path):
    eng = await _make_live_engine(tmp_path, cash=990.0)
    day_tid = "mystic_ETH/USDT_1795200378182"
    _seed_live_fill(eng.db_path, tid=day_tid, sym="ETH/USDT", side="BUY", oid="1702302208", qty=0.0597, px=1627.0, fee=1.194e-5, fee_asset="ETH")
    _seed_live_fill(eng.db_path, tid=day_tid, sym="ETH/USDT", side="SELL", oid="1702379221", qty=0.0596, px=1615.85, fee=0.01926, fee_asset="USDT")
    with sqlite3.connect(eng.db_path) as conn:
        pt_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
        if "engine_id" not in pt_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN engine_id TEXT")
        conn.execute(
            "INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status, strategy_id, engine_id) "
            "VALUES (?, 'live', 'live', 'ETH/USDT', 'BUY', 0.05968806, 1627.0, 8.806e-05, '2026-11-20T18:46:18+00:00', 'executed', 'day', 'DAY_V2')",
            (day_tid,),
        )
        conn.commit()

    bal = {"BTC": 0.0, "ETH": 8.806e-05, "SOL": 0.0, "XRP": 0.0, "USDT": 990.0}
    await eng.run_live_reconcile(bal, free_balances=dict(bal))

    assert all(p.engine_id != "LEGACY_DAY_LIVE" and not str(p.trade_id).startswith("dust_exchange:") for p in eng.open_positions.values())
    held = held_lots(eng.db_path, "ETH/USDT")
    assert len(held) == 1
    assert held[0]["engine_id"] == "DAY_V2"
    assert held[0]["source_trade_id"] == day_tid
    assert held[0]["quantity"] == pytest.approx(8.806e-05)


async def _setup_sell_engine(tmp_path: Path, *, tid: str, engine_id: str, qty: float, entry_px: float) -> tuple[PortfolioEngine, OpenPosition]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / f"{tid.replace('/', '_')}.db"
    initialize_paper_trading_schema(str(db_path))
    eng = PortfolioEngine(db_path=str(db_path), principal=1000.0, test_mode=True)
    eng._ensure_db_schema()
    await eng.initialize_from_db()
    now = time.time() - 120.0
    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=qty,
        entry_price=entry_px,
        entry_time=now,
        trade_id=tid,
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=entry_px + 100.0,
        lowest_price=entry_px - 20.0,
        sleeve=Sleeve.ACTIVE.value,
        entry_strategy_id="binance_scalp_paper" if engine_id == "SCALP_V2" else "day",
        engine_id=engine_id,
        entry_thesis="orderbook",
        status="ACTIVE",
        quantity_exact=f"{qty:.8f}",
    )
    key = make_position_key(engine_id, "BTC/USDT")
    eng.open_positions[key] = pos
    await eng._persist_position_to_sqlite(pos)
    with sqlite3.connect(str(db_path)) as conn:
        pt_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
        if "engine_id" not in pt_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN engine_id TEXT")
        conn.execute(
            "INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status, strategy_id, engine_id) "
            "VALUES (?, 'live', 'live', 'BTC/USDT', 'BUY', ?, ?, ?, datetime('now'), 'executed', ?, ?)",
            (tid, qty, entry_px, qty, pos.entry_strategy_id, engine_id),
        )
        conn.commit()
    eng._positions_initialized = True
    eng._live_execution_enabled = False
    eng._paper_service = MagicMock(paper_run_id="live")
    eng._paper_service.place_order = AsyncMock(return_value={"status": "ok"})
    eng._check_kill_switch_sell = MagicMock(return_value=(True, ""))
    eng._record_reject = AsyncMock()
    eng._clear_all_quarantines = AsyncMock()
    eng._update_coin_performance = AsyncMock()
    eng._record_thesis_regime_outcome = MagicMock()
    eng._record_day_bucket_outcome = MagicMock()
    eng._persist_quality_cooldowns = AsyncMock()
    eng.record_sell_cooldown = MagicMock()
    eng._get_loss_hold_until = AsyncMock(return_value=None)
    eng._set_loss_hold_until = AsyncMock()
    eng.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
    eng._validate_invariants = AsyncMock(return_value=True)
    eng._record_audit = AsyncMock()
    eng._ensure_symbol_constraints = AsyncMock()
    eng._symbol_constraints["BTC/USDT"] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 5.0}
    return eng, pos


async def _run_sell_fill(eng: PortfolioEngine, *, engine_id: str, sold_qty: float, fill_px: float) -> dict:
    eng._normalize_order_amount = MagicMock(return_value=(sold_qty, "ok", sold_qty))
    eng._dust_check = MagicMock(return_value=(False, sold_qty, "", sold_qty * fill_px))
    mock_pf = MagicMock(passed=True, expected_avg_fill=fill_px, to_audit_dict=MagicMock(return_value={"passed": True}))
    sell_eval = {
        "symbol": "BTC/USDT",
        "reason": "SCALP_V2_LEARNED_CONTINUATION",
        "exit_branch": "strategy",
        "avg_entry_price": 57814.01,
        "mark_price": fill_px,
        "mark_source": "test",
        "mark_age_seconds": 0.0,
        "gross_pct": 0.0015,
        "effective_exit_cost_pct": 0.0002,
        "effective_roundtrip_cost_pct": 0.0004,
        "net_exit_pct": 0.0011,
        "roundtrip_net_pct": 0.0011,
        "required_profit_buffer_pct": 0.0,
        "allowed": True,
        "block_reason": "",
        "emergency_flag": False,
        "strategy_authority": True,
        "measured_cost_sample_count": 5,
        "measured_cost_p75": 0.0004,
    }
    with (
        patch.object(eng, "_evaluate_sell_profitability", AsyncMock(return_value=sell_eval)),
        patch("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=mock_pf)),
        patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True),
        patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=eng._paper_service),
        patch.object(eng, "_fetch_mtm_prices_for_open_positions", AsyncMock(return_value={"BTC/USDT": fill_px})),
    ):
        res = await eng.execute_sell_fifo(
            "BTC/USDT",
            sold_qty,
            fill_px,
            ExitType.STRATEGY,
            "SCALP_V2_LEARNED_CONTINUATION",
            engine_id=engine_id,
        )
    assert res is not None
    return res


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tid", "net_custody", "fills", "expected_residual"),
    [
        ("scalp_v2_BTCUSDT_1795252798012", 0.00197960, [0.00062, 0.00135], 0.00000960),
        ("scalp_v2_BTCUSDT_1795253013393", 0.00203959, [0.00043, 0.00062, 0.00098], 0.00000959),
    ],
)
async def test_defect2_and_section18_multi_fill_btc_exits_record_sold_qty_and_preserve_scalp_dust(
    tmp_path: Path,
    tid: str,
    net_custody: float,
    fills: list[float],
    expected_residual: float,
):
    """Regression for historical 0.00197 BTC (2 fills) and 0.00203 BTC (3 fills) exits."""
    eng, _pos = await _setup_sell_engine(tmp_path, tid=tid, engine_id="SCALP_V2", qty=net_custody, entry_px=57814.01)

    for idx, fill_qty in enumerate(fills):
        # Simulate live memory custody application followed by execute_sell_fifo learning write
        await _run_sell_fill(eng, engine_id="SCALP_V2", sold_qty=fill_qty, fill_px=57910.0 + idx * 5.0)

    # Verify SCALP_V2 dust residual preserved in memory & SQLite (not floored to 0 and lost!)
    key = make_position_key("SCALP_V2", "BTC/USDT")
    assert key in eng.open_positions
    assert eng.open_positions[key].engine_id == "SCALP_V2"
    assert eng.open_positions[key].status == "DUST_PENDING"
    assert eng.open_positions[key].quantity == pytest.approx(expected_residual, abs=1e-11)

    # Verify trade_learning_outcomes recorded actual sold_quantity per fill (never residual, 0, or NULL)
    with sqlite3.connect(eng.db_path) as conn:
        rows = conn.execute("SELECT quantity, close_reason, extra_json FROM trade_learning_outcomes ORDER BY id ASC").fetchall()
    # Filter to execution close rows (exclude synthetic trade-final aggregate if emitted)
    exec_rows = [r for r in rows if "sell_qty" in (json.loads(r[2] or "{}") or {})]
    assert [r[0] for r in exec_rows] == [pytest.approx(q) for q in fills]
    assert all(r[0] is not None and r[0] > 0 for r in exec_rows)

    # Verify scalp_learning_outcomes ingests the sold quantities cleanly
    ingest_scalp_outcomes(eng.db_path)
    with sqlite3.connect(eng.db_path) as conn:
        slo_qtys = [r[0] for r in conn.execute("SELECT quantity FROM scalp_learning_outcomes ORDER BY id ASC").fetchall()]
    assert len(slo_qtys) >= 1
    assert all(q > 1e-4 for q in slo_qtys)  # Never 9.6e-6 dust or 0.0!


@pytest.mark.asyncio
async def test_defect2_full_close_with_and_without_dust_records_sold_quantity_never_null_or_residual(tmp_path: Path):
    # Case 1: Full tradable close leaving 9.61e-6 BTC dust (scalp_v2_BTCUSDT_1795253585802)
    eng1, _ = await _setup_sell_engine(tmp_path / "c1", tid="scalp_v2_BTCUSDT_1795253585802", engine_id="SCALP_V2", qty=0.00196961, entry_px=57984.43)
    await _run_sell_fill(eng1, engine_id="SCALP_V2", sold_qty=0.00196, fill_px=57951.33)
    ingest_scalp_outcomes(eng1.db_path)
    with sqlite3.connect(eng1.db_path) as conn:
        tlo_qty = conn.execute("SELECT quantity FROM trade_learning_outcomes ORDER BY id ASC LIMIT 1").fetchone()[0]
        slo_qty = conn.execute("SELECT quantity FROM scalp_learning_outcomes ORDER BY id ASC LIMIT 1").fetchone()[0]
    assert tlo_qty == pytest.approx(0.00196)
    assert slo_qty == pytest.approx(0.00196)

    # Case 2: Exact zero-residual full close
    eng2, _ = await _setup_sell_engine(tmp_path / "c2", tid="mystic_BTC/USDT_1795300000000", engine_id="DAY_V2", qty=0.00200, entry_px=58000.0)
    await _run_sell_fill(eng2, engine_id="DAY_V2", sold_qty=0.00200, fill_px=58200.0)
    with sqlite3.connect(eng2.db_path) as conn:
        tlo_qty2 = conn.execute("SELECT quantity FROM trade_learning_outcomes ORDER BY id ASC LIMIT 1").fetchone()[0]
    assert tlo_qty2 == pytest.approx(0.00200)


@pytest.mark.asyncio
async def test_stale_protected_preexisting_dust_cleared_on_flat_and_capped_by_venue_surplus(tmp_path: Path):
    """Prevents Trade #5 avoidable dust bug (0.0669 vs 0.0671 ETH when prior dust already converted)."""
    eng = await _make_live_engine(tmp_path, cash=990.55)
    stamp_protected_preexisting_dust(eng.db_path, "ETH/USDT", "0.00026")
    stamp_protected_preexisting_dust(eng.db_path, "XRP/USDT", "3.10946")
    assert float(load_protected_preexisting_dust(eng.db_path, "ETH/USDT")) == pytest.approx(0.00026)

    # Reconciling flat exchange clears stale protected_preexisting_dust
    flat = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "XRP": 0.0, "USDT": 990.55}
    await eng.run_live_reconcile(flat, free_balances=dict(flat))
    assert float(load_protected_preexisting_dust(eng.db_path, "ETH/USDT")) == 0.0
    assert float(load_protected_preexisting_dust(eng.db_path, "XRP/USDT")) == 0.0


@pytest.mark.asyncio
async def test_partial_close_active_lot_records_confirmed_sold_qty_not_remaining_active_qty(tmp_path: Path):
    """Partial close leaving an ACTIVE 0.00130 BTC lot must record sold_quantity=0.00070, not remaining 0.00130."""
    eng, _ = await _setup_sell_engine(tmp_path / "partial", tid="mystic_BTC/USDT_1795400000000", engine_id="DAY_V2", qty=0.00200, entry_px=58000.0)
    await _run_sell_fill(eng, engine_id="DAY_V2", sold_qty=0.00070, fill_px=58300.0)
    key = make_position_key("DAY_V2", "BTC/USDT")
    assert eng.open_positions[key].status == "ACTIVE"
    assert eng.open_positions[key].quantity == pytest.approx(0.00130)
    with sqlite3.connect(eng.db_path) as conn:
        tlo_qty = conn.execute("SELECT quantity FROM trade_learning_outcomes ORDER BY id ASC LIMIT 1").fetchone()[0]
    assert tlo_qty == pytest.approx(0.00070)


def test_historical_post_227e036_quantity_backfill_and_tlo_12736_cleanliness(tmp_path: Path):
    db = str(tmp_path / "backfill.db")
    initialize_paper_trading_schema(db)
    _ensure_table(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE live_exchange_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT, symbol TEXT, side TEXT, exchange_order_id TEXT,
                executed_qty REAL, avg_fill_price REAL, fee_amount REAL, fee_asset TEXT, mystic_trade_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE scalp_learning_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER UNIQUE,
                symbol TEXT,
                quantity REAL,
                net_pnl_usd REAL,
                net_pnl_pct REAL
            )
            """
        )
        # 1) Provable post-227e036 row (stored residual 9.61e-6 instead of sold 0.00196)
        pt_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
        if "order_id" not in pt_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN order_id TEXT")
        if "code_sha" not in pt_cols:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN code_sha TEXT")
        conn.execute(
            "INSERT INTO live_exchange_fills (venue, symbol, side, exchange_order_id, executed_qty, avg_fill_price, fee_amount, fee_asset, mystic_trade_id) "
            "VALUES ('binanceus', 'BTC/USDT', 'SELL', '1914089766', 0.00196, 57951.33, 0.0227, 'USDT', 'scalp_v2_BTCUSDT_1795253585802')"
        )
        conn.execute(
            "INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, pnl, timestamp, status, order_id, code_sha) "
            "VALUES ('s1', 'live', 'live', 'BTC/USDT', 'SELL', 0.00196, 57951.33, -0.1100965, '2026-11-21T09:42:37+00:00', 'executed', '1914089766', ?)",
            (POST_227E036_SHA,),
        )
        conn.execute(
            """
            INSERT INTO trade_learning_outcomes (
                id, written_at_utc, mode, symbol, entry_timestamp, exit_timestamp,
                entry_price, exit_price, quantity, fees_paid, net_profit_usd, net_profit_pct,
                close_reason, engine_id, event_key, code_sha, extra_json
            ) VALUES (12735, '2026-11-21 09:42:38', 'live', 'BTC/USDT', 1795253586.0, 1795254158.0,
                      57984.43, 57951.33, 9.61e-06, 0.0454, -0.1100965, -0.0009687,
                      'LEARNED_CONTINUATION_EXIT', 'SCALP_V2', 'live-close:1914089766', ?, ?)
            """,
            (POST_227E036_SHA, json.dumps({"sell_qty": 0.00196, "residual_qty": 9.61e-6})),
        )
        conn.execute("INSERT INTO scalp_learning_outcomes (id, source_id, symbol, quantity, net_pnl_usd, net_pnl_pct) VALUES (5664, 12735, 'BTC/USDT', 9.61e-06, -0.1100965, -0.0009687)")
        # 2) Unprovable row (no fill or paper_trades row) -> must NOT be invented
        conn.execute(
            """
            INSERT INTO trade_learning_outcomes (
                id, written_at_utc, mode, symbol, entry_timestamp, exit_timestamp,
                entry_price, exit_price, quantity, net_profit_usd, net_profit_pct,
                close_reason, engine_id, event_key, code_sha, extra_json
            ) VALUES (12999, '2026-11-21 10:00:00', 'live', 'ETH/USDT', 1795255000.0, 1795255500.0,
                      1800.0, 1805.0, NULL, 0.25, 0.002,
                      'LEARNED_CONTINUATION_EXIT', 'DAY_V2', 'live-close:999999999', ?, '{}')
            """,
            (POST_227E036_SHA,),
        )
        # 3) TLO 12736 spurious DUST_WRITEOFF for venue-converted dust_exchange:BTCUSDT
        conn.execute(
            """
            INSERT INTO trade_learning_outcomes (
                id, written_at_utc, mode, symbol, entry_timestamp, exit_timestamp,
                entry_price, quantity, dust_remaining_qty, close_reason, decision_reason,
                engine_id, realized_profit_unknown, event_key, extra_json
            ) VALUES (12736, '2026-11-21 18:34:40', 'live', 'BTC/USDT', 1795253381.8, 1795286080.0,
                      58024.96, 1.919e-05, 1.919e-05, 'DUST_WRITEOFF', 'engine_close:DUST_WRITEOFF',
                      'LEGACY_DAY_LIVE', 1, 'DUST_WRITEOFF|LEGACY_DAY_LIVE|BTC/USDT|dust_exchange:BTCUSDT|1.919e-05',
                      ?)
            """,
            (json.dumps({"bad_trade": True, "good_trade": False, "is_dust": True, "trade_id": "dust_exchange:BTCUSDT"}),),
        )
        conn.commit()

    res = backfill_post_227e036_learning_quantities(db)
    assert len(res["tlo_updated"]) == 1
    assert res["tlo_updated"][0]["id"] == 12735
    assert res["tlo_updated"][0]["new_quantity"] == pytest.approx(0.00196)
    assert len(res["slo_updated"]) == 1
    assert res["slo_updated"][0]["new_quantity"] == pytest.approx(0.00196)
    assert len(res["unprovable_rows"]) == 1
    assert res["unprovable_rows"][0]["id"] == 12999

    clean = verify_and_clean_venue_converted_dust_learning_row(db, tlo_id=12736, venue_conversions=BTC_CONVERSION_LOG)
    assert clean["reclassified"] is True
    assert clean["venue_ref"] == "2483316880"
    assert all(v == 0 for v in clean["adaptive_impact"].values())

    with sqlite3.connect(db) as conn:
        r12735 = conn.execute("SELECT quantity, net_profit_usd, net_profit_pct, fees_paid, entry_price, exit_price FROM trade_learning_outcomes WHERE id=12735").fetchone()
        assert r12735[0] == pytest.approx(0.00196)
        assert r12735[1] == pytest.approx(-0.1100965)
        assert r12735[2] == pytest.approx(-0.0009687)
        assert r12735[3] == pytest.approx(0.0454)
        assert r12735[4] == pytest.approx(57984.43)
        assert r12735[5] == pytest.approx(57951.33)

        r12999 = conn.execute("SELECT quantity FROM trade_learning_outcomes WHERE id=12999").fetchone()
        assert r12999[0] is None

        r12736 = conn.execute("SELECT close_reason, extra_json FROM trade_learning_outcomes WHERE id=12736").fetchone()
        assert r12736[0] == EVENT_CONVERSION
        ex12736 = json.loads(r12736[1])
        assert ex12736["bad_trade"] is False
        assert ex12736["venue_ref"] == "2483316880"
