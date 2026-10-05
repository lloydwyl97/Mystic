"""A reload or valuation must never pair a just-committed BUY lot with pre-debit cash.

Ocean 2026-10-05 01:39:27: SCALP_V2 bought 94.58 XRP for $143.97. The lot was
visible before the cash debit and the MTM loop persisted total_equity 1284.27
against a true 1140.30 until its next cycle.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

import backend.services.portfolio_engine as pe
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, make_position_key

KEY = make_position_key("SCALP_V2", "XRP/USDT")


def _engine(tmp_path) -> PortfolioEngine:
    eng = PortfolioEngine(db_path=str(tmp_path / "buy.db"), principal=1164.16, test_mode=True)
    eng._ensure_db_schema()
    return eng


def _lot() -> OpenPosition:
    return OpenPosition(
        symbol="XRP/USDT",
        quantity=94.58108,
        entry_price=1.5222,
        entry_time=time.time(),
        trade_id="scalp_v2_XRPUSDT_1790648367000",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        engine_id="SCALP_V2",
    )


async def test_reload_inside_a_buy_commit_does_not_adopt_the_lot_before_the_debit(tmp_path):
    eng = _engine(tmp_path)
    await eng._persist_position_to_sqlite(_lot())
    eng.open_positions = {}
    with eng._buy_commit_window():
        await eng._load_positions_from_sqlite(allow_mutations=False)
        assert eng.open_positions == {}
    await eng._load_positions_from_sqlite(allow_mutations=False)
    assert eng.open_positions[KEY].quantity == pytest.approx(94.58108)


async def test_reload_overlapping_a_buy_publish_keeps_the_published_lot(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    eng.open_positions = {}
    lot = _lot()
    real_ro = pe.connect_ro

    def _ro_with_buy_publishing(*a, **k):
        with eng._buy_commit_window():
            eng.open_positions = {KEY: lot}
        return real_ro(*a, **k)

    monkeypatch.setattr(pe, "connect_ro", _ro_with_buy_publishing)
    await eng._load_positions_from_sqlite(allow_mutations=False)
    assert eng.open_positions[KEY] is lot


async def test_valuation_straddling_a_buy_publish_is_discarded(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    eng.open_positions = {}
    eng.cash_balance = 680.60
    eng._positions_value = 0.0
    eng._total_equity = 680.60
    lot = _lot()
    notional = lot.quantity * lot.entry_price
    real = eng._compute_positions_value_and_cost_basis

    def _value_while_buy_publishes(prices=None):
        stale = real(prices)
        with eng._buy_commit_window():
            eng.open_positions = {KEY: lot}
            eng.cash_balance = 680.60 - notional
            eng._positions_value = notional
            eng._total_equity = 680.60
        return stale

    monkeypatch.setattr(eng, "_compute_positions_value_and_cost_basis", _value_while_buy_publishes)
    await eng._recompute_positions_values({"XRP/USDT": lot.entry_price})
    assert eng._total_equity == pytest.approx(680.60)
    assert eng._total_equity == pytest.approx(eng.cash_balance + eng._positions_value)


def test_buy_commit_window_releases_on_failure_and_bumps_generation(tmp_path):
    eng = _engine(tmp_path)
    with pytest.raises(RuntimeError), eng._buy_commit_window():
        assert eng._buys_inflight == 1
        raise RuntimeError("commit failed")
    assert eng._buys_inflight == 0
    assert eng._buy_seq == 1


def test_both_buy_paths_commit_and_publish_inside_the_window():
    tree = ast.parse(Path(pe.__file__).read_text())
    published: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute) and i.context_expr.func.attr == "_buy_commit_window" for i in node.items):
            continue
        referenced = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        assigned = {t.attr for n in ast.walk(node) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Attribute)}
        for commit in ("_commit_atomic_day_open_sync", "_scalp_v2_commit_buy_sync"):
            if commit in referenced:
                published[commit] = assigned
    required = {"open_positions", "cash_balance", "_positions_value", "_total_equity"}
    assert set(published) == {"_commit_atomic_day_open_sync", "_scalp_v2_commit_buy_sync"}
    assert all(required <= assigned for assigned in published.values())
