from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.endpoints import paper_trading_endpoints, portfolio_engine_endpoints
from backend.services.admin_auth import require_admin_key
from backend.services.portfolio_engine import PortfolioEngine
from backend.utils.redis_helpers import SHARED_ATOMIC_WRITER_ROLES, WRITER_ROLES

ROOT = Path(__file__).resolve().parents[1]


def _route_has_admin_dependency(router, path: str, method: str) -> bool:
    for route in router.routes:
        if route.path == path and method.upper() in route.methods:
            return any(dep.call is require_admin_key for dep in route.dependant.dependencies)
    raise AssertionError(f"route not found: {method} {path}")


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/portfolio-engine/sync-from-binance", "POST"),
        ("/api/portfolio-engine/sync", "POST"),
        ("/api/portfolio-engine/execution-mode", "POST"),
    ],
)
def test_portfolio_mutation_routes_require_admin_key(path: str, method: str) -> None:
    assert _route_has_admin_dependency(portfolio_engine_endpoints.router, path, method)


def test_paper_order_cancellation_requires_admin_key() -> None:
    assert _route_has_admin_dependency(
        paper_trading_endpoints.router,
        "/api/paper-trading/orders/{order_id}",
        "DELETE",
    )


def test_admin_dependency_accepts_dashboard_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setenv("ADMIN_TOKEN", "operator-secret")
    require_admin_key(x_api_key=None, authorization="Bearer operator-secret")
    with pytest.raises(HTTPException) as exc:
        require_admin_key(x_api_key=None, authorization="Bearer wrong")
    assert exc.value.status_code == 401


def test_buy_cash_gate_precedes_trade_row_commit() -> None:
    source = inspect.getsource(PortfolioEngine._execute_buy_fifo_locked)
    final_gate = source.index("pending_other = self._pending_buy_notional(exclude_symbol=symbol)")
    trade_commit = source.index("_commit_atomic_day_open_sync")
    cash_apply = source.index("self.cash_balance = committed_cash")
    assert final_gate < trade_commit < cash_apply


def test_tp1_latch_occurs_only_after_successful_sell_result() -> None:
    source = inspect.getsource(PortfolioEngine._check_exit_conditions)
    sell = source.index("_tp1_result = await self.execute_sell_fifo")
    success = source.index("if _tp1_result is not None:")
    latch = source.index("remaining_position.tp1_hit = True")
    assert sell < success < latch


@pytest.mark.asyncio
async def test_get_ledger_does_not_heal_or_persist_on_read(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
    engine._ensure_db_schema()
    engine._realized_pnl = 0.0
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (
                trade_id, paper_run_id, mode, symbol, side, quantity, price,
                pnl, timestamp, status
            ) VALUES ('sell-1', 'test', 'paper', 'BTC/USDT', 'SELL', 1, 100, 25,
                      '2026-08-01T00:00:00+00:00', 'executed')
            """
        )
        conn.commit()

    engine._persist_ledger_to_sqlite = AsyncMock()
    result = await engine.get_ledger()

    assert result["realized_pnl"] == 0.0
    assert engine._realized_pnl == 0.0
    engine._persist_ledger_to_sqlite.assert_not_awaited()


def test_writer_topology_distinguishes_exclusive_and_shared_atomic_roles() -> None:
    assert WRITER_ROLES == {
        "MARKET_DATA": "market_data_writer",
        "AI_SIGNALS": "ai_signal_writer",
        "DECISION_ROUTER": "decision_router",
    }
    assert SHARED_ATOMIC_WRITER_ROLES == {
        "RATE_LIMITER": "binance_weight_limiter",
    }


def test_market_and_decision_services_acquire_exclusive_writer_locks() -> None:
    market_source = (ROOT / "backend/services/live_market_data.py").read_text()
    decision_source = (ROOT / "backend/services/portfolio_engine_integration.py").read_text()
    assert 'WriterLock(WRITER_ROLES["MARKET_DATA"]' in market_source
    assert 'WriterLock(WRITER_ROLES["DECISION_ROUTER"]' in decision_source
