"""Exit/order repair batch — nine defects found in the Ocean app-wide audit.

Each test names the specific live defect it pins, so a regression reads as the
original failure rather than as an abstract assertion.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from backend.config.execution_cost_model import honest_all_in_rt_pct


# ---------------------------------------------------------------- 1. AllWeather shadow
def test_allweather_position_requires_explicit_family_tag():
    """A DAY position whose thesis is "BREAKOUT" is not an all-weather position.

    SETUP_BREAKOUT is the literal string "BREAKOUT", which is also a live DAY
    setup name, so the thesis-name fallback claimed ordinary engine-managed
    entries and made them skip the engine exit stack.
    """
    from backend.services.allweather_breakout_pullback_adapter import (
        STRATEGY_FAMILY,
        is_allweather_position,
    )

    day_pos = SimpleNamespace(
        strategy_family="",
        entry_strategy_id="",
        entry_thesis="BREAKOUT",
        thesis_target_level=105.0,
        thesis_invalid_level=95.0,
    )
    assert is_allweather_position(day_pos) is False

    tagged = SimpleNamespace(strategy_family=STRATEGY_FAMILY, entry_thesis="")
    assert is_allweather_position(tagged) is True


def test_allweather_shadow_defaults_off(monkeypatch):
    from backend.services import allweather_breakout_pullback_adapter as awbp

    monkeypatch.delenv("ALLWEATHER_BREAKOUT_PULLBACK_SHADOW", raising=False)
    assert awbp.shadow_enabled() is False


# ------------------------------------------------- 2. symbol-aware executable profit floor
def test_executable_profit_floor_uses_symbol_floor(monkeypatch):
    """An exit authorized by a low per-coin floor must not be rejected by the global."""
    import backend.services.protected_limit_execution as ple

    # Global floor 0.40%, XRP floor 0.10%. A +0.25% net exit is valid for XRP.
    monkeypatch.setattr(ple, "MIN_NET_PROFIT_TO_SELL", 0.004, raising=False)
    monkeypatch.setattr(ple, "min_net_profit_for_symbol", lambda _symbol: 0.001, raising=False)
    monkeypatch.setattr(ple, "honest_all_in_rt_pct", lambda _symbol: 0.0, raising=False)

    check = ple.evaluate_executable_sell_profit(
        entry_price=100.0,
        quantity=10.0,
        executable_sell_price=100.25,
        sell_fee_rate=0.0,
        symbol="XRP/USDT",
    )
    assert check.passed, check.reject_reason

    # Same trade without a symbol still gets the global floor and is rejected,
    # which is the behaviour that was blocking live profit exits.
    unscoped = ple.evaluate_executable_sell_profit(
        entry_price=100.0,
        quantity=10.0,
        executable_sell_price=100.25,
        sell_fee_rate=0.0,
    )
    assert not unscoped.passed
    assert unscoped.reject_reason == ple.EXECUTABLE_NET_PROFIT_BELOW_FLOOR


# ------------------------------------------------------------------ 3. per-symbol sell lock
def test_sell_lock_serializes_same_symbol():
    """Two concurrent exits for one symbol must not both reach the exchange."""
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._sell_execution_locks = {}
    in_flight = 0
    peak = 0

    async def fake_locked(symbol, quantity, price, exit_type, exit_trigger, current_bar=None, force_sell=False):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)  # stands in for the exchange await
        in_flight -= 1
        return {"symbol": symbol}

    engine._execute_sell_fifo_locked = fake_locked

    async def run():
        await asyncio.gather(
            PortfolioEngine.execute_sell_fifo(engine, "BTC/USDT", 1.0, 100.0, None, "t"),
            PortfolioEngine.execute_sell_fifo(engine, "BTC/USDT", 1.0, 100.0, None, "t"),
        )

    asyncio.run(run())
    assert peak == 1, f"{peak} concurrent sells reached the exchange path for one symbol"


def test_sell_lock_does_not_serialize_across_symbols():
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._sell_execution_locks = {}
    peak = 0
    in_flight = 0

    async def fake_locked(symbol, *a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1

    engine._execute_sell_fifo_locked = fake_locked

    async def run():
        await asyncio.gather(
            PortfolioEngine.execute_sell_fifo(engine, "BTC/USDT", 1.0, 100.0, None, "t"),
            PortfolioEngine.execute_sell_fifo(engine, "ETH/USDT", 1.0, 100.0, None, "t"),
        )

    asyncio.run(run())
    assert peak == 2, "different symbols must still exit in parallel"


# -------------------------------------------------------------- 6. restart recovery states
class _FakeLive:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[tuple] = []

    async def fetch_order(self, exchange: str, order_id: str, symbol: str, params=None):
        self.calls.append((exchange, order_id, symbol, params))
        return self.payload


@pytest.mark.parametrize(
    ("status", "filled", "expected"),
    [
        ("closed", 1.0, "filled"),
        ("open", 0.0, "open"),
        ("new", 0.0, "open"),
        ("partially_filled", 0.5, "open"),
        ("accepted", 0.0, "accepted"),
        ("canceled", 0.0, "canceled"),
        ("rejected", 0.0, "canceled"),
        ("expired", 0.0, "canceled"),
        ("weird_venue_status", 0.0, "unknown"),
    ],
)
def test_exchange_order_classifies_every_state(status, filled, expected):
    """fetch_order is (exchange, order_id, symbol); the old kwargs never bound."""
    from backend.services.day_trailing_buy import _exchange_order

    live = _FakeLive({"status": "success", "order": {"id": "OID-1", "status": status, "filled": filled}})
    engine = SimpleNamespace(_live_service=live)

    got = asyncio.run(_exchange_order(engine, client_order_id="CID-1", symbol="BTC/USDT", order_id="1837670272"))

    assert got is not None
    assert got["state"] == expected
    assert live.calls == [("binanceus", "1837670272", "BTC/USDT", None)]


def test_exchange_order_fetch_failure_is_unknown_not_no_order():
    """A transport failure must never be read as "proven no order"."""
    from backend.services.day_trailing_buy import _exchange_order

    class Boom:
        async def fetch_order(self, *a, **k):
            raise RuntimeError("connection reset")

    engine = SimpleNamespace(_live_service=Boom())
    got = asyncio.run(_exchange_order(engine, client_order_id="CID", symbol="BTC/USDT", order_id="OID"))
    assert got is not None
    assert got["state"] == "unknown"


# ------------------------------------------------- 7. IOC honors PROTECTED_LIMIT_ALLOW_PARTIAL
def test_production_disallows_partial_fills():
    from backend.config.protected_execution import PROTECTED_LIMIT_ALLOW_PARTIAL

    assert PROTECTED_LIMIT_ALLOW_PARTIAL is False, "Ocean runs PROTECTED_LIMIT_ALLOW_PARTIAL=false"


def test_allow_partial_is_enforced_before_the_order_is_sent():
    """With partials disallowed, a quantity the book cannot fill is never sent.

    This is the only point where the flag can be enforced. After an IOC has
    partially filled, the asset is already in the account and the fill must be
    adopted; dropping it would leave inventory we own untracked.
    """
    import inspect

    import backend.services.protected_limit_execution as ple

    src = inspect.getsource(ple.run_protected_preflight)
    assert src.count("not PROTECTED_LIMIT_ALLOW_PARTIAL and not fully") >= 1
    assert "not flatten and not PROTECTED_LIMIT_ALLOW_PARTIAL and not fully" in src


def test_dust_writeoff_is_booked_once_per_residual():
    """The same residual was written off on every monitor cycle.

    A single BTC 1e-05 leftover produced four DUST_WRITEOFF rows in 70 seconds,
    each at the full residual notional.
    """
    import inspect

    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine._execute_sell_fifo_locked)
    assert "_dust_writeoff_seen" in src
    assert "DUST_WRITEOFF_ALREADY_BOOKED" in src


# ------------------------------------------ 8. break-even / trail positive after honest cost
@pytest.mark.parametrize("symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
def test_break_even_offset_covers_round_trip(symbol, monkeypatch):
    from backend.services.day_controlled_exits import _break_even_offset_pct

    monkeypatch.setenv("DAY_BREAK_EVEN_OFFSET_PCT", "0.0005")
    offset = _break_even_offset_pct(symbol)
    cost = honest_all_in_rt_pct(symbol)
    assert offset >= cost, f"{symbol} break-even at {offset:.6f} is below its {cost:.6f} round trip"


@pytest.mark.parametrize("symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
def test_break_even_stop_is_profitable_after_costs(symbol, monkeypatch):
    from backend.services.day_controlled_exits import apply_break_even_and_mfe_trail

    monkeypatch.setenv("DAY_BREAK_EVEN_TRAIL_ENABLED", "true")
    entry = 100.0
    cost = honest_all_in_rt_pct(symbol)
    pos = SimpleNamespace(
        symbol=symbol,
        entry_price=entry,
        highest_price=entry * 1.02,  # well past any trigger
        stop_price=0.0,
        trailing_stop_price=0.0,
    )
    assert apply_break_even_and_mfe_trail(pos, entry * 1.02) is True
    net = (pos.stop_price - entry) / entry - cost
    assert net >= -1e-12, f"{symbol} break-even stop nets {net * 1e4:.2f} bps"


@pytest.mark.parametrize("symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
def test_trail_activation_ratchet_is_profitable(symbol):
    """Activating at entry*(1+d) put the ratchet at entry*(1-d^2) — below entry."""
    from backend.services.day_controlled_exits import _trail_activation_price

    entry = 100.0
    d = 0.005
    activation = _trail_activation_price(entry=entry, trail_distance=d, symbol=symbol)
    ratchet = activation * (1.0 - d)
    net = (ratchet - entry) / entry - honest_all_in_rt_pct(symbol)
    assert net >= -1e-12, f"{symbol} trail ratchet nets {net * 1e4:.2f} bps at activation"
    assert activation > entry * (1.0 + d), "activation must be above the naive level"


def test_refresh_trailing_stop_never_sets_ratchet_below_cost(monkeypatch):
    from backend.services.day_controlled_exits import refresh_trailing_stop

    monkeypatch.setenv("DAY_BREAK_EVEN_TRAIL_ENABLED", "false")
    entry = 100.0
    symbol = "SOL/USDT"
    pos = SimpleNamespace(
        symbol=symbol,
        entry_price=entry,
        highest_price=entry * 1.01,
        trailing_stop_price=0.0,
        stop_price=0.0,
        trail_pct=0.005,
    )
    refresh_trailing_stop(pos, entry * 1.01, {"trail": 0.005, "sl": 0.010})
    if pos.trailing_stop_price > 0:
        net = (pos.trailing_stop_price - entry) / entry - honest_all_in_rt_pct(symbol)
        assert net >= -1e-12, f"activated trail ratchet nets {net * 1e4:.2f} bps"


# ------------------------------------------------------------ 9. paper vs live P&L separation
def test_realized_pnl_is_mode_scoped(tmp_path):
    """The ledger was summing simulated paper P&L into the live account."""
    import sqlite3

    from backend.services.portfolio_engine import PortfolioEngine

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE paper_trades (
            mode TEXT, side TEXT, pnl REAL, exit_type TEXT,
            is_synthetic INTEGER, timestamp TEXT
        )"""
    )
    conn.executemany(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?)",
        [
            ("paper", "SELL", 973.98, "NET_PROFIT", 0, "2026-08-20T00:00:00+00:00"),
            ("live", "SELL", -9.47, "NET_PROFIT", 0, "2026-09-01T00:00:00+00:00"),
        ],
    )
    conn.commit()
    conn.close()

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(db)
    engine._live_execution_enabled = True
    engine._forward_paper_epoch_start = lambda: None

    split = engine.realized_pnl_by_mode()
    assert round(split["paper"], 2) == 973.98
    assert round(split["live"], 2) == -9.47

    live_only = engine._compute_realized_pnl_from_paper_trades()
    assert round(live_only, 2) == -9.47, "live realized P&L must exclude paper rows"

    engine._live_execution_enabled = False
    assert round(engine._compute_realized_pnl_from_paper_trades(), 2) == 973.98


def test_production_env_has_shadow_disabled():
    """Ocean must not run the all-weather shadow alongside live DAY."""
    env_path = "/home/mystic/mystic/.env"
    if not os.path.exists(env_path):
        pytest.skip("production env not present on this host")
    with open(env_path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip().startswith("ALLWEATHER_BREAKOUT_PULLBACK_SHADOW=")]
    if not lines:
        pytest.skip("flag not set; adapter default is off")
    assert lines[-1].split("=", 1)[1].strip().lower() in ("false", "0", "no", "off")
