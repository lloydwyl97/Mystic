"""Tests for DAY V2 direct live entry (no trailing-buy confirmation stack).

A qualified closed-15m setup must proceed straight to protected live
execution: no WAIT_DIP, no 20bp dip, no 6bp rebound, no 26bp retention
bracket, no 5m confirmation, no 60m entry TTL. Setup definitions,
frequency guards, hard safety, sizing, exits, and SCALP are unchanged.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
import tempfile
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from backend.services.day_v2.live_entry import (
    DAY_DIRECT_ENTRY_V1,
    submit_day_v2_direct_entry,
)


@pytest.fixture(autouse=True)
def _day_v2_enabled():
    import backend.services.day_v2.live_entry as _le

    with patch.object(_le, "DAY_V2_ENABLED", True):
        yield


from backend.services.day_v2.live_signal import (
    ENABLED_SETUPS,
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_EXHAUSTION_MR,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    SETUP_VWAP_REVERSION,
    DayV2Signal,
    _opportunity_id,
    _signal_bar_ts,
    evaluate_entry_signal,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEngine:
    """Minimal engine surface for submit_day_v2_direct_entry."""

    def __init__(self, db_path: str, *, can_open: tuple[bool, str] = (True, ""), fill: bool = True):
        self.db_path = db_path
        self._can_open = can_open
        self._fill = fill
        self.open_positions: dict[str, Any] = {}
        self.last_buy_reject_reason = ""
        self.last_buy_outcome = ""
        self._trading_paused = False
        self._pause_reason = ""
        self._available_balance = 500.0
        self._entry_reservations: dict[str, Any] = {}
        self.buy_calls: list[dict[str, Any]] = []

    def _check_kill_switch_buy(self) -> tuple[bool, str]:
        return True, ""

    async def _can_open_position(self, symbol: str, notional: float, *, decision_id: str = "") -> tuple[bool, str]:
        return self._can_open

    def _pending_buy_order_symbols(self) -> set[str]:
        return set()

    def _own_entry_reservation(self, symbol: str, decision_id: str) -> tuple[dict[str, Any], str]:
        return {}, ""

    def _pending_buy_notional(self, **kwargs: Any) -> float:
        return 0.0

    async def execute_buy_fifo(self, **kwargs: Any) -> dict[str, Any] | None:
        self.buy_calls.append(dict(kwargs))
        if not self._fill:
            self.last_buy_reject_reason = "BINANCE:-2010:Account has insufficient balance for requested action."
            return None
        return {
            "order_id": "test-order-1",
            "fill_id": "test-fill-1",
            "trade_id": "test-trade-1",
            "price": kwargs.get("price"),
            "quantity": kwargs.get("quantity"),
            "filled": True,
        }


def _make_db() -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def _make_signal(**overrides: Any) -> DayV2Signal:
    base = {
        "symbol": "BTCUSDT",
        "setup": SETUP_HTF_TREND_PULLBACK,
        "regime": "bull",
        "structural_anchor": 84000.0,
        "target_price": 86000.0,
        "atr": 150.0,
        "signal_bar_ts": 1790520000,
        "h1_bullish": True,
        "opportunity_id": _opportunity_id("BTCUSDT", SETUP_HTF_TREND_PULLBACK, 84000.0),
    }
    base.update(overrides)
    return DayV2Signal(**base)


def _bull_bars(n: int = 60) -> tuple[list, list, list]:
    bars_15m = []
    ts = int(time.time()) - n * 900
    c = 50000.0
    for i in range(n):
        if i < n - 5:
            c = c * 1.0005
        elif i < n - 1:
            c = c * 0.998
        else:
            c = c * 1.002
        bars_15m.append({"ts": ts + i * 900, "open": c * 0.999, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1000.0})
    bars_1h = []
    for i in range(20):
        c_1h = 50000.0 * (1.0 + i * 0.001)
        bars_1h.append({"ts": ts + i * 3600, "open": c_1h, "high": c_1h * 1.005, "low": c_1h * 0.995, "close": c_1h, "volume": 5000.0})
    bars_4h = []
    c_4h = 50000.0
    for i in range(15):
        c_4h = c_4h * 1.002
        bars_4h.append({"ts": ts + i * 14400, "open": c_4h, "high": c_4h * 1.01, "low": c_4h * 0.99, "close": c_4h, "volume": 20000.0})
    return bars_15m, bars_1h, bars_4h


# ---------------------------------------------------------------------------
# 1-7: direct entry, no confirmation stack
# ---------------------------------------------------------------------------


class TestDirectEntry:
    @pytest.mark.asyncio
    async def test_valid_setup_proceeds_directly_to_execution(self):
        db = _make_db()
        engine = FakeEngine(db)
        signal = _make_signal()
        result = await submit_day_v2_direct_entry(engine, signal=signal, ask_price=85000.0, quantity=0.001)
        assert result is not None
        assert len(engine.buy_calls) == 1
        call = engine.buy_calls[0]
        assert call["symbol"] == "BTCUSDT"
        assert call["price"] == pytest.approx(85000.0)
        assert call["entry_authority"] == "DAY_V2_CONFIRMED"
        assert result["entry_policy_version"] == DAY_DIRECT_ENTRY_V1

    @pytest.mark.asyncio
    async def test_no_wait_dip_intent_created(self):
        """A direct entry must not create any WAIT_DIP/TRAIL_LOW intent row."""
        db = _make_db()
        engine = FakeEngine(db)
        signal = _make_signal()
        await submit_day_v2_direct_entry(engine, signal=signal, ask_price=85000.0, quantity=0.001)
        with sqlite3.connect(db) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "day_trailing_buy_intents" not in tables

    @pytest.mark.asyncio
    async def test_no_dip_rebound_retention_or_5m_gate(self):
        """Submit succeeds immediately on an empty DB (no 5m bars, no dip
        tracking) — proving none of the confirmation stack is consulted."""
        db = _make_db()
        engine = FakeEngine(db)
        signal = _make_signal()
        # Ask is ABOVE any dip level and no market history exists at all.
        result = await submit_day_v2_direct_entry(engine, signal=signal, ask_price=86000.0, quantity=0.001)
        assert result is not None
        assert engine.buy_calls[0]["price"] == pytest.approx(86000.0)

    def test_direct_entry_module_has_no_5m_or_trailing_dependency(self):
        """The direct path must not consult the confirmation machinery.

        Docstring prose is ignored — only executable code is checked.
        """
        import ast as _ast

        import backend.services.day_v2.live_entry as le

        tree = _ast.parse(inspect.getsource(le.submit_day_v2_direct_entry))
        parts: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Constant) and isinstance(node.value.value, str):
                continue  # docstring prose is not executable code
            if isinstance(node, (_ast.Assign, _ast.Expr, _ast.Await, _ast.Call, _ast.If, _ast.Return)):
                parts.append(_ast.unparse(node))
        code = "\n".join(parts)
        assert "five_min_confirm" not in code
        assert "WAIT_DIP" not in code
        assert "TRAIL_LOW" not in code
        assert "create_intent" not in code
        assert "expires_at" not in code

    def test_no_entry_ttl_constant_used_by_direct_path(self):
        import backend.services.day_v2.live_entry as le

        src = inspect.getsource(le.submit_day_v2_direct_entry)
        assert "STRUCTURAL_OPPORTUNITY_LIFETIME_SEC" not in src
        assert "expires_at" not in src


# ---------------------------------------------------------------------------
# 8: setup definitions unchanged
# ---------------------------------------------------------------------------


class TestSetupDefinitionsUnchanged:
    def test_all_five_families_enabled(self):
        assert (
            frozenset(
                {
                    SETUP_HTF_TREND_PULLBACK,
                    SETUP_BREAKOUT_CONTINUATION,
                    SETUP_RANGE_BOUNCE,
                    SETUP_VWAP_REVERSION,
                    SETUP_EXHAUSTION_MR,
                }
            )
            == ENABLED_SETUPS
        )

    def test_range_bounce_still_fires_on_oversold_bounce(self):
        """Engineered decline + green bounce near the 20-bar low fires RANGE_BOUNCE."""
        bars_15m = []
        ts = int(time.time()) - 51 * 900
        c = 100.0
        for i in range(50):
            c = c * 0.99785  # steady bleed: RSI collapses, %B hugs the floor
            bars_15m.append({"ts": ts + i * 900, "open": c * 1.0005, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1000.0})
        c = c * 1.003  # first green reaction bar, still at the lows
        bars_15m.append({"ts": ts + 50 * 900, "open": c / 1.003, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1000.0})
        bars_4h = [{"ts": ts + i * 14400, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 5000.0} for i in range(15)]
        signal = evaluate_entry_signal("BTCUSDT", bars_15m, [], bars_4h)
        assert signal is not None
        assert signal.setup == SETUP_RANGE_BOUNCE

    def test_flat_bars_still_produce_no_signal(self):
        bars = [{"ts": int(time.time()) - (60 - i) * 900, "open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 10.0} for i in range(60)]
        assert evaluate_entry_signal("BTCUSDT", bars, [], []) is None


# ---------------------------------------------------------------------------
# 9-15: guards unchanged
# ---------------------------------------------------------------------------


class TestGuardsUnchanged:
    def test_frequency_guard_caps_unchanged(self):
        from backend.services.day_v2.frequency_guard import (
            DAY_V2_MAX_FILLS_PER_SYMBOL_24H,
            DAY_V2_MAX_FILLS_TOTAL_24H,
            check_frequency_limit,
        )

        assert DAY_V2_MAX_FILLS_PER_SYMBOL_24H == 2
        assert DAY_V2_MAX_FILLS_TOTAL_24H == 8
        allowed, _ = check_frequency_limit(_make_db(), "BTCUSDT")
        assert allowed is True

    def test_stale_data_safety_still_rejects(self):
        from backend.services.day_v2.candle_wait import evaluate_candle_gate

        out = evaluate_candle_gate(
            completed_bar_count=0,
            minimum_bars=32,
            latest_open=None,
            required_open=float(int(time.time()) // 900 * 900 - 900),
            executable_price=50000.0,
            book_age_sec=1.0,
            book_stale_sec=30.0,
            now=time.time(),
        )
        assert out["action"] in {"pending", "reject", "timeout"}
        assert out["action"] != "proceed"

    def test_missing_price_still_rejects(self):
        from backend.services.day_v2.cycle_gate import cycle_decision

        out = cycle_decision(
            completed_bar_count=60,
            minimum_bars=32,
            executable_price=0.0,
            book_age_sec=None,
            book_stale_sec=30.0,
            already_evaluated=False,
            retried=True,
        )
        assert out["action"] == "reject"

    @pytest.mark.asyncio
    async def test_capacity_block_still_holds(self):
        db = _make_db()
        engine = FakeEngine(db, can_open=(False, "MAX_POSITIONS_REACHED"))
        signal = _make_signal()
        result = await submit_day_v2_direct_entry(engine, signal=signal, ask_price=85000.0, quantity=0.001)
        assert result is None
        assert engine.buy_calls == []

    def test_cross_engine_ownership_still_blocks(self):
        from backend.services.two_engine_claim import claim_symbol

        db = _make_db()
        positions = {"BTC/USDT": SimpleNamespace(status="ACTIVE", engine_id="SCALP_V2", quantity=0.01)}
        ok, reason, _ = claim_symbol(db, "BTCUSDT", "DAY_V2", "opp-1", 30.0, positions=positions)
        assert ok is False
        assert reason == "SYMBOL_OCCUPIED_BY_OTHER_ENGINE"

    def test_max_combined_positions_still_blocks(self):
        from backend.services.two_engine_claim import claim_symbol

        db = _make_db()
        positions = {f"C{i}/USDT": SimpleNamespace(status="ACTIVE", engine_id="SCALP_V2", quantity=1.0) for i in range(4)}
        ok, reason, _ = claim_symbol(db, "BTCUSDT", "DAY_V2", "opp-1", 30.0, positions=positions)
        assert ok is False
        assert reason == "MAX_COMBINED_POSITIONS"

    def test_reservation_release_and_consume(self):
        from backend.services.day_entry_reservations import (
            consume_reservation,
            create_reservation,
            release_reservation,
        )

        db = _make_db()
        ok, _, rid = create_reservation(db, decision_id="d1", symbol="BTC/USDT", notional_usd=30.0, sleeve="DAY_V2")
        assert ok is True
        assert release_reservation(db, reservation_id=rid, decision_id="d1", symbol="BTC/USDT") is True
        ok2, _, rid2 = create_reservation(db, decision_id="d2", symbol="ETH/USDT", notional_usd=30.0, sleeve="DAY_V2")
        assert ok2 is True
        assert consume_reservation(db, reservation_id=rid2, decision_id="d2", symbol="ETH/USDT") is True
        # second consume is a no-op
        assert consume_reservation(db, reservation_id=rid2, decision_id="d2", symbol="ETH/USDT") is False


# ---------------------------------------------------------------------------
# 16: fill creates correct DAY_V2 position wiring
# ---------------------------------------------------------------------------


class TestFillWiring:
    @pytest.mark.asyncio
    async def test_fill_passes_day_v2_ownership_and_thesis(self):
        db = _make_db()
        engine = FakeEngine(db)
        signal = _make_signal()
        await submit_day_v2_direct_entry(engine, signal=signal, ask_price=85000.0, quantity=0.001, stop_price=84000.0)
        call = engine.buy_calls[0]
        assert call["fill_engine_id"] == "DAY_V2"
        assert call["fill_opportunity_id"] == signal.opportunity_id
        assert call["bar_timestamp"] == 1790520000
        exp = call["explainability"]
        assert float(exp.thesis_invalid_level) == pytest.approx(84000.0)
        assert float(exp.thesis_target_level) == pytest.approx(86000.0)

    def test_execute_path_threads_fill_engine_id(self):
        import inspect as _inspect

        from backend.services import portfolio_engine as _pe

        for fn_name in ("execute_buy_fifo", "_execute_buy_fifo_locked"):
            params = _inspect.signature(getattr(_pe.PortfolioEngine, fn_name)).parameters
            assert "fill_engine_id" in params, fn_name
            assert "fill_opportunity_id" in params, fn_name
        src = _inspect.getsource(_pe)
        assert 'str(fill_engine_id or "") == "DAY_V2"' in src


# ---------------------------------------------------------------------------
# 17: signal timestamp persists non-zero
# ---------------------------------------------------------------------------


class TestSignalTimestamp:
    def test_ts_epoch_preferred(self):
        bar = {"ts": "2026-09-27 13:30:00.000000", "ts_epoch": 1790520000.0}
        assert _signal_bar_ts(bar) == 1790520000

    def test_datetime_string_parsed(self):
        bar = {"ts": "2026-09-27 13:30:00.000000"}
        assert _signal_bar_ts(bar) == 1790515800

    def test_numeric_ts_used(self):
        assert _signal_bar_ts({"ts": 1790520000}) == 1790520000

    def test_signal_bar_ts_nonzero_from_loader_format(self):
        bars_15m, bars_1h, bars_4h = _bull_bars()
        epoch = int(time.time()) - 60 * 900
        for i, b in enumerate(bars_15m):
            b["ts_epoch"] = float(epoch + i * 900)
            b["ts"] = "2026-09-27 00:00:00.000000"
        signal = evaluate_entry_signal("BTCUSDT", bars_15m, bars_1h, bars_4h)
        if signal is not None:
            assert int(signal.signal_bar_ts) > 0


# ---------------------------------------------------------------------------
# 18: exchange rejection diagnostics persist
# ---------------------------------------------------------------------------


class TestRejectDiagnostics:
    @pytest.mark.asyncio
    async def test_reject_reason_available_with_order_params(self):
        db = _make_db()
        engine = FakeEngine(db, fill=False)
        signal = _make_signal()
        result = await submit_day_v2_direct_entry(engine, signal=signal, ask_price=85000.0, quantity=0.001)
        assert result is None
        assert "BINANCE" in engine.last_buy_reject_reason or "insufficient" in engine.last_buy_reject_reason.lower()

    def test_reject_decision_record_round_trip(self):
        from backend.services.day_v2.decision_log import record_day_decision

        db = _make_db()
        reject = "BINANCE:-2010:Account has insufficient balance for requested action."
        unmet = [
            f"exchange_code={reject}",
            f"message={reject}",
            "symbol=BTCUSDT",
            "qty=0.00100000",
            "price=85000.00000000",
            "notional=85.0000",
            "decision_id=dec-1",
            "opportunity_id=opp-1",
            "bar_ts=1790520000",
        ]
        record_day_decision(db, "BTCUSDT", f"SUBMIT_REJECTED:{reject}", cycle_ts=1790520000.0, closest="HTF_TREND_PULLBACK", unmet=unmet)
        with sqlite3.connect(db) as con:
            row = con.execute("SELECT result, unmet_json FROM day_v2_decisions").fetchone()
        assert row[0].startswith("SUBMIT_REJECTED:")
        stored = json.loads(row[1])
        assert any("exchange_code=" in u for u in stored)
        assert any("qty=" in u for u in stored)
        assert any("notional=" in u for u in stored)
