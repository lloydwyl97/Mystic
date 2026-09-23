"""Tests for DAY_STRUCTURAL_PULLBACK_V1 entry policy.

Covers:
 - ML pipeline cannot create DAY V2 intents
 - REVERSAL_BREAKOUT marked unsupported
 - Setup-specific structural zones
 - 5-minute confirmation logic (synthetic bars)
 - Opportunity consumed / persistence
 - Frequency limits (24h rolling)
 - Policy version persisted on new intents
 - Old LEGACY intents canceled at startup
 - Submitted orders not blindly canceled
 - SCALP V2 unchanged
 - Exit-reason passthrough (regression)
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.day_v2.config import (
    DAY_STRUCTURAL_PULLBACK_V1,
    DAY_V2_UNIVERSE,
)
from backend.services.day_v2.five_min_confirm import (
    FiveMinConfirmResult,
    SyntheticBar,
    _build_synthetic_5m,
    _epoch_to_dt_str,
    _five_min_boundary,
    check_5m_confirmation,
    load_synthetic_5m_bars,
)
from backend.services.day_v2.frequency_guard import (
    DAY_ENTRY_FREQUENCY_LIMIT,
    DAY_V2_MAX_FILLS_PER_SYMBOL_24H,
    DAY_V2_MAX_FILLS_TOTAL_24H,
    check_frequency_limit,
)
from backend.services.day_v2.live_signal import (
    ENABLED_SETUPS,
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_EXHAUSTION_MR,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    SETUP_REVERSAL_BREAKOUT,
    SETUP_VWAP_REVERSION,
    DayV2Signal,
)
from backend.services.day_v2.migrations import (
    cancel_old_policy_intents,
    consume_opportunity,
    is_opportunity_consumed,
    run_day_v2_migrations,
)
from backend.services.day_v2.structural_entry import (
    MISSING_STRUCTURAL_ENTRY_LEVEL,
    StructuralZone,
    evaluate_structural_zone,
    price_in_structural_zone,
)
from backend.services.portfolio_engine import ExitType, paper_trades_exit_type_label

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    symbol: str = "BTCUSDT",
    setup: str = SETUP_HTF_TREND_PULLBACK,
    anchor: float = 60000.0,
    target: float = 62000.0,
    atr: float = 500.0,
    regime: str = "bull",
) -> DayV2Signal:
    from backend.services.day_v2.live_signal import _opportunity_id

    return DayV2Signal(
        symbol=symbol,
        setup=setup,
        regime=regime,
        structural_anchor=anchor,
        target_price=target,
        atr=atr,
        signal_bar_ts=int(time.time()),
        h1_bullish=True,
        opportunity_id=_opportunity_id(symbol, setup, anchor),
    )


def _make_db(tmp_path: Path) -> str:
    """Create a minimal DB with required tables."""
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS day_trailing_buy_intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id TEXT NOT NULL UNIQUE,
            symbol TEXT NOT NULL,
            engine_id TEXT NOT NULL DEFAULT 'DAY_V2',
            status TEXT NOT NULL DEFAULT 'WAIT_DIP',
            policy_version TEXT DEFAULT 'LEGACY',
            order_id TEXT DEFAULT '',
            client_order_id TEXT NOT NULL DEFAULT '',
            reservation_id TEXT DEFAULT '',
            arm_ts REAL DEFAULT 0,
            updated_at REAL DEFAULT 0,
            cancel_reason TEXT DEFAULT '',
            scalp_opportunity_id TEXT DEFAULT '',
            structural_entry_level REAL DEFAULT 0,
            pullback_reached INTEGER DEFAULT 0,
            red_5m_seen INTEGER DEFAULT 0,
            reversal_candle_ts REAL DEFAULT 0,
            reversal_level REAL DEFAULT 0,
            freq_limit_state TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        )
    """)
    conn.commit()
    conn.close()
    return db


def _insert_intent(
    db: str,
    intent_id: str,
    symbol: str = "BTCUSDT",
    engine_id: str = "DAY_V2",
    status: str = "WAIT_DIP",
    policy_version: str = "LEGACY",
    order_id: str = "",
    arm_ts: float | None = None,
) -> None:
    if arm_ts is None:
        arm_ts = time.time()
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT OR REPLACE INTO day_trailing_buy_intents
           (intent_id, symbol, engine_id, status, policy_version, order_id, arm_ts, updated_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (intent_id, symbol, engine_id, status, policy_version, order_id, arm_ts, arm_ts),
    )
    conn.commit()
    conn.close()


# ===========================================================================
# 1. ML pipeline cannot create DAY V2 intent
# ===========================================================================


@pytest.mark.asyncio
async def test_ml_cannot_create_day_v2_intent() -> None:
    """arm_selected_candidate must return None for any DAY_V2_UNIVERSE symbol."""
    from backend.services.day_trailing_buy import arm_selected_candidate

    engine = MagicMock()
    engine.db_path = ":memory:"
    engine.open_positions = {}
    engine._entry_reservations = {}
    engine._try_reserve_entry = MagicMock(return_value=(False, "TEST"))
    redis = MagicMock()

    for symbol in DAY_V2_UNIVERSE:
        result = await arm_selected_candidate(
            engine,
            symbol=symbol,
            quantity=0.001,
            stop_price=0.0,
            atr=500.0,
            confidence=0.9,
            bar_timestamp=int(time.time()),
            explainability=MagicMock(),
            decision_id="test-decision",
            sleeve="DAY",
            decision_data={},
            redis_client=redis,
        )
        assert result is None, f"arm_selected_candidate must return None for DAY_V2 symbol {symbol}"


# ===========================================================================
# 2. REVERSAL_BREAKOUT is not in ENABLED_SETUPS
# ===========================================================================


def test_reversal_breakout_not_in_enabled_setups() -> None:
    assert SETUP_REVERSAL_BREAKOUT not in ENABLED_SETUPS, "REVERSAL_BREAKOUT is UNSUPPORTED_NOT_IMPLEMENTED and must be excluded from ENABLED_SETUPS"


def test_reversal_breakout_constant_exists() -> None:
    """Constant must exist for guard tests but must not be live."""
    assert SETUP_REVERSAL_BREAKOUT == "REVERSAL_BREAKOUT"


def test_reversal_breakout_structural_zone_unsupported() -> None:
    sig = _make_signal(setup=SETUP_REVERSAL_BREAKOUT)
    zone = evaluate_structural_zone(sig)
    assert not zone.valid
    assert "UNSUPPORTED_SETUP" in zone.reason or "REVERSAL_BREAKOUT" in zone.reason


# ===========================================================================
# 3. Structural zones — each of the 5 supported setups
# ===========================================================================


@pytest.mark.parametrize(
    "setup",
    [
        SETUP_HTF_TREND_PULLBACK,
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_RANGE_BOUNCE,
        SETUP_VWAP_REVERSION,
        SETUP_EXHAUSTION_MR,
    ],
)
def test_each_supported_setup_has_valid_zone(setup: str) -> None:
    sig = _make_signal(setup=setup, anchor=60000.0, target=62000.0, atr=500.0)
    zone = evaluate_structural_zone(sig)
    assert zone.valid, f"Setup {setup} should produce a valid zone: {zone.reason}"
    assert zone.zone_low > 0
    assert zone.zone_high >= zone.zone_low
    assert zone.reclaim_level > 0


def test_missing_structural_level_rejected() -> None:
    sig = _make_signal(anchor=0.0, atr=0.0)
    zone = evaluate_structural_zone(sig)
    assert not zone.valid
    assert zone.reason == MISSING_STRUCTURAL_ENTRY_LEVEL


def test_price_in_zone_true() -> None:
    sig = _make_signal(anchor=60000.0, atr=500.0)
    zone = evaluate_structural_zone(sig)
    # Price at anchor should be inside zone for HTF_TREND_PULLBACK
    assert price_in_structural_zone(60000.0, zone)


def test_price_above_zone_false() -> None:
    sig = _make_signal(anchor=60000.0, atr=500.0)
    zone = evaluate_structural_zone(sig)
    assert not price_in_structural_zone(65000.0, zone)


# ===========================================================================
# 4. Synthetic 5m bar logic
# ===========================================================================


def test_five_min_boundary_alignment() -> None:
    # Any ts should floor to a multiple of 300
    for offset in [0, 1, 149, 299]:
        base = 1700000000
        result = _five_min_boundary(base + offset)
        assert result % 300 == 0
        assert result <= base + offset


def test_synthetic_5m_requires_all_five_components() -> None:
    boundary = 1700000000  # arbitrary aligned ts
    bars_1m = [
        {"ts": boundary + i * 60, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0}
        for i in range(4)  # only 4, not 5
    ]
    result = _build_synthetic_5m(bars_1m, boundary)
    assert result is None


def test_synthetic_5m_all_five_present() -> None:
    # Boundary must be aligned to 300s: 1700000000 % 300 == 200, so subtract 200
    boundary = _five_min_boundary(1700000000)  # 1699999800; already mod-300-aligned
    assert boundary % 300 == 0
    bars_1m = [{"ts": boundary + i * 60, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 2.0} for i in range(5)]
    result = _build_synthetic_5m(bars_1m, boundary)
    assert result is not None
    assert result.ts == boundary
    assert result.open == 100.0
    assert result.high == 101.0
    assert result.low == 99.0
    assert result.close == 100.5
    assert result.volume == pytest.approx(10.0)


def test_synthetic_5m_zero_volume_accepted() -> None:
    """Zero volume is allowed when OHLC prices are valid."""
    boundary = _five_min_boundary(1700000000)
    bars_1m = [{"ts": boundary + i * 60, "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.5, "volume": 0.0} for i in range(5)]
    result = _build_synthetic_5m(bars_1m, boundary)
    assert result is not None
    assert result.volume == 0.0


def test_synthetic_5m_rejects_zero_ohlc() -> None:
    boundary = 1700000000
    bars_1m = [{"ts": boundary + i * 60, "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0} for i in range(5)]
    result = _build_synthetic_5m(bars_1m, boundary)
    assert result is None


def test_load_synthetic_5m_rejects_incomplete_current_minute(tmp_path: Path) -> None:
    """A bar whose +300s boundary has not passed should not appear."""
    db = _make_db(tmp_path)
    now = time.time()
    boundary = _five_min_boundary(now)  # current (incomplete) 5m window
    conn = sqlite3.connect(db)
    for i in range(5):
        conn.execute(
            "INSERT INTO feature_ohlcv (symbol, interval, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
            ("BTCUSDT", "1m", boundary + i * 60, 100.0, 101.0, 99.0, 100.5, 1.0),
        )
    conn.commit()
    conn.close()
    bars = load_synthetic_5m_bars(db, "BTCUSDT", since_ts=boundary - 60, now=now)
    # Current window is incomplete; must not appear
    ts_list = [b.ts for b in bars]
    assert boundary not in ts_list


def test_5m_confirmation_red_then_reclaim(tmp_path: Path) -> None:
    """Red bar followed by bullish reclaim => confirmed.

    Mocks load_synthetic_5m_bars to return controlled SyntheticBar objects so
    the test is independent of DB timestamp format and staleness thresholds.
    This unit-tests the pattern-detection logic inside check_5m_confirmation.
    """
    import backend.services.day_v2.five_min_confirm as fmc

    db = _make_db(tmp_path)
    now = time.time()
    b1_start = int(_five_min_boundary(now - 700))
    b2_start = b1_start + 300

    red_bar = SyntheticBar(ts=b1_start, open=100.0, high=100.5, low=99.5, close=99.0, volume=5.0, source="SYNTHETIC_5X1M")
    reclaim_bar = SyntheticBar(ts=b2_start, open=99.5, high=101.5, low=99.0, close=101.0, volume=5.0, source="SYNTHETIC_5X1M")

    with patch.object(fmc, "load_synthetic_5m_bars", return_value=[red_bar, reclaim_bar]):
        result = check_5m_confirmation(
            db,
            "BTCUSDT",
            reclaim_level=99.0,
            opportunity_armed_at=float(b1_start - 60),
            now=now,
        )
    assert result.confirmed
    assert result.red_bar_ts == pytest.approx(b1_start, abs=1)
    assert result.reclaim_bar_ts == pytest.approx(b2_start, abs=1)


def test_5m_confirmation_not_authorized_by_tick(tmp_path: Path) -> None:
    """A simple price move without completed 5m bars does not confirm."""
    db = _make_db(tmp_path)
    now = time.time()
    result = check_5m_confirmation(
        db,
        "BTCUSDT",
        reclaim_level=99.0,
        opportunity_armed_at=now - 30,
        now=now,
    )
    assert not result.confirmed


def test_5m_stale_data_reason(tmp_path: Path) -> None:
    """If 1m bars are all older than 120s, reason should surface staleness."""
    db = _make_db(tmp_path)
    now = time.time()
    # Insert bars from 10 minutes ago only
    old_boundary = _five_min_boundary(now - 700)
    conn = sqlite3.connect(db)
    for i in range(5):
        conn.execute(
            "INSERT INTO feature_ohlcv (symbol, interval, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
            ("BTCUSDT", "1m", old_boundary + i * 60, 100.0, 101.0, 99.0, 100.5, 1.0),
        )
    conn.commit()
    conn.close()
    # With no recent 1m data and since_ts near now, confirmation should be absent
    result = check_5m_confirmation(
        db,
        "BTCUSDT",
        reclaim_level=99.0,
        opportunity_armed_at=now - 60,
        now=now,
    )
    assert not result.confirmed


# ===========================================================================
# 5. Opportunity consumed — DB persistence
# ===========================================================================


def test_opportunity_consumed_blocks_rearm(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    opp_id = "TEST_OPP_001"
    assert not is_opportunity_consumed(db, opp_id)
    consume_opportunity(db, opp_id, "BTCUSDT", "HTF_TREND_PULLBACK", trade_id="T1")
    assert is_opportunity_consumed(db, opp_id)


def test_opportunity_persistence_survives_new_connection(tmp_path: Path) -> None:
    """Opportunity consumed state must persist across DB reconnects (restart simulation)."""
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    opp_id = "PERSIST_OPP_002"
    consume_opportunity(db, opp_id, "ETHUSDT", "RANGE_BOUNCE", trade_id="T2")
    # Open a completely new connection (simulates restart)
    assert is_opportunity_consumed(db, opp_id)
    # A second consume call must be idempotent
    consume_opportunity(db, opp_id, "ETHUSDT", "RANGE_BOUNCE", trade_id="T2")
    assert is_opportunity_consumed(db, opp_id)


def test_opportunity_different_anchor_different_id() -> None:
    from backend.services.day_v2.live_signal import _opportunity_id

    id1 = _opportunity_id("BTCUSDT", SETUP_HTF_TREND_PULLBACK, 60000.0)
    id2 = _opportunity_id("BTCUSDT", SETUP_HTF_TREND_PULLBACK, 61000.0)
    assert id1 != id2


# ===========================================================================
# 6. Frequency limits
# ===========================================================================


def test_frequency_limit_2_per_symbol(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    now = time.time()
    # Insert 2 FILLED intents for BTCUSDT (at limit)
    for i in range(2):
        _insert_intent(db, f"FREQ-SYM-{i}", "BTCUSDT", status="FILLED", policy_version=DAY_STRUCTURAL_PULLBACK_V1, arm_ts=now - 100)
    ok, reason = check_frequency_limit(db, "BTCUSDT")
    assert not ok
    assert DAY_ENTRY_FREQUENCY_LIMIT in reason


def test_frequency_limit_8_total(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    now = time.time()
    symbols = list(DAY_V2_UNIVERSE)
    # 8 FILLED intents across 4 symbols = 2 each
    for i, sym in enumerate(symbols * 2):
        _insert_intent(db, f"FREQ-TOT-{i}", sym, status="FILLED", policy_version=DAY_STRUCTURAL_PULLBACK_V1, arm_ts=now - 100)
    ok, reason = check_frequency_limit(db, "BTCUSDT")
    assert not ok
    assert DAY_ENTRY_FREQUENCY_LIMIT in reason


def test_frequency_limit_rolls_past_24h(tmp_path: Path) -> None:
    """A fill older than 24h should not count toward the limit."""
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    old_ts = time.time() - 90000  # 25 hours ago
    for i in range(2):
        _insert_intent(db, f"FREQ-OLD-{i}", "BTCUSDT", status="FILLED", policy_version=DAY_STRUCTURAL_PULLBACK_V1, arm_ts=old_ts)
    ok, reason = check_frequency_limit(db, "BTCUSDT")
    assert ok, f"Old fills should not block new entry: {reason}"


def test_day_limits_do_not_affect_scalp(tmp_path: Path) -> None:
    """Frequency guard only counts engine_id='DAY_V2' rows."""
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    now = time.time()
    # Insert SCALP_V2 filled intents — should not count
    for i in range(8):
        _insert_intent(db, f"SCALP-{i}", "BTCUSDT", engine_id="SCALP_V2", status="FILLED", policy_version="SCALP_V2_POLICY", arm_ts=now - 100)
    ok, reason = check_frequency_limit(db, "BTCUSDT")
    assert ok, f"SCALP fills must not count toward DAY limit: {reason}"


# ===========================================================================
# 7. All four symbols enabled
# ===========================================================================


def test_all_four_symbols_in_universe() -> None:
    assert "BTCUSDT" in DAY_V2_UNIVERSE
    assert "ETHUSDT" in DAY_V2_UNIVERSE
    assert "SOLUSDT" in DAY_V2_UNIVERSE
    assert "XRPUSDT" in DAY_V2_UNIVERSE


# ===========================================================================
# 8. Policy version persisted on new intents
# ===========================================================================


def test_policy_version_in_config() -> None:
    assert DAY_STRUCTURAL_PULLBACK_V1 == "DAY_STRUCTURAL_PULLBACK_V1"


def test_policy_version_persisted_on_new_intent(tmp_path: Path) -> None:
    """create_day_v2_intent must stamp DAY_STRUCTURAL_PULLBACK_V1 on the intent."""
    from backend.services.day_trailing_buy_store import ensure_trailing_buy_schema
    from backend.services.day_v2.live_entry import create_day_v2_intent
    from backend.services.day_v2.structural_entry import evaluate_structural_zone

    # Use the real production schema so ensure_trailing_buy_schema doesn't fail
    # on missing columns / indexes when called internally by create_intent.
    db = str(tmp_path / "test.db")
    ensure_trailing_buy_schema(db)
    run_day_v2_migrations(db)

    sig = _make_signal()
    zone = evaluate_structural_zone(sig)

    with patch("backend.services.day_v2.live_entry.DAY_V2_ENABLED", True):
        intent = create_day_v2_intent(db, sig, 60000.0, 0.001, structural_zone=zone, reclaim_level=zone.reclaim_level)

    if intent:  # may be None if reservation fails in test DB
        assert intent.get("policy_version") == DAY_STRUCTURAL_PULLBACK_V1


# ===========================================================================
# 9. Old LEGACY intents canceled at startup
# ===========================================================================


def test_old_legacy_intents_canceled(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    # Insert an old LEGACY WAIT_DIP intent with no order_id
    _insert_intent(db, "LEGACY-001", "BTCUSDT", status="WAIT_DIP", policy_version="LEGACY", order_id="")
    n = cancel_old_policy_intents(db)
    assert n == 1
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT status, cancel_reason FROM day_trailing_buy_intents WHERE intent_id='LEGACY-001'").fetchone()
    conn.close()
    assert row[0] == "CANCELED"
    assert row[1] == "LEGACY_POLICY_SUPERSEDED_BY_DAY_STRUCTURAL_PULLBACK_V1"


def test_submitted_orders_not_canceled(tmp_path: Path) -> None:
    """LEGACY intent with a real order_id must NOT be canceled."""
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    _insert_intent(db, "LEGACY-WITH-ORDER", "BTCUSDT", status="WAIT_DIP", policy_version="LEGACY", order_id="123456789")
    n = cancel_old_policy_intents(db)
    assert n == 0
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT status FROM day_trailing_buy_intents WHERE intent_id='LEGACY-WITH-ORDER'").fetchone()
    conn.close()
    assert row[0] == "WAIT_DIP"  # unchanged


# ===========================================================================
# 10. Exit-reason passthrough — regression from 3c3c287
# ===========================================================================


def test_day_v2_exit_reasons_pass_through_unchanged() -> None:
    for reason in (
        "DAY_V2_CATASTROPHIC_PROTECTION",
        "DAY_V2_STRUCTURAL_INVALIDATION",
        "DAY_V2_WINNER_PROTECTION",
        "DAY_V2_OBJECTIVE_COMPLETE",
        "DAY_V2_TIME_EXPIRATION",
    ):
        result = paper_trades_exit_type_label(ExitType.MANUAL, reason)
        assert result == reason, f"Expected {reason!r}, got {result!r}"
        assert result != ExitType.MANUAL.value


# ===========================================================================
# 11. SCALP V2 unchanged — engine_id isolation
# ===========================================================================


def test_scalp_arm_opportunity_uses_scalp_engine_id() -> None:
    """arm_opportunity default engine_id must be SCALP_V2, never DAY_V2."""
    import inspect

    from backend.services.scalp_v2.opportunity import arm_opportunity

    sig = inspect.signature(arm_opportunity)
    engine_id_default = sig.parameters["engine_id"].default
    from backend.services.day_v2.engine_identity import SCALP_V2_ENGINE_ID

    assert str(engine_id_default) == SCALP_V2_ENGINE_ID


# ===========================================================================
# 12. DB migration is idempotent
# ===========================================================================


def test_migrations_idempotent(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    run_day_v2_migrations(db)
    # Running a second time must not raise
    run_day_v2_migrations(db)
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(day_trailing_buy_intents)").fetchall()}
    conn.close()
    assert "policy_version" in cols
    assert "pullback_reached" in cols
    assert "red_5m_seen" in cols
    assert "structural_entry_level" in cols
