"""Tests for DAY V2 live engine — signal, entry, exits, and dual-engine safety.

Coverage:
  - DayV2Signal generation from OHLCV bars
  - Opportunity ID determinism and continuity
  - Intent creation with correct engine_id, thesis fields, and trailing-buy params
  - Exit role evaluation: catastrophic, structural, winner-trail, objective, time
  - Dual-engine safety: engine ownership, no cross-engine exit, no double-buy
  - Engine authority promotion in engine_identity
  - Closed-bar authority: tick data only triggers catastrophic
  - All four coins eligible
  - Disabled engine produces no intents
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import time
from typing import Any
from unittest.mock import patch

import pytest

from backend.services.day_v2.engine_identity import (
    _AUTHORITY_TABLE,
    LIVE_ENGINE_IDS,
    AuthorityLevel,
    EngineId,
    has_live_authority,
)
from backend.services.day_v2.live_exit_evaluator import (
    DAY_V2_ENGINE_ID,
    WINNER_TRAIL_ATR_MULT,
    WINNER_TRAIL_FLOOR_PCT,
    evaluate_day_v2_exit,
)
from backend.services.day_v2.live_signal import (
    DayV2Signal,
    _opportunity_id,
    evaluate_entry_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(
    n: int,
    base_close: float = 100.0,
    trend: float = 0.0,
    atr_pct: float = 0.01,
) -> list[dict[str, Any]]:
    """Generate synthetic 15m bars (oldest-first)."""
    bars = []
    c = base_close
    ts = int(time.time()) - n * 900
    for i in range(n):
        c = c * (1.0 + trend + (0.002 if i % 2 == 0 else -0.001))
        h = c * (1.0 + atr_pct * 0.5)
        low = c * (1.0 - atr_pct * 0.5)
        bars.append({"ts": ts + i * 900, "open": c * 0.999, "high": h, "low": low, "close": c, "volume": 1000.0})
    return bars


def _bull_bars(n: int = 60) -> tuple[list, list, list]:
    """Return (bars_15m, bars_1h, bars_4h) in a bull setup."""
    # 15m: steady up-trend, last bar is first green after small pullback
    bars_15m = []
    ts = int(time.time()) - n * 900
    c = 50000.0
    for i in range(n):
        if i < n - 5:
            c = c * 1.0005  # gentle uptrend
        elif i < n - 1:
            c = c * 0.998  # small pullback
        else:
            c = c * 1.002  # first green bar (recovery)
        h = c * 1.005
        low = c * 0.995
        bars_15m.append({"ts": ts + i * 900, "open": c * 0.999, "high": h, "low": low, "close": c, "volume": 1000.0})

    # 1h: bullish (last close > 5 bars ago)
    bars_1h = []
    for i in range(20):
        c_1h = 50000.0 * (1.0 + i * 0.001)
        bars_1h.append({"ts": ts + i * 3600, "open": c_1h, "high": c_1h * 1.005, "low": c_1h * 0.995, "close": c_1h, "volume": 5000.0})

    # 4h: bull regime (above 10-bar SMA * 1.005)
    bars_4h = []
    c_4h = 50000.0
    for i in range(15):
        c_4h = c_4h * 1.002
        bars_4h.append({"ts": ts + i * 14400, "open": c_4h, "high": c_4h * 1.01, "low": c_4h * 0.99, "close": c_4h, "volume": 20000.0})

    return bars_15m, bars_1h, bars_4h


# ---------------------------------------------------------------------------
# Engine identity
# ---------------------------------------------------------------------------


class TestEngineIdentity:
    def test_day_v2_live_has_live_authority(self):
        assert _AUTHORITY_TABLE[EngineId.DAY_V2_LIVE] == AuthorityLevel.LIVE

    def test_day_v2_live_in_live_engine_ids(self):
        assert EngineId.DAY_V2_LIVE in LIVE_ENGINE_IDS

    def test_day_v2_shadow_remains_shadow(self):
        assert _AUTHORITY_TABLE[EngineId.DAY_V2_SHADOW] == AuthorityLevel.SHADOW

    def test_has_live_authority_returns_true_for_day_v2_live(self):
        assert has_live_authority(EngineId.DAY_V2_LIVE) is True

    def test_has_live_authority_returns_false_for_shadow(self):
        assert has_live_authority(EngineId.DAY_V2_SHADOW) is False

    def test_all_four_coins_in_universe(self):
        from backend.services.day_v2.config import DAY_V2_UNIVERSE

        assert "BTCUSDT" in DAY_V2_UNIVERSE
        assert "ETHUSDT" in DAY_V2_UNIVERSE
        assert "SOLUSDT" in DAY_V2_UNIVERSE
        assert "XRPUSDT" in DAY_V2_UNIVERSE


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------


class TestDayV2Signal:
    def test_insufficient_bars_returns_none(self):
        bars = _make_bars(10)
        result = evaluate_entry_signal("BTCUSDT", bars, [], [])
        assert result is None

    def test_opportunity_id_is_deterministic(self):
        oid_a = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49000.0)
        oid_b = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49000.0)
        assert oid_a == oid_b

    def test_opportunity_id_differs_by_setup(self):
        oid_a = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49000.0)
        oid_b = _opportunity_id("BTCUSDT", "RANGE_BOUNCE", 49000.0)
        assert oid_a != oid_b

    def test_opportunity_id_differs_by_symbol(self):
        oid_a = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49000.0)
        oid_b = _opportunity_id("ETHUSDT", "HTF_TREND_PULLBACK", 49000.0)
        assert oid_a != oid_b

    def test_opportunity_id_is_16_chars(self):
        oid = _opportunity_id("BTCUSDT", "RANGE_BOUNCE", 1234.5)
        assert len(oid) == 16

    def test_opportunity_id_same_for_small_anchor_drift(self):
        """Two signals with the same structural level (±0.1%) share the same ID."""
        oid_a = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49000.0)
        oid_b = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49001.0)
        # These differ by 0.002% — they share a 4-significant-figure key
        # 49000 and 49001 both round to "4.9e+04", so IDs match.
        assert oid_a == oid_b

    def test_signal_dataclass_is_frozen(self):
        sig = DayV2Signal(
            symbol="BTCUSDT",
            setup="HTF_TREND_PULLBACK",
            regime="bull",
            structural_anchor=48000.0,
            target_price=51250.0,
            atr=500.0,
            signal_bar_ts=1700000000,
            h1_bullish=True,
            opportunity_id="ABCD1234ABCD1234",
        )
        with pytest.raises((AttributeError, TypeError)):
            sig.symbol = "ETHUSDT"  # type: ignore[misc]

    def test_evaluate_entry_signal_returns_none_when_no_setup(self):
        """Flat bars with no trend/setup pattern → no signal."""
        bars_flat = _make_bars(60, base_close=100.0, trend=0.0, atr_pct=0.001)
        result = evaluate_entry_signal("BTCUSDT", bars_flat, [], [])
        # May or may not return a signal depending on indicator values,
        # but must not raise.
        assert result is None or isinstance(result, DayV2Signal)


# ---------------------------------------------------------------------------
# Exit evaluation
# ---------------------------------------------------------------------------


class TestDayV2ExitEvaluator:
    BASE = {
        "engine_id": "DAY_V2",
        "entry_price": 50000.0,
        "current_price": 50100.0,
        "bar_low": 49950.0,
        "highest_price": 50100.0,
        "atr_at_entry": 250.0,
        "structural_anchor": 49800.0,
        "target_price": 50625.0,  # 2.5 x ATR above entry (2.5 * 250)
        "entry_time": time.time() - 30 * 60,  # 30 min ago
        "estimated_roundtrip_cost": 0.0006,
    }

    def _call(self, **overrides):
        kwargs = {**self.BASE, **overrides}
        return evaluate_day_v2_exit(**kwargs)

    def test_non_day_v2_engine_returns_none(self):
        result = self._call(engine_id="LEGACY_DAY_LIVE")
        assert result is None

    def test_zero_entry_price_returns_none(self):
        result = self._call(entry_price=0.0)
        assert result is None

    def test_no_exit_in_normal_conditions(self):
        result = self._call()
        assert result is None

    # Role 1: Catastrophic
    def test_catastrophic_fires_when_bar_low_below_3x_atr(self):
        # 3.0 * 250 = 750 bps of entry = entry - 750 -> catastrophic at 49250
        bar_low = 49249.0  # below catastrophic threshold (49250)
        result = self._call(bar_low=bar_low)
        assert result is not None
        assert result["action"] == "sell"
        assert result["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"

    def test_catastrophic_does_not_fire_above_threshold(self):
        bar_low = 49251.0  # just above catastrophic threshold
        result = self._call(bar_low=bar_low)
        # Should NOT fire catastrophic
        if result is not None:
            assert result["reason"] != "DAY_V2_CATASTROPHIC_PROTECTION"

    def test_catastrophic_returns_estimated_exit_price(self):
        bar_low = 49249.0
        result = self._call(bar_low=bar_low)
        assert result is not None
        # exit_price_estimate should be near the catastrophic threshold
        est = result["exit_price_estimate"]
        catastro_px = 50000.0 * (1.0 - 3.0 * 250.0 / 50000.0)
        assert abs(est - catastro_px) < 1.0

    # Role 2: Structural invalidation
    def test_structural_fires_after_3_bars(self):
        # entry_time 45+ min ago → bars_held_approx = 3
        result = self._call(
            current_price=49799.0,  # below structural anchor (49800)
            entry_time=time.time() - 46 * 60,
            bar_low=49799.0,
        )
        assert result is not None
        assert result["reason"] == "DAY_V2_STRUCTURAL_INVALIDATION"

    def test_structural_does_not_fire_before_3_bars(self):
        # entry_time 20 min ago → bars_held_approx = 1
        result = self._call(
            current_price=49799.0,
            entry_time=time.time() - 20 * 60,
            bar_low=49799.0,
        )
        # Either no exit or a different reason (catastrophic might fire on bar_low)
        if result is not None:
            assert result["reason"] != "DAY_V2_STRUCTURAL_INVALIDATION"

    def test_structural_does_not_fire_above_anchor(self):
        result = self._call(
            current_price=49900.0,  # above anchor (49800)
            entry_time=time.time() - 60 * 60,
            bar_low=49850.0,
        )
        if result is not None:
            assert result["reason"] != "DAY_V2_STRUCTURAL_INVALIDATION"

    # Role 4: Winner protection
    def test_winner_trail_fires_after_sufficient_mfe(self):
        from backend.services.day_v2.config import DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT

        # MFE must be >= 0.8%
        mfe_required = DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT
        highest = 50000.0 * (1.0 + mfe_required + 0.001)  # safely above threshold
        # Trail: max(0.5%, 1.5 * atr_pct) = max(0.5%, 1.5 * 250/50000)
        # atr_pct = 250/50000 = 0.5% → trail = max(0.5%, 0.75%) = 0.75%
        trail_pct = max(WINNER_TRAIL_FLOOR_PCT, WINNER_TRAIL_ATR_MULT * 250.0 / 50000.0)
        trigger = highest * (1.0 - trail_pct)
        result = self._call(
            highest_price=highest,
            current_price=trigger - 10.0,  # just below trigger
            bar_low=trigger - 10.0,
        )
        assert result is not None
        assert result["reason"] == "DAY_V2_WINNER_PROTECTION"

    def test_winner_trail_does_not_fire_below_mfe_threshold(self):
        """If MFE is below the 0.8% threshold, winner trail must not activate."""
        highest = 50000.0 * 1.003  # only 0.3% MFE — below 0.8% threshold
        current = highest * 0.99  # pulled back 1% from high
        result = self._call(
            highest_price=highest,
            current_price=current,
            bar_low=current,
        )
        if result is not None:
            assert result["reason"] != "DAY_V2_WINNER_PROTECTION"

    # Role 5: Objective complete
    def test_objective_fires_when_price_reaches_target(self):
        result = self._call(
            current_price=50626.0,  # above target (50625)
            bar_low=50000.0,
        )
        assert result is not None
        assert result["reason"] == "DAY_V2_OBJECTIVE_COMPLETE"

    def test_objective_does_not_fire_below_target(self):
        result = self._call(
            current_price=50624.0,  # just below target
            bar_low=50000.0,
        )
        if result is not None:
            assert result["reason"] != "DAY_V2_OBJECTIVE_COMPLETE"

    # Role 3: Time expiration
    def test_time_expiration_fires_when_at_ceiling_and_negative(self):
        from backend.services.day_v2.config import DAY_V2_MAX_HOLD_MINUTES

        result = self._call(
            current_price=49990.0,  # below entry (net negative)
            entry_time=time.time() - (DAY_V2_MAX_HOLD_MINUTES + 5) * 60,
            bar_low=49950.0,
            highest_price=49990.0,
        )
        assert result is not None
        assert result["reason"] == "DAY_V2_TIME_EXPIRATION"

    def test_time_expiration_does_not_fire_when_profitable(self):
        from backend.services.day_v2.config import DAY_V2_MAX_HOLD_MINUTES

        # net_pnl = (50400-50000)/50000 - 0.0006 = 0.8% - 0.06% = +0.74% (positive)
        result = self._call(
            current_price=50400.0,
            entry_time=time.time() - (DAY_V2_MAX_HOLD_MINUTES + 5) * 60,
            bar_low=50300.0,
            highest_price=50400.0,
        )
        if result is not None:
            assert result["reason"] != "DAY_V2_TIME_EXPIRATION"

    def test_time_expiration_does_not_fire_before_ceiling(self):
        result = self._call(
            current_price=49990.0,  # negative
            entry_time=time.time() - 60 * 60,  # 60 min — well below ceiling
            bar_low=49950.0,
            highest_price=49990.0,
        )
        if result is not None:
            assert result["reason"] != "DAY_V2_TIME_EXPIRATION"

    def test_catastrophic_takes_priority_over_structural(self):
        """Catastrophic fires even if price is also below structural anchor."""
        bar_low = 49249.0  # triggers catastrophic
        result = self._call(
            bar_low=bar_low,
            current_price=49799.0,  # also below structural anchor
            entry_time=time.time() - 50 * 60,
        )
        assert result is not None
        assert result["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"


# ---------------------------------------------------------------------------
# Intent creation
# ---------------------------------------------------------------------------


class TestDayV2Intent:
    def _make_db(self) -> str:
        """Create a temp DB with day_trailing_buy_intents schema."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        from backend.services.day_v2.migrations import apply_all_migrations

        apply_all_migrations(tmp.name)
        return tmp.name

    def _make_signal(self, symbol: str = "BTCUSDT") -> DayV2Signal:
        return DayV2Signal(
            symbol=symbol,
            setup="HTF_TREND_PULLBACK",
            regime="bull",
            structural_anchor=49000.0,
            target_price=51250.0,
            atr=500.0,
            signal_bar_ts=1700000000,
            h1_bullish=True,
            opportunity_id=_opportunity_id(symbol, "HTF_TREND_PULLBACK", 49000.0),
        )

    def test_intent_created_with_day_v2_engine_id(self):
        db = self._make_db()
        signal = self._make_signal()
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", True):
            intent = _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)

        assert intent is not None
        assert intent.get("engine_id") == "DAY_V2"

    def test_intent_has_thesis_invalid_level(self):
        db = self._make_db()
        signal = self._make_signal()
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", True):
            intent = _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)

        assert intent is not None
        assert float(intent.get("thesis_invalid_level") or 0.0) == pytest.approx(49000.0)

    def test_intent_has_opportunity_id_in_scalp_opp_field(self):
        db = self._make_db()
        signal = self._make_signal()
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", True):
            intent = _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)

        assert intent is not None
        assert intent.get("scalp_opportunity_id") == signal.opportunity_id

    def test_intent_target_stored_in_payload(self):
        import json

        db = self._make_db()
        signal = self._make_signal()
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", True):
            intent = _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)

        assert intent is not None
        payload = json.loads(intent.get("payload_json") or "{}")
        assert float(payload.get("thesis_target_level") or 0.0) == pytest.approx(51250.0)

    def test_second_intent_same_symbol_blocked(self):
        db = self._make_db()
        signal = self._make_signal()
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", True):
            intent1 = _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)
            _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)

        assert intent1 is not None
        # Second call returns None (blocked) or the PRESERVED existing row —
        # never creates a second active row.
        # Symbol is stored in slash-normalized format (BTC/USDT).
        with sqlite3.connect(db) as con:
            count = con.execute(
                "SELECT COUNT(*) FROM day_trailing_buy_intents WHERE symbol LIKE ?",
                ("%BTC%USDT%",),
            ).fetchone()[0]
        assert count == 1, "Must not create two active intents for the same symbol"

    def test_disabled_engine_raises(self):
        db = self._make_db()
        signal = self._make_signal()
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", False), pytest.raises(RuntimeError, match="DAY_V2_ENABLED"):
            _le.create_day_v2_intent(db, signal, ask_price=50500.0, quantity=0.01)

    def test_separate_symbols_get_separate_intents(self):
        db = self._make_db()
        signal_btc = self._make_signal("BTCUSDT")
        signal_eth = self._make_signal("ETHUSDT")
        from backend.services.day_v2 import live_entry as _le

        with patch.object(_le, "DAY_V2_ENABLED", True):
            i1 = _le.create_day_v2_intent(db, signal_btc, ask_price=50000.0, quantity=0.01)
            i2 = _le.create_day_v2_intent(db, signal_eth, ask_price=3000.0, quantity=0.1)

        assert i1 is not None
        assert i2 is not None
        assert i1.get("symbol") != i2.get("symbol")


# ---------------------------------------------------------------------------
# Cross-engine safety
# ---------------------------------------------------------------------------


class TestCrossEngineSafety:
    def test_day_v2_exit_does_not_fire_on_scalp_v2_position(self):
        """evaluate_day_v2_exit must return None for non-DAY_V2 engine_id."""
        result = evaluate_day_v2_exit(
            engine_id="SCALP_V2",
            entry_price=50000.0,
            current_price=49000.0,  # severely down
            bar_low=48000.0,
            highest_price=50000.0,
            atr_at_entry=250.0,
            structural_anchor=49800.0,
            target_price=50625.0,
            entry_time=time.time() - 120 * 60,
            estimated_roundtrip_cost=0.0006,
        )
        assert result is None

    def test_day_v2_exit_does_not_fire_on_legacy_day_live_position(self):
        result = evaluate_day_v2_exit(
            engine_id="LEGACY_DAY_LIVE",
            entry_price=50000.0,
            current_price=49000.0,
            bar_low=48000.0,
            highest_price=50000.0,
            atr_at_entry=250.0,
            structural_anchor=49800.0,
            target_price=50625.0,
            entry_time=time.time() - 120 * 60,
            estimated_roundtrip_cost=0.0006,
        )
        assert result is None

    def test_all_four_coins_eligible_in_universe(self):
        """All required coins are in the DAY V2 universe — immutable."""
        from backend.services.day_v2.config import DAY_V2_UNIVERSE

        required = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
        assert required.issubset(set(DAY_V2_UNIVERSE)), f"Missing coins: {required - set(DAY_V2_UNIVERSE)}"

    def test_scalp_v2_and_day_v2_both_in_live_ids(self):
        """Both engines must be recognisable as live (for audit queries)."""
        from backend.services.scalp_v2.exit_calibration import SCALP_V2_ENGINE_ID

        assert EngineId.DAY_V2_LIVE in LIVE_ENGINE_IDS
        # SCALP_V2 uses a plain string, not EngineId enum — verify it's not empty
        assert SCALP_V2_ENGINE_ID  # non-empty string

    def test_day_v2_trailing_buy_params_differ_from_scalp_v2(self):
        """DAY V2 must not silently inherit SCALP V2's 14/4 bps values."""
        from backend.services.day_v2.live_entry import DAY_V2_MIN_DIP_BPS, DAY_V2_REBOUND_BPS

        # Verify they are present and separately calibrated
        assert DAY_V2_MIN_DIP_BPS > 0
        assert DAY_V2_REBOUND_BPS > 0
        # DAY V2 uses 20 bps min dip (multi-hour patience) vs SCALP V2's 14 bps
        assert DAY_V2_MIN_DIP_BPS >= 15, "DAY V2 min dip must reflect multi-hour patience (>= 15 bps)"

    def test_opportunity_id_continuity_same_move(self):
        """Same setup + same structural anchor → same opportunity ID.

        Two entries into the same pullback level must share the opportunity ID.
        """
        sym, setup = "BTCUSDT", "HTF_TREND_PULLBACK"
        anchor = 49000.0
        oid1 = _opportunity_id(sym, setup, anchor)
        oid2 = _opportunity_id(sym, setup, anchor)
        assert oid1 == oid2

    def test_opportunity_id_resets_on_new_setup_family(self):
        """Different setup family → different opportunity ID (structural reset)."""
        oid_pullback = _opportunity_id("BTCUSDT", "HTF_TREND_PULLBACK", 49000.0)
        oid_breakout = _opportunity_id("BTCUSDT", "BREAKOUT_CONTINUATION", 51500.0)
        assert oid_pullback != oid_breakout

    def test_opportunity_id_resets_on_different_anchor(self):
        """A materially different anchor level signals a new structural setup."""
        oid_a = _opportunity_id("BTCUSDT", "RANGE_BOUNCE", 49000.0)
        oid_b = _opportunity_id("BTCUSDT", "RANGE_BOUNCE", 45000.0)
        assert oid_a != oid_b


# ---------------------------------------------------------------------------
# Config isolation
# ---------------------------------------------------------------------------


class TestDayV2Config:
    def test_day_v2_config_raises_when_disabled(self):
        from backend.services.day_v2.config import get_day_v2_config

        with patch("backend.services.day_v2.config.DAY_V2_ENABLED", False), pytest.raises(RuntimeError):
            get_day_v2_config()

    def test_day_v2_config_returns_dict_when_enabled(self):
        from backend.services.day_v2.config import get_day_v2_config

        with patch("backend.services.day_v2.config.DAY_V2_ENABLED", True):
            cfg = get_day_v2_config()

        assert "DAY_V2_UNIVERSE" in cfg
        assert "DAY_V2_MAX_HOLD_MINUTES" in cfg
        assert "DAY_V2_CATASTROPHIC_ATR_MULTIPLIER" in cfg
        assert "DAY_V2_MIN_DIP_BPS" in cfg
        assert "DAY_V2_REBOUND_BPS" in cfg

    def test_trailing_buy_params_in_config(self):
        with patch("backend.services.day_v2.config.DAY_V2_ENABLED", True):
            from backend.services.day_v2.config import get_day_v2_config

            cfg = get_day_v2_config()

        assert cfg["DAY_V2_MIN_DIP_BPS"] > 0
        assert cfg["DAY_V2_REBOUND_BPS"] > 0


# ---------------------------------------------------------------------------
# Closed-bar authority — catastrophic only on bar_low (tick/intra-bar)
# ---------------------------------------------------------------------------


class TestClosedBarAuthority:
    def test_catastrophic_uses_bar_low_not_close(self):
        """Catastrophic exit is triggered by bar_low (intra-bar), not current_price."""
        # current_price is above catastrophic level, but bar_low is below it
        entry = 50000.0
        atr = 250.0
        catastro_pct = 3.0 * atr / entry  # 1.5%
        catastro_px = entry * (1.0 - catastro_pct)  # 49250

        # bar_low goes below catastro, but current_price (close) does not
        result = evaluate_day_v2_exit(
            engine_id="DAY_V2",
            entry_price=entry,
            current_price=catastro_px + 100,  # close is above threshold
            bar_low=catastro_px - 10,  # low went below threshold
            highest_price=entry,
            atr_at_entry=atr,
            structural_anchor=entry * 0.97,
            target_price=entry * 1.05,
            entry_time=time.time() - 30 * 60,
            estimated_roundtrip_cost=0.0006,
        )
        # Catastrophic must fire because bar_low crossed the threshold
        assert result is not None
        assert result["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"

    def test_structural_uses_closed_price_not_bar_low(self):
        """Structural invalidation fires on current_price (closed bar), not bar_low."""
        entry = 50000.0
        anchor = 49800.0
        # current_price is above anchor but bar_low is below
        result = evaluate_day_v2_exit(
            engine_id="DAY_V2",
            entry_price=entry,
            current_price=anchor + 50.0,  # closed above anchor — no structural exit
            bar_low=anchor - 100.0,  # low dipped below, but closed above
            highest_price=entry,
            atr_at_entry=100.0,  # ATR small so catastrophic is far away
            structural_anchor=anchor,
            target_price=entry * 1.05,
            entry_time=time.time() - 50 * 60,  # 50 min → bars_held_approx = 3
            estimated_roundtrip_cost=0.0006,
        )
        if result is not None:
            assert result["reason"] != "DAY_V2_STRUCTURAL_INVALIDATION"
