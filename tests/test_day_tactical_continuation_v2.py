"""Semantic and regression tests for DAY_V2 tactical continuation & population isolation."""

from __future__ import annotations

import pytest

from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services import adaptive_learning as al
from backend.services import continuation_surface as cs
from backend.services.continuation_checkpoint import recover_continuation, write_checkpoint
from backend.services.day_v2.day_policy_decision import decide_day_continuation
from tests.day_entry_value_support import install_day_surface

T0 = 1_790_000_000.0


def test_day_hold_adv_population_isolation_prevents_market_fallback_contamination(tmp_path):
    """Unconditional *__MARKET continuation losses do not contaminate DETECTOR_FIRED posteriors."""
    db = str(tmp_path / "pop_iso.db")
    install_day_surface(db, how="tactical")
    cost = canonical_roundtrip_cost_pct()
    f0 = cs.state_features(entry=100.0, mark=100.0, net=-cost, mfe=0.0, mae=0.0, high_water=100.0, prev_net=None, age_sec=0.0)

    # Seed genuine detector-fired setup on BTCUSDT
    for h in cs.DAY_ADVANTAGE_HORIZONS:
        cs.record_advantage(
            db,
            engine="DAY_V2",
            symbol="BTCUSDT",
            setup="VWAP_REVERSION",
            regime="btcup_vollo",
            horizon=h,
            features=f0,
            advantage=0.0018,
            weight=3.0,
            now=T0,
        )

    before_det = al.estimate(db, "DAY_V2", "ETHUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", "hold_adv_1800", now=T0 + 20.0, include_engine=True)
    assert float(before_det["mean"]) > 0.0

    # Flood *__MARKET fallback setups with massive negative continuation drift (33x larger weight)
    for idx, stp in enumerate(("VWAP_REVERSION__MARKET", "RANGE_BOUNCE__MARKET", "BREAKOUT_CONTINUATION__MARKET")):
        for h in cs.DAY_ADVANTAGE_HORIZONS:
            cs.record_advantage(
                db,
                engine="DAY_V2",
                symbol="ETHUSDT",
                setup=stp,
                regime="btcup_vollo",
                horizon=h,
                features=f0,
                advantage=-0.0065,
                weight=80.0,
                now=T0 + 10.0 + idx,
            )

    after_det = al.estimate(db, "DAY_V2", "ETHUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", "hold_adv_1800", now=T0 + 20.0, include_engine=True)
    mkt_view = al.estimate(db, "DAY_V2", "ETHUSDT", "BREAKOUT_CONTINUATION__MARKET", "btcup_vollo", "hold_adv_1800", now=T0 + 20.0, include_engine=True)

    # Detector-fired setup engine prior is completely shielded from *__MARKET flood
    assert float(after_det["mean"]) == pytest.approx(float(before_det["mean"]), abs=1e-9)
    assert float(mkt_view["mean"]) < -0.0040


def test_tactical_continuation_eliminates_90s_choke_on_detector_setups_while_guarding_market_fallback(tmp_path):
    """Genuine detector setup with +15m/+30m/+1h intraday edge and -4h macro drift holds at 90s under tactical."""
    db = str(tmp_path / "tactical_90s.db")
    install_day_surface(db, how="tactical")
    cost = canonical_roundtrip_cost_pct()
    f0 = cs.state_features(entry=60000.0, mark=60000.0, net=-cost, mfe=0.0, mae=0.0, high_water=60000.0, prev_net=None, age_sec=0.0)

    # Exact live production profile observed on BTCUSDT BREAKOUT_CONTINUATION btcup_vollo:
    # 15m=+6.2 bps, 30m=+12.1 bps, 1h=+1.7 bps, 2h=-1.7 bps, 4h=-13.7 bps, 6h=-15.0 bps, 12h=+9.9 bps
    live_profile = {
        900: 0.00062,
        1800: 0.00121,
        3600: 0.00017,
        7200: -0.00017,
        14400: -0.00137,
        21600: -0.00150,
        43200: 0.00099,
    }
    for h, adv in live_profile.items():
        cs.record_advantage(
            db,
            engine="DAY_V2",
            symbol="BTCUSDT",
            setup="BREAKOUT_CONTINUATION",
            regime="btcup_vollo",
            horizon=h,
            features=f0,
            advantage=adv,
            weight=25.0,
            now=T0,
        )
        cs.record_advantage(
            db,
            engine="DAY_V2",
            symbol="BTCUSDT",
            setup="BREAKOUT_CONTINUATION__MARKET",
            regime="btcup_vollo",
            horizon=h,
            features=f0,
            advantage=adv,
            weight=25.0,
            now=T0,
        )

    f_90s = cs.state_features(entry=60000.0, mark=60005.0, net=0.00008 - cost, mfe=0.00008, mae=0.0, high_water=60005.0, prev_net=-cost, age_sec=90.0)

    # Legacy blended chokes at 90s due to 2*h14400 double-counting (-27 bps)
    legacy_tr = cs.surface_trace(db, "DAY_V2", "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", f_90s, T0 + 90.0, how="blended")
    assert legacy_tr is not None and float(legacy_tr["advantage"]) < -0.0015

    # Tactical holds genuine detector-fired setup at 90s because 15m (+6.2 bps) and 30m (+12.1 bps) are strongly positive
    dec_det = decide_day_continuation(
        db,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_vollo",
        entry_price=60000.0,
        mark=60005.0,
        low=59990.0,
        high=60005.0,
        prev_net=-cost,
        age_sec=90.0,
        atr_at_entry=300.0,
        structural_anchor=59400.0,
        target_price=60900.0,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0 + 90.0,
    )
    assert dec_det["action"] == "hold"
    assert float(dec_det["advantage"]) > 0.0005

    # Unconditional *__MARKET fallback keeps conservative 4h discipline
    dec_mkt = decide_day_continuation(
        db,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="btcup_vollo",
        entry_price=60000.0,
        mark=60005.0,
        low=59990.0,
        high=60005.0,
        prev_net=-cost,
        age_sec=90.0,
        atr_at_entry=300.0,
        structural_anchor=59400.0,
        target_price=60900.0,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0 + 90.0,
    )
    assert dec_mkt["action"] == "exit"
    assert float(dec_mkt["advantage"]) < 0.0


def test_tactical_continuation_age_zero_continuity_rollover_guard_and_checkpoint_persistence(tmp_path):
    """Tactical aggregator is strictly C^0 continuous at age=0, locks in profitable rollovers, and survives recovery."""
    db = str(tmp_path / "tact_cont.db")
    install_day_surface(db, how="blended")
    cost = canonical_roundtrip_cost_pct()
    f0 = cs.state_features(entry=1650.0, mark=1650.0, net=-cost, mfe=0.0, mae=0.0, high_water=1650.0, prev_net=None, age_sec=0.0)

    for h, adv in ((900, 0.0009), (1800, 0.0014), (3600, 0.0018), (7200, 0.0011), (14400, -0.0012), (21600, -0.0018), (43200, -0.0025)):
        cs.record_advantage(
            db,
            engine="DAY_V2",
            symbol="ETHUSDT",
            setup="VWAP_REVERSION",
            regime="btcup_vollo",
            horizon=h,
            features=f0,
            advantage=adv,
            weight=20.0,
            now=T0,
        )

    write_checkpoint(db)
    assert cs.set_installed_aggregator(db, "DAY_V2", "tactical") is True
    assert cs.installed_aggregator(db, "DAY_V2") == "tactical"
    assert recover_continuation(db)["status"] == "ok"
    assert cs.installed_aggregator(db, "DAY_V2") == "tactical"

    def _eval_at(age_s: float) -> dict:
        return decide_day_continuation(
            db,
            symbol="ETHUSDT",
            setup="VWAP_REVERSION",
            regime="btcup_vollo",
            entry_price=1650.0,
            mark=1650.0,
            low=1650.0,
            high=1650.0,
            prev_net=None,
            age_sec=age_s,
            atr_at_entry=20.0,
            structural_anchor=1625.0,
            target_price=1690.0,
            entry_time=T0,
            roundtrip_cost=cost,
            now=T0 + age_s,
        )

    c0 = _eval_at(0.0)
    ceps = _eval_at(1e-4)
    c1s = _eval_at(1.0)
    c30s = _eval_at(30.0)
    entry_view = al.day_decision(db, "ETHUSDT", "VWAP_REVERSION", "btcup_vollo", now=T0)

    assert c0["terminal"] == pytest.approx(entry_view["continuation_terminal_net"], abs=1e-12)
    assert abs(float(ceps["terminal"]) - float(c0["terminal"])) < 1e-6
    assert abs(float(c1s["terminal"]) - float(c0["terminal"])) < 1e-5
    assert abs(float(c30s["terminal"]) - float(c0["terminal"])) < 2e-4

    # Profitable rollover lock-in: mfe >= 20 bps, giveback >= 35% * mfe, net > 0, slope <= 0 -> exits cleanly
    rolled = decide_day_continuation(
        db,
        symbol="ETHUSDT",
        setup="VWAP_REVERSION",
        regime="btcup_vollo",
        entry_price=1650.0,
        mark=1653.0,
        low=1649.5,
        high=1656.0,
        prev_net=0.0020,
        age_sec=900.0,
        atr_at_entry=20.0,
        structural_anchor=1625.0,
        target_price=1690.0,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0 + 900.0,
    )
    assert rolled["action"] == "exit"
