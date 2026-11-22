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


def test_v3_rebuild_causally_aligns_pretactical_detector_counterfactuals(tmp_path):
    """Pre-tactical choked DETECTOR_FIRED counterfactuals re-evaluate causally on 1m bars without mutating policy_episodes."""
    import json
    import sqlite3
    import time

    from backend.services.day_v2.entry_policy_value_v3 import entry_policy_value_v3, rebuild_entry_policy_value_v3
    from backend.services.exact_policy_capture import CAPTURE, ensure_capture_schema

    db = str(tmp_path / "v3_tact_align.db")
    f_zero = cs.state_features(entry=100.0, mark=100.0, net=-0.00066, mfe=0.0, mae=0.0, high_water=100.0, prev_net=-0.00066, age_sec=0.0)
    for h_s, adv in ((900, 0.0008), (1800, 0.0015), (3600, 0.0012), (7200, 0.0005), (14400, -0.0025)):
        cs.record_advantage(
            db,
            engine="DAY_V2",
            symbol="BTCUSDT",
            setup="BREAKOUT_CONTINUATION",
            regime="btcup_vollo",
            horizon=h_s,
            features=f_zero,
            advantage=adv,
            weight=12.0,
            now=T0,
        )

    conn = sqlite3.connect(db)
    ensure_capture_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS continuation_learning_meta (
            engine_id TEXT NOT NULL,
            economic_version TEXT NOT NULL,
            learning_version TEXT NOT NULL,
            source TEXT NOT NULL,
            anchor_commit TEXT NOT NULL,
            anchor_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            observations INTEGER NOT NULL,
            PRIMARY KEY (engine_id, economic_version)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_ohlcv (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL DEFAULT 10.0
        )
        """
    )
    cutover_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + 20_000.0))
    conn.execute(
        "INSERT OR REPLACE INTO continuation_learning_meta VALUES (?, ?, ?, ?, '7c4e990', ?, ?, 100)",
        ("DAY_V2", al.current_economic_version("DAY_V2"), cs.ADVANTAGE_VERSION, "horizon_hold_advantage:tactical", cutover_iso, cutover_iso),
    )

    # Insert 1m bars showing a clean breakout rally (+35 bps over 10m, then mild giveback triggering tactical rollover lock-in)
    dec_t = T0 + 1800.0
    for m in range(25):
        bar_open_t = dec_t + m * 60.0
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(bar_open_t))
        px = 100.0 + min(m, 10) * 0.04 - max(0, m - 10) * 0.018
        hi = px + 0.01
        lo = px - 0.01
        conn.execute(
            "INSERT INTO feature_ohlcv (symbol, interval, ts, open, high, low, close, volume) VALUES (?, '1m', ?, ?, ?, ?, ?, 10.0)",
            ("BTCUSDT", ts_str, px, hi, lo, px),
        )

    # Pre-tactical episode was choked at 90s with -0.0009 (-9 bps)
    conn.execute(
        """
        INSERT INTO policy_episodes (
            candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask, entry_bid,
            entry_spread, roundtrip_cost, snapshot_json, status, exit_bid, exit_at, label_available_at,
            exit_reason, gross, net, planning_net, label_taker_fee, capture_version, train_eligible, funded
        ) VALUES (?, 'COUNTERFACTUAL_POLICY_OUTCOME', 'DAY_V2', 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo',
                  ?, 100.0, 99.99, 0.0001, 0.00066, ?, 'CLOSED', 99.97, ?, ?, 'LEARNED_CONTINUATION_EXIT',
                  -0.0003, -0.00096, -0.00096, 0.00033, ?, 1, 0)
        """,
        (101, dec_t, json.dumps({"features": {"sma20_dist": 0.008}}), dec_t + 90.0, dec_t + 90.0, CAPTURE),
    )
    conn.commit()
    conn.close()

    # 1. At cutoff = dec_t + 120s (before tactical trajectory closes), strictly causal: 0 episodes folded!
    early_reb = rebuild_entry_policy_value_v3(db, now=dec_t + 120.0)
    assert early_reb["applied"] == 0

    # 2. At cutoff = dec_t + 9000s (after tactical trajectory closes), V3 learns the positive tactical outcome
    late_reb = rebuild_entry_policy_value_v3(db, now=dec_t + 9000.0)
    assert late_reb["applied"] == 1
    pred = entry_policy_value_v3(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", now=dec_t + 9000.0)
    assert pred["expected_policy_net"] > 0.0

    # Original policy_episodes.net was NEVER mutated
    check_conn = sqlite3.connect(db)
    orig_net = check_conn.execute("SELECT net FROM policy_episodes WHERE candidate_id=101").fetchone()[0]
    check_conn.close()
    assert orig_net == pytest.approx(-0.00096)


def test_scalp_zero_claim_negative_market_edge_gate_and_same_setup_calibration(tmp_path):
    """Exact reproduction of live id=4156/4157: zero-claim rejected setup with market_edge <= 0 cannot buy, and negative micro drags entry value."""
    from backend.services.portfolio_engine import PortfolioEngine
    from backend.services.scalp_v2 import continuation_value as cv
    from backend.services.scalp_v2.executable_edge import scalp_executable_edge

    db = str(tmp_path / "scalp_gate.db")
    ver = al.current_strategy_version("SCALP_V2")

    # Same-setup negative policy_calibration propagates across sibling symbols/regimes without cross-setup engine deadlock
    for _ in range(8):
        al.observe(
            db,
            engine="SCALP_V2",
            symbol="ETHUSDT",
            setup="RANGE_BOUNCE_SCALP",
            regime="btcup_vollo",
            metric="policy_calibration",
            value=-0.0030,
            strategy_version=ver,
            now=T0,
        )
    cal = al.scalp_claim_calibration(db, "BTCUSDT", "RANGE_BOUNCE_SCALP", "btcup_vollo", now=T0 + 10.0)
    assert cal["key_policy_calibration"] < 0.0
    cold_other = al.scalp_claim_calibration(db, "BTCUSDT", "TREND_PULLBACK_MICRO", "btcup_vollo", now=T0 + 10.0)
    assert cold_other["key_policy_calibration"] == pytest.approx(0.0, abs=1e-12)

    # Zero-claim (strategy_passed=False, NOT_NEAR_SUPPORT) with negative market_edge (-0.24 bps) is blocked even if entry_value > cost
    view = {
        "claim_gross_setup": -0.000108,
        "claim_gross_mean": 0.000656,
        "claim_capture": 0.0,
        "micro_residual": -0.000077,
        "confidence": 0.95,
        "risk_estimate": 0.0007,
    }
    edge = scalp_executable_edge(
        view,
        raw_expected_move_pct=0.0,
        spread_pct=0.000005,
        impact_pct=0.0,
        edge_source="STRATEGY_CLAIM",
        value_of=lambda _ent: 0.000735,
    )
    assert edge.market_edge_pct < 0.0
    assert not edge.eligible

    # Winner at +12 bps net (mfe = 18 bps gross) is NOT prematurely dumped by ro_drag on a 6.5 bps wiggle
    model = cv.ValueModel()
    x0 = cv.vector(cv.entry_position_state(0.0006), {"cost": 0.0006}, "BTCUSDT", "ORDERBOOK_TAPE_SCALP", "btcup_vollo")
    model.fold([(120, x0, 0.0004, T0, 1.0)] * 30, T0)
    hold_mid = cv.value(
        "unused",
        "BTCUSDT",
        "ORDERBOOK_TAPE_SCALP",
        "btcup_vollo",
        {"net": 0.0012, "mfe": 0.0018, "mae": 0.0002, "giveback": 0.00065, "age_min": 2.5},
        {"calibrated": 0.0008, "adaptive": 0.0002, "cost": 0.0006},
        model=model,
    )
    assert hold_mid["advantage"] > 0.0

    # Default constraints include both slash and non-slash forms
    pe = PortfolioEngine.__new__(PortfolioEngine)
    defaults = pe._get_default_symbol_constraints()
    assert defaults["BTC/USDT"]["qty_step"] == pytest.approx(0.00001)
    assert defaults["BTCUSDT"]["qty_step"] == pytest.approx(0.00001)


def test_v3_pretactical_reconstruction_blocks_future_hindsight_state(tmp_path):
    """Future continuation_advantage_log labels (label_time > bar_close_ts) must never leak into earlier episode reconstruction, and funded episodes remain immutable."""
    import json
    import sqlite3 as _sq
    import time

    from backend.services import adaptive_learning as al
    from backend.services.continuation_surface import ADVANTAGE_VERSION
    from backend.services.day_v2.entry_policy_value_v3 import rebuild_entry_policy_value_v3
    from backend.services.exact_policy_capture import CAPTURE, ensure_capture_schema

    def _build_base_db(db_file: str, *, inject_future_poison: bool) -> float:
        dec_t = T0 + 3600.0
        conn = _sq.connect(db_file)
        ensure_capture_schema(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_learning_meta (
                engine_id TEXT NOT NULL,
                economic_version TEXT NOT NULL,
                learning_version TEXT NOT NULL,
                source TEXT NOT NULL,
                code_sha TEXT NOT NULL,
                installed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                label_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (engine_id, economic_version)
            )
            """
        )
        cutover_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dec_t + 80000.0))
        conn.execute(
            "INSERT OR REPLACE INTO continuation_learning_meta VALUES (?, ?, ?, ?, '7c4e990', ?, ?, 100)",
            ("DAY_V2", al.current_economic_version("DAY_V2"), ADVANTAGE_VERSION, "horizon_hold_advantage:tactical", cutover_iso, cutover_iso),
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_advantage_log (
                engine_id TEXT, trade_id TEXT, snapshot_time REAL, horizon INTEGER,
                symbol TEXT, setup TEXT, regime TEXT, features_json TEXT,
                advantage REAL, weight REAL, label_time REAL, feature_version TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_ohlcv (
                symbol TEXT NOT NULL, interval TEXT NOT NULL, ts TEXT NOT NULL,
                open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER UNIQUE, kind TEXT, engine_id TEXT, symbol TEXT,
                setup TEXT, regime TEXT, decided_at REAL, entry_ask REAL, entry_bid REAL,
                entry_spread REAL, roundtrip_cost REAL, snapshot_json TEXT, status TEXT,
                exit_bid REAL, exit_at REAL, label_available_at REAL, exit_reason TEXT,
                gross REAL, net REAL, planning_net REAL, label_taker_fee REAL,
                capture_version TEXT, train_eligible INTEGER DEFAULT 1, funded INTEGER DEFAULT 0,
                entry_policy_learned INTEGER DEFAULT 0, entry_policy_v2_learned INTEGER DEFAULT 0,
                entry_policy_v3_learned INTEGER DEFAULT 0
            )
            """
        )
        # Causal past continuation labels (label_time <= dec_t): positive tactical hold advantage (+8..+15 bps)
        f_e0 = {"net": -0.00066, "mfe": 0.0, "mae": 0.0, "giveback": 0.0, "dist_high": 0.0, "slope": 0.0, "age_min": 0.0}
        f_p1 = {"net": 0.0008, "mfe": 0.0014, "mae": 0.0001, "giveback": 0.0, "dist_high": 0.0, "slope": 0.0003, "age_min": 3.0}
        f_p2 = {"net": -0.0009, "mfe": 0.0002, "mae": 0.0012, "giveback": 0.0002, "dist_high": 0.0010, "slope": -0.0003, "age_min": 12.0}
        for h, adv in ((900, 0.0008), (1800, 0.0015), (3600, 0.0012), (7200, 0.0005), (14400, -0.0025)):
            conn.execute(
                """
                INSERT INTO continuation_advantage_log
                (engine_id, trade_id, snapshot_time, horizon, symbol, setup, regime, features_json, advantage, weight, label_time, feature_version)
                VALUES ('DAY_V2', 'past_e0', ?, ?, 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo', ?, ?, 12.0, ?, ?)
                """,
                (T0, h, json.dumps(f_e0), adv, T0 + 1800.0, ADVANTAGE_VERSION),
            )
            conn.execute(
                """
                INSERT INTO continuation_advantage_log
                (engine_id, trade_id, snapshot_time, horizon, symbol, setup, regime, features_json, advantage, weight, label_time, feature_version)
                VALUES ('DAY_V2', 'past_p1', ?, ?, 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo', ?, ?, 4.0, ?, ?)
                """,
                (T0 + 60.0, h, json.dumps(f_p1), adv, T0 + 1860.0, ADVANTAGE_VERSION),
            )
            conn.execute(
                """
                INSERT INTO continuation_advantage_log
                (engine_id, trade_id, snapshot_time, horizon, symbol, setup, regime, features_json, advantage, weight, label_time, feature_version)
                VALUES ('DAY_V2', 'past_p2', ?, ?, 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo', ?, -0.0005, 4.0, ?, ?)
                """,
                (T0 + 120.0, h, json.dumps(f_p2), T0 + 1920.0, ADVANTAGE_VERSION),
            )
        if inject_future_poison:
            # Future hindsight continuation labels (label_time = dec_t + 50000s, AFTER trajectory finished): massive -500 bps
            for h in (900, 1800, 3600, 7200):
                conn.execute(
                    """
                    INSERT INTO continuation_advantage_log
                    (engine_id, trade_id, snapshot_time, horizon, symbol, setup, regime, features_json, advantage, weight, label_time, feature_version)
                    VALUES ('DAY_V2', 'future_poison', ?, ?, 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo', ?, -0.0500, 200.0, ?, ?)
                    """,
                    (dec_t + 40000.0, h, json.dumps(f_e0), dec_t + 50000.0, ADVANTAGE_VERSION),
                )
        # 1m bars rising +50 bps over 10m, then rolling over so winner rollover guard locks profit around m=17 (+25 bps net)
        for m in range(30):
            bar_open_t = dec_t + m * 60.0
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(bar_open_t))
            px = 100.0 + min(m, 10) * 0.05 - max(0, m - 10) * 0.025
            conn.execute(
                "INSERT INTO feature_ohlcv (symbol, interval, ts, open, high, low, close, volume) VALUES (?, '1m', ?, ?, ?, ?, ?, 10.0)",
                ("BTCUSDT", ts_str, px, px + 0.01, px - 0.01, px),
            )
        # Update cutover timestamp in continuation_learning_meta so dec_t < tactical_cutover_ts
        conn.execute(
            "UPDATE continuation_learning_meta SET updated_at=? WHERE engine_id='DAY_V2'",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dec_t + 80000.0)),),
        )
        # Unfunded pre-cutover DETECTOR_FIRED episode
        conn.execute(
            """
            INSERT INTO policy_episodes (
                candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask, entry_bid,
                entry_spread, roundtrip_cost, snapshot_json, status, exit_bid, exit_at, label_available_at,
                exit_reason, gross, net, planning_net, label_taker_fee, capture_version, train_eligible, funded
            ) VALUES (201, 'COUNTERFACTUAL_POLICY_OUTCOME', 'DAY_V2', 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo',
                      ?, 100.0, 99.99, 0.0001, 0.00066, '{}', 'CLOSED', 99.95, ?, ?, 'LEARNED_CONTINUATION_EXIT',
                      -0.0005, -0.00116, -0.00116, 0.00033, ?, 1, 0)
            """,
            (dec_t, dec_t + 90.0, dec_t + 90.0, CAPTURE),
        )
        # Funded historical episode in a separate 15m bar (must remain 100% immutable!)
        conn.execute(
            """
            INSERT INTO policy_episodes (
                candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask, entry_bid,
                entry_spread, roundtrip_cost, snapshot_json, status, exit_bid, exit_at, label_available_at,
                exit_reason, gross, net, planning_net, label_taker_fee, capture_version, train_eligible, funded
            ) VALUES (202, 'FUNDED_POLICY_OUTCOME', 'DAY_V2', 'BTCUSDT', 'BREAKOUT_CONTINUATION', 'btcup_vollo',
                      ?, 100.0, 99.99, 0.0001, 0.00066, '{}', 'CLOSED', 100.25, ?, ?, 'LEARNED_CONTINUATION_EXIT',
                      0.0025, 0.00184, 0.00184, 0.00033, ?, 1, 1)
            """,
            (dec_t + 1800.0, dec_t + 2400.0, dec_t + 2400.0, CAPTURE),
        )
        conn.commit()
        conn.close()
        # Rebuild at cutoff = dec_t + 90000s (when future_poison is ALREADY in DB <= cutoff, yet > episode exit!)
        rebuild_entry_policy_value_v3(db_file, now=dec_t + 90000.0)
        rc = _sq.connect(db_file)
        t201 = rc.execute("SELECT target FROM entry_policy_v3_observations WHERE episode_id=(SELECT id FROM policy_episodes WHERE candidate_id=201)").fetchone()[0]
        t202 = rc.execute("SELECT target FROM entry_policy_v3_observations WHERE episode_id=(SELECT id FROM policy_episodes WHERE candidate_id=202)").fetchone()[0]
        rc.close()
        assert t202 == pytest.approx(0.00184, abs=1e-12)
        return float(t201)

    clean_target = _build_base_db(str(tmp_path / "clean.db"), inject_future_poison=False)
    poisoned_target = _build_base_db(str(tmp_path / "poisoned.db"), inject_future_poison=True)
    assert clean_target > 0.0
    assert poisoned_target == pytest.approx(clean_target, abs=1e-12)
