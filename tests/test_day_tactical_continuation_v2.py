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

    # Insert 1m bars showing a clean breakout rally (+45 bps net over 10m, then giveback triggering tactical rollover lock-in)
    dec_t = T0 + 1800.0
    for m in range(25):
        bar_open_t = dec_t + m * 60.0
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(bar_open_t))
        px = 100.0 + min(m, 10) * 0.052 - max(0, m - 10) * 0.022
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


def test_breakout_detection_and_progressive_runner_profit_lock(monkeypatch):
    """Verify big-outbreak visibility (DAY rebound/high-RSI + SCALP >60bps wave) and progressive hold-until-fall profit-taking."""
    import asyncio
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.binance_scalp.market_reader import MarketSnapshot
    from backend.services.binance_scalp.strategies.base import StrategyMarketContext
    from backend.services.binance_scalp.strategies.breakout_momentum import BreakoutMomentumStrategy
    from backend.services.continuation_surface import (
        _aggregate_informed,
        _apply_winner_rollover_guard,
        day_progressive_rollover_frac,
        rolling_position_prev_net,
    )
    from backend.services.day_v2.live_signal import SETUP_BREAKOUT_CONTINUATION, evaluate_all_entry_signals
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration
    from backend.services.scalp_v2 import continuation_value as cv

    # 1. DAY_V2 sees rebound breakout out of bear 4H when 1H turned bullish and 15m RSI > 72
    ts0 = 1_795_000_000
    bars_15m = [{"ts_epoch": ts0 + i * 900, "open": 100.0 + (i % 3) * 0.05, "high": 100.25, "low": 99.90, "close": 100.05 + (i % 2) * 0.05, "volume": 100.0} for i in range(45)]
    # Strong breakout impulse pushing RSI > 72
    bars_15m.append({"ts_epoch": ts0 + 45 * 900, "open": 100.10, "high": 100.55, "low": 100.08, "close": 100.50, "volume": 250.0})
    bars_15m.append({"ts_epoch": ts0 + 46 * 900, "open": 100.50, "high": 101.85, "low": 100.48, "close": 101.80, "volume": 500.0})
    bars_1h = [{"ts_epoch": ts0 + i * 3600, "open": 99.0 + i * 0.25, "high": 99.5 + i * 0.25, "low": 98.8 + i * 0.25, "close": 99.2 + i * 0.25, "volume": 800.0} for i in range(12)]
    bars_4h = [{"ts_epoch": ts0 - (12 - i) * 14400, "open": 105.0, "high": 105.5, "low": 96.0, "close": 105.0 if i < 11 else 98.0, "volume": 2000.0} for i in range(12)]
    sigs = evaluate_all_entry_signals("BTCUSDT", bars_15m, bars_1h, bars_4h)
    assert any(s.setup == SETUP_BREAKOUT_CONTINUATION for s in sigs)

    # 2. SCALP_V2 BreakoutMomentumStrategy no longer rejects a +95 bps high-volume breakout as MOVE_EXHAUSTED_CHASE
    bars_1m = [{"open": 100.0, "high": 100.35, "low": 99.95, "close": 100.20, "volume": 10.0} for _ in range(5)] + [
        {"open": 100.20, "high": 100.60, "low": 100.18, "close": 100.55, "volume": 35.0},
        {"open": 100.55, "high": 100.85, "low": 100.50, "close": 100.80, "volume": 45.0},
        {"open": 100.80, "high": 100.92, "low": 100.75, "close": 100.90, "volume": 55.0},
    ]
    snap = MarketSnapshot(
        symbol="BTC/USDT",
        symbol_bus="BTCUSDT",
        best_bid=100.915,
        best_ask=100.925,
        mid=100.92,
        spread_pct=0.0001,
        bids=[[100.915, 50.0]],
        asks=[[100.925, 50.0]],
        redis_spread_pct=0.0001,
        order_book_imbalance=0.35,
        book_source="ws",
        orderbook_age_sec=0.2,
    )
    econ = ScalpEconomics.from_env()
    cfg = get_scalp_config()
    mom = SimpleNamespace(
        mid_change_5s=0.0008,
        mid_change_15s=0.0018,
        mid_change_30s=0.0035,
        mid_change_60s=0.0060,
        bid_change_15s=0.0017,
        ask_change_15s=0.0018,
        velocity_bps_per_s=1.2,
        microprice_pressure=0.0002,
    )
    ctx = StrategyMarketContext(symbol="BTC/USDT", snap=snap, bars_1m=bars_1m, mom=mom, notional_usd=125.0, econ=econ, config=cfg)
    bm_sig = BreakoutMomentumStrategy().evaluate(ctx)
    assert bm_sig.passed is True
    assert bm_sig.expected_move_pct > 0.006

    # 3. Progressive rollover curve holds climbing +400 bps breakout, then locks profit on an ~18.7% pullback (without waiting for 35% = 140 bps loss of profit)
    assert day_progressive_rollover_frac(0.0040) == pytest.approx(0.38)
    assert day_progressive_rollover_frac(0.0400) < 0.20
    # Climbing at +400 bps MFE, +360 bps net, positive slope -> HOLD even at 2.5h age!
    informed = [{"horizon": 7200, "value": 0.0010, "evidence": 15.0, "rem_fade": 0.10}]
    climb_feats = {"net": 0.0360, "mfe": 0.0400, "mae": 0.0005, "giveback": 0.0030, "dist_high": 0.0030, "slope": 0.0012, "age_min": 150.0}
    assert _aggregate_informed(informed, engine_id="DAY_V2", setup="BREAKOUT_CONTINUATION", features=climb_feats, age_sec=9000.0, how="tactical") > 0.0
    # Starts falling: +400 bps MFE pulls back 80 bps (20% > 18.67% progressive threshold) with negative slope -> TAKE PROFIT IMMEDIATELY!
    fall_feats = {"net": 0.0314, "mfe": 0.0400, "mae": 0.0005, "giveback": 0.0080, "dist_high": 0.0080, "slope": -0.0020, "age_min": 152.0}
    assert _apply_winner_rollover_guard(0.0020, fall_feats) < 0.0

    # 4. SCALP_V2 extends active climbing runner past the old 10m cliff, then exits when progressive rollover triggers
    assert cv.scalp_progressive_rollover_frac(0.0040) == pytest.approx(0.40)
    assert cv.scalp_progressive_rollover_frac(0.0250) < 0.23
    vm = cv.ValueModel()
    # Seed mildly negative base horizon bias (-1 bps) with strong positive entry claim (+15 bps)
    z0 = cv.vector({"net": 0.001, "mfe": 0.001, "mae": 0.0, "giveback": 0.0, "age_min": 2.0}, {"cost": 0.0006}, "BTCUSDT", "BREAKOUT_MOMENTUM", "btcup_vollo")
    vm.fold([(300, z0, -0.0001, 20.0, 1000.0)], 1000.0)
    ent_run = {"calibrated": 0.0015, "adaptive": 0.0, "micro": 0.0, "cost": 0.0006}
    # At age=15m while climbing near highs (+220 bps MFE, only 15 bps giveback) -> still positive advantage (not killed at 10m cliff!)
    v_climb = cv.value("", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcup_vollo", {"net": 0.0199, "mfe": 0.0220, "mae": 0.0, "giveback": 0.0015, "age_min": 15.0}, ent_run, model=vm)
    assert v_climb is not None and v_climb["advantage"] > 0.0
    # Once it starts falling (+220 bps MFE, 55 bps giveback = 25% > progressive ~22.9%) -> negative advantage -> SELL!
    v_fall = cv.value("", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcup_vollo", {"net": 0.0159, "mfe": 0.0220, "mae": 0.0, "giveback": 0.0055, "age_min": 16.0}, ent_run, model=vm)
    assert v_fall is not None and v_fall["advantage"] < 0.0

    # 5. Rolling prev_net gives live positive slope while climbing & negative slope when falling before 5m heartbeats exist
    dummy_pos = SimpleNamespace()
    assert rolling_position_prev_net(dummy_pos, 0.0010, 1000.0) is None
    prev1 = rolling_position_prev_net(dummy_pos, 0.0035, 1008.0)
    assert prev1 == pytest.approx(0.0010)
    assert (0.0035 - prev1) > 0.0  # positive slope while climbing!

    # 6. Fast 2s DAY_V2 winner high-water ratchet & immediate rollover wake inside _scalp_fast_exit_wait
    day_pos = SimpleNamespace(
        engine_id="DAY_V2",
        status="ACTIVE",
        symbol="BTC/USDT",
        quantity=0.002,
        entry_price=100.0,
        highest_price=100.0,
        lowest_price=100.0,
        atr_at_entry=0.8,
        thesis_invalid_level=98.5,
    )
    book_seq = iter([(103.0, 103.02), (102.10, 102.12)])  # spikes +300 bps, then pulls back 90 bps (30% > 21.3% ro_frac) -> wakes immediately!
    monkeypatch.setattr("backend.services.exact_policy_capture.executable_book", lambda _s: next(book_seq, (102.10, 102.12)))
    monkeypatch.setattr("backend.services.scalp_v2.exit_evaluator.SCALP_V2_EXIT_MONITOR_INTERVAL_SEC", 0.5)
    integ = object.__new__(PortfolioEngineIntegration)
    integ.is_running = True
    integ.engine = SimpleNamespace(open_positions={"d": day_pos})
    integ._monitor_positions_once = AsyncMock(return_value=[])
    t_start = time.monotonic()
    asyncio.run(integ._scalp_fast_exit_wait(10.0))
    elapsed = time.monotonic() - t_start
    assert day_pos.highest_price == pytest.approx(103.0)
    assert elapsed < 3.0  # Woke in ~1.0s instead of sleeping the full 10s/45s!


def test_scalp_directional_claim_unclipped_and_market_population_separated(tmp_path, monkeypatch):
    """Verify SCALP >60bps directional_move_pct reaches executable_edge, unclaimed rejects stay in __MARKET, and sub-step dust retires."""
    import sqlite3
    from types import SimpleNamespace

    import backend.services.adaptive_learning as al
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.binance_scalp.market_reader import MarketSnapshot
    from backend.services.binance_scalp.scalp_candidate_ranking import rank_setup_signal
    from backend.services.binance_scalp.strategies.base import StrategyMarketContext
    from backend.services.binance_scalp.strategies.breakout_momentum import BreakoutMomentumStrategy
    from backend.services.binance_scalp.strategies.common import reject_signal

    db = str(tmp_path / "scalp_pop.db")
    cfg = SimpleNamespace(**{**get_scalp_config().__dict__, "database_path": db})
    monkeypatch.setattr("backend.services.binance_scalp.config.get_scalp_config", lambda: cfg)
    monkeypatch.setattr("backend.services.microstructure_engine.compute_features", lambda *_a, **_k: {})

    bars_1m = [{"open": 100.0, "high": 100.55, "low": 99.85, "close": 100.20, "volume": 10.0} for _ in range(5)] + [
        {"open": 100.20, "high": 100.75, "low": 100.18, "close": 100.70, "volume": 40.0},
        {"open": 100.70, "high": 101.05, "low": 100.65, "close": 101.00, "volume": 50.0},
        {"open": 101.00, "high": 101.20, "low": 100.95, "close": 101.15, "volume": 65.0},
    ]
    snap = MarketSnapshot(
        symbol="BTC/USDT",
        symbol_bus="BTCUSDT",
        best_bid=101.155,
        best_ask=101.165,
        mid=101.16,
        spread_pct=0.0001,
        bids=[[101.155, 50.0]],
        asks=[[101.165, 50.0]],
        redis_spread_pct=0.0001,
        order_book_imbalance=0.35,
        book_source="ws",
        orderbook_age_sec=0.2,
    )
    mom = SimpleNamespace(
        mid_change_5s=0.0008,
        mid_change_15s=0.0018,
        mid_change_30s=0.0035,
        mid_change_60s=0.0060,
        bid_change_15s=0.0017,
        ask_change_15s=0.0018,
        velocity_bps_per_s=1.2,
        microprice_pressure=0.0002,
        insufficient_history=False,
    )
    ctx = StrategyMarketContext(symbol="BTC/USDT", snap=snap, bars_1m=bars_1m, mom=mom, notional_usd=125.0, econ=ScalpEconomics.from_env(), config=cfg)

    # 1. Breakout directional_move_pct is NOT clipped back down to 0.006 inside pass_signal / rank_setup_signal
    sig = BreakoutMomentumStrategy().evaluate(ctx)
    assert sig.passed is True
    assert sig.directional_move_pct > 0.006
    rc_pass = rank_setup_signal(sig, regime="bull_trend", ctx=ctx)
    assert rc_pass.executable_edge["raw_expected_move_pct"] == pytest.approx(sig.directional_move_pct)
    assert rc_pass.executable_edge["raw_expected_move_pct"] > 0.006

    # 2. Unclaimed pattern rejects (NO_BREAKOUT) write to BREAKOUT_MOMENTUM__MARKET and never dilute BREAKOUT_MOMENTUM
    mkt_key = al.scalp_learned_setup("BREAKOUT_MOMENTUM", claim_fired=False)
    fired_key = al.scalp_learned_setup("BREAKOUT_MOMENTUM", claim_fired=True)
    assert mkt_key == "BREAKOUT_MOMENTUM__MARKET"
    assert fired_key == "BREAKOUT_MOMENTUM"
    for _ in range(40):
        al.learn_claim_label(db, symbol="BTCUSDT", setup=mkt_key, regime="", strategy_version=al.current_strategy_version(al.SCALP_ENGINE), raw=0.0, gross=-0.0050)
    for _ in range(15):
        al.learn_claim_label(db, symbol="BTCUSDT", setup=fired_key, regime="", strategy_version=al.current_strategy_version(al.SCALP_ENGINE), raw=0.0080, gross=0.0040)

    rej_sig = reject_signal(ctx, "breakout_momentum", "NO_BREAKOUT", mechanical=False, score=0.78)
    rc_rej = rank_setup_signal(rej_sig, regime="bull_trend", ctx=ctx)
    rc_fired = rank_setup_signal(sig, regime="bull_trend", ctx=ctx)
    assert rc_rej.adaptive_decision["setup"] == "BREAKOUT_MOMENTUM__MARKET"
    assert rc_fired.adaptive_decision["setup"] == "BREAKOUT_MOMENTUM"
    assert rc_fired.executable_edge["calibrated_move_pct"] > 0.0020
    assert rc_rej.executable_edge["calibrated_move_pct"] < -0.0020

    # 3. FIFO reconcile retires sub-step (<$0.10 & <1% qty) post-SELL dust while keeping real unclosed BUY lots
    from backend.services.portfolio_engine import PortfolioEngine

    eng_db = tmp_path / "fifo_dust.db"
    eng = object.__new__(PortfolioEngine)
    eng.db_path = str(eng_db)
    eng.open_positions = {}
    with sqlite3.connect(str(eng_db)) as conn:
        conn.execute(
            """CREATE TABLE paper_trades (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   trade_id TEXT, paper_run_id TEXT, mode TEXT, symbol TEXT,
                   side TEXT, quantity REAL, price REAL, remaining_position REAL,
                   timestamp TEXT, status TEXT, order_id TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status)
               VALUES ('sol_closed', 'r1', 'live', 'SOL/USDT', 'BUY', 3.854229, 71.43, 0.000229, '2026-11-16T02:01:18+00:00', 'executed')"""
        )
        conn.execute(
            """INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status)
               VALUES ('sol_sell', 'r1', 'live', 'SOL/USDT', 'SELL', 3.854, 71.55, 0.0, '2026-11-16T02:03:09+00:00', 'executed')"""
        )
        conn.execute(
            """INSERT INTO paper_trades (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position, timestamp, status)
               VALUES ('eth_real_orphan', 'r1', 'live', 'ETH/USDT', 'BUY', 0.10, 2500.0, 0.10, '2026-11-16T02:05:00+00:00', 'executed')"""
        )
        conn.commit()
    eng._sync_paper_fifo_remaining_to_engine_positions_sync()
    with sqlite3.connect(str(eng_db)) as conn:
        sol_rem = conn.execute("SELECT remaining_position FROM paper_trades WHERE trade_id='sol_closed'").fetchone()[0]
        eth_rem = conn.execute("SELECT remaining_position FROM paper_trades WHERE trade_id='eth_real_orphan'").fetchone()[0]
    assert sol_rem == pytest.approx(0.0)
    assert eth_rem == pytest.approx(0.10)  # Real unclosed lot is NEVER zeroed!


@pytest.mark.asyncio
async def test_live_forensic_repairs_trend_hold_market_block_breakout_capture_and_ioc_cushion(tmp_path):
    """Verify all 5 live profit repairs: DAY trend hold + constrained ridge, SCALP __MARKET block + >60bps capture + rollover guard, and IOC cushion."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.economics import ScalpEconomics
    from backend.services.binance_scalp.market_reader import MarketSnapshot
    from backend.services.binance_scalp.scalp_candidate_ranking import rank_setup_signal
    from backend.services.binance_scalp.strategies.base import StrategyMarketContext
    from backend.services.binance_scalp.strategies.common import reject_signal
    from backend.services.binance_scalp.strategies.trend_pullback_micro import TrendPullbackMicroStrategy
    from backend.services.protected_limit_execution import ExecutableBook, run_protected_preflight

    db = str(tmp_path / "forensic_repairs.db")
    install_day_surface(db, how="tactical")
    cost = canonical_roundtrip_cost_pct()
    f0 = cs.state_features(entry=71.42, mark=71.42, net=-cost, mfe=0.0, mae=0.0, high_water=71.42, prev_net=None, age_sec=0.0)

    # 1. DAY_V2 BREAKOUT_CONTINUATION with negative short horizons but strong +6h/+12h holds through 4.8m -4 bps breath AND 8m -59 bps pullback
    sol_profile = {900: -0.0026, 1800: -0.0025, 3600: -0.0026, 7200: -0.0027, 14400: -0.0017, 21600: 0.0061, 43200: 0.0238}
    for h, adv in sol_profile.items():
        cs.record_advantage(db, engine="DAY_V2", symbol="SOLUSDT", setup="BREAKOUT_CONTINUATION", regime="btcup_volhi", horizon=h, features=f0, advantage=adv, weight=20.0, now=T0)

    # Seed noisy collinear observations that previously made unconstrained OLS slope beta negative
    st_r = cs._blank_ridge()
    cs._update_ridge(st_r, {"net": 0.0020, "mfe": 0.0030, "mae": 0.0, "giveback": 0.0005, "dist_high": 0.0005, "slope": 0.0025, "age_min": 5.0}, -0.0040, 3.0, T0, "DAY_V2")
    cs._update_ridge(st_r, {"net": -0.0005, "mfe": 0.0010, "mae": 0.0010, "giveback": 0.0010, "dist_high": 0.0011, "slope": -0.0015, "age_min": 10.0}, 0.0030, 3.0, T0 + 60, "DAY_V2")
    # Positive surge candle must NEVER receive negative dot penalty from noisy slope
    surge_dot = cs._ridge_dot(st_r, {"net": 0.0040, "mfe": 0.0050, "mae": 0.0, "giveback": 0.0, "dist_high": 0.0, "slope": 0.0035, "age_min": 8.0}, T0 + 120, "DAY_V2", setup="BREAKOUT_CONTINUATION")
    assert surge_dot >= -1e-9

    # 4.8m +18.5 bps winner with 19.7 bps breath (-4.1 bps slope) holds for +6h/+12h breakout wave
    dec_4m = decide_day_continuation(
        db,
        symbol="SOLUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_volhi",
        entry_price=71.42,
        mark=71.58,
        low=71.42,
        high=71.72,
        prev_net=0.0020,
        age_sec=291.0,
        atr_at_entry=0.35,
        structural_anchor=70.60,
        target_price=74.00,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0 + 291.0,
    )
    assert dec_4m["action"] == "hold"

    # 7.9m -59 bps minor pullback above structural anchor ALSO holds instead of panic-shaking out
    dec_dip = decide_day_continuation(
        db,
        symbol="SOLUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_volhi",
        entry_price=72.20,
        mark=71.80,
        low=71.80,
        high=72.24,
        prev_net=-0.0040,
        age_sec=472.0,
        atr_at_entry=0.56,
        structural_anchor=70.80,
        target_price=75.00,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0 + 472.0,
    )
    assert dec_dip["action"] == "hold"

    # 2. SCALP_V2 unclaimed *__MARKET reject is NEVER entry_eligible even if residual/value is positive
    sc_db = str(tmp_path / "scalp_mkt_guard.db")
    sc_ver = al.current_strategy_version(al.SCALP_ENGINE)
    for step in range(25):
        al.learn_claim_label(sc_db, symbol="SOLUSDT", setup="VWAP_EMA_RECLAIM__MARKET", regime="btcup_volhi", strategy_version=sc_ver, raw=0.0, gross=0.0035, now=T0 + step)
    cfg = SimpleNamespace(**{**get_scalp_config().__dict__, "database_path": sc_db})
    snap = MarketSnapshot(
        symbol="SOL/USDT",
        symbol_bus="SOLUSDT",
        best_bid=75.945,
        best_ask=75.955,
        mid=75.95,
        spread_pct=0.00013,
        bids=[[75.945, 100.0]],
        asks=[[75.955, 100.0]],
        redis_spread_pct=0.00013,
        order_book_imbalance=0.1,
        book_source="ws",
        orderbook_age_sec=0.1,
    )
    mom = SimpleNamespace(
        mid_change_5s=0.0,
        mid_change_15s=0.0001,
        mid_change_30s=0.0001,
        mid_change_60s=0.0001,
        bid_change_15s=0.0001,
        ask_change_15s=0.0001,
        velocity_bps_per_s=0.1,
        microprice_pressure=0.0,
        insufficient_history=False,
    )
    bars_up = [{"open": 75.0 + i * 0.08, "high": 75.1 + i * 0.08, "low": 74.95 + i * 0.08, "close": 75.05 + i * 0.08, "volume": 20.0} for i in range(15)]
    ctx = StrategyMarketContext(symbol="SOL/USDT", snap=snap, bars_1m=bars_up, mom=mom, notional_usd=100.0, econ=ScalpEconomics.from_env(), config=cfg)
    with patch("backend.services.binance_scalp.config.get_scalp_config", return_value=cfg), patch("backend.services.microstructure_engine.compute_features", return_value={}):
        rej = reject_signal(ctx, "vwap_ema_reclaim", "NOT_NEAR_SUPPORT", mechanical=False, score=0.75)
        rc_mkt = rank_setup_signal(rej, regime="bull_trend", ctx=ctx)
    assert rc_mkt.adaptive_decision["setup"] == "VWAP_EMA_RECLAIM__MARKET"
    assert not rc_mkt.entry_eligible
    assert rc_mkt.hard_block == "NO_EXECUTABLE_NET_EDGE"

    # 3. SCALP_V2 isolate_regime inherits setup capture prior and >60bps breakout claims get un-amputated capture
    for i, (r_raw, r_gross) in enumerate(((0.0030, 0.0008), (0.0090, 0.0032), (0.0150, 0.0056)) * 6):
        al.learn_claim_label(sc_db, symbol="BTCUSDT", setup="VOLUME_IMPULSE_CONTINUATION", regime="btcup_vollo", strategy_version=sc_ver, raw=r_raw, gross=r_gross, now=T0 + i)
    # Flat-raw regime cell on SOL btcup_volhi should inherit positive setup slope instead of 0.0
    for i in range(5):
        al.learn_claim_label(sc_db, symbol="SOLUSDT", setup="VOLUME_IMPULSE_CONTINUATION", regime="btcup_volhi", strategy_version=sc_ver, raw=0.0092, gross=0.0010, now=T0 + 50 + i)
    cal_vic = al.scalp_claim_calibration(sc_db, "SOLUSDT", "VOLUME_IMPULSE_CONTINUATION", "btcup_volhi", now=T0 + 100)
    assert cal_vic["claim_capture"] > 0.10
    mv_big = al.scalp_expected_gross(cal_vic, 0.0160)
    assert mv_big["calibrated"] > cost * 1.5

    # 4. TrendPullbackMicroStrategy rejects post-blowoff falling-knife rollover (-200 bps off 12m high)
    rollover_bars = [{"open": 1.024 - i * 0.0018, "high": 1.025 - i * 0.0016, "low": 1.022 - i * 0.0018, "close": 1.023 - i * 0.0018, "volume": 50.0} for i in range(12)]
    xrp_snap = MarketSnapshot(
        symbol="XRP/USDT",
        symbol_bus="XRPUSDT",
        best_bid=1.0049,
        best_ask=1.0051,
        mid=1.0050,
        spread_pct=0.0002,
        bids=[[1.0049, 5000.0]],
        asks=[[1.0051, 5000.0]],
        redis_spread_pct=0.0002,
        order_book_imbalance=0.0,
        book_source="ws",
        orderbook_age_sec=0.1,
    )
    xrp_ctx = StrategyMarketContext(symbol="XRP/USDT", snap=xrp_snap, bars_1m=rollover_bars, mom=mom, notional_usd=100.0, econ=ScalpEconomics.from_env(), config=cfg)
    assert TrendPullbackMicroStrategy().evaluate(xrp_ctx).passed is False

    # 5. Protected IOC preflight adds tick cushion above walked ask levels while staying <= MAX_ORDERBOOK_PRICE_IMPACT_PCT
    now_b = T0
    eth_book = ExecutableBook(bids=[[1743.80, 2.0]], asks=[[1743.90, 0.10], [1744.05, 1.0]], last_update_id=1, source_ts=now_b, receive_ts=now_b)
    with (
        patch("backend.services.protected_limit_execution._fetch_order_book", AsyncMock(return_value=eth_book)),
        patch("backend.services.protected_limit_execution.time.time", return_value=now_b + 0.1),
    ):
        pf_eth = await run_protected_preflight(symbol="ETH/USDT", side="BUY", quantity=0.1626, reference_price=1743.90, live_capable=True)
    assert pf_eth.passed is True
    assert pf_eth.protected_limit_price >= 1744.05  # Clears second walked book level!
    assert pf_eth.protected_limit_price <= 1743.90 * 1.0005 + 1e-9

    # 6. Immediate blow-off top failure (-125 bps with mfe=0) exits cleanly instead of riding -330 bps
    d_blowoff = decide_day_continuation(
        db,
        symbol="SOLUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_volhi",
        entry_price=77.56,
        mark=76.59,
        low=76.55,
        high=77.56,
        prev_net=-0.0080,
        age_sec=180.0,
        atr_at_entry=0.80,
        structural_anchor=76.32,
        target_price=80.03,
        entry_time=T0,
        roundtrip_cost=cost,
        now=T0 + 180.0,
    )
    assert d_blowoff["action"] == "exit"
