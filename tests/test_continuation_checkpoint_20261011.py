"""The installed hold-advantage surface round-trips through a checkpoint and later folds."""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.services.continuation_checkpoint import (
    RECOVERY_FAILED,
    day_entry_block_reason,
    documents_match,
    events_after,
    feature_contract,
    latest_checkpoint,
    live_document,
    materialize,
    recover_continuation,
    state_hash,
    write_checkpoint,
)
from backend.services.continuation_surface import ADVANTAGE_VERSION, DAY_ADVANTAGE_HORIZONS, FEATURE_NAMES, record_advantage, surface_advantage

DAY = "DAY_V2"
_NOW = 1_800_000_000.0


def _ready(db: str) -> None:
    from backend.services.adaptive_learning import _connect, current_economic_version

    _connect(db).close()
    conn = sqlite3.connect(db)
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
        INSERT OR REPLACE INTO continuation_learning_meta
        (engine_id, economic_version, learning_version, source, anchor_commit, anchor_at, updated_at, observations)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (DAY, current_economic_version(DAY), ADVANTAGE_VERSION, "horizon_hold_advantage:blended", "test", "2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z", 1),
    )
    conn.commit()
    conn.close()


def _fold(db: str, *, setup: str, horizon: int, now: float, advantage: float = -0.0004, weight: float = 1.0) -> None:
    record_advantage(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup=setup,
        regime="btcup_vollo",
        horizon=horizon,
        features=dict.fromkeys(FEATURE_NAMES, 0.001 * (horizon / 900.0)),
        advantage=advantage,
        weight=weight,
        now=now,
    )


def _grid(features: dict[str, float]) -> list[tuple[str, str, str, dict[str, float]]]:
    return [("BTCUSDT", f"S{i}", "btcup_vollo", features) for i in range(88)]


def test_checkpoint_roundtrip_matches_live_surface(tmp_path):
    db = str(tmp_path / "live.db")
    _ready(db)
    for i in range(88):
        for horizon in DAY_ADVANTAGE_HORIZONS:
            _fold(db, setup=f"S{i}", horizon=horizon, now=_NOW + i)
    zeros = dict.fromkeys(FEATURE_NAMES, 0.0)
    before = surface_advantage(db, DAY, "BTCUSDT", "S0", "btcup_vollo", zeros, _NOW + 10_000)
    written = write_checkpoint(db)
    after = surface_advantage(db, DAY, "BTCUSDT", "S0", "btcup_vollo", zeros, _NOW + 10_000)
    assert before == pytest.approx(after)
    conn = sqlite3.connect(db)
    saved = latest_checkpoint(conn)
    current = live_document(conn)
    conn.close()
    assert saved["checksum_ok"] is True
    assert saved["state_hash"] == state_hash(current) == written["state_hash"]
    assert saved["key_count"] == 88 * len(DAY_ADVANTAGE_HORIZONS)
    assert saved["feature_contract"] == feature_contract()
    other = str(tmp_path / "loaded.db")
    rebuilt = materialize(other, saved["document"], [])
    assert documents_match(current, rebuilt)
    signs = exits = 0
    max_delta = 0.0
    blank = dict.fromkeys(FEATURE_NAMES, 0.0)
    for features in (
        blank,
        blank | {"age_min": 0.0, "net": -0.0004, "mae": 0.0002},
        blank | {"age_min": 30.0, "net": 0.002, "mfe": 0.004, "mae": 0.001, "giveback": 0.002, "dist_high": 0.001, "slope": 0.0001},
    ):
        for symbol, setup, regime, feats in _grid(features):
            live = surface_advantage(db, DAY, symbol, setup, regime, feats, _NOW + 10_000)
            loaded = surface_advantage(other, DAY, symbol, setup, regime, feats, _NOW + 10_000)
            assert live is not None and loaded is not None
            max_delta = max(max_delta, abs(live - loaded))
            if (live > 0) != (loaded > 0):
                signs += 1
            if (live > 0) != (loaded > 0):
                exits += 1
    assert signs == 0 and exits == 0 and max_delta < 1e-9


def test_checkpoint_plus_events_matches_incremental_updates(tmp_path):
    db = str(tmp_path / "inc.db")
    _ready(db)
    _fold(db, setup="S0", horizon=900, now=_NOW, advantage=-0.0002)
    write_checkpoint(db)
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once

    labels = [
        AdvantageLabel(
            engine=DAY,
            symbol="BTCUSDT",
            setup="S0",
            regime="btcup_vollo",
            trade_id=f"lot-{i}",
            snapshot_time=_NOW + 100 + i,
            label_time=_NOW + 1000 + i,
            horizon=900,
            advantage=-0.0001 * (i + 1),
            weight=0.5,
            features=dict.fromkeys(FEATURE_NAMES, 0.002),
        )
        for i in range(4)
    ]
    assert fold_advantages_once(db, labels) == 4
    assert fold_advantages_once(db, labels) == 0
    conn = sqlite3.connect(db)
    saved = latest_checkpoint(conn)
    current = live_document(conn)
    pending = events_after(conn, saved["last_event_id"])
    conn.close()
    assert len(pending) == 4
    assert all(row["features_json"] and row["weight"] == 0.5 and row["label_time"] for row in pending)
    rebuilt = materialize(str(tmp_path / "replay.db"), saved["document"], pending)
    assert documents_match(current, rebuilt)
    assert state_hash(current) == state_hash(rebuilt)


def test_crash_before_commit_leaves_the_surface_unchanged(tmp_path, monkeypatch):
    from backend.services import continuation_backfill as cb
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once

    db = str(tmp_path / "crash.db")
    _ready(db)
    label = AdvantageLabel(DAY, "BTCUSDT", "S0", "btcup_vollo", "lot", _NOW, _NOW + 10, 900, -0.0003, 1.0, dict.fromkeys(FEATURE_NAMES, 0.0))

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("crash before the learner commit")

    monkeypatch.setattr(cb, "record_advantage", _boom)
    with pytest.raises(sqlite3.OperationalError):
        fold_advantages_once(db, [label])
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM continuation_advantage_log").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM adaptive_metric_state WHERE metric LIKE 'hold_adv_%'").fetchone()[0] == 0
    conn.close()


def test_corrupt_checkpoint_blocks_day_entries_and_keeps_rows(tmp_path):
    db = str(tmp_path / "bad.db")
    _ready(db)
    _fold(db, setup="S0", horizon=900, now=_NOW)
    write_checkpoint(db)
    before = sqlite3.connect(db).execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='hold_adv_900'").fetchone()
    conn = sqlite3.connect(db)
    conn.execute("UPDATE continuation_checkpoint SET checksum='dead'")
    conn.commit()
    conn.close()
    result = recover_continuation(db)
    assert result["status"] == RECOVERY_FAILED
    assert day_entry_block_reason(db) == RECOVERY_FAILED
    after = sqlite3.connect(db).execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='hold_adv_900'").fetchone()
    assert after == before


def test_feature_contract_mismatch_fails_closed(tmp_path):
    db = str(tmp_path / "ver.db")
    _ready(db)
    _fold(db, setup="S0", horizon=900, now=_NOW)
    write_checkpoint(db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE continuation_checkpoint SET feature_contract='other'")
    conn.commit()
    conn.close()
    assert recover_continuation(db)["reason"] == "feature_contract"
    assert day_entry_block_reason(db) == RECOVERY_FAILED


def test_recover_from_checkpoint_when_live_rows_were_lost(tmp_path):
    db = str(tmp_path / "lost.db")
    _ready(db)
    _fold(db, setup="S0", horizon=900, now=_NOW, advantage=0.001)
    write_checkpoint(db)
    conn = sqlite3.connect(db)
    saved = latest_checkpoint(conn)
    conn.execute("DELETE FROM adaptive_metric_state WHERE metric LIKE 'hold_adv_%'")
    conn.execute("DELETE FROM adaptive_linear_model WHERE model LIKE 'continuation_surface%'")
    conn.commit()
    conn.close()
    result = recover_continuation(db)
    assert result["status"] == "ok" and result.get("restored") is True
    conn = sqlite3.connect(db)
    restored = live_document(conn)
    conn.close()
    assert documents_match(saved["document"], restored)


def test_second_checkpoint_is_the_one_recovery_uses(tmp_path):
    db = str(tmp_path / "seq.db")
    _ready(db)
    _fold(db, setup="S0", horizon=900, now=_NOW)
    first = write_checkpoint(db)
    _fold(db, setup="S1", horizon=900, now=_NOW + 5, advantage=0.002)
    second = write_checkpoint(db)
    assert second["id"] == first["id"] + 1
    assert second["key_count"] > first["key_count"]
    assert recover_continuation(db)["checkpoint_id"] == second["id"]


def test_real_and_counterfactual_call_the_same_continuation_decision():
    import inspect

    from backend.services import exact_policy_capture as ep

    src = inspect.getsource(ep)
    assert src.count("decide_day_continuation(") >= 2


def test_historical_log_without_features_is_not_replayable():
    db = ":memory:"
    conn = sqlite3.connect(db)
    from backend.services.continuation_checkpoint import ensure_event_columns

    ensure_event_columns(conn)
    conn.execute("INSERT INTO continuation_advantage_log (engine_id, trade_id, snapshot_time, horizon) VALUES ('DAY_V2','old',1,900)")
    conn.commit()
    assert events_after(conn, 0) == []
    conn.close()


def test_surface_install_syncs_existing_checkpoint_and_never_blocks_day(tmp_path):
    from backend.services import continuation_backfill as cb
    from backend.services.continuation_surface import ContinuationMemory, install_surface

    target = str(tmp_path / "live.db")
    _ready(target)
    _fold(target, setup="RANGE_BOUNCE", horizon=900, now=_NOW, advantage=0.0010)
    cp1 = write_checkpoint(target)
    assert recover_continuation(target)["status"] == "ok"
    assert day_entry_block_reason(target) == ""

    # Rebuild/install a modified surface onto target where a checkpoint already exists
    state = str(tmp_path / "rebuilt.db")
    mem = ContinuationMemory()
    mem.update(DAY, "BTCUSDT", "RANGE_BOUNCE", "btcup_vollo", 900, dict.fromkeys(FEATURE_NAMES, 0.001), 0.0018, 2.0, _NOW + 60)
    mem.update(DAY, "ETHUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", 900, dict.fromkeys(FEATURE_NAMES, 0.001), 0.0022, 3.0, _NOW + 60)
    cb.write_surface(state, mem)
    install_surface(target, state, {DAY: {"anchor_commit": "d856d9f", "anchor_at": "2026-11-01T00:00:00Z", "observations": 2}}, how="blended", engines=(DAY,))

    rec = recover_continuation(target)
    assert rec["status"] == "ok"
    assert rec["checkpoint_id"] > cp1["id"]
    assert day_entry_block_reason(target) == ""


def test_passive_clocks_do_not_age_out_active_trade_or_policy_calibration_evidence(tmp_path):
    from backend.services import adaptive_learning as al

    db = str(tmp_path / "clocks.db")
    al._connect(db).close()
    # Active SCALP policy_calibration at _NOW (n=2.6 >= 1.8, ewma=-14 bps)
    for step in range(3):
        al.observe(
            db,
            engine="SCALP_V2",
            symbol="ETHUSDT",
            setup="BREAKOUT_MOMENTUM",
            regime="btcflat_volhi",
            metric="policy_calibration",
            value=-0.0014,
            strategy_version=al.current_strategy_version("SCALP_V2"),
            now=_NOW + step,
            weight=0.9,
        )
    # Passive 1m markouts advance 48 hours later (_NOW + 172800)
    for step in range(20):
        al.learn_claim_label(
            db,
            symbol="ETHUSDT",
            setup="BREAKOUT_MOMENTUM",
            regime="btcflat_volhi",
            strategy_version=al.current_strategy_version("SCALP_V2"),
            raw=0.0010,
            gross=0.0005,
            now=_NOW + 172_800 + step * 60,
            weight=1.0,
        )
    cal = al.scalp_claim_calibration(db, "ETHUSDT", "BREAKOUT_MOMENTUM", "btcflat_volhi")
    assert cal["claim_gross_mean"] < 0.0

    # Active DAY trade_net at _NOW vs passive lifecycle_net 10 days later
    al.observe(
        db,
        engine="DAY_V2",
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_vollo",
        metric="trade_net",
        value=0.0025,
        strategy_version=al.current_strategy_version("DAY_V2"),
        now=_NOW,
        weight=4.0,
    )
    al.observe(
        db,
        engine="DAY_V2",
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_vollo",
        metric="lifecycle_net",
        value=-0.0010,
        strategy_version=al.current_strategy_version("DAY_V2"),
        now=_NOW + 10 * 86400,
        weight=10.0,
    )
    exp = al.day_net_expectancy(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo")
    assert exp["n_trade"] == pytest.approx(4.0)
    rep = al.adaptive_state_report(db)
    day_row = next(r for r in rep["engines"]["DAY_V2"] if r["symbol"] == "BTCUSDT" and r["setup"] == "BREAKOUT_CONTINUATION")
    live_dec = al.day_decision(db, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo")
    assert day_row["n_trade_net"] == pytest.approx(live_dec["n_trade_net"])


def test_zero_non_claims_and_overlapping_day_horizons_and_end_to_end_loops(tmp_path):
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, canonical_roundtrip_cost_pct
    from backend.services import adaptive_learning as al
    from backend.services.day_v2.day_policy_decision import decide_day_continuation
    from backend.services.scalp_v2 import continuation_value as cv
    from backend.services.scalp_v2.executable_edge import scalp_executable_edge

    db = str(tmp_path / "e2e.db")
    _ready(db)
    cost = canonical_roundtrip_cost_pct()
    assert cost == pytest.approx(ESTIMATED_ROUNDTRIP_COST)

    # 1. Zero non-claim rows (95%) + positive claims (5% at +30 bps) must not dilute claim_raw_center to ~0
    for i in range(95):
        al.learn_claim_label(
            db,
            symbol="SOLUSDT",
            setup="VWAP_EMA_RECLAIM",
            regime="btcdown_vollo",
            strategy_version=al.current_strategy_version("SCALP_V2"),
            raw=0.0,
            gross=-0.0002,
            now=_NOW + i,
            weight=1.0,
        )
    for i in range(5):
        al.learn_claim_label(
            db,
            symbol="SOLUSDT",
            setup="VWAP_EMA_RECLAIM",
            regime="btcdown_vollo",
            strategy_version=al.current_strategy_version("SCALP_V2"),
            raw=0.0030,
            gross=0.0002,
            now=_NOW + 100 + i,
            weight=1.0,
        )
    s_view = al.scalp_decision(db, "SOLUSDT", "VWAP_EMA_RECLAIM", "btcdown_vollo")
    assert s_view["claim_raw_center"] == pytest.approx(0.0030, rel=0.05)
    s_edge = scalp_executable_edge(s_view, raw_expected_move_pct=0.0030, spread_pct=0.0001, impact_pct=0.0, edge_source="STRATEGY_CLAIM")
    assert not s_edge.eligible

    # 2. DAY long overlapping horizons (6h/12h) cannot overpower inadequate 900s base horizon at entry
    for h, adv in ((900, 0.0003), (1800, 0.0004), (3600, 0.0005), (7200, 0.0010), (14400, 0.0025), (21600, 0.0045), (43200, 0.0070)):
        record_advantage(
            db,
            engine=DAY,
            symbol="SOLUSDT",
            setup="EXHAUSTION_MR__MARKET",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=adv,
            weight=5.0,
            now=_NOW,
        )
    bad_entry = al.day_decision(db, "SOLUSDT", "EXHAUSTION_MR__MARKET", "btcdown_volhi")
    assert bad_entry["expected_net"] < 0.0

    # 3. Legitimate DAY candidate with strong 900s and multi-horizon advantage qualifies, holds, rolls over, and survives restart
    for h, adv in ((900, 0.0012), (1800, 0.0015), (3600, 0.0022), (7200, 0.0030), (14400, 0.0035), (21600, 0.0040), (43200, 0.0045)):
        record_advantage(
            db,
            engine=DAY,
            symbol="XRPUSDT",
            setup="BREAKOUT_CONTINUATION__MARKET",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=adv,
            weight=8.0,
            now=_NOW,
        )
        # Train intra-hold rollover sensitivity on ridge
        record_advantage(
            db,
            engine=DAY,
            symbol="XRPUSDT",
            setup="BREAKOUT_CONTINUATION__MARKET",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": 0.0035, "mfe": 0.0050, "mae": 0.0020, "giveback": 0.0020, "dist_high": 0.0020, "slope": -0.0010, "age_min": 20.0},
            advantage=-0.0030,
            weight=8.0,
            now=_NOW + 10,
        )
        record_advantage(
            db,
            engine=DAY,
            symbol="XRPUSDT",
            setup="BREAKOUT_CONTINUATION__MARKET",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": 0.0010, "mfe": 0.0015, "mae": 0.0002, "giveback": 0.0, "dist_high": 0.0, "slope": 0.0005, "age_min": 5.0},
            advantage=0.0050,
            weight=8.0,
            now=_NOW + 20,
        )
    write_checkpoint(db)
    good_before = al.day_decision(db, "XRPUSDT", "BREAKOUT_CONTINUATION__MARKET", "btcdown_volhi")
    assert good_before["expected_net"] > 0.0
    # Simulate restart + recovery
    al._invalidate_adaptive_caches(db)
    assert recover_continuation(db)["status"] == "ok"
    good_after = al.day_decision(db, "XRPUSDT", "BREAKOUT_CONTINUATION__MARKET", "btcdown_volhi")
    assert good_after["expected_net"] == pytest.approx(good_before["expected_net"])
    # Hold scenario at 5m while rising
    hold_dec = decide_day_continuation(
        db,
        symbol="XRPUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="btcdown_volhi",
        entry_price=1.0000,
        mark=1.0020,
        low=0.9998,
        high=1.0020,
        prev_net=0.0008,
        age_sec=300.0,
        atr_at_entry=0.0050,
        structural_anchor=0.9900,
        target_price=1.0150,
        entry_time=_NOW,
        roundtrip_cost=cost,
        now=_NOW + 300.0,
    )
    assert hold_dec["action"] == "hold"

    # 4. SCALP profitable rollover lock-in in continuation_value (including fast rollover past 0)
    s_model = cv.ValueModel()
    x0 = cv.vector(cv.entry_position_state(cost), {"cost": cost}, "SOLUSDT", "ORDERBOOK_TAPE_SCALP", "btcdown_volhi")
    s_model.fold([(120, x0, 0.0002, _NOW, 1.0)] * 20, _NOW)
    ent = {"calibrated": 0.0018, "adaptive": 0.0004, "cost": cost}
    rising = cv.value("unused", "SOLUSDT", "ORDERBOOK_TAPE_SCALP", "btcdown_volhi", {"net": 0.0025, "mfe": 0.0032, "mae": 0.0, "giveback": 0.0002, "age_min": 3.0}, ent, model=s_model)
    rolled = cv.value("unused", "SOLUSDT", "ORDERBOOK_TAPE_SCALP", "btcdown_volhi", {"net": 0.0012, "mfe": 0.0032, "mae": 0.0, "giveback": 0.0014, "age_min": 3.0}, ent, model=s_model)
    rolled_red = cv.value("unused", "SOLUSDT", "ORDERBOOK_TAPE_SCALP", "btcdown_volhi", {"net": -0.0010, "mfe": 0.0032, "mae": 0.0005, "giveback": 0.0032, "age_min": 3.0}, ent, model=s_model)
    assert rising["advantage"] > 0.0
    assert rolled["advantage"] < 0.0
    assert rolled_red["advantage"] < 0.0

    # 5. DAY rollover lock-in when mfe >= 20 bps, giveback >= 35% * mfe, net > 0, and slope <= 0
    day_ro = decide_day_continuation(
        db,
        symbol="XRPUSDT",
        setup="BREAKOUT_CONTINUATION__MARKET",
        regime="btcdown_volhi",
        entry_price=1.0000,
        mark=1.0015,
        low=0.9998,
        high=1.0035,
        prev_net=0.0020,
        age_sec=600.0,
        atr_at_entry=0.0050,
        structural_anchor=0.9900,
        target_price=1.0150,
        entry_time=_NOW,
        roundtrip_cost=cost,
        now=_NOW + 600.0,
    )
    assert day_ro["action"] == "exit"

    # 6. SCALP ValueModel multi-horizon information retention under fractional _label_weight
    import numpy as np

    from backend.services.ai_learning_ingestion import record_position_heartbeat
    from backend.services.mystic_maintenance import maintenance_status_summary

    rng = np.random.default_rng(11)
    vm = cv.ValueModel()
    for h in cv.HORIZONS_SEC:
        w = (60.0 / float(h)) / 22.0
        rows_h = []
        for _ in range(400):
            net_s = -rng.uniform(0.0, 0.012)
            mfe_s = rng.uniform(0.0, 0.006)
            mae_s = max(0.0, -net_s - cost)
            gb_s = rng.uniform(0.0, mfe_s)
            age_s = rng.uniform(0.5, 20.0)
            st_s = {"net": net_s, "mfe": mfe_s, "mae": mae_s, "giveback": gb_s, "age_min": age_s}
            ent_s = {"cost": cost, "m_ofi": rng.normal(0.0, 1.0), "m_adverse": rng.uniform(0.0, 1.0)}
            x_s = cv.vector(st_s, ent_s, "SOLUSDT", "ORDERBOOK_TAPE_SCALP", "btcdown_volhi")
            y_s = -0.25 * (net_s + 0.003) + 0.0008 * ent_s["m_ofi"] - 0.0006 * ent_s["m_adverse"] + rng.normal(0.0, 0.0005)
            rows_h.append((h, x_s, y_s, _NOW, w))
        vm.fold(rows_h, _NOW)
    for h in cv.HORIZONS_SEC:
        bh = vm.beta(h)
        assert bh is not None
        assert sum(1 for v in bh[1:] if abs(float(v)) > 1e-8) >= 8
        assert float(bh[15]) <= 0.0

    # 7. maintenance_status_summary(cfg=None) does not raise NoneType backup_dir error
    ms = maintenance_status_summary(None)
    assert isinstance(ms, dict) and "error" not in ms

    # 8. record_position_heartbeat preserves intra-interval highest_price and lowest_price watermarks
    record_position_heartbeat(
        symbol="SOL/USDT",
        trade_id="test_hb_4064",
        entry_price=94.99,
        mark=94.96,
        entry_time_epoch=_NOW - 300.0,
        highest_price=95.41,
        lowest_price=94.795,
        db_path=db,
    )
    hb_row = sqlite3.connect(db).execute("SELECT highest_since_entry, lowest_since_entry, mfe_pct, mae_pct FROM ai_position_heartbeats WHERE trade_id='test_hb_4064'").fetchone()
    assert hb_row is not None
    assert hb_row[0] == pytest.approx(95.41)
    assert hb_row[1] == pytest.approx(94.795)
    assert hb_row[2] == pytest.approx((95.41 - 94.99) / 94.99)
