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


def test_successful_recovery_advances_the_checkpoint_without_rewriting_state(tmp_path):
    from backend.services.adaptive_learning import day_decision
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once

    db = str(tmp_path / "advance.db")
    _ready(db)
    features = dict.fromkeys(FEATURE_NAMES, 0.001)

    def _label(setup: str, horizon: int, when: float, advantage: float) -> AdvantageLabel:
        return AdvantageLabel(DAY, "BTCUSDT", setup, "btcup_vollo", f"lot-{setup}", when, when, horizon, advantage, 1.0, features)

    assert fold_advantages_once(db, [_label("S0", 900, _NOW, 0.0011)]) == 1
    first = write_checkpoint(db)
    assert fold_advantages_once(db, [_label("S1", 1800, _NOW + 5, 0.0022)]) == 1
    conn = sqlite3.connect(db)
    before = conn.execute("SELECT symbol, setup, regime, metric, n, ewma, m2 FROM adaptive_metric_state ORDER BY 1, 2, 3, 4").fetchall()
    covered = conn.execute("SELECT MAX(rowid) FROM continuation_advantage_log").fetchone()[0]
    conn.close()
    expected = day_decision(db, "BTCUSDT", "S1", "btcup_vollo")["expected_net"]
    result = recover_continuation(db)
    assert result["status"] == "ok"
    assert result["events"] == 1
    assert result["advanced_to_event"] == covered
    assert result["checkpoint_id"] == first["id"]
    conn = sqlite3.connect(db)
    after = conn.execute("SELECT symbol, setup, regime, metric, n, ewma, m2 FROM adaptive_metric_state ORDER BY 1, 2, 3, 4").fetchall()
    saved = latest_checkpoint(conn)
    count = conn.execute("SELECT COUNT(*) FROM continuation_checkpoint").fetchone()[0]
    conn.close()
    assert after == before
    assert saved["last_event_id"] == covered
    assert saved["id"] == result["advanced_checkpoint_id"]
    assert day_decision(db, "BTCUSDT", "S1", "btcup_vollo")["expected_net"] == expected
    again = recover_continuation(db)
    assert again["status"] == "ok"
    assert again.get("advanced_checkpoint_id") is None
    assert again["events"] == 0
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM continuation_checkpoint").fetchone()[0] == count
    conn.close()


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
            y_s = 0.25 * (net_s + 0.003) + 0.0008 * ent_s["m_ofi"] - 0.0006 * ent_s["m_adverse"] + rng.normal(0.0, 0.0005)
            rows_h.append((h, x_s, y_s, _NOW, w))
        vm.fold(rows_h, _NOW)
    for h in cv.HORIZONS_SEC:
        bh = vm.beta(h)
        assert bh is not None
        assert sum(1 for v in bh[1:] if abs(float(v)) > 1e-8) >= 8
        assert float(bh[15]) <= 0.0
        assert float(bh[14]) <= 0.0

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

    # 9. Entry-learning & calibration regression tests (Step 17):
    # (a) Passive opportunity evidence cannot override contradictory funded realized calibration / coin-regime drift
    db_cal = str(tmp_path / "entry_cal_reg.db")
    _ready(db_cal)
    for stp_m, adv_m in (("BREAKOUT_CONTINUATION__MARKET", -0.0020), ("RANGE_BOUNCE__MARKET", -0.0025), ("HTF_TREND_PULLBACK__MARKET", -0.0022)):
        for h in DAY_ADVANTAGE_HORIZONS:
            record_advantage(db_cal, engine=DAY, symbol="ETHUSDT", setup=stp_m, regime="btcdown_vollo", horizon=h, features={"net": -cost, "age_min": 0.0}, advantage=adv_m, weight=4.0, now=_NOW)
    for h, adv_v in ((900, 0.0014), (1800, 0.0010), (3600, 0.0012), (7200, 0.0009), (14400, -0.0005), (21600, -0.0010), (43200, 0.0035)):
        record_advantage(
            db_cal,
            engine=DAY,
            symbol="ETHUSDT",
            setup="VWAP_REVERSION__MARKET",
            regime="btcdown_vollo",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=adv_v,
            weight=1.5,
            now=_NOW,
        )
    vwap_dec = al.day_decision(db_cal, "ETHUSDT", "VWAP_REVERSION__MARKET", "btcdown_vollo", now=_NOW + 10)
    assert vwap_dec["expected_net"] < 0.0

    # (b) A losing funded trade / negative realized observation causally lowers the next comparable prediction
    before_loss = al.day_decision(db, "XRPUSDT", "BREAKOUT_CONTINUATION__MARKET", "btcdown_volhi", now=_NOW + 30)["expected_net"]
    for h in DAY_ADVANTAGE_HORIZONS:
        record_advantage(
            db,
            engine=DAY,
            symbol="XRPUSDT",
            setup="BREAKOUT_CONTINUATION__MARKET",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=-0.0040,
            weight=4.0,
            now=_NOW + 40,
        )
    after_loss = al.day_decision(db, "XRPUSDT", "BREAKOUT_CONTINUATION__MARKET", "btcdown_volhi", now=_NOW + 50)["expected_net"]
    assert after_loss < before_loss

    # (c) Cold SCALP cells receive conservative parent/pooled policy_calibration instead of 0.0 default, while DAY/SCALP stay isolated
    db_sc = str(tmp_path / "scalp_cold_cal.db")
    al._connect(db_sc).close()
    sc_ver = al.current_strategy_version("SCALP_V2")
    for step in range(4):
        al.learn_claim_label(db_sc, symbol="SOLUSDT", setup="BREAKOUT_MOMENTUM", regime="btcdown_vollo", strategy_version=sc_ver, raw=0.0015, gross=0.0009, now=_NOW + step, weight=1.0)
        al.learn_claim_label(db_sc, symbol="ETHUSDT", setup="BREAKOUT_MOMENTUM", regime="btcdown_vollo", strategy_version=sc_ver, raw=0.0015, gross=0.0002, now=_NOW + step, weight=2.0)
    before_parent_cal = al.scalp_claim_calibration(db_sc, "SOLUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", now=_NOW + 10)
    # Record losing realized policy_calibration on sibling symbol ETHUSDT (SOLUSDT cell itself remains cold on policy_calibration, n=0)
    for step in range(3):
        al.observe(
            db_sc,
            engine="SCALP_V2",
            symbol="ETHUSDT",
            setup="BREAKOUT_MOMENTUM",
            regime="btcdown_vollo",
            metric="policy_calibration",
            value=-0.0030,
            strategy_version=sc_ver,
            now=_NOW + 20 + step,
            weight=1.0,
        )
    after_parent_cal = al.scalp_claim_calibration(db_sc, "SOLUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", now=_NOW + 30)
    assert after_parent_cal["key_policy_calibration"] < 0.0
    assert after_parent_cal["claim_gross_mean"] < before_parent_cal["claim_gross_mean"]
    # Verify DAY state in db_sc is completely untouched by SCALP learning
    assert al.day_decision(db_sc, "SOLUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", now=_NOW + 30)["expected_net"] == pytest.approx(-cost)


def test_scalp_symmetric_claim_damping_and_hold_policy_calibration_20261102(tmp_path):
    import numpy as np

    from backend.services import adaptive_learning as al
    from backend.services.continuation_surface import surface_trace
    from backend.services.scalp_v2 import continuation_value as cv

    # Test 1: Symmetric 0.12 claim slope damping across raw > center, raw == center, raw < center, and raw == 0
    setup_mean = 0.000383
    gross_mean = 0.000446
    capture = 0.50
    center_v = 0.000726
    key_pol_cal = -0.000252
    view = {
        "claim_gross_mean": gross_mean + key_pol_cal,
        "claim_gross_uncalibrated": gross_mean,
        "claim_gross_setup": setup_mean,
        "claim_capture": capture,
        "claim_raw_center": center_v,
        "key_policy_calibration": key_pol_cal,
    }
    for raw_v in (0.001200, center_v, 0.000400, 0.0):
        move = al.scalp_expected_gross(view, raw_v)
        expected_cal = setup_mean + 0.12 * capture * (raw_v - center_v)
        assert move["calibrated"] == pytest.approx(expected_cal, abs=1e-12)
        assert move["expected"] == pytest.approx((gross_mean + key_pol_cal) + 0.12 * capture * (raw_v - center_v), abs=1e-12)

    # Test 2: Position with key_policy_calibration < 0 includes it at entry (age = 0) AND during hold (age > 0)
    vm = cv.ValueModel()
    rng = np.random.default_rng(20261102)
    cost = 0.0006015
    for h in cv.HORIZONS_SEC:
        w = (60.0 / float(h)) / 22.0
        rows_h = []
        for _ in range(120):
            net_s = rng.uniform(-0.002, 0.002)
            st_s = {"net": net_s, "mfe": max(0.0, net_s + cost), "mae": max(0.0, -net_s - cost), "giveback": 0.0, "age_min": rng.uniform(0.5, 10.0)}
            ent_s = {"cost": cost}
            x_s = cv.vector(st_s, ent_s, "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo")
            rows_h.append((h, x_s, 0.0002 - 0.15 * net_s, _NOW, w))
        vm.fold(rows_h, _NOW)

    view_zero_cal = {**view, "claim_gross_mean": gross_mean, "key_policy_calibration": 0.0}
    move_neg = al.scalp_expected_gross(view, 0.0)
    move_zero = al.scalp_expected_gross(view_zero_cal, 0.0)
    ent_neg = {
        "calibrated": move_neg["calibrated"],
        "adaptive": move_neg["adaptive"],
        "raw_claim": 0.0,
        "cost": cost,
        "key_policy_calibration": move_neg["key_policy_calibration"],
        "uncalibrated_gross": move_neg["uncalibrated_gross"],
        "uncalibrated_slope_term": move_neg["uncalibrated_slope_term"],
    }
    ent_zero = {
        "calibrated": move_zero["calibrated"],
        "adaptive": move_zero["adaptive"],
        "raw_claim": 0.0,
        "cost": cost,
        "key_policy_calibration": 0.0,
        "uncalibrated_gross": move_zero["uncalibrated_gross"],
        "uncalibrated_slope_term": move_zero["uncalibrated_slope_term"],
    }
    ev_neg = cv.entry_value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", ent_neg, model=vm)
    ev_zero = cv.entry_value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", ent_zero, model=vm)
    assert ev_neg - ev_zero == pytest.approx(key_pol_cal, abs=1e-12)

    hold_state = {"net": 0.000077, "mfe": 0.000678, "mae": 0.0, "giveback": 0.0, "age_min": 3.58}
    fade_358 = 1.0 - 3.58 / 10.0
    hv_neg = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", hold_state, ent_neg, model=vm)["advantage"]
    hv_zero = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", hold_state, ent_zero, model=vm)["advantage"]
    assert hv_neg - hv_zero == pytest.approx(key_pol_cal * fade_358, abs=1e-12)

    # Test 3: Deterministic reproduction of trade 82654 lifecycle (key_policy_calibration = -2.52 bps exits at age=3.58m for +0.77 bps instead of holding)
    ent_82654_repaired = {
        "calibrated": 0.0000383 - 0.12 * 0.0003229,
        "adaptive": (0.0003082 + key_pol_cal) - 0.0000383,
        "raw_claim": 0.0,
        "cost": cost,
        "key_policy_calibration": key_pol_cal,
        "uncalibrated_gross": 0.0003082,
        "uncalibrated_slope_term": -0.0003229,
    }
    ent_82654_dropped = {**ent_82654_repaired, "key_policy_calibration": 0.0}
    # At age = 3.58m (u_net = +0.77 bps, mfe = +6.78 bps), ValueModel horizon max is +0.75 bps:
    # Without key_policy_calibration: hold_cal_sum = -0.147 bps -> advantage = +0.75 - 0.147 * 0.642 = +0.656 bps > 0 (fails to exit)
    # With key_policy_calibration = -2.52 bps: hold_cal_sum = -2.667 bps -> advantage = +0.75 - 2.667 * 0.642 = -0.962 bps < 0 (exits at +0.77 bps)
    adv_dropped_358 = 0.000075 + (ent_82654_dropped["uncalibrated_gross"] + ent_82654_dropped["uncalibrated_slope_term"] + ent_82654_dropped["key_policy_calibration"]) * fade_358
    adv_repaired_358 = 0.000075 + (ent_82654_repaired["uncalibrated_gross"] + ent_82654_repaired["uncalibrated_slope_term"] + ent_82654_repaired["key_policy_calibration"]) * fade_358
    assert adv_dropped_358 > 0.0
    assert adv_repaired_358 < 0.0
    read_rep = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", hold_state, ent_82654_repaired, model=vm)
    read_drp = cv.value("unused", "BTCUSDT", "BREAKOUT_MOMENTUM", "btcdown_vollo", hold_state, ent_82654_dropped, model=vm)
    assert read_rep["advantage"] - read_drp["advantage"] == pytest.approx(key_pol_cal * fade_358, abs=1e-12)

    # Test 4 & Test 5: DAY_V2 outputs are byte/value-identical and completely isolated from SCALP_V2 adaptive state
    db_iso = str(tmp_path / "day_scalp_iso.db")
    _ready(db_iso)
    day_before = al.day_decision(db_iso, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", now=_NOW + 5)
    trace_before = surface_trace(db_iso, DAY, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", {"net": 0.001, "mfe": 0.002, "mae": 0.0005, "giveback": 0.001, "age_min": 15.0}, _NOW + 5)
    sc_ver = al.current_strategy_version("SCALP_V2")
    for i in range(6):
        al.learn_claim_label(db_iso, symbol="BTCUSDT", setup="BREAKOUT_MOMENTUM", regime="btcup_vollo", strategy_version=sc_ver, raw=0.0010, gross=0.0008, now=_NOW + i, weight=1.0)
        al.observe(
            db_iso,
            engine="SCALP_V2",
            symbol="BTCUSDT",
            setup="BREAKOUT_MOMENTUM",
            regime="btcup_vollo",
            metric="policy_calibration",
            value=-0.0020,
            strategy_version=sc_ver,
            now=_NOW + i,
            weight=1.0,
        )
    day_after = al.day_decision(db_iso, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", now=_NOW + 5)
    trace_after = surface_trace(db_iso, DAY, "BTCUSDT", "BREAKOUT_CONTINUATION", "btcup_vollo", {"net": 0.001, "mfe": 0.002, "mae": 0.0005, "giveback": 0.001, "age_min": 15.0}, _NOW + 5)
    assert json.dumps(day_before, sort_keys=True) == json.dumps(day_after, sort_keys=True)
    assert json.dumps(trace_before, sort_keys=True) == json.dumps(trace_after, sort_keys=True)


def test_day_term_structure_and_scalp_policy_calibration_deadlock_repairs_20261104(tmp_path):
    from backend.services import adaptive_learning as al
    from backend.services.continuation_surface import DAY_ADVANTAGE_HORIZONS, record_advantage, surface_trace
    from backend.services.scalp_v2 import continuation_value as cv

    cost = 0.00066

    # =========================================================================
    # DAY REGRESSION TESTS (1 - 4)
    # =========================================================================
    # 1. Coherent positive 15m-4h upward term structure is not penalized as dispersion
    #    merely because longer horizons are stronger, and stale negative 12h does not destroy it.
    db_up = str(tmp_path / "day_upward.db")
    _ready(db_up)
    upward_vals = {900: 0.000452, 1800: 0.000751, 3600: 0.001014, 7200: 0.001662, 14400: 0.002181, 21600: 0.003058, 43200: 0.001653}
    for h, v in upward_vals.items():
        record_advantage(
            db_up,
            engine=DAY,
            symbol="BTCUSDT",
            setup="EXHAUSTION_MR",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=v,
            weight=80.0,
            now=_NOW,
        )
    tr_up = surface_trace(db_up, DAY, "BTCUSDT", "EXHAUSTION_MR", "btcdown_volhi", {"net": -cost, "age_min": 0.0}, _NOW + 1, how="blended")
    assert tr_up is not None
    h900_up = next(float(item["value"]) for item in tr_up["horizons"] if int(item["horizon"]) == 900)
    assert tr_up["advantage"] > h900_up
    assert tr_up["advantage"] > cost

    # Stale negative 12h posterior on coherent positive 15m-6h VWAP_REVERSION does not destroy entry
    eth_vwap_vals = {900: 0.000832, 1800: 0.000571, 3600: 0.001322, 7200: 0.000335, 14400: 0.001562, 21600: 0.001800, 43200: -0.002467}
    for h, v in eth_vwap_vals.items():
        record_advantage(
            db_up,
            engine=DAY,
            symbol="ETHUSDT",
            setup="VWAP_REVERSION",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=v,
            weight=80.0,
            now=_NOW,
        )
    tr_eth_vwap = surface_trace(db_up, DAY, "ETHUSDT", "VWAP_REVERSION", "btcdown_volhi", {"net": -cost, "age_min": 0.0}, _NOW + 1, how="blended")
    assert tr_eth_vwap is not None
    assert tr_eth_vwap["advantage"] > cost

    # 2. Opposing-sign / negative longer-horizon structure (+8, +7, +2, -10, -25 bps) remains strongly penalized
    down_vals = {900: 0.00080, 1800: 0.00070, 3600: 0.00020, 7200: -0.00100, 14400: -0.00250, 21600: -0.00250, 43200: -0.00250}
    for h, v in down_vals.items():
        record_advantage(
            db_up,
            engine=DAY,
            symbol="SOLUSDT",
            setup="RANGE_BOUNCE",
            regime="btcup_vollo",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=v,
            weight=80.0,
            now=_NOW,
        )
    tr_down = surface_trace(db_up, DAY, "SOLUSDT", "RANGE_BOUNCE", "btcup_vollo", {"net": -cost, "age_min": 0.0}, _NOW + 1, how="blended")
    assert tr_down is not None
    assert tr_down["advantage"] < -0.0040

    # 3. Old 6h/12h overlap subsidy remains prevented (e.g. ERA_5 SOL EXHAUSTION_MR with h900=+4.55, h14400=+20.60, h43200=+58.31 bps)
    overlap_vals = {900: 0.000455, 1800: 0.000699, 3600: 0.001518, 7200: 0.003097, 14400: 0.002060, 21600: 0.003459, 43200: 0.005831}
    for h, v in overlap_vals.items():
        record_advantage(
            db_up,
            engine=DAY,
            symbol="SOLUSDT",
            setup="EXHAUSTION_MR",
            regime="btcdown_volhi",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=v,
            weight=80.0,
            now=_NOW,
        )
    tr_overlap = surface_trace(db_up, DAY, "SOLUSDT", "EXHAUSTION_MR", "btcdown_volhi", {"net": -cost, "age_min": 0.0}, _NOW + 1, how="blended")
    assert tr_overlap is not None
    assert tr_overlap["advantage"] < cost

    # 4. Previous ERA_3 low-conviction losing cohort (e.g. h900=+2.35, h1800=+0.90, h3600=+3.52, h7200=-0.50, h14400=+9.00, h21600=+30.63) remains rejected
    era3_vals = {900: 0.000235, 1800: 0.000090, 3600: 0.000352, 7200: -0.000050, 14400: 0.000900, 21600: 0.003063, 43200: 0.002442}
    for h in DAY_ADVANTAGE_HORIZONS:
        record_advantage(
            db_up,
            engine=DAY,
            symbol="XRPUSDT",
            setup="HTF_TREND_PULLBACK",
            regime="btcdown_vollo",
            horizon=h,
            features={"net": -cost, "age_min": 0.0},
            advantage=era3_vals[h],
            weight=80.0,
            now=_NOW,
        )
    tr_era3 = surface_trace(db_up, DAY, "XRPUSDT", "HTF_TREND_PULLBACK", "btcdown_vollo", {"net": -cost, "age_min": 0.0}, _NOW + 1, how="blended")
    assert tr_era3 is not None
    assert tr_era3["advantage"] < 0.0

    # =========================================================================
    # SCALP REGRESSION TESTS (1 - 6)
    # =========================================================================
    db_sc = str(tmp_path / "scalp_deadlock_repair.db")
    al._connect(db_sc).close()
    sc_ver = al.current_strategy_version("SCALP_V2")

    # Seed heavy negative policy_calibration on OTHER setups (simulating stale October engine-wide -7.41 bps prior)
    for other_stp in ("FAILED_BREAKOUT_REVERSAL", "BREAKOUT_MOMENTUM", "VWAP_EMA_RECLAIM"):
        for step in range(15):
            al.observe(
                db_sc,
                engine="SCALP_V2",
                symbol="ETHUSDT",
                setup=other_stp,
                regime="btcdown_volhi",
                metric="policy_calibration",
                value=-0.0020,
                strategy_version=sc_ver,
                now=_NOW + step,
                weight=1.0,
            )

    # 1. Cold setup/cell (TREND_PULLBACK_MICRO) does NOT inherit other setups' engine-wide negative policy_calibration
    for step in range(8):
        al.learn_claim_label(
            db_sc,
            symbol="BTCUSDT",
            setup="TREND_PULLBACK_MICRO",
            regime="btcup_volhi",
            strategy_version=sc_ver,
            raw=0.0020,
            gross=0.0009,
            now=_NOW + 20 + step,
            weight=1.0,
        )
    cold_cal = al.scalp_claim_calibration(db_sc, "BTCUSDT", "TREND_PULLBACK_MICRO", "btcup_volhi", now=_NOW + 30)
    assert cold_cal["key_policy_calibration"] == pytest.approx(0.0, abs=1e-12)
    assert cold_cal["claim_gross_mean"] == pytest.approx(cold_cal["claim_gross_uncalibrated"], abs=1e-12)

    # 2. Exact/setup negative realized evidence still lowers comparable future entries within the same setup
    sibling_cal = al.scalp_claim_calibration(db_sc, "SOLUSDT", "FAILED_BREAKOUT_REVERSAL", "btcdown_volhi", now=_NOW + 30)
    exact_cal = al.scalp_claim_calibration(db_sc, "ETHUSDT", "FAILED_BREAKOUT_REVERSAL", "btcdown_volhi", now=_NOW + 30)
    assert sibling_cal["key_policy_calibration"] < -0.0005
    assert exact_cal["key_policy_calibration"] <= sibling_cal["key_policy_calibration"] < -0.0015

    # 3. Old zero-dilution / FAILED_BREAKOUT_REVERSAL / raw=0 BREAKOUT_MOMENTUM losing cohorts remain rejected
    for step in range(8):
        al.learn_claim_label(
            db_sc,
            symbol="ETHUSDT",
            setup="FAILED_BREAKOUT_REVERSAL",
            regime="btcdown_volhi",
            strategy_version=sc_ver,
            raw=0.0022,
            gross=0.0003,
            now=_NOW + 20 + step,
            weight=1.0,
        )
        al.learn_claim_label(
            db_sc,
            symbol="BTCUSDT",
            setup="BREAKOUT_MOMENTUM",
            regime="btcflat_volhi",
            strategy_version=sc_ver,
            raw=0.0012,
            gross=0.0003,
            now=_NOW + 20 + step,
            weight=1.0,
        )
    fbr_view = al.scalp_claim_calibration(db_sc, "ETHUSDT", "FAILED_BREAKOUT_REVERSAL", "btcdown_volhi", now=_NOW + 35)
    fbr_move = al.scalp_expected_gross(fbr_view, 0.0022)
    assert fbr_move["expected"] - cost < 0.0

    bm_view = al.scalp_claim_calibration(db_sc, "BTCUSDT", "BREAKOUT_MOMENTUM", "btcflat_volhi", now=_NOW + 35)
    bm_move_raw0 = al.scalp_expected_gross(bm_view, 0.0)
    assert bm_move_raw0["expected"] - cost < 0.0

    # 4. Positive claim cells can become reachable when current evidence genuinely overcomes specific negative calibration
    for step in range(12):
        al.learn_claim_label(
            db_sc,
            symbol="XRPUSDT",
            setup="FAILED_BREAKDOWN_REVERSAL",
            regime="btcup_volhi",
            strategy_version=sc_ver,
            raw=0.0028,
            gross=0.0014,
            now=_NOW + 40 + step,
            weight=1.0,
        )
    al.observe(
        db_sc,
        engine="SCALP_V2",
        symbol="SOLUSDT",
        setup="FAILED_BREAKDOWN_REVERSAL",
        regime="btcflat_volhi",
        metric="policy_calibration",
        value=-0.0015,
        strategy_version=sc_ver,
        now=_NOW,
        weight=1.0,
    )
    pos_view = al.scalp_claim_calibration(db_sc, "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", now=_NOW + 60)
    pos_move = al.scalp_expected_gross(pos_view, 0.0028)
    assert pos_view["key_policy_calibration"] < 0.0
    assert pos_move["expected"] > cost

    # 5 & 6. 1d29a26 symmetric 0.12 claim damping and hold key_policy_calibration preservation remain unchanged
    m_above = al.scalp_expected_gross(pos_view, pos_view["claim_raw_center"] + 0.0005)
    m_below = al.scalp_expected_gross(pos_view, max(0.0, pos_view["claim_raw_center"] - 0.0005))
    assert m_above["calibrated"] - pos_view["claim_gross_setup"] == pytest.approx(0.12 * pos_view["claim_capture"] * 0.0005, abs=1e-12)
    assert pos_view["claim_gross_setup"] - m_below["calibrated"] == pytest.approx(0.12 * pos_view["claim_capture"] * 0.0005, abs=1e-12)

    vm = cv.ValueModel()
    hold_st = {"net": 0.0001, "mfe": 0.0005, "mae": 0.0, "giveback": 0.0, "age_min": 4.0}
    ent_with = {
        "calibrated": pos_move["calibrated"],
        "adaptive": pos_move["adaptive"],
        "raw_claim": 0.0028,
        "cost": cost,
        "key_policy_calibration": pos_move["key_policy_calibration"],
        "uncalibrated_gross": pos_move["uncalibrated_gross"],
        "uncalibrated_slope_term": pos_move["uncalibrated_slope_term"],
    }
    ent_without = {**ent_with, "key_policy_calibration": 0.0}
    # Fold one dummy observation so ValueModel has informed horizons
    x_d = cv.vector(hold_st, ent_with, "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi")
    for h in cv.HORIZONS_SEC:
        vm.fold([(h, x_d, 0.0001, _NOW, 1.0)], _NOW)
    v_with = cv.value("unused", "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", hold_st, ent_with, model=vm)["advantage"]
    v_without = cv.value("unused", "XRPUSDT", "FAILED_BREAKDOWN_REVERSAL", "btcup_volhi", hold_st, ent_without, model=vm)["advantage"]
    assert v_with - v_without == pytest.approx(pos_move["key_policy_calibration"] * 0.60, abs=1e-12)


def test_periodic_checkpoint_matches_the_live_surface_and_does_not_rewrite_it(tmp_path, monkeypatch):
    from backend.services.adaptive_learning import day_decision
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once
    from backend.services.continuation_checkpoint import checkpoint_status, maybe_checkpoint_live_surface

    db = str(tmp_path / "periodic.db")
    _ready(db)
    features = dict.fromkeys(FEATURE_NAMES, 0.001)

    def _label(setup: str, when: float) -> AdvantageLabel:
        return AdvantageLabel(DAY, "BTCUSDT", setup, "btcup_vollo", f"lot-{setup}", when, when, 900, 0.001, 1.0, features)

    assert fold_advantages_once(db, [_label("S0", _NOW)]) == 1
    write_checkpoint(db)
    assert fold_advantages_once(db, [_label("S1", _NOW + 5)]) == 1
    assert maybe_checkpoint_live_surface(db) is None
    monkeypatch.setattr("backend.services.continuation_checkpoint.CHECKPOINT_EVENT_LAG_LIMIT", 1)
    conn = sqlite3.connect(db)
    before = conn.execute("SELECT symbol, setup, regime, metric, n, ewma, m2 FROM adaptive_metric_state ORDER BY 1,2,3,4").fetchall()
    covered = conn.execute("SELECT MAX(rowid) FROM continuation_advantage_log").fetchone()[0]
    conn.close()
    expected = day_decision(db, "BTCUSDT", "S1", "btcup_vollo")["expected_net"]
    written = maybe_checkpoint_live_surface(db)
    assert written is not None
    status = checkpoint_status(db)
    assert status["event_lag"] == 0
    assert status["checkpoint_event_id"] == covered
    assert status["latest_event_id"] == covered
    assert status["last_verification_status"] in (None, "ok")
    conn = sqlite3.connect(db)
    payload = json.loads(conn.execute("SELECT payload_json, last_event_id FROM continuation_checkpoint ORDER BY id DESC LIMIT 1").fetchone()[0])
    column_event = conn.execute("SELECT last_event_id FROM continuation_checkpoint ORDER BY id DESC LIMIT 1").fetchone()[0]
    after = conn.execute("SELECT symbol, setup, regime, metric, n, ewma, m2 FROM adaptive_metric_state ORDER BY 1,2,3,4").fetchall()
    conn.close()
    assert column_event == covered
    assert payload["last_event_id"] == covered
    assert after == before
    assert day_decision(db, "BTCUSDT", "S1", "btcup_vollo")["expected_net"] == expected
    assert maybe_checkpoint_live_surface(db) is None


def test_checkpoint_insert_failure_keeps_the_previous_row(tmp_path, monkeypatch):
    from backend.services import continuation_checkpoint as checkpoint

    db = str(tmp_path / "crash.db")
    _ready(db)
    _fold(db, setup="S0", horizon=900, now=_NOW)
    write_checkpoint(db)
    conn = sqlite3.connect(db)
    before = conn.execute("SELECT COUNT(*), MAX(last_event_id) FROM continuation_checkpoint").fetchone()
    conn.close()
    real_connect = checkpoint._connect

    def _connect_commit_fails(path: str):
        inner = real_connect(path)

        class _Proxy:
            def execute(self, *args, **kwargs):
                return inner.execute(*args, **kwargs)

            def commit(self):
                raise sqlite3.OperationalError("injected before commit")

            def rollback(self):
                inner.rollback()

            def close(self):
                inner.close()

        return _Proxy()

    monkeypatch.setattr(checkpoint, "_connect", _connect_commit_fails)
    with pytest.raises(sqlite3.OperationalError):
        write_checkpoint(db)
    conn = sqlite3.connect(db)
    after = conn.execute("SELECT COUNT(*), MAX(last_event_id) FROM continuation_checkpoint").fetchone()
    conn.close()
    assert after == before


def test_changed_hold_advantage_fails_closed_without_advancing(tmp_path):
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once

    db = str(tmp_path / "mismatch.db")
    _ready(db)
    features = dict.fromkeys(FEATURE_NAMES, 0.001)
    fold_advantages_once(db, [AdvantageLabel(DAY, "BTCUSDT", "S0", "btcup_vollo", "lot", _NOW, _NOW, 900, 0.001, 1.0, features)])
    write_checkpoint(db)
    fold_advantages_once(db, [AdvantageLabel(DAY, "BTCUSDT", "S1", "btcup_vollo", "lot2", _NOW + 5, _NOW + 5, 900, 0.002, 1.0, features)])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE adaptive_metric_state SET ewma = ewma + 0.01 WHERE metric LIKE 'hold_adv_%'")
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM continuation_checkpoint").fetchone()[0]
    conn.close()
    result = recover_continuation(db)
    assert result["status"] == RECOVERY_FAILED
    assert result.get("advanced_checkpoint_id") is None
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM continuation_checkpoint").fetchone()[0] == count
    conn.close()


def test_batched_scratch_replay_matches_per_event_replay(tmp_path):
    from backend.services.adaptive_learning import _connect as connect_learning
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once
    from backend.services.continuation_checkpoint import _connect, _install_document, apply_event, events_after, live_document, materialize

    db = str(tmp_path / "live.db")
    _ready(db)
    features = dict.fromkeys(FEATURE_NAMES, 0.002)
    fold_advantages_once(db, [AdvantageLabel(DAY, "BTCUSDT", "S0", "btcup_vollo", "a", _NOW, _NOW, 900, 0.001, 1.0, features)])
    saved = write_checkpoint(db)
    fold_advantages_once(
        db,
        [
            AdvantageLabel(DAY, "BTCUSDT", "S1", "btcup_vollo", "b", _NOW + 5, _NOW + 5, 1800, 0.002, 1.0, features),
            AdvantageLabel(DAY, "ETHUSDT", "S2", "btcup_vollo", "c", _NOW + 6, _NOW + 6, 900, -0.001, 1.0, features),
        ],
    )
    conn = _connect(db)
    try:
        events = events_after(conn, int(saved["last_event_id"]) if "last_event_id" in saved else 0)
        base = latest_checkpoint(conn)["document"]
        # write_checkpoint return does not include last_event_id; read it.
        base_event = int(conn.execute("SELECT last_event_id FROM continuation_checkpoint ORDER BY id DESC LIMIT 1").fetchone()[0])
        # The checkpoint above was taken before the two extra folds, so re-read that row.
        first = conn.execute("SELECT last_event_id, payload_json FROM continuation_checkpoint ORDER BY id ASC LIMIT 1").fetchone()
        events = events_after(conn, int(first[0]))
        base = json.loads(first[1])
        current = live_document(conn)
    finally:
        conn.close()
    batched = materialize(str(tmp_path / "batch.db"), base, events)
    sequential_path = str(tmp_path / "seq.db")
    connect_learning(sequential_path).close()
    seq = _connect(sequential_path)
    seq.execute("BEGIN IMMEDIATE")
    _install_document(seq, base)
    seq.commit()
    seq.close()
    for event in events:
        apply_event(sequential_path, event)
    seq = _connect(sequential_path)
    try:
        sequential = live_document(seq)
    finally:
        seq.close()
    assert documents_match(batched, sequential)
    assert documents_match(batched, current)
    assert base_event >= 0
