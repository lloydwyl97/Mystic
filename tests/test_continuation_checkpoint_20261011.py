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
