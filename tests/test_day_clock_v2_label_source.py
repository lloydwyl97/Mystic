"""V5 3h label authority, retry semantics, comparability. Research only."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from backend.services.day_clock_v2_label_source import (
    INTERVAL_SEC,
    INVALID_MISMATCH,
    INVALID_NO_BARS,
    INVALID_REST_TRANSIENT,
    LABEL_SOURCE_VERSION,
    SOURCE_REDIS,
    SOURCE_REST,
    STATUS_COMPLETE,
    STATUS_PENDING_LABEL_SOURCE,
    STATUS_PENDING_NOT_MATURE,
    HorizonCandle,
    candle_close_ts,
    last_closed_open_ts,
    ohlcv_equal,
    parse_redis_rows,
    pit_ok,
    resolve_v5_horizon_candle,
    select_closed_horizon_candle,
)
from backend.services.day_clock_v2_labels import (
    INVALID_IMMATURE,
    INVALID_NOT_AVAILABLE,
    TABLE_V5_LABELS,
    TABLE_V5_LABELS_HISTORY,
    TARGET_NAME,
    build_v5_label,
    group_label_status,
    hold_label,
    load_v5_label_presence,
    persist_v5_labels,
    required_actions,
    run_v5_label_batch,
)
from backend.services.day_clock_v2_partition import DEVELOPMENT
from backend.services.day_path_clock_v2 import (
    PRIMARY_TARGET_HORIZON_SEC,
    REQUIRED_CLOCK_V2_FIELDS_V5,
    v5_listed_features_from_blob,
)
from backend.services.day_path_clock_v2_capture import TABLE_ARTIFACT, ensure_artifact_schema, group_comparability_v5
from backend.services.day_path_clock_v2_readiness import evaluate_clock_v2_v5_readiness

DECISION = datetime(2026, 9, 6, 0, 0, 10, tzinfo=timezone.utc)
HORIZON = DECISION + timedelta(seconds=PRIMARY_TARGET_HORIZON_SEC)
TARGET_OPEN = datetime(2026, 9, 6, 2, 59, 0, tzinfo=timezone.utc)
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _candle(open_ts: datetime, close: float = 100.0, volume: float = 1.0) -> HorizonCandle:
    return HorizonCandle(
        open_ts=open_ts,
        close_ts=candle_close_ts(open_ts),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _redis_row(open_ts: datetime, close: float = 100.0, volume: float = 1.0) -> list[float]:
    return [open_ts.timestamp(), close, close, close, close, volume]


def _rest_row(open_ts: datetime, close: float = 100.0, volume: float = 1.0) -> list[object]:
    open_ms = int(open_ts.timestamp() * 1000)
    return [open_ms, str(close), str(close), str(close), str(close), str(volume), open_ms + 59999]


class _FakeRedis:
    def __init__(self, by_symbol: dict[str, list[list[float]]]):
        self.by_symbol = by_symbol

    def get(self, key: str):
        symbol = key.split(":")[1]
        rows = self.by_symbol.get(symbol)
        return json.dumps(rows) if rows is not None else None


def _matching_books(close: float = 101.5, volume: float = 0.0) -> tuple[_FakeRedis, object]:
    rows = [_redis_row(TARGET_OPEN, close, volume)]
    redis = _FakeRedis(dict.fromkeys(COINS, rows))

    def rest(symbol: str, start_ms: int, end_ms: int):
        del symbol, start_ms, end_ms
        return [_rest_row(TARGET_OPEN, close, volume)]

    return redis, rest


def test_target_candle_is_last_closed_at_or_before_horizon():
    assert datetime(2026, 9, 6, 3, 0, 10, tzinfo=timezone.utc) == HORIZON
    assert last_closed_open_ts(HORIZON) == TARGET_OPEN
    forming = _candle(datetime(2026, 9, 6, 3, 0, 0, tzinfo=timezone.utc))
    chosen = select_closed_horizon_candle(
        [_candle(TARGET_OPEN - timedelta(minutes=1)), _candle(TARGET_OPEN), forming],
        horizon_at=HORIZON,
        now=HORIZON + timedelta(hours=4),
    )
    assert chosen is not None
    assert chosen.open_ts == TARGET_OPEN
    assert chosen.close_ts == datetime(2026, 9, 6, 3, 0, 0, tzinfo=timezone.utc)
    assert chosen.close_ts <= HORIZON


def test_horizon_exactly_on_minute_close():
    exact = datetime(2026, 9, 6, 3, 0, 0, tzinfo=timezone.utc)
    assert last_closed_open_ts(exact) == TARGET_OPEN
    chosen = select_closed_horizon_candle(
        [_candle(TARGET_OPEN), _candle(exact)],
        horizon_at=exact,
        now=exact + timedelta(hours=1),
    )
    assert chosen is not None
    assert chosen.open_ts == TARGET_OPEN
    assert chosen.close_ts == exact


def test_forming_and_future_candles_are_rejected():
    now = HORIZON
    forming = _candle(datetime(2026, 9, 6, 3, 0, 0, tzinfo=timezone.utc))
    future = _candle(datetime(2026, 9, 6, 3, 1, 0, tzinfo=timezone.utc))
    assert (
        select_closed_horizon_candle([forming, future], horizon_at=HORIZON, now=now) is None
    )
    chosen = select_closed_horizon_candle(
        [_candle(TARGET_OPEN), forming, future],
        horizon_at=HORIZON,
        now=HORIZON + timedelta(hours=1),
    )
    assert chosen is not None
    assert chosen.open_ts == TARGET_OPEN


def test_zero_volume_genuine_exchange_candle_is_kept():
    redis, rest = _matching_books(volume=0.0)
    out = resolve_v5_horizon_candle("ETHUSDT", HORIZON, now=HORIZON + timedelta(hours=4), redis_client=redis, rest_fetch=rest)
    assert out["ok"] is True
    assert out["candle"].volume == 0.0
    assert out["label_source"] == SOURCE_REDIS
    assert out["source_verified"] is True


def test_redis_hit_with_matching_rest():
    redis, rest = _matching_books()
    out = resolve_v5_horizon_candle("BTCUSDT", HORIZON, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=rest)
    assert out["ok"] is True
    assert out["label_source"] == SOURCE_REDIS
    assert out["source_verified"] is True
    assert out["target_bar_open_ts"] == TARGET_OPEN.isoformat()


def test_redis_aged_out_uses_rest_fallback():
    redis = _FakeRedis({"BTCUSDT": []})

    def rest(symbol: str, start_ms: int, end_ms: int):
        del symbol, start_ms, end_ms
        return [_rest_row(TARGET_OPEN, 99.0)]

    out = resolve_v5_horizon_candle("BTCUSDT", HORIZON, now=HORIZON + timedelta(hours=12), redis_client=redis, rest_fetch=rest)
    assert out["ok"] is True
    assert out["label_source"] == SOURCE_REST
    assert out["candle"].close == 99.0


def test_redis_rest_equality_and_mismatch():
    left = _candle(TARGET_OPEN, 100.0)
    right = _candle(TARGET_OPEN, 100.0)
    assert ohlcv_equal(left, right)
    redis = _FakeRedis({"ETHUSDT": [_redis_row(TARGET_OPEN, 100.0)]})

    def rest(symbol: str, start_ms: int, end_ms: int):
        del symbol, start_ms, end_ms
        return [_rest_row(TARGET_OPEN, 101.0)]

    out = resolve_v5_horizon_candle("ETHUSDT", HORIZON, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=rest)
    assert out["ok"] is False
    assert out["reason"] == INVALID_MISMATCH
    assert out["status"] == "TERMINAL_INVALID"


def test_missing_exchange_candle_and_rest_transient():
    redis = _FakeRedis({"SOLUSDT": []})

    def empty(symbol: str, start_ms: int, end_ms: int):
        del symbol, start_ms, end_ms
        return []

    missing = resolve_v5_horizon_candle("SOLUSDT", HORIZON, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=empty)
    assert missing["ok"] is False
    assert missing["reason"] == INVALID_NO_BARS
    assert missing["status"] == STATUS_PENDING_LABEL_SOURCE

    def boom(symbol: str, start_ms: int, end_ms: int):
        del symbol, start_ms, end_ms
        raise TimeoutError("rest timeout")

    transient = resolve_v5_horizon_candle("SOLUSDT", HORIZON, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=boom)
    assert transient["reason"] == INVALID_REST_TRANSIENT
    assert transient["status"] == STATUS_PENDING_LABEL_SOURCE


def test_pit_rejects_future_and_inverted_horizon():
    candle = _candle(TARGET_OPEN)
    assert pit_ok(decision_ts=DECISION, horizon_at=HORIZON, candle=candle) is True
    late = _candle(datetime(2026, 9, 6, 3, 0, 0, tzinfo=timezone.utc))
    assert late.close_ts > HORIZON
    assert pit_ok(decision_ts=DECISION, horizon_at=HORIZON, candle=late) is False
    assert pit_ok(decision_ts=HORIZON, horizon_at=DECISION, candle=candle) is False


def test_build_label_uses_frozen_formula_without_feature_ohlcv(tmp_path):
    redis, rest = _matching_books(close=110.0)
    lab = build_v5_label(
        db_path=tmp_path / "unused.db",
        decision_group_id="g1",
        symbol="ETHUSDT",
        decision_ts=DECISION.isoformat(),
        action_available=True,
        entry_px=100.0,
        spread_bps=1.2,
        now=HORIZON + timedelta(minutes=5),
        redis_client=redis,
        rest_fetch=rest,
    )
    assert lab["label_valid"] is True
    assert lab["label_source_version"] == LABEL_SOURCE_VERSION
    assert lab["target_name"] == TARGET_NAME
    assert lab["target_horizon_sec"] == 10800
    gross = (110.0 - 100.0) / 100.0 * 1e4
    assert lab["executable_gross_bps_3h"] == pytest.approx(gross)
    assert lab["executable_net_bps_3h"] == pytest.approx(gross - lab["all_in_cost_bps"])
    assert lab["spread_bps"] == 1.2


def test_hold_does_not_complete_group_alone():
    arts = [
        {"decision_group_id": "g", "symbol": "ETHUSDT", "action_available": 1, "decision_timestamp": DECISION.isoformat()},
        {"decision_group_id": "g", "symbol": "HOLD", "action_available": 1, "decision_timestamp": DECISION.isoformat()},
    ]
    assert required_actions(arts) == ["HOLD", "ETHUSDT"]
    labels = {("g", "HOLD"): {"label_valid": True, "label_source_version": LABEL_SOURCE_VERSION}}
    assert group_label_status(arts, labels, now=HORIZON + timedelta(hours=1)) == STATUS_PENDING_LABEL_SOURCE


def test_one_missing_coin_keeps_group_pending_and_ineligible_does_not_block():
    arts = [
        {"decision_group_id": "g", "symbol": "BTCUSDT", "action_available": 1},
        {"decision_group_id": "g", "symbol": "ETHUSDT", "action_available": 0},
        {"decision_group_id": "g", "symbol": "HOLD", "action_available": 1},
    ]
    assert required_actions(arts) == ["HOLD", "BTCUSDT"]
    labels = {
        ("g", "HOLD"): {"label_valid": True, "label_source_version": LABEL_SOURCE_VERSION},
        ("g", "BTCUSDT"): {"label_valid": False, "label_invalid_reason": INVALID_NO_BARS},
    }
    assert group_label_status(arts, labels, now=HORIZON + timedelta(hours=1)) == STATUS_PENDING_LABEL_SOURCE
    labels[("g", "BTCUSDT")] = {"label_valid": True, "label_source_version": LABEL_SOURCE_VERSION}
    assert group_label_status(arts, labels, now=HORIZON + timedelta(hours=1)) == STATUS_COMPLETE


def test_unavailable_and_immature_paths(tmp_path):
    early = build_v5_label(
        db_path=tmp_path / "x.db",
        decision_group_id="g",
        symbol="ETHUSDT",
        decision_ts=DECISION.isoformat(),
        action_available=True,
        entry_px=100.0,
        now=DECISION + timedelta(hours=1),
    )
    assert early["label_invalid_reason"] == INVALID_IMMATURE
    assert early["label_status"] == STATUS_PENDING_NOT_MATURE
    blocked = build_v5_label(
        db_path=tmp_path / "x.db",
        decision_group_id="g",
        symbol="ETHUSDT",
        decision_ts=DECISION.isoformat(),
        action_available=False,
        entry_px=100.0,
        now=HORIZON + timedelta(hours=1),
    )
    assert blocked["label_invalid_reason"] == INVALID_NOT_AVAILABLE


def _seed_group(db, gid="daygrp_1", *, available=True):
    ensure_artifact_schema(db)
    conn = sqlite3.connect(str(db))
    quote = json.dumps({"entry_px": 100.0, "mid": 100.0, "best_ask": 100.1, "spread_bps": 1.0})
    feats = dict.fromkeys(REQUIRED_CLOCK_V2_FIELDS_V5, 0.1)
    feats.update(
        {
            "spread_bps": 1.0,
            "commission_rt_bps": 4.0,
            "expected_slippage_bps": 1.4,
            "final_rank_score": 0.001,
            "legacy_path_ev": 0.001,
        }
    )
    hold_feats = {
        "symbol": "HOLD",
        "legacy_path_ev": 0.0,
        "final_rank_score": 0.0,
        "spread_bps": 0.0,
        "estimated_all_in_cost_bps": 0.0,
    }
    for sym in (*COINS, "HOLD"):
        payload = hold_feats if sym == "HOLD" else {**feats, "symbol": sym}
        feats_json = json.dumps(payload)
        conn.execute(
            f"""
            INSERT INTO {TABLE_ARTIFACT}(
                decision_group_id, symbol, created_at, decision_timestamp,
                feature_schema_version, feature_contract_version, eligible,
                feature_json, quote_json, action_available, clock_v2_partition
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                gid,
                sym,
                DECISION.isoformat(),
                DECISION.isoformat(),
                "v",
                "v",
                1,
                feats_json,
                quote,
                1 if (sym == "HOLD" or available) else 0,
                DEVELOPMENT,
            ),
        )
    conn.commit()
    conn.close()


def test_hold_valid_v1_does_not_skip_retry_and_batch_is_idempotent(tmp_path):
    db = tmp_path / "retry.db"
    _seed_group(db)
    conn = sqlite3.connect(str(db))
    hold = hold_label(decision_group_id="daygrp_1", decision_ts=DECISION.isoformat())
    hold["label_source_version"] = "day_clock_v2_target_3h_v1"
    persist_v5_labels(db, [hold])
    conn.execute(
        f"UPDATE {TABLE_V5_LABELS} SET label_source_version='legacy_v1' WHERE symbol='HOLD'"
    )
    conn.commit()
    conn.close()
    redis, rest = _matching_books()
    first = run_v5_label_batch(db, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=rest)
    assert first["groups_scanned"] == 1
    assert first["valid"] == 5
    presence = load_v5_label_presence(db)
    assert all(presence[("daygrp_1", sym)] for sym in (*COINS, "HOLD"))
    hist = sqlite3.connect(str(db)).execute(f"SELECT COUNT(*) FROM {TABLE_V5_LABELS_HISTORY}").fetchone()[0]
    assert hist >= 1
    second = run_v5_label_batch(db, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=rest)
    assert second["groups_scanned"] == 0
    rows = sqlite3.connect(str(db)).execute(f"SELECT COUNT(*) FROM {TABLE_V5_LABELS}").fetchone()[0]
    assert rows == 5


def test_comparability_and_calibration_count_active_version_only(tmp_path):
    db = tmp_path / "cmp.db"
    _seed_group(db, gid="daygrp_fill")
    redis, rest = _matching_books()
    run_v5_label_batch(db, now=HORIZON + timedelta(hours=1), redis_client=redis, rest_fetch=rest)
    arts = []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    for row in conn.execute(f"SELECT * FROM {TABLE_ARTIFACT}"):
        payload = dict(row)
        payload["features"] = json.loads(payload["feature_json"])
        payload["action_available"] = bool(payload["action_available"])
        arts.append(payload)
    presence = load_v5_label_presence(db)
    labels = {sym: bool(presence.get(("daygrp_fill", sym))) for sym in (*COINS, "HOLD")}
    cmp = group_comparability_v5(arts, labels_by_symbol=labels)
    assert cmp["FULLY_COMPARABLE"] is True
    assert "final_rank_score" not in v5_listed_features_from_blob(arts[0]["features"])


def test_obsolete_final_rank_blob_is_ignored_by_v5_export():
    blob = {"legacy_path_ev": 0.2, "final_rank_score": 0.2, "p_buy": 0.3, "ret_5m": 0.01}
    exported = v5_listed_features_from_blob(blob)
    assert exported["legacy_path_ev"] == 0.2
    assert "final_rank_score" not in exported
    snap_inputs = evaluate_clock_v2_v5_readiness.__doc__
    assert snap_inputs  # readiness exists and does not train
    from backend.services.day_path_clock_v2 import clock_v2_v5_readiness_requirements

    assert "final_rank_score" not in clock_v2_v5_readiness_requirements()["listed_inputs"]


def test_parse_redis_rejects_duplicates():
    open_ts = TARGET_OPEN
    rows = parse_redis_rows([_redis_row(open_ts), _redis_row(open_ts)])
    from backend.services.day_clock_v2_label_source import redis_history_ok

    assert redis_history_ok(rows) is False
