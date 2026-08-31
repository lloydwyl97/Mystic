"""Compact tests for signal timestamp/freshness parsing hygiene (data only, no strategy changes)."""

import time
from datetime import datetime, timezone

from backend.services.ai_entry_context_gate import (
    _coerce_to_epoch,
    evaluate_signal_hash_for_entry,
)


def test_coerce_iso_with_tz():
    ts = _coerce_to_epoch("2026-07-05T17:00:00+00:00")
    # Must be a plausible epoch for mid-2026 (around 1.78e9)
    assert ts is not None and 1.7e9 < ts < 1.9e9


def test_coerce_iso_z():
    ts = _coerce_to_epoch("2026-07-05T17:00:00Z")
    assert ts is not None


def test_coerce_epoch_seconds():
    now = time.time()
    assert abs(_coerce_to_epoch(str(now)) - now) < 0.1


def test_coerce_epoch_ms():
    now_ms = int(time.time() * 1000)
    ts = _coerce_to_epoch(str(now_ms))
    assert ts is not None
    assert abs(ts - (now_ms / 1000.0)) < 1


def test_coerce_nested_explain_like():
    # simulate pulling from explain json path that may carry ISO
    ts = _coerce_to_epoch("2026-07-05T12:00:00+00:00")
    assert ts is not None


def test_fallback_to_age_when_no_ts():
    now_iso = datetime.now(timezone.utc).isoformat()
    dd = {
        "content_fresh": "1",
        "signal_content_age_sec": "10",
        "context_fresh": "1",
        "ctx_age_sec": "5",
        "ctx_ts_utc": now_iso,
        "feature_version": "1",
    }
    ok, reason, _detail = evaluate_signal_hash_for_entry(dd)
    # age 10s is well under max; must not fail closed on timestamp missing/parse
    assert reason not in ("SIGNAL_CONTENT_TIMESTAMP_MISSING", "SIGNAL_CONTENT_TIMESTAMP_PARSE")
    assert ok is True


def test_fallback_to_zero_content_age_when_no_ts():
    """Live Redis hashes often emit content_age_sec=0; that must not mean MISSING."""
    now_iso = datetime.now(timezone.utc).isoformat()
    dd = {
        "content_fresh": "1",
        "content_age_sec": "0",
        "context_fresh": "1",
        "ctx_age_sec": "5",
        "ctx_ts_utc": now_iso,
        "feature_version": "1",
    }
    ok, reason, _detail = evaluate_signal_hash_for_entry(dd)
    assert reason != "SIGNAL_CONTENT_TIMESTAMP_MISSING"
    assert ok is True


def test_missing_ts_gives_missing():
    dd = {"content_fresh": "1"}  # no timestamp and no usable age
    ok, reason, _detail = evaluate_signal_hash_for_entry(dd)
    assert ok is False
    assert reason in {"SIGNAL_CONTENT_TIMESTAMP_MISSING", "SIGNAL_CONTENT_TIMESTAMP_PARSE"}


def test_malformed_ts_gives_parse():
    dd = {"timestamp": "not-a-time", "content_fresh": "1"}
    ok, reason, _detail = evaluate_signal_hash_for_entry(dd)
    assert ok is False
    assert reason in ("SIGNAL_CONTENT_TIMESTAMP_PARSE", "SIGNAL_CONTENT_TIMESTAMP_MISSING")


def test_real_stale_still_rejected():
    # very old epoch
    old = str(time.time() - 100000)
    dd = {"timestamp": old, "content_fresh": "1", "signal_content_stale": "0"}
    ok, reason, _detail = evaluate_signal_hash_for_entry(dd)
    # Should hit AGE_EXCEEDED (or STALE if flag), not let it pass
    assert ok is False
    assert reason in ("SIGNAL_CONTENT_AGE_EXCEEDED", "SIGNAL_CONTENT_STALE")
