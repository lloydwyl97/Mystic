"""continuity_scan: one read, O(N) set checks, same report as the two-query version."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import and_, asc, create_engine, select, text
from sqlalchemy.orm import sessionmaker

from backend.config.canonical_candle_intervals import interval_ms
from backend.services import canonical_candle_store as store
from backend.services.canonical_candle_store import (
    api_symbol,
    expected_open_ms_range,
    is_aligned_open_ms,
    load_aligned_candles,
    open_dt_from_ms,
    open_ms_from_dt,
)
from backend.services.feature_store import FeatureOHLCV

INTERVALS = ("1m", "5m", "15m", "1h")
START = 1_700_006_400_000  # aligned to 1h


def _legacy_report(symbol: str, interval: str, *, start_ms: int, end_completed_ms: int) -> dict[str, Any]:
    """Pre-refactor continuity_report (two DB reads, list membership for ``extra``)."""
    db_sym = store.db_symbol(symbol)
    raw_open_ms: list[int] = []
    invalid_boundary = 0
    with store.SessionLocal() as session:
        conds = [
            FeatureOHLCV.symbol == db_sym,
            FeatureOHLCV.interval == interval,
            FeatureOHLCV.ts >= open_dt_from_ms(start_ms),
            FeatureOHLCV.ts <= open_dt_from_ms(end_completed_ms),
        ]
        stamps = session.execute(select(FeatureOHLCV.ts).where(and_(*conds)).order_by(asc(FeatureOHLCV.ts))).scalars().all()
    for ts in stamps:
        open_ms = open_ms_from_dt(ts)
        if open_ms is None or not is_aligned_open_ms(open_ms, interval):
            invalid_boundary += 1
            continue
        raw_open_ms.append(int(open_ms))
    rows = load_aligned_candles(symbol, interval, start_ms=start_ms, end_ms=end_completed_ms)
    have = [int(r["open_ms"]) for r in rows]
    if not raw_open_ms:
        raw_open_ms = list(have)
    expected = expected_open_ms_range(start_ms, end_completed_ms, interval)
    have_set = set(have)
    missing = [ts for ts in expected if ts not in have_set]
    extra = [ts for ts in have if ts not in set(expected)]
    duplicates = len(raw_open_ms) - len(set(raw_open_ms))
    out_of_order = sum(1 for i in range(1, len(have)) if have[i] < have[i - 1])
    latest = rows[-1] if rows else None
    return {
        "symbol": api_symbol(symbol),
        "interval": interval,
        "row_count": len(rows),
        "unique_count": len(have_set),
        "expected_count": len(expected),
        "expected_first_ms": expected[0] if expected else None,
        "expected_last_ms": expected[-1] if expected else None,
        "actual_first_ms": have[0] if have else None,
        "actual_last_ms": have[-1] if have else None,
        "missing_count": len(missing),
        "missing_timestamps": missing[:50],
        "duplicate_count": duplicates,
        "invalid_boundary_count": invalid_boundary,
        "out_of_order_count": out_of_order,
        "extra_unaligned_or_outside": extra[:20],
        "oldest_open_ms": have[0] if have else None,
        "newest_completed_ms": have[-1] if have else None,
        "latest_ohlcv": latest,
    }


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        # No unique index: legacy tables can hold duplicate open times.
        conn.execute(
            text("CREATE TABLE feature_ohlcv (id INTEGER PRIMARY KEY, symbol VARCHAR(32), interval VARCHAR(8), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, ts DATETIME NOT NULL)")
        )
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(store, "SessionLocal", session_factory)

    def insert(symbol: str, interval: str, open_ms_list: list[int]) -> None:
        with session_factory() as s:
            for i, ms in enumerate(open_ms_list):
                s.add(FeatureOHLCV(symbol=store.db_symbol(symbol), interval=interval, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i, volume=1.0 + i, ts=open_dt_from_ms(ms)))
            s.commit()

    return insert


def _scenario(name: str, interval: str) -> tuple[list[int], int]:
    step = interval_ms(interval)
    n = 120
    full = [START + i * step for i in range(n)]
    end = full[-1]
    if name == "complete":
        return full, end
    if name == "missing":
        return [ts for i, ts in enumerate(full) if i not in (3, 4, 50, 119)], end
    if name == "missing_over_50":
        return full[:10] + full[80:], end
    if name == "extra":
        # Unaligned bars inside the window plus bars after the completed end.
        return [*full, full[5] + 1_000, full[-1] + step, full[-1] + 2 * step], end
    if name == "duplicate":
        return [*full, full[7], full[7], full[30]], end
    if name == "out_of_order":
        return list(reversed(full[:60])) + full[60:], end
    raise AssertionError(name)


SCENARIOS = ("complete", "missing", "missing_over_50", "extra", "duplicate", "out_of_order")


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scan_matches_legacy_report(db, interval, scenario):
    stamps, end = _scenario(scenario, interval)
    db("BTCUSDT", interval, stamps)
    db("ETHUSDT", interval, [START])  # other symbols never leak in
    new_report, missing_all = store.continuity_scan("BTCUSDT", interval, start_ms=START, end_completed_ms=end)
    legacy = _legacy_report("BTCUSDT", interval, start_ms=START, end_completed_ms=end)
    assert new_report == legacy
    assert store.continuity_report("BTCUSDT", interval, start_ms=START, end_completed_ms=end) == legacy
    expected = expected_open_ms_range(START, end, interval)
    have = {int(r["open_ms"]) for r in load_aligned_candles("BTCUSDT", interval, start_ms=START, end_ms=end)}
    assert missing_all == [ts for ts in expected if ts not in have]
    assert len(missing_all) == legacy["missing_count"]


def test_gap_detection_still_reports_counts(db):
    stamps, end = _scenario("missing", "15m")
    db("SOLUSDT", "15m", [*stamps, stamps[0]])
    report, missing = store.continuity_scan("SOLUSDT", "15m", start_ms=START, end_completed_ms=end)
    step = interval_ms("15m")
    assert missing == [START + 3 * step, START + 4 * step, START + 50 * step, START + 119 * step]
    assert report["missing_count"] == 4
    assert report["duplicate_count"] == 1
    assert report["expected_count"] == 120


def test_scan_reads_the_window_once(db, monkeypatch):
    stamps, end = _scenario("complete", "1m")
    db("XRPUSDT", "1m", stamps)
    calls = {"load": 0}
    real = store.load_aligned_candles

    def counting(*a, **k):
        calls["load"] += 1
        return real(*a, **k)

    monkeypatch.setattr(store, "load_aligned_candles", counting)
    store.continuity_scan("XRPUSDT", "1m", start_ms=START, end_completed_ms=end)
    assert calls["load"] == 0


def test_expected_set_is_built_once(db, monkeypatch):
    stamps, end = _scenario("complete", "1m")
    db("BTCUSDT", "1m", stamps)
    built: list[int] = []
    real_set = set

    class _CountingSet(real_set):
        def __init__(self, *a):
            built.append(1)
            super().__init__(*a)

    monkeypatch.setitem(store.__dict__, "set", _CountingSet)
    store.continuity_scan("BTCUSDT", "1m", start_ms=START, end_completed_ms=end)
    # have_set, expected_set, set(raw_open_ms); not one per candle.
    assert len(built) <= 4


@pytest.mark.asyncio
async def test_repair_gaps_skips_identical_rescan_and_integrity_reuses_it(db, monkeypatch):
    from backend.services import canonical_candle_pipeline as cp

    stamps, end = _scenario("complete", "1m")
    db("BTCUSDT", "1m", stamps)
    pipe = cp.CanonicalCandlePipeline()
    scans = {"report": 0}
    real_report = cp.continuity_report

    def counting(*a, **k):
        scans["report"] += 1
        return real_report(*a, **k)

    monkeypatch.setattr(cp, "continuity_report", counting)
    result = await pipe.repair_gaps("BTCUSDT", "1m", START, end)
    assert scans["report"] == 0
    assert result["after"] == result["before"] == _legacy_report("BTCUSDT", "1m", start_ms=START, end_completed_ms=end)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(pipe, "_completed_open_ms", lambda _interval: end)
    monkeypatch.setattr(pipe, "publish_integrity", _noop, raising=False)
    try:
        await pipe.write_integrity("BTCUSDT", "1m", start_ms=START, repair=result)
    except Exception:
        pass
    assert scans["report"] == 0
    monkeypatch.setattr(pipe, "_completed_open_ms", lambda _interval: end + 60_000)
    try:
        await pipe.write_integrity("BTCUSDT", "1m", start_ms=START, repair=result)
    except Exception:
        pass
    assert scans["report"] == 1  # different window -> fresh scan


@pytest.mark.asyncio
async def test_repair_gaps_rescans_after_backfill(db, monkeypatch):
    from backend.services import canonical_candle_pipeline as cp

    stamps, end = _scenario("missing", "1m")
    db("BTCUSDT", "1m", stamps)
    pipe = cp.CanonicalCandlePipeline()
    fetched: list[tuple[int, int]] = []

    async def fake_backfill(symbol, interval, start_ms, end_ms):
        fetched.append((start_ms, end_ms))
        db(symbol, interval, list(range(start_ms, end_ms + 1, 60_000)))
        return {"fetched": (end_ms - start_ms) // 60_000 + 1}

    monkeypatch.setattr(pipe, "backfill_range", fake_backfill)
    result = await pipe.repair_gaps("BTCUSDT", "1m", START, end)
    assert result["before"]["missing_count"] == 4
    assert result["after"]["missing_count"] == 0
    assert fetched
