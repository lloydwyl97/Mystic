"""
Regression/stress test (final pre-push audit item 6): concurrent reads of the
ai_signal:<strategy>:<SYMBOL> Redis hash must never see a missing
signal-content timestamp merely due to a transient/cold-start fetch race.

Real incident traced: 13 consecutive SIGNAL_CONTENT_TIMESTAMP_MISSING rejects
for one symbol immediately following a process restart, self-resolving ~13
minutes later with zero further occurrences. The writer's HMSET+EXPIRE
pipeline was confirmed atomic and successful every cycle throughout that
window (ruling out a partial/torn write), which is consistent with a cold
sync-Redis-client connection-pool establishment race in the consumer's fetch
path outlasting the previous single 50ms retry. This test drives concurrent
readers against a real local Redis instance under repeated writer churn
(mimicking the TTL-expire-and-rewrite cycle) and asserts zero missing
authentic timestamps once the bounded retry budget is applied.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

pytest.importorskip("redis")
import redis as redis_lib

from backend.services.live_strategy_contracts import redis_ai_signal_key


def _redis_available() -> bool:
    try:
        client = redis_lib.Redis(host="localhost", port=6379, socket_connect_timeout=1.0)
        return bool(client.ping())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="requires a local Redis instance")


@pytest.fixture()
def engine_with_real_redis(monkeypatch):
    import tempfile
    from pathlib import Path

    from backend.services.portfolio_engine import PortfolioEngine

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "sigfetch.db"
        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        engine._ensure_db_schema()

        real_client = redis_lib.Redis(host="localhost", port=6379, decode_responses=False)
        monkeypatch.setattr("backend.config.redis_config.get_redis_client", lambda: real_client)
        yield engine, real_client


def test_concurrent_reads_never_see_missing_timestamp_under_writer_churn(engine_with_real_redis):
    engine, client = engine_with_real_redis
    strategy_id = "day"
    symbol_bus = f"STRESSTEST{uuid.uuid4().hex[:8].upper()}"
    key = redis_ai_signal_key(strategy_id, symbol_bus)

    stop = threading.Event()
    write_errors: list[Exception] = []

    def _writer_loop() -> None:
        """Simulate the real writer: periodic DELETE (TTL-preserve path) followed
        shortly by a fresh atomic HMSET+EXPIRE — the exact race window a naive
        consumer fetch could observe as an empty hash."""
        try:
            while not stop.is_set():
                client.delete(key)
                time.sleep(0.01)
                mapping = {
                    "timestamp": str(time.time()),
                    "writer_timestamp": str(int(time.time())),
                    "content_fresh": "1",
                    "signal_content_stale": "0",
                    "feature_version": "5",
                }
                pipe = client.pipeline(transaction=True)
                pipe.hmset(key, mapping)
                pipe.expire(key, 30)
                pipe.execute()
                time.sleep(0.02)
        except Exception as exc:  # pragma: no cover - surfaced via write_errors
            write_errors.append(exc)

    writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    writer_thread.start()
    try:
        # Give the writer a moment to perform its first write before readers start,
        # matching the real scenario (readers arrive after the writer is already
        # cycling, not before any data has ever existed).
        time.sleep(0.05)

        results: list[dict[str, str]] = []
        results_lock = threading.Lock()

        def _reader() -> None:
            sig = engine._fetch_redis_ai_signal_string_map(strategy_id, symbol_bus)
            with results_lock:
                results.append(sig)

        reader_threads = [threading.Thread(target=_reader) for _ in range(24)]
        for t in reader_threads:
            t.start()
        for t in reader_threads:
            t.join(timeout=10)
    finally:
        stop.set()
        writer_thread.join(timeout=5)
        client.delete(key)

    assert not write_errors, f"writer thread errors: {write_errors}"
    assert len(results) == 24
    missing = [r for r in results if not r.get("timestamp")]
    assert not missing, f"{len(missing)}/24 concurrent reads got no signal-content timestamp under writer churn — the bounded retry budget must survive delete-then-rewrite race windows"


def test_fetch_survives_cold_connection_establishment_latency(monkeypatch):
    """
    Simulate a cold Redis client whose first call raises (connection not yet
    established) — the bounded multi-attempt backoff must still recover the
    real data on a later attempt within its budget, not give up after one try.
    """
    import tempfile
    from pathlib import Path

    from backend.services.portfolio_engine import PortfolioEngine

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "cold.db"
        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        engine._ensure_db_schema()

    call_count = {"n": 0}

    class _ColdThenReadyClient:
        def hgetall(self, _key):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("connection pool not yet established")
            return {b"timestamp": b"1234567890.0", b"content_fresh": b"1"}

    monkeypatch.setattr("backend.config.redis_config.get_redis_client", _ColdThenReadyClient)
    sig = engine._fetch_redis_ai_signal_string_map("day", "BTCUSDT")

    assert sig.get("timestamp") == "1234567890.0", "must recover once the connection pool comes up within the retry budget"
    assert call_count["n"] == 3
