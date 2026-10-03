"""REST pollers survive transient network failures and still stop on shutdown."""

from __future__ import annotations

import asyncio
import re
import socket
import subprocess
from pathlib import Path

import httpx
import pytest

from backend.services import live_market_data as lmd
from backend.services.canonical_candle_pipeline import CanonicalCandlePipeline

REPO = Path(__file__).resolve().parents[1]


def _ticker_payload() -> dict:
    return {
        "lastPrice": "100",
        "bidPrice": "99",
        "askPrice": "101",
        "highPrice": "110",
        "lowPrice": "90",
        "volume": "1",
        "quoteVolume": "100",
        "priceChange": "1",
        "priceChangePercent": "1",
        "closeTime": 1_700_000_000_000,
    }


def _kline_payload() -> list:
    return [[1_700_000_000_000, "1", "2", "0.5", "1.5", "10"]]


class _Response:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid/market")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("status", request=request, response=response)

    def json(self):
        return self._payload


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.invalid/market")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("status", request=request, response=response)


class _Limiter:
    async def consume(self, *_a, **_k):
        return None


class _ScriptedClient:
    def __init__(self, script: list):
        self.script = script

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def get(self, *_a, **_k):
        if not self.script:
            raise AssertionError("poll continued past the script")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(
        "backend.utils.binance_credentials.get_binance_us_credentials",
        lambda: ("key", "secret"),
    )
    svc = lmd.LiveMarketDataService()
    svc.binance = object()
    svc.watchlist_ccxt = ["BTC/USDT"]
    svc.ticker_interval = 0
    svc.ohlcv_interval = 0
    svc._running = True
    return svc


def _patch_sleep(monkeypatch, stop):
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *_a, **_k):
        recorded.append(float(delay))
        stop(recorded)
        await real_sleep(0)

    monkeypatch.setattr(lmd.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(lmd.random, "random", lambda: 0.0)
    return recorded


def _patch_client(monkeypatch, script: list):
    monkeypatch.setattr(lmd.httpx, "AsyncClient", lambda *_a, **_k: _ScriptedClient(script))


async def _run_ticker(service, monkeypatch, script: list, stop):
    _patch_client(monkeypatch, script)
    recorded = _patch_sleep(monkeypatch, stop)

    async def limiter():
        return _Limiter()

    monkeypatch.setattr(service, "_get_limiter", limiter)
    await service._ticker_loop()
    return recorded


async def _run_ohlcv(service, monkeypatch, script: list, stop):
    _patch_client(monkeypatch, script)
    recorded = _patch_sleep(monkeypatch, stop)
    persisted: list = []
    heartbeats: list[str] = []

    class _Guard:
        r = None

        async def mark_market_update(self, source):
            heartbeats.append(source)

    service._cache_guard = _Guard()

    async def persist(sym, rows):
        persisted.append((sym, list(rows)))
        return True

    async def limiter():
        return _Limiter()

    monkeypatch.setattr(service, "_persist_latest_1m_candle", persist)
    monkeypatch.setattr(service, "_get_limiter", limiter)
    await service._ohlcv_loop()
    return recorded, persisted, heartbeats


def _stop_when_cached(service, key):
    def stop(_recorded):
        cache = service._ticker_cache if key == "ticker" else service._ohlcv_cache
        if cache:
            service._running = False

    return stop


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timeout"),
        ConnectionResetError("reset"),
        socket.gaierror("dns"),
        _status_error(503),
        httpx.ConnectError("transport"),
    ],
)
@pytest.mark.asyncio
async def test_ticker_loop_recovers_from_transient_transport_errors(service, monkeypatch, exc):
    script = [exc, _Response(_ticker_payload())]
    await _run_ticker(service, monkeypatch, script, _stop_when_cached(service, "ticker"))
    cached = service._ticker_cache["BTC/USDT"]
    assert cached["last"] == 100.0
    assert service._ticker_cache_at["BTC/USDT"] > 0
    assert service._ticker_transport_backoff == 0.0
    assert script == []


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timeout"),
        ConnectionResetError("reset"),
        socket.gaierror("dns"),
        _status_error(503),
        httpx.ConnectError("transport"),
    ],
)
@pytest.mark.asyncio
async def test_ohlcv_loop_recovers_and_resumes_candles(service, monkeypatch, exc):
    script = [exc, _Response(_kline_payload())]
    _recorded, persisted, heartbeats = await _run_ohlcv(
        service,
        monkeypatch,
        script,
        _stop_when_cached(service, "ohlcv"),
    )
    assert service._ohlcv_cache["BTC/USDT"][0][4] == 1.5
    assert persisted and persisted[0][0] == "BTC/USDT"
    assert heartbeats == ["live_market_data"]
    assert service._ohlcv_transport_backoff == 0.0
    assert script == []


@pytest.mark.asyncio
async def test_repeated_ticker_failures_use_bounded_backoff_then_reset(service, monkeypatch, caplog):
    script = [
        httpx.ConnectError("secret-url-must-not-log"),
        ConnectionResetError("secret-url-must-not-log"),
        httpx.ReadTimeout("secret-url-must-not-log"),
        socket.gaierror("secret-url-must-not-log"),
        _status_error(500),
        httpx.ConnectError("secret-url-must-not-log"),
        httpx.ConnectError("secret-url-must-not-log"),
        _Response(_ticker_payload()),
        httpx.ConnectError("secret-url-must-not-log"),
    ]

    def stop(recorded):
        if service._ticker_cache and len(recorded) >= 10:
            service._running = False

    with caplog.at_level("WARNING"):
        recorded = await _run_ticker(service, monkeypatch, script, stop)
    pauses = recorded[1:]
    assert pauses[:7] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert pauses[7] == 0.0
    assert pauses[8] == 1.0
    assert service._ticker_transport_backoff == 1.0
    assert "secret-url-must-not-log" not in caplog.text
    assert "https://" not in caplog.text
    assert "LIVE_MD_TICKER_TRANSPORT" in caplog.text


@pytest.mark.asyncio
async def test_ticker_failure_does_not_kill_ohlcv_poller(service, monkeypatch):
    service.watchlist_ccxt = ["BTC/USDT"]

    class _Router:
        def __init__(self):
            self.ticker_errors = 0
            self.ohlcv_ok = 0

        def __call__(self, *_a, **_k):
            router = self

            class _Client:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_a):
                    return False

                async def get(self, url, *_a, **_k):
                    if "ticker/24hr" in url:
                        router.ticker_errors += 1
                        raise httpx.ConnectError("ticker-down")
                    router.ohlcv_ok += 1
                    return _Response(_kline_payload())

            return _Client()

    router = _Router()
    monkeypatch.setattr(lmd.httpx, "AsyncClient", router)
    monkeypatch.setattr(lmd.random, "random", lambda: 0.0)
    real_sleep = asyncio.sleep

    async def fake_sleep(_delay, *_a, **_k):
        if router.ticker_errors >= 1 and router.ohlcv_ok >= 1:
            service._running = False
        await real_sleep(0)

    monkeypatch.setattr(lmd.asyncio, "sleep", fake_sleep)

    async def limiter():
        return _Limiter()

    monkeypatch.setattr(service, "_get_limiter", limiter)
    persisted = []

    async def persist(sym, rows):
        persisted.append(sym)
        return True

    monkeypatch.setattr(service, "_persist_latest_1m_candle", persist)
    heartbeats = []

    class _Guard:
        r = None

        async def mark_market_update(self, source):
            heartbeats.append(source)

    service._cache_guard = _Guard()
    ticker_task = asyncio.create_task(service._ticker_loop())
    ohlcv_task = asyncio.create_task(service._ohlcv_loop())
    await asyncio.gather(ticker_task, ohlcv_task)
    assert ticker_task.exception() is None
    assert ohlcv_task.exception() is None
    assert router.ticker_errors >= 1
    assert "BTC/USDT" not in service._ticker_cache
    assert service._ohlcv_cache["BTC/USDT"]
    assert "BTC/USDT" in persisted
    assert "live_market_data" in heartbeats


@pytest.mark.asyncio
async def test_one_symbol_transport_failure_does_not_drop_the_batch(service, monkeypatch):
    service.watchlist_ccxt = ["BTC/USDT", "ETH/USDT"]
    service.ticker_interval = 60
    script = [
        httpx.ConnectError("btc-down"),
        _Response(_ticker_payload()),
    ]

    def stop(_recorded):
        if "ETH/USDT" in service._ticker_cache:
            service._running = False

    await _run_ticker(service, monkeypatch, script, stop)
    assert "ETH/USDT" in service._ticker_cache
    assert "BTC/USDT" not in service._ticker_cache
    assert service._ticker_transport_backoff == 0.0


@pytest.mark.asyncio
async def test_programmer_error_still_kills_the_ticker_loop(service, monkeypatch):
    script = [ZeroDivisionError("malformed-internal")]

    def stop(_recorded):
        return None

    with pytest.raises(ZeroDivisionError):
        await _run_ticker(service, monkeypatch, script, stop)


@pytest.mark.asyncio
async def test_client_error_status_is_not_retried(service, monkeypatch):
    script = [_status_error(400)]

    def stop(_recorded):
        return None

    with pytest.raises(httpx.HTTPStatusError):
        await _run_ticker(service, monkeypatch, script, stop)


@pytest.mark.asyncio
async def test_cancellation_exits_the_poller(service):
    service._running = True
    task = asyncio.create_task(service._ticker_loop())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stop_flag_exits_without_raising(service, monkeypatch):
    script = [httpx.ConnectError("down"), httpx.ConnectError("down")]

    def stop(recorded):
        if len(recorded) >= 2:
            service._running = False

    await _run_ticker(service, monkeypatch, script, stop)
    assert service._running is False


@pytest.mark.asyncio
async def test_gap_repair_loop_continues_after_a_transport_error(monkeypatch):
    stub = CanonicalCandlePipeline.__new__(CanonicalCandlePipeline)
    stub._running = True
    stub._hydrate_done = True
    stub.calls = 0

    def _completed(interval):
        return 60_000

    def _recent(interval):
        return 0

    stub._completed_open_ms = _completed
    stub._recent_window_start_ms = _recent

    async def repair_gaps(_symbol, _interval, start_ms, end_ms):
        stub.calls += 1
        if stub.calls == 1:
            raise httpx.ConnectError("gap-fetch-down")
        return {"repaired_fetched": 1, "start_ms": start_ms, "end_ms": end_ms}

    async def write_integrity(*_a, **_k):
        return {}

    stub.repair_gaps = repair_gaps
    stub.write_integrity = write_integrity
    long_sleeps = {"n": 0}
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *_a, **_k):
        if float(delay) >= 100:
            long_sleeps["n"] += 1
            if long_sleeps["n"] >= 2:
                stub._running = False
        await real_sleep(0)

    monkeypatch.setattr("backend.services.canonical_candle_pipeline.asyncio.sleep", fake_sleep)
    await stub._integrity_loop()
    assert stub.calls > 1
    assert long_sleeps["n"] >= 2


def test_restart_preserves_prior_log_and_stays_inside_retention(tmp_path):
    text = (REPO / "start_mystic.sh").read_text()
    match = re.search(r"preserve_log\(\) \{.*?\n\}", text, re.S)
    assert match
    for name in (
        "mystic_backend.log",
        "mystic_live_md.log",
        "mystic_signal.log",
        "mystic_portfolio.log",
        "mystic_learning.log",
        "mystic_ai_context.log",
    ):
        assert f"preserve_log /home/mystic/mystic/logs/{name}" in text
        assert f"> /home/mystic/mystic/logs/{name}" in text
    assert ">> /home/mystic/mystic/logs/mystic_portfolio.log" in text
    log = tmp_path / "mystic_backend.log"
    log.write_text("prior-audit-window\n")
    empty = tmp_path / "mystic_signal.log"
    empty.write_text("")
    subprocess.run(["bash", "-c", match.group(0) + '\npreserve_log "$1"\npreserve_log "$2"', "bash", str(log), str(empty)], check=True)
    kept = list(tmp_path.glob("mystic_backend.log.prerestart.*"))
    assert len(kept) == 1
    assert kept[0].read_text() == "prior-audit-window\n"
    assert empty.exists() and empty.read_text() == ""
    import fnmatch

    from backend.services.mystic_maintenance import LOG_SNAPSHOT_GLOBS

    assert any(fnmatch.fnmatch(kept[0].name, pattern) for pattern in LOG_SNAPSHOT_GLOBS)


def test_maintenance_entry_does_not_write_bytecode_before_importing_the_app():
    text = (REPO / "scripts" / "mystic_maintenance.py").read_text()
    assert text.index("sys.dont_write_bytecode = True") < text.index("from backend.services import mystic_maintenance")


def test_transient_classifier_rejects_cancellation_and_client_errors():
    assert lmd.is_transient_market_data_transport_error(httpx.ConnectError("x"))
    assert lmd.is_transient_market_data_transport_error(TimeoutError())
    assert lmd.is_transient_market_data_transport_error(_status_error(500))
    assert not lmd.is_transient_market_data_transport_error(_status_error(404))
    assert not lmd.is_transient_market_data_transport_error(asyncio.CancelledError())
    assert not lmd.is_transient_market_data_transport_error(ZeroDivisionError())
    assert lmd.next_market_data_transport_backoff(0) == 1.0
    assert lmd.next_market_data_transport_backoff(16) == 30.0
    assert lmd.next_market_data_transport_backoff(30) == 30.0
