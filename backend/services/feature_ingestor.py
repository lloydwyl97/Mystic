"""
Feature Ingestor
----------------
Writes live ticker snapshots and periodic OHLCV into the feature store.

Design goals:
- Async, side-effect-contained loops (ticker + ohlcv) with stable cadence
- Minimal external deps, no REST fallbacks here; relies on provided services
- Resilient to per-symbol errors (continues ingesting others)
- Configurable via env vars; sensible defaults
- Lightweight metrics with safe fallbacks (no-ops if metrics module unavailable)
- Status/introspection helpers and idempotent start/stop
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timezone
from typing import Any

from backend.services.feature_store import init_feature_store, insert_ohlcv, insert_tick
from backend.services.live_market_data import live_market_data_service

# Optional imports - try at top level
try:
    from backend.services.task_manager import create_tracked_task
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    create_tracked_task = None

# ----- metrics (safe fallbacks) -----
try:
    from backend.metrics import Counter, Histogram  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover

    class Counter:  # type: ignore[misc]
        def __init__(self, *_: Any, **__: Any) -> None: ...

        def labels(self, *_: Any, **__: Any) -> Counter:
            return self

        def inc(self, *_: Any, **__: Any) -> None: ...

    class Histogram:  # type: ignore[misc]
        def __init__(self, *_: Any, **__: Any) -> None: ...

        def labels(self, *_: Any, **__: Any) -> Histogram:
            return self

        def observe(self, *_: Any, **__: Any) -> None: ...


# Optional legacy aggregate metric (kept for compatibility with existing dashboards)
try:
    from backend.metrics import metrics  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    metrics = None  # type: ignore[assignment]

# Concrete metrics for this ingestor (namespaced and labeled)
feat_ticks_total = Counter("feature_ingest_ticks_total", "Ticker ingested rows", ["symbol"])  # type: ignore[arg-type]
feat_ohlcv_total = Counter("feature_ingest_ohlcv_total", "OHLCV ingested rows", ["symbol", "interval"])  # type: ignore[arg-type]
feat_errors_total = Counter("feature_ingest_errors_total", "Ingest errors", ["stage", "symbol"])  # type: ignore[arg-type]
feat_loop_latency = Histogram("feature_ingest_loop_seconds", "Ingest loop cycle duration", ["type"])  # type: ignore[arg-type]

# ----- helpers -----


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _parse_watchlist() -> list[str]:
    """
    Resolve watchlist from env FEATURE_INGESTOR_WATCHLIST or live_market_data_service.watchlist_human,
    else use the top-4 Mystic day-trade scope.
    Accepts comma-separated variants like: BTC-USDT,ETH-USDT or BTC-USD,ETH-USD or BTCUSDT,ETHUSDT.
    """
    raw = os.getenv("FEATURE_INGESTOR_WATCHLIST", "").strip()
    items = [s.strip() for s in raw.split(",") if s.strip()] if raw else getattr(live_market_data_service, "watchlist_human", []) or ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"]
    return [_normalize_symbol(s) for s in items]


def _normalize_symbol(sym: str) -> str:
    """
    Normalize symbols to human 'BASE-USDT' format (used by live_market_data_service in many code paths).
    Accepts: 'BTC-USDT', 'BTC-USD', 'BTCUSDT', 'BTC/USD', 'btc usdt'
    Returns: 'BTC-USDT'
    """
    s = (sym or "").strip().upper().replace(" ", "")
    if "/" in s:
        base, quote = s.split("/", 1)
        quote = "USDT" if quote in ("USDT", "USD") else quote
        return f"{base}-USDT"
    if "-" in s:
        base, quote = s.split("-", 1)
        quote = "USDT" if quote in ("USDT", "USD") else quote
        return f"{base}-USDT"
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USDT"
    if s.endswith("USD"):
        base = s[:-3]
        return f"{base}-USDT"
    # Default to USDT quote
    return f"{s}-USDT" if s else s


class FeatureIngestor:
    def __init__(self) -> None:
        self._running = False
        self._tasks: list[asyncio.Task] = []
        # Cadence (seconds) - OPTIMIZED for 90% weight utilization
        self.ticker_interval_s: float = _env_float("FEATURE_INGESTOR_TICKER_INTERVAL_S", 10.0)  # 60 weight/min
        self.ohlcv_interval_s: float = _env_float("FEATURE_INGESTOR_OHLCV_INTERVAL_S", 10.0)  # 60 weight/min (was 60s)
        # Watchlist
        self.watchlist: list[str] = _parse_watchlist()
        # Status
        self._last_ticker_run: str | None = None
        self._last_ohlcv_run: str | None = None
        self._last_error: str | None = None

        # Initialize feature store once on construct (safe if already initialized)
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Defer to start() if init failed here
            init_feature_store()

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._running:
            return
        # Ensure feature store is ready
        try:
            init_feature_store()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self._last_error = f"init_feature_store failed: {e}"
        # Refresh watchlist each start so env overrides take effect
        self.watchlist = _parse_watchlist()
        self._running = True
        if create_tracked_task is None:
            msg = "create_tracked_task not available"
            raise RuntimeError(msg)

        self._tasks = [
            await create_tracked_task(self._ticker_loop(), "feature_ingestor_ticker"),
            await create_tracked_task(self._ohlcv_loop(), "feature_ingestor_ohlcv"),
        ]

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for t in self._tasks:
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                t.cancel()
        self._tasks.clear()

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    # ---- control / configuration ----

    def set_watchlist(self, symbols: list[str]) -> None:
        self.watchlist = [_normalize_symbol(s) for s in symbols or []]

    def set_intervals(self, ticker_s: float | None = None, ohlcv_s: float | None = None) -> None:
        if ticker_s is not None:
            self.ticker_interval_s = float(max(0.1, ticker_s))
        if ohlcv_s is not None:
            self.ohlcv_interval_s = float(max(1.0, ohlcv_s))

    # ---- one-shot helpers (useful for tests) ----

    async def run_ticker_pass(self) -> dict[str, Any]:
        """
        Execute a single ticker ingestion pass across the current watchlist.
        Returns a summary with per-symbol results.
        """
        results: dict[str, Any] = {"items": [], "ts": _now_iso()}
        for sym in list(self.watchlist):
            try:
                data = await live_market_data_service.get_ticker(sym)
                if isinstance(data, dict):
                    insert_tick(sym, data)
                    try:
                        feat_ticks_total.labels(symbol=sym).inc()
                        if metrics and getattr(metrics, "feature_ingest", None):
                            metrics.feature_ingest.inc()  # type: ignore[attr-defined]
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                    results["items"].append({"symbol": sym, "status": "ok"})
                else:
                    results["items"].append({"symbol": sym, "status": "skip"})
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    feat_errors_total.labels(stage="ticker", symbol=sym).inc()
                results["items"].append({"symbol": sym, "status": "error", "error": str(e)})
                self._last_error = f"ticker:{sym}:{e}"
        self._last_ticker_run = _now_iso()
        return results

    async def run_ohlcv_pass(self, interval: str = "1m") -> dict[str, Any]:
        """
        Execute a single OHLCV ingestion pass (latest candle only) across the current watchlist.
        """
        results: dict[str, Any] = {"items": [], "ts": _now_iso(), "interval": interval}
        for sym in list(self.watchlist):
            try:
                candles = await live_market_data_service.get_ohlcv(sym, interval, limit=1)
                if isinstance(candles, list) and candles:
                    c = candles[-1]
                    candle = {
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    }
                    insert_ohlcv(sym, interval, candle)
                    try:
                        feat_ohlcv_total.labels(symbol=sym, interval=interval).inc()
                        if metrics and getattr(metrics, "feature_ingest", None):
                            metrics.feature_ingest.inc()  # type: ignore[attr-defined]
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                    results["items"].append({"symbol": sym, "status": "ok"})
                else:
                    results["items"].append({"symbol": sym, "status": "skip"})
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    feat_errors_total.labels(stage="ohlcv", symbol=sym).inc()
                results["items"].append({"symbol": sym, "status": "error", "error": str(e)})
                self._last_error = f"ohlcv:{sym}:{e}"
        self._last_ohlcv_run = _now_iso()
        return results

    # ---- background loops ----

    async def _ticker_loop(self) -> None:
        """
        Periodic ticker ingestion loop with stable cadence.
        """
        while self._running:
            start = _now_utc()
            try:
                await self.run_ticker_pass()
            except asyncio.CancelledError:
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self._last_error = f"ticker_loop:{e}"
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    feat_errors_total.labels(stage="ticker_loop", symbol="-").inc()
            finally:
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    feat_loop_latency.labels(type="ticker").observe(max(0.0, elapsed))
                # Maintain stable cadence
                sleep_for = max(0.0, float(self.ticker_interval_s) - elapsed)
                try:
                    await asyncio.sleep(sleep_for)
                except asyncio.CancelledError:
                    break

    async def _ohlcv_loop(self) -> None:
        """
        Periodic OHLCV ingestion loop (latest 1m candle).
        """
        interval = "1m"
        while self._running:
            start = _now_utc()
            try:
                await self.run_ohlcv_pass(interval=interval)
            except asyncio.CancelledError:
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self._last_error = f"ohlcv_loop:{e}"
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    feat_errors_total.labels(stage="ohlcv_loop", symbol="-").inc()
            finally:
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    feat_loop_latency.labels(type="ohlcv").observe(max(0.0, elapsed))
                # Maintain stable cadence
                sleep_for = max(0.0, float(self.ohlcv_interval_s) - elapsed)
                try:
                    await asyncio.sleep(sleep_for)
                except asyncio.CancelledError:
                    break

    # ---- status / introspection ----

    def get_status(self) -> dict[str, Any]:
        return {
            "running": bool(self._running),
            "watchlist": list(self.watchlist),
            "ticker_interval_s": float(self.ticker_interval_s),
            "ohlcv_interval_s": float(self.ohlcv_interval_s),
            "last_ticker_run": self._last_ticker_run,
            "last_ohlcv_run": self._last_ohlcv_run,
            "last_error": self._last_error,
            "timestamp": _now_iso(),
        }


# Global instance + accessor
feature_ingestor = FeatureIngestor()


def get_feature_ingestor() -> FeatureIngestor:
    return feature_ingestor
