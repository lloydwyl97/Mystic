"""
Binance.US spot universe eligibility — research only, no live promotion.

Scans exchangeInfo + ticker/24hr + bookTicker to build a liquid eligible universe.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE_URL = "https://api.binance.us"
MIN_DAILY_VOLUME_USD = 50_000.0
MAX_HALF_SPREAD_PCT = 0.0015
MIN_CANDLE_COVERAGE = 0.90
INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def _curl_json(url: str, timeout: int = 45) -> Any:
    proc = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def ccxt_symbol(api_sym: str) -> str:
    if api_sym.endswith("USDT"):
        return f"{api_sym[:-4]}/USDT"
    if api_sym.endswith("USD"):
        return f"{api_sym[:-3]}/USD"
    return api_sym


def api_symbol(ccxt_sym: str) -> str:
    return ccxt_sym.replace("/", "")


def fetch_exchange_info() -> list[dict]:
    data = _curl_json(f"{BASE_URL}/api/v3/exchangeInfo")
    if not isinstance(data, dict):
        return []
    return list(data.get("symbols") or [])


def fetch_all_24hr_tickers() -> list[dict]:
    data = _curl_json(f"{BASE_URL}/api/v3/ticker/24hr")
    return list(data) if isinstance(data, list) else []


def fetch_all_book_tickers() -> dict[str, dict]:
    data = _curl_json(f"{BASE_URL}/api/v3/ticker/bookTicker")
    out: dict[str, dict] = {}
    if isinstance(data, list):
        for row in data:
            sym = row.get("symbol")
            if sym:
                out[sym] = row
    return out


def half_spread_from_book(row: dict | None) -> float:
    if not row:
        return 0.001
    try:
        bid = float(row["bidPrice"])
        ask = float(row["askPrice"])
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 0.001
        return (ask - bid) / mid / 2.0
    except (KeyError, TypeError, ValueError):
        return 0.001


def fetch_klines(api_sym: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    sec = INTERVAL_SEC.get(interval, 3600)
    bars: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        url = f"{BASE_URL}/api/v3/klines?symbol={api_sym}&interval={interval}&startTime={cursor}&endTime={end_ms}&limit=1000"
        rows = _curl_json(url)
        if not isinstance(rows, list) or not rows:
            break
        for r in rows:
            bars.append(
                {
                    "ts": int(r[0]) // 1000,
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
            )
        last_ms = int(rows[-1][0])
        if last_ms <= cursor:
            break
        cursor = last_ms + sec * 1000
        time.sleep(0.04)
    return bars


def fetch_klines_cached(
    api_sym: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{api_sym}_{interval}_{start_ms}_{end_ms}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    bars = fetch_klines(api_sym, interval, start_ms, end_ms)
    if bars:
        cache_path.write_text(json.dumps(bars))
    return bars


def candle_coverage_1h(api_sym: str, days: int = 90, cache_dir: Path | None = None) -> float:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 2)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    if cache_dir:
        bars = fetch_klines_cached(api_sym, "1h", start_ms, end_ms, cache_dir)
    else:
        bars = fetch_klines(api_sym, "1h", start_ms, end_ms)
    expected = days * 24
    if expected <= 0:
        return 0.0
    return min(1.0, len(bars) / expected)


def scan_eligible_universe(
    *,
    min_daily_volume_usd: float = MIN_DAILY_VOLUME_USD,
    max_half_spread_pct: float = MAX_HALF_SPREAD_PCT,
    min_candle_coverage: float = MIN_CANDLE_COVERAGE,
    history_days: int = 90,
    check_candle_coverage: bool = True,
    cache_dir: Path | None = None,
    max_coverage_checks: int = 120,
) -> dict[str, Any]:
    symbols_info = fetch_exchange_info()
    tickers = {t["symbol"]: t for t in fetch_all_24hr_tickers()}
    books = fetch_all_book_tickers()

    scanned = 0
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    coverage_checks = 0

    for info in symbols_info:
        api_sym = info.get("symbol", "")
        scanned += 1
        quote = info.get("quoteAsset", "")
        status = info.get("status", "")
        reasons: list[str] = []

        if status != "TRADING":
            reasons.append("not_trading")
        if quote not in ("USDT", "USD"):
            reasons.append("quote_not_usd_usdt")
        if not info.get("isSpotTradingAllowed", True):
            reasons.append("spot_not_allowed")

        ticker = tickers.get(api_sym, {})
        try:
            vol_quote = float(ticker.get("quoteVolume") or 0)
            last_price = float(ticker.get("lastPrice") or ticker.get("weightedAvgPrice") or 0)
        except (TypeError, ValueError):
            vol_quote = 0.0
            last_price = 0.0

        if vol_quote < min_daily_volume_usd:
            reasons.append("low_volume")

        half_sp = half_spread_from_book(books.get(api_sym))
        if half_sp > max_half_spread_pct:
            reasons.append("spread_too_wide")

        coverage = 1.0
        if not reasons and check_candle_coverage and coverage_checks < max_coverage_checks:
            coverage_checks += 1
            coverage = candle_coverage_1h(api_sym, history_days, cache_dir)
            if coverage < min_candle_coverage:
                reasons.append("insufficient_candle_history")

        row = {
            "api_symbol": api_sym,
            "ccxt_symbol": ccxt_symbol(api_sym),
            "quote_asset": quote,
            "status": status,
            "daily_volume_usd": round(vol_quote, 2),
            "last_price": last_price,
            "half_spread_pct": round(half_sp, 6),
            "full_spread_pct": round(half_sp * 2, 6),
            "candle_coverage_1h_90d": round(coverage, 4),
        }

        if reasons:
            row["reject_reasons"] = reasons
            rejected.append(row)
        else:
            accepted.append(row)

    accepted.sort(key=lambda x: -float(x.get("daily_volume_usd") or 0))
    avg_spread = sum(float(a.get("half_spread_pct") or 0) for a in accepted) / max(len(accepted), 1)
    avg_vol = sum(float(a.get("daily_volume_usd") or 0) for a in accepted) / max(len(accepted), 1)
    avg_cov = sum(float(a.get("candle_coverage_1h_90d") or 0) for a in accepted) / max(len(accepted), 1)

    return {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "total_pairs_scanned": scanned,
        "pairs_accepted": len(accepted),
        "pairs_rejected": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "summary": {
            "avg_half_spread_pct": round(avg_spread, 6),
            "avg_daily_volume_usd": round(avg_vol, 2),
            "avg_candle_coverage": round(avg_cov, 4),
            "min_daily_volume_usd": min_daily_volume_usd,
            "max_half_spread_pct": max_half_spread_pct,
            "min_candle_coverage": min_candle_coverage,
        },
    }
