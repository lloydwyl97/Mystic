"""
Canonical wiring for legacy 124-block “fundamental” slots (FEATURE_MAPPING 81-90, 0-based 80-89).

These are merged into the ``sentiment`` dict passed to ``build_feature_vector_124`` so live inference,
training collection, and persistence see the **same** population logic.

Priority (later steps override earlier only where appropriate — fear/greed stays owned by Redis/context):
  1. Per-coin env: ``MYSTIC_AI_<KEY>_<BASE>`` (e.g. MYSTIC_AI_MARKET_CAP_BTC)
  2. Global env: ``MYSTIC_AI_<KEY>``
  3. Redis: ``news_impact:{PAIR}``, ``social_sentiment:{BASE}`` (when upstream services populated them)
  4. **Inline live fetch** (when ``MYSTIC_AI_CANONICAL_INLINE_SENTIMENT`` is true, default): NewsAPI via
     ``crypto_news_articles`` (symbol query + **market-wide** NewsAPI fallback); Reddit hot titles + alias
     match + **global Reddit search** when OAuth is set. Empty APIs still leave slots unset (0).
  5. ``ai_context`` cross-fill: dominance from ``ctx_btc_dominance_proxy``; optional flow proxy for put/call
  6. CoinGecko public ``/coins/markets`` (cached, no API key on free tier)
  7. OHLCV-derived VIX proxy (ATR%) when ``vix`` still unset

No silent “fake” constants: keys are omitted until a real source supplies them (feature_builder leaves 0).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Base symbol -> CoinGecko id (DAY top-4 universe)
COINGECKO_ID_BY_BASE: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
}

_COINGECKO_CACHE_TS: float = 0.0
_COINGECKO_CACHE_ROWS: list[dict[str, Any]] | None = None
_COINGECKO_TTL_SEC = 120.0

_SENTIMENT_ENV_KEYS = (
    ("social_sentiment", "MYSTIC_AI_SOCIAL_SENTIMENT"),
    ("news_sentiment", "MYSTIC_AI_NEWS_SENTIMENT"),
    ("put_call_ratio", "MYSTIC_AI_PUT_CALL_RATIO"),
    ("vix", "MYSTIC_AI_VIX"),
    ("market_cap", "MYSTIC_AI_MARKET_CAP"),
    ("supply", "MYSTIC_AI_SUPPLY"),
    ("circulating_supply", "MYSTIC_AI_CIRCULATING_SUPPLY"),
    ("max_supply", "MYSTIC_AI_MAX_SUPPLY"),
    ("market_dominance", "MYSTIC_AI_MARKET_DOMINANCE"),
)


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _parse_env_float(key: str) -> float | None:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        x = float(raw)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _env_fundamentals(base_upper: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for fname, gkey in _SENTIMENT_ENV_KEYS:
        per = _parse_env_float(f"{gkey}_{base_upper}")
        if per is not None:
            out[fname] = per
            continue
        glob = _parse_env_float(gkey)
        if glob is not None:
            out[fname] = glob
    return out


def _vix_proxy_from_ohlcv(ohlcv: list[list] | None) -> float | None:
    if not ohlcv or len(ohlcv) < 16:
        return None
    try:
        import numpy as np

        arr = np.asarray(ohlcv, dtype=float)
        hi = arr[:, 2]
        lo = arr[:, 3]
        cl = arr[:, 4]
        tr = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
        if tr.size < 14:
            return None
        atr = float(np.mean(tr[-14:]))
        px = float(cl[-1])
        if px <= 0:
            return None
        atr_pct = atr / px
        # Map to a VIX-like scale ~10-80 for typical crypto intraday ATR%
        vix_like = float(min(90.0, max(8.0, atr_pct * 8500.0)))
        return vix_like
    except Exception:
        return None


async def _fetch_coingecko_markets_rows() -> list[dict[str, Any]]:
    global _COINGECKO_CACHE_TS, _COINGECKO_CACHE_ROWS
    now = time.monotonic()
    if _COINGECKO_CACHE_ROWS is not None and (now - _COINGECKO_CACHE_TS) < _COINGECKO_TTL_SEC:
        return _COINGECKO_CACHE_ROWS
    ids = ",".join(sorted(set(COINGECKO_ID_BY_BASE.values())))
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ids,
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            rows = data if isinstance(data, list) else []
    except Exception as e:
        logger.debug("CoinGecko markets fetch failed: %s", e)
        rows = []
    _COINGECKO_CACHE_ROWS = rows
    _COINGECKO_CACHE_TS = now
    return rows


def _coingecko_row_for_base(rows: list[dict[str, Any]], base_upper: str) -> dict[str, Any] | None:
    want = COINGECKO_ID_BY_BASE.get(base_upper)
    if not want:
        return None
    for row in rows:
        if str(row.get("id") or "") == want:
            return row
    return None


def _redis_float_from_news_impact_payload(jd: dict[str, Any]) -> float | None:
    for k in ("impact", "sentiment", "score", "polarity"):
        v = jd.get(k)
        if v is None:
            continue
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            continue
    return None


async def _redis_news_social(
    redis_client: Any | None,
    base_upper: str,
    pair_upper: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if redis_client is None:
        return out
    bu = (base_upper or "BTC").replace("USDT", "").strip().upper()
    pair_norm = (pair_upper or f"{bu}USDT").replace("/", "").strip().upper()
    if not pair_norm.endswith("USDT"):
        pair_norm = f"{pair_norm}USDT"
    news_keys = sorted({f"news_impact:{pair_norm}", f"news_impact:{bu}", f"news_impact:{bu}USDT"})
    for nk in news_keys:
        if "news_sentiment" in out:
            break
        try:
            raw_n = await redis_client.get(nk)
            if not raw_n:
                continue
            s = raw_n.decode() if isinstance(raw_n, bytes) else raw_n
            jd = json.loads(s)
            imp = _redis_float_from_news_impact_payload(jd)
            if imp is not None:
                out["news_sentiment"] = float(imp)
        except Exception as e:
            logger.debug("redis news_impact read failed key=%s: %s", nk, e)

    social_keys = sorted({f"social_sentiment:{bu}", f"social_sentiment:{bu}USDT"})
    for sk in social_keys:
        if "social_sentiment" in out:
            break
        try:
            raw_s = await redis_client.get(sk)
            if not raw_s:
                continue
            s = raw_s.decode() if isinstance(raw_s, bytes) else raw_s
            jd = json.loads(s)
            soc = jd.get("sentiment")
            if soc is None:
                soc = jd.get("polarity")
            if soc is not None:
                out["social_sentiment"] = float(soc)
        except Exception as e:
            logger.debug("redis social_sentiment read failed key=%s: %s", sk, e)

    ai_sent_key = f"ai_sentiment:{pair_norm}"
    try:
        h = await redis_client.hgetall(ai_sent_key)
        if h:

            def _hf(k: str) -> float | None:
                v = h.get(k) or h.get(k.encode() if isinstance(k, str) else k)
                if v is None:
                    return None
                s = v.decode() if isinstance(v, bytes) else str(v)
                if not s.strip():
                    return None
                try:
                    return float(s)
                except (TypeError, ValueError):
                    return None

            if "news_sentiment" not in out:
                ns = _hf("news_sentiment_score")
                if ns is not None:
                    out["news_sentiment"] = ns
            if "social_sentiment" not in out:
                ss = _hf("social_sentiment_score")
                if ss is not None:
                    out["social_sentiment"] = ss
            fg = _hf("fear_greed_index")
            if fg is not None:
                out["fear_greed_index"] = fg
    except Exception as e:
        logger.debug("redis ai_sentiment hash read failed key=%s: %s", ai_sent_key, e)

    return out


async def _write_canonical_sentiment_status(
    redis_client: Any | None,
    base_upper: str,
    pair_upper: str,
    payload: dict[str, Any],
) -> None:
    if redis_client is None:
        return
    bu = (base_upper or "BTC").replace("USDT", "").strip().upper()
    try:
        raw = json.dumps(payload, separators=(",", ":"))
        await redis_client.setex(f"mystic:canonical_sentiment_status:{bu}", 300, raw)
    except Exception as e:
        logger.debug("canonical sentiment status redis write failed: %s", e)


def _ctx_float(ctx: dict[str, str] | None, key: str) -> float | None:
    if not ctx:
        return None
    raw = ctx.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        x = float(raw)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


async def build_canonical_fundamental_sentiment(
    *,
    base_symbol: str,
    pair_symbol: str,
    ctx_for_overlay: dict[str, str] | None,
    redis_client: Any | None,
    ohlcv_1m: list[list] | None,
) -> dict[str, float]:
    """
    Build sentiment-map entries for social/news/options proxy/vix/market_cap/supply/dominance slots.
    Does **not** set fear_greed_index (owned by Redis ``ai_sentiment:fear_greed`` + ctx overlay).
    """
    base = (base_symbol or "BTC").replace("USDT", "").strip().upper()
    pair = (pair_symbol or f"{base}USDT").strip().upper()
    if not pair.endswith("USDT"):
        pair = f"{pair}USDT"

    merged: dict[str, float] = {}
    social_path = "unset"
    news_path = "unset"

    # 1) Explicit env (highest priority for manual overrides)
    env_f = _env_fundamentals(base)
    merged.update(env_f)
    if "social_sentiment" in env_f:
        social_path = "env_override"
    if "news_sentiment" in env_f:
        news_path = "env_override"

    # 2) Redis news/social when upstream writers populated keys (news_sentiment.py / market_intel)
    rs = await _redis_news_social(redis_client, base, pair)
    for k, v in rs.items():
        merged.setdefault(k, v)
    if "social_sentiment" in rs:
        social_path = "redis"
    if "news_sentiment" in rs:
        news_path = "redis"

    # 2b) News from collector cache only — never inline NewsAPI (single writer: ActiveSentimentCollector)
    if "news_sentiment" not in merged:
        try:
            from backend.services.news_sentiment import read_news_sentiment_from_collector

            ns, news_path = await read_news_sentiment_from_collector(redis_client, base)
            if ns is not None:
                merged["news_sentiment"] = float(ns)
            elif news_path == "news_unavailable_rate_limited":
                news_path = "news_unavailable_rate_limited"
            elif news_path == "news_ok_no_symbol_match":
                news_path = "news_ok_no_symbol_match"
            elif news_path == "news_cache_empty":
                news_path = "news_cache_empty"
        except Exception as e:
            logger.debug("read_news_sentiment_from_collector failed: %s", e)
            news_path = "news_collector_read_error"

    if _truthy(os.getenv("MYSTIC_AI_CANONICAL_INLINE_SENTIMENT", "true")):
        if "social_sentiment" not in merged:
            try:
                from backend.services.reddit_social_sentiment_live import fetch_reddit_social_polarity

                tup = await fetch_reddit_social_polarity(base)
                if tup is not None:
                    soc, n_posts = tup
                    merged.setdefault("social_sentiment", float(soc))
                    social_path = "reddit_inline"
                    if redis_client is not None:
                        try:
                            for sk in sorted({f"social_sentiment:{base}", f"social_sentiment:{base}USDT"}):
                                await redis_client.setex(
                                    sk,
                                    300,
                                    json.dumps(
                                        {
                                            "sentiment": float(soc),
                                            "posts": int(n_posts),
                                            "sources": "reddit_hot_canonical_inline",
                                        },
                                    ),
                                )
                        except Exception as e:
                            logger.debug("redis write social_sentiment inline failed: %s", e)
                else:
                    rid = (os.getenv("REDDIT_CLIENT_ID") or "").strip()
                    rsec = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
                    social_path = "reddit_oauth_missing" if (not rid or not rsec) else "reddit_no_matching_posts"
            except Exception as e:
                logger.debug("inline reddit social failed: %s", e)
                social_path = "reddit_inline_error"
        elif social_path == "unset":
            social_path = "already_resolved"

    # 3) ai_context cross-fill
    dom = _ctx_float(ctx_for_overlay, "ctx_btc_dominance_proxy")
    if dom is not None and "market_dominance" not in merged:
        # Same 0-1 scale as v2 ctx_btc_dominance_proxy (exchange-volume proxy, not CoinGecko global %).
        merged["market_dominance"] = float(min(1.0, max(0.0, dom)))

    # put_call_ratio: no synthetic depth proxy on crypto spot (audit marks UNSUPPORTED_FOR_SPOT).
    if _truthy(os.getenv("MYSTIC_AI_COINGECKO_ENABLE", "true")):
        rows = await _fetch_coingecko_markets_rows()
        row = _coingecko_row_for_base(rows, base)
        if row:
            mc = row.get("market_cap")
            if mc is not None and "market_cap" not in merged:
                merged["market_cap"] = float(mc)
            circ = row.get("circulating_supply")
            if circ is not None and "circulating_supply" not in merged:
                merged["circulating_supply"] = float(circ)
            total = row.get("total_supply")
            if total is not None and "supply" not in merged:
                merged["supply"] = float(total)
            mx = row.get("max_supply")
            if mx is not None and "max_supply" not in merged:
                merged["max_supply"] = float(mx)

            # Effective max supply: CoinGecko often returns 0 / null for unlimited or stale mint metadata.
            # Canonical policy: at minimum the circulating float (structural floor), never a meaningless 0 cap.
            circ_v = merged.get("circulating_supply")
            max_v = merged.get("max_supply")
            if circ_v is not None:
                cf = float(circ_v)
                mf = float(max_v) if max_v is not None else 0.0
                if mf <= 0 or mf < cf:
                    merged["max_supply"] = cf

    # 5) VIX proxy from intraday ATR if nothing else set
    if "vix" not in merged:
        vx = _vix_proxy_from_ohlcv(ohlcv_1m)
        if vx is not None:
            merged["vix"] = vx

    if "social_sentiment" not in merged and social_path == "unset":
        social_path = "unresolved_no_source"
    if "news_sentiment" not in merged and news_path == "unset":
        news_path = "unresolved_no_source"

    await _write_canonical_sentiment_status(
        redis_client,
        base,
        pair,
        {
            "ts_monotonic": time.monotonic(),
            "base": base,
            "pair": pair,
            "social_sentiment_value": merged.get("social_sentiment"),
            "news_sentiment_value": merged.get("news_sentiment"),
            "social_path": social_path,
            "news_path": news_path,
            "social_resolved": "social_sentiment" in merged,
            "news_resolved": "news_sentiment" in merged,
        },
    )

    return merged


async def merge_canonical_sentiment_payload(
    *,
    base_symbol: str,
    pair_symbol: str,
    ctx_for_overlay: dict[str, str] | None,
    redis_client: Any | None,
    ohlcv_1m: list[list] | None,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge fundamental map into an existing sentiment dict (preserves fear_greed_index from ``existing``)."""
    fund = await build_canonical_fundamental_sentiment(
        base_symbol=base_symbol,
        pair_symbol=pair_symbol,
        ctx_for_overlay=ctx_for_overlay,
        redis_client=redis_client,
        ohlcv_1m=ohlcv_1m,
    )
    out: dict[str, Any] = dict(existing or {})
    for k, v in fund.items():
        out.setdefault(k, v)
    return out


__all__ = [
    "COINGECKO_ID_BY_BASE",
    "build_canonical_fundamental_sentiment",
    "merge_canonical_sentiment_payload",
]
