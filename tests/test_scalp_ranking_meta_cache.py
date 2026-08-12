"""Item p22: last-ranking-meta-per-symbol cache used by the unified EV contract endpoint.

Also covers the p28 restart-verification fix: the scalp paper runner and the
API/uvicorn process are separate OS processes with no shared memory, so the
plain in-process `_LAST_RANKING_META_BY_SYMBOL` dict is invisible to the API
process in real deployment. `get_last_ranking_meta` must fall back to a
cross-process Redis snapshot for that case (see `_publish_ranking_meta_to_redis`
/ `ranking_meta_key`)."""

from __future__ import annotations

import sys
from pathlib import Path

import redis as redis_lib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp import scalp_strategy_router as ssr

_TEST_REDIS_URL = "redis://127.0.0.1:6379/0"
_TEST_PREFIX = "scalp_test_p28"


def setup_function(_fn):
    ssr._LAST_RANKING_META_BY_SYMBOL.clear()


def test_get_last_ranking_meta_none_when_unseen():
    assert ssr.get_last_ranking_meta("BTCUSDT") is None


def test_get_last_ranking_meta_returns_cached_row():
    ssr._LAST_RANKING_META_BY_SYMBOL["BTCUSDT"] = {
        "symbol": "BTCUSDT",
        "strategy_passed": True,
        "arm_penalty_mult": 0.9,
    }
    result = ssr.get_last_ranking_meta("BTCUSDT")
    assert result["strategy_passed"] is True
    assert result["arm_penalty_mult"] == 0.9


def test_get_last_ranking_meta_returns_a_copy_not_the_live_dict():
    ssr._LAST_RANKING_META_BY_SYMBOL["ETHUSDT"] = {"arm_penalty_mult": 1.0}
    result = ssr.get_last_ranking_meta("ETHUSDT")
    result["arm_penalty_mult"] = 0.1
    assert ssr._LAST_RANKING_META_BY_SYMBOL["ETHUSDT"]["arm_penalty_mult"] == 1.0


def _redis_reachable() -> bool:
    try:
        redis_lib.from_url(_TEST_REDIS_URL).ping()
        return True
    except Exception:
        return False


def test_publish_ranking_meta_to_redis_is_json_safe_and_readable():
    if not _redis_reachable():
        return
    client = redis_lib.from_url(_TEST_REDIS_URL, decode_responses=True)
    client.delete(f"{_TEST_PREFIX}:ranking_meta:BTCUSDT")
    row = {
        "symbol": "BTCUSDT",
        "rank_score": 0.42,
        "entry_eligible": True,
        "multi_horizon_ev": {"available": False, "composite_ev_pct": None},
    }
    ssr._publish_ranking_meta_to_redis("BTCUSDT", row, redis_url=_TEST_REDIS_URL, prefix=_TEST_PREFIX)
    raw = client.get(f"{_TEST_PREFIX}:ranking_meta:BTCUSDT")
    assert raw is not None
    import json

    parsed = json.loads(raw)
    assert parsed["rank_score"] == 0.42
    assert parsed["entry_eligible"] is True
    client.delete(f"{_TEST_PREFIX}:ranking_meta:BTCUSDT")


def test_get_last_ranking_meta_falls_back_to_redis_when_in_process_cache_is_empty():
    """Simulates the real cross-process scenario: a fresh reader (like the
    API/uvicorn process) with an empty in-process cache must still see the
    ranking row the (separate-process) scalp runner published to Redis."""
    if not _redis_reachable():
        return
    client = redis_lib.from_url(_TEST_REDIS_URL, decode_responses=True)
    key = f"{_TEST_PREFIX}:ranking_meta:SOLUSDT"
    client.delete(key)
    row = {"symbol": "SOLUSDT", "rank_score": 0.77, "entry_eligible": False}
    ssr._publish_ranking_meta_to_redis("SOLUSDT", row, redis_url=_TEST_REDIS_URL, prefix=_TEST_PREFIX)

    assert "SOLUSDT" not in ssr._LAST_RANKING_META_BY_SYMBOL  # proves this is the cross-process path
    result = ssr.get_last_ranking_meta("SOLUSDT", redis_url=_TEST_REDIS_URL, prefix=_TEST_PREFIX)
    assert result is not None
    assert result["rank_score"] == 0.77
    client.delete(key)


def test_get_last_ranking_meta_prefers_in_process_cache_over_redis():
    if not _redis_reachable():
        return
    client = redis_lib.from_url(_TEST_REDIS_URL, decode_responses=True)
    key = f"{_TEST_PREFIX}:ranking_meta:XRPUSDT"
    client.delete(key)
    ssr._publish_ranking_meta_to_redis("XRPUSDT", {"symbol": "XRPUSDT", "rank_score": 0.11}, redis_url=_TEST_REDIS_URL, prefix=_TEST_PREFIX)
    ssr._LAST_RANKING_META_BY_SYMBOL["XRPUSDT"] = {"symbol": "XRPUSDT", "rank_score": 0.99}

    result = ssr.get_last_ranking_meta("XRPUSDT", redis_url=_TEST_REDIS_URL, prefix=_TEST_PREFIX)
    assert result["rank_score"] == 0.99  # in-process wins, no Redis round-trip needed
    client.delete(key)
