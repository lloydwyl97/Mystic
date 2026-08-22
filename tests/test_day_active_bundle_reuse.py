"""DAY active bundle must keep complete prior TFs when a live fetch returns empty."""

from __future__ import annotations

import pytest

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, min_bars_for_day_tf
from backend.services.day_active_market_bundle import (
    _fetch_day_active_ohlcv_bundle_raw,
    validate_day_active_bundle,
)


def _rows(n: int) -> list[list]:
    return [[i, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(n)]


class _Svc:
    def __init__(self, live: dict[str, list]):
        self.live = live

    async def get_ohlcv(self, symbol, tf, lim):
        return list(self.live.get(tf) or [])


@pytest.mark.asyncio
async def test_empty_live_fetch_reuses_complete_prior_tf():
    prior = {tf: _rows(min_bars_for_day_tf(tf) + 10) for tf in DAY_ACTIVE_TIMEFRAMES}
    prior_ts = {tf: 1.0 for tf in DAY_ACTIVE_TIMEFRAMES}
    svc = _Svc({"1m": [], "5m": [], "15m": [], "30m": [], "1h": prior["1h"], "4h": prior["4h"], "8h": prior["8h"], "12h": prior["12h"], "1d": prior["1d"], "1w": prior["1w"]})
    bundle, _ = await _fetch_day_active_ohlcv_bundle_raw(
        svc,
        "BTC/USDT",
        prior_bundle=prior,
        prior_tf_fetched_at=prior_ts,
    )
    ok, miss = validate_day_active_bundle(bundle)
    assert ok, miss
    assert len(bundle["1m"]) >= min_bars_for_day_tf("1m")
    assert len(bundle["5m"]) >= min_bars_for_day_tf("5m")
    assert len(bundle["15m"]) >= min_bars_for_day_tf("15m")
    assert len(bundle["30m"]) >= min_bars_for_day_tf("30m")


@pytest.mark.asyncio
async def test_truly_missing_tf_still_fails_contract():
    prior = {tf: [] for tf in DAY_ACTIVE_TIMEFRAMES}
    svc = _Svc({tf: [] for tf in DAY_ACTIVE_TIMEFRAMES})
    bundle, _ = await _fetch_day_active_ohlcv_bundle_raw(
        svc,
        "BTC/USDT",
        prior_bundle=prior,
        prior_tf_fetched_at={},
    )
    ok, miss = validate_day_active_bundle(bundle)
    assert ok is False
    assert any("missing_tf:1m" in m for m in miss)
