"""
Redis ai_signal:* TTL and executor stale-age alignment.

The signal generator refreshes each symbol once per loop; a full loop can exceed 120s wall
clock (many symbols x inference). The consumer's flush-on-unblock must not delete keys that
are still the authoritative feed and within Redis TTL.
"""

from __future__ import annotations

import os

# Written on every canonical ``ai_signal:<strategy_id>:<BUS>`` hash by RealTimeAISignalGenerator.
AI_SIGNAL_REDIS_TTL_SEC = int(os.getenv("AI_SIGNAL_REDIS_TTL_SEC", "300"))

# Stale flush / SIGNAL_STALE_SKIP / canonical executor visibility. Defaults to TTL so behavior
# matches key lifetime; override MAX_SIGNAL_AGE_SEC alone for stricter late-fill windows.
MAX_SIGNAL_AGE_SEC = int(os.getenv("MAX_SIGNAL_AGE_SEC", str(AI_SIGNAL_REDIS_TTL_SEC)))

# Written when portfolio integration enqueues a BUY after consuming ai_signal:* / hot / rule;
# deleted after bar close processing. Mirrors enqueued BUY state for post-consume
# visibility (Redis signal hash is intentionally removed on CANDIDATE_ADDED).
PE_BUY_CANDIDATE_KEY_PREFIX = "pe_buy_candidate:"


def pe_buy_candidate_redis_key(bus_symbol: str) -> str:
    return f"{PE_BUY_CANDIDATE_KEY_PREFIX}{str(bus_symbol).strip().upper()}"
