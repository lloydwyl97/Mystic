"""
Retry utilities with decorrelated jitter backoff for resilience.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any


async def with_jittered_backoff(
    op: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.25,
    max_delay: float = 3.0,
) -> Any:
    attempt = 0
    delay = max(0.0, float(base_delay))
    last_exc: Exception | None = None
    while attempt < max_attempts:
        try:
            return await op()
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            last_exc = e
            attempt += 1
            if attempt >= max_attempts:
                break
            delay = min(max_delay, random.uniform(base_delay, delay * 3.0))
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
