"""
Data Fetchers Package
Contains modules for fetching data from various sources.

Provides `DataFetcherManager` for legacy compatibility. The app factory
expects an object with an async `start_all()` method. We delegate to the
unified fetcher pipeline where applicable.
"""

from __future__ import annotations

import asyncio
from typing import Any


class DataFetcherManager:
    def __init__(self, shared_cache: Any | None = None) -> None:
        # Lazy: avoid importing heavy fetchers at import time
        self._shared_cache = shared_cache
        self._task: asyncio.Task[Any] | None = None

    async def start_all(self) -> None:
        # No-op on purpose: the app factory launches a comprehensive fetcher
        # background task separately. Keep this lightweight to avoid duplicates.
        await asyncio.sleep(0)
