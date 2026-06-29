"""
DEPRECATED — SQLite OHLCV persist + market_data:last_update heartbeat now run inside
LiveMarketDataService (start_live_market_data.py / mystic-market-data.service).

This file remains as a stub so old scripts fail loudly instead of duplicating Binance calls.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.error("live_data_collector.py is deprecated. Use start_live_market_data.py or systemctl --user start mystic-market-data.service")
    sys.exit(0)
