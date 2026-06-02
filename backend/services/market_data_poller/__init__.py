"""
Market Data Poller Module
========================

This module provides market data polling functionality for the Mystic AI Trading Platform.
"""

__version__ = "1.0.0"
__author__ = "Mystic AI Team"

from .middleware_router import MarketDataPoller

__all__ = ["MarketDataPoller"]
