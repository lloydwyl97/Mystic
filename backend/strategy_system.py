#!/usr/bin/env python3
"""
Strategy System Module
Re-exports StrategyManager from endpoints for compatibility
"""

# Import the actual StrategyManager from the endpoints location
from backend.endpoints.strategies.strategy_manager import StrategyManager

# Re-export for compatibility with existing imports
__all__ = ["StrategyManager"]
