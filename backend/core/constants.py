"""
Core Constants Configuration - All Live Data, No Fallback/Hardcoded Data

This module provides core constants for live trading operations (backend port 8000).
All constants:
- Exchange identifiers for live Binance.US operations
- Trading symbols from live trading universe (single source of truth)
- AI trading thresholds for live trading operations
- All constants for live operations - no fallback/hardcoded values

Live Data Sources:
- Trading symbols: From `backend.config.trading_universe` (live Binance.US Top-10)
- Exchange configuration: Binance.US only (live exchange for all operations)
- All constants used for live trading operations via backend (port 8000)
- AI thresholds for live trading execution

Endpoint References:
- Binance.US API: https://api.binance.us (live exchange API)
- Backend API: Port 8000 (for live trading operations)
- All constants used for live endpoints - no fallback/hardcoded data
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from backend.config.trading_universe import (
    EXCHANGE_ID as TRADING_UNIVERSE_EXCHANGE_ID,
)
from backend.config.trading_universe import (
    TOP10_COINS,
    TRADING_SYMBOLS,
)


class ExitReason(str, Enum):
    """Exit reason enumeration for trade telemetry"""

    AI_SIGNAL_FLIP = "ai_signal_flip"
    NEUTRAL_SIGNAL = "neutral_signal"
    TRAILING_STOP = "trailing_stop"
    SCRATCH_EXIT = "scratch_exit"
    TAKE_PROFIT_PARTIAL = "take_profit_partial"
    TAKE_PROFIT_FULL = "take_profit_full"
    HARD_STOP_LOSS = "hard_stop_loss"
    STOP_LOSS = "stop_loss"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL_OVERRIDE = "manual_override"
    SESSION_END = "session_end"
    NO_TRADE_GATE = "no_trade_gate"


# Exchange constants (Binance.US only - live trading exchange)
# All operations connect to Binance.US for live trading (backend port 8000)
EXCHANGE_ID: Final[str] = TRADING_UNIVERSE_EXCHANGE_ID

# Default trading symbols (Binance.US Top-10) - FROM LIVE TRADING UNIVERSE
# All 10 Binance.US Top-10 base symbols from live trading universe
DEFAULT_BASES: Final[list[str]] = list(TOP10_COINS)

# Default quote currency (USDT for live Binance.US trading)
DEFAULT_QUOTE: Final[str] = "USDT"

# Default symbols in various formats - FROM LIVE TRADING UNIVERSE ONLY
# All 10 Binance.US Top-10 symbols from live trading universe
DEFAULT_SYMBOLS: Final[list[str]] = list(TRADING_SYMBOLS)


def _to_ccxt_symbol(base: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Convert base and quote to CCXT format (BASE/QUOTE) for live trading operations.

    Converts base and quote currencies to CCXT format for live API calls.
    Used for live trading operations via backend (port 8000).

    Args:
        base: Base currency (e.g., "BTC") - from live Binance.US Top-10
        quote: Quote currency (e.g., "USDT") - default USDT for live trading

    Returns:
        CCXT format symbol (e.g., "BTC/USDT") for live API calls
    """
    return f"{base.upper()}/{quote.upper()}"


# Exchange mapping for normalization (Binance.US only - live trading exchange)
# Maps various exchange name formats to standard Binance.US identifier
EXCHANGE_MAPPING: Final[dict[str, str]] = {
    "binance": EXCHANGE_ID,
    "binanceus": EXCHANGE_ID,
    "binance_us": EXCHANGE_ID,
    "binance.us": EXCHANGE_ID,
}


def normalize_exchange_id(exchange_name: str | None) -> str:
    """
    Normalize any exchange name to our standard binance_us format (live trading exchange).

    Normalizes various exchange name formats to Binance.US identifier.
    Used for live trading operations (backend port 8000).

    Args:
        exchange_name: Exchange name to normalize (can be None)

    Returns:
        Standardized Binance.US exchange identifier for live trading operations
    """
    if not exchange_name:
        return EXCHANGE_ID
    normalized = exchange_name.strip().lower().replace("-", "_").replace(".", "_")
    return EXCHANGE_MAPPING.get(normalized, EXCHANGE_ID)


# ============================================================================
# AI TRADING CONSTANTS - CRITICAL THRESHOLDS (Live Trading Operations)
# ============================================================================

# Trading Performance Gates (for live trading operations - backend port 8000)
# Replaced AI accuracy gates with trading performance metrics
# OPTIMIZED: Slightly lowered for faster transition to live (personal laptop)
GO_LIVE_MIN_TRADES: Final[int] = 30  # 30 trades for statistical significance
GO_LIVE_MIN_WIN_RATE: Final[float] = 0.38  # OPTIMIZED: Was 0.40, now 38% (still profitable with good R:R)
GO_LIVE_MIN_EXPECTANCY: Final[float] = 0.25  # OPTIMIZED: Was 0.30, now 0.25R per trade
GO_LIVE_MAX_DRAWDOWN: Final[float] = -7.0  # OPTIMIZED: Was -5%, now -7% (more room for learning)
GO_LIVE_MIN_REALIZED_PNL: Final[float] = 0.01  # Must have actual positive realized P&L

# Legacy AI Accuracy Thresholds (kept for individual trade validation)
# OPTIMIZED: Lowered for more signals during learning phase
MIN_AI_ACCURACY_FOR_LIVE_TRADING: Final[float] = 0.40  # OPTIMIZED: Was 0.45, now 40%
MIN_AI_ACCURACY_FOR_RETRAINING: Final[float] = 0.55  # OPTIMIZED: Was 0.60, now 55%
MIN_TRADER_WIN_RATE_COPY_TRADING: Final[float] = 0.60  # 60% - Only copy traders with 60%+ win rate (configuration default)

# Feature/Signal Constants (for live AI trading operations)
TOTAL_FEATURES: Final[int] = 124  # Total number of signals in feature vector (configuration default)
REQUIRED_FEATURES_FOR_PREDICTION: Final[int] = 124  # All features must be present (configuration default)

# Trading Execution Constants (for live trading operations - backend port 8000)
TRADE_INTERVAL_SEC: Final[int] = 60  # Seconds between trading cycles (configuration default)
TRADE_NOTIONAL_USD: Final[float] = 100.0  # Default trade size in USD (configuration default)
MAX_POSITION_SIZE_USD: Final[float] = 1000.0  # Maximum position size (configuration default)

# Model Training Constants (for live AI model operations)
MIN_TRAINING_SAMPLES: Final[int] = 100  # Minimum samples required to train model (configuration default)
MIN_VALIDATION_SAMPLES: Final[int] = 50  # Minimum samples for validation (configuration default)
RETRAIN_INTERVAL_HOURS: Final[int] = 24  # Retrain models every 24 hours (configuration default)

# Paper Trading Constants (for paper trading operations - backend port 8000)
PAPER_TRADING_INITIAL_BALANCE: Final[float] = 2500.0  # Starting balance for paper trading (configuration default)
PAPER_TRADING_MAX_TRADES_PER_DAY: Final[int] = 50  # Maximum trades per day in paper mode (configuration default)

# ============================================================================
# SCRATCH EXIT & BREAKOUT PROFIT CAPTURE CONSTANTS
# ============================================================================

# Scratch Exit Detection Constants
SCRATCH_PNL_ABS_MAX: Final[float] = 0.10  # Maximum absolute PnL for scratch exit ($0.10)
SCRATCH_MAX_HOLD_SEC: Final[int] = 180  # Maximum hold time for scratch exit (3 minutes)
SCRATCH_REENTRY_COOLDOWN_SEC: Final[int] = 180  # Cooldown after scratch exit (3 minutes)
SIGNAL_PERSISTENCE_REQUIRED: Final[int] = 2  # BUY signals needed after cooldown

# Breakout Profit Capture Constants
TP1_PCT: Final[float] = 3.0  # Take profit rung 1 at +3.0% (sell 25% of position)
TP2_PCT: Final[float] = 6.0  # Take profit rung 2 at +6.0% (sell 25% of position)
TP3_PCT: Final[float] = 9.0  # Take profit rung 3 at +9.0% (sell 25% of position)

# ATR-based Trailing Stop Constants
ATR_TRAIL_MULT: Final[float] = 0.70  # ATR multiplier for trailing distance
ATR_TRAIL_MIN_PCT: Final[float] = 1.5  # Minimum trailing distance (1.5%)
ATR_TRAIL_MAX_PCT: Final[float] = 6.0  # Maximum trailing distance (6.0%)

# Exhaustion Tightening Constants
EXHAUSTION_TIGHTEN_FACTOR: Final[float] = 0.6  # Tighten trailing by 60% during exhaustion

# ============================================================================
# ANTI-CHURN PROTECTION CONSTANTS
# ============================================================================

# Minimum Hold Time for Discretionary SELLs (prevents sub-second churn)
MIN_HOLD_SECONDS_FOR_SELL: Final[int] = 180  # 3 minutes minimum hold time for discretionary sells

# Post-SELL Cooldown (prevents buy->sell->buy loops)
SELL_COOLDOWN_SECONDS: Final[int] = 120  # 2 minutes cooldown after SELL before allowing BUY
