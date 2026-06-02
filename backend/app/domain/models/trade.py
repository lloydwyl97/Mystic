"""
Trade Domain Models - All Live Data, No Fallback/Hardcoded Data

This module defines Pydantic domain models for trade-related data from the backend API (port 8000).
All models:
- Represent live trade data from backend endpoints (port 8000)
- Field defaults are schema defaults only, not fallback data
- All data validated from live API responses
- Used for serialization/deserialization of live trade operations

Live Data Sources:
- Trades: From live order execution via backend (port 8000)
- Trade status: From live exchange API (Binance.US) via backend endpoints
- Trade prices and quantities: From live order execution
- All data from live trading operations - no fallback/hardcoded data

Endpoint References:
- /api/trades - Live trade operations
- /api/trades/create - Live trade creation
- /api/trades/update - Live trade updates
- All connected to backend running on port 8000
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class TradeSide(str, Enum):
    """
    Trade side enum for live trade operations.

    Used to categorize live trades from backend API (port 8000).
    """

    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    """
    Trade status enum for live trade operations.

    Used to represent status of live trades from backend API (port 8000).
    """

    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TradeBase(BaseModel):
    """
    Base trade model for live trade data.

    Represents core trade fields from live order execution via backend (port 8000).

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT") from live order
        side: Trade side (BUY or SELL) from live order
        quantity: Trade quantity (from live order execution)
        price: Trade price (from live order execution)
    """

    symbol: str
    side: TradeSide
    quantity: float = Field(..., gt=0)  # From live order execution
    price: float = Field(..., gt=0)  # From live order execution


class TradeCreate(TradeBase):
    """
    Trade creation model for live trade operations.

    Used to create new trades via backend API (port 8000).
    All fields from live order execution - no fallback/hardcoded data.
    """


class TradeUpdate(BaseModel):
    """
    Trade update model for live trade operations.

    Used to update existing trades via backend API (port 8000).
    All fields from live order execution - no fallback/hardcoded data.

    Attributes:
        status: Trade status (optional, from live order execution)
        price: Trade price (optional, from live order execution)
        quantity: Trade quantity (optional, from live order execution)
    """

    status: TradeStatus | None = None  # From live order execution
    price: float | None = Field(None, gt=0)  # From live order execution
    quantity: float | None = Field(None, gt=0)  # From live order execution


class Trade(TradeBase):
    """
    Trade model with all fields for live trade data.

    Represents complete trade information from backend API (port 8000).
    All data from live order execution - no fallback/hardcoded data.

    Attributes:
        id: Unique trade identifier
        user_id: User identifier
        symbol: Trading symbol (from live order)
        side: Trade side (BUY or SELL, from live order)
        quantity: Trade quantity (from live order execution)
        price: Trade price (from live order execution)
        status: Trade status (from live order execution)
        created_at: Creation timestamp (from live order)
        updated_at: Last update timestamp (optional, from live order)
        completed_at: Completion timestamp (optional, from live order)
    """

    id: UUID
    user_id: str
    status: TradeStatus
    created_at: datetime  # From live order execution
    updated_at: datetime | None = None  # From live order execution
    completed_at: datetime | None = None  # From live order execution

    class Config:
        from_attributes = True  # Pydantic v2 syntax (replaces orm_mode)
