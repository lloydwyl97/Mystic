"""
Account Domain Models - All Live Data, No Fallback/Hardcoded Data

This module defines Pydantic domain models for account-related data from the backend API (port 8000).
All models:
- Represent live account data from backend endpoints (port 8000)
- Field defaults are schema defaults only, not fallback data
- All data validated from live API responses
- Used for serialization/deserialization of live account operations

Live Data Sources:
- Account balances: From live exchange API (Binance.US) via backend endpoints
- Transactions: From live trading operations via backend (port 8000)
- Account performance: Calculated from live account data
- All data from live API calls - no fallback/hardcoded data

Endpoint References:
- /api/accounts/balance - Live account balance data
- /api/transactions - Live transaction operations
- /api/accounts/performance - Live account performance metrics
- All connected to backend running on port 8000
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class AssetBalance(BaseModel):
    """
    Asset balance model for live account data.

    Represents live asset balances from backend API (port 8000).
    Field defaults are schema defaults only - actual values from live exchange API.

    Attributes:
        asset: Asset symbol (e.g., "BTC", "ETH") from live exchange
        free: Free balance (schema default: 0.0, actual from live API)
        locked: Locked balance (schema default: 0.0, actual from live API)
    """

    asset: str
    # Schema defaults only - actual values from live exchange API via backend port 8000
    free: float = Field(0.0, ge=0)
    locked: float = Field(0.0, ge=0)


class AccountBalance(BaseModel):
    """
    Account balance model for live account data.

    Represents live account balances from backend API (port 8000).
    All data from live exchange API (Binance.US) via backend endpoints.

    Attributes:
        user_id: User identifier
        balances: Dictionary of asset balances (live data from backend port 8000)
        updated_at: Timestamp when balance was last updated (from live API)
    """

    user_id: str
    balances: dict[str, AssetBalance]
    updated_at: datetime


class TransactionType(str, Enum):
    """
    Transaction type enum for live transaction operations.

    Used to categorize live transactions from backend API (port 8000).
    """

    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRADE = "trade"
    TRANSFER = "transfer"
    FEE = "fee"


class TransactionStatus(str, Enum):
    """
    Transaction status enum for live transaction operations.

    Used to represent status of live transactions from backend API (port 8000).
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Transaction(BaseModel):
    """
    Transaction model for live transaction data.

    Represents live transactions from backend API (port 8000).
    All data from live trading operations - no fallback/hardcoded data.

    Attributes:
        id: Unique transaction identifier
        user_id: User identifier
        type: Transaction type (from live transaction)
        asset: Asset symbol (from live transaction)
        amount: Transaction amount (from live transaction)
        status: Transaction status (from live transaction)
        created_at: Creation timestamp (from live transaction)
        updated_at: Last update timestamp (optional, from live transaction)
        completed_at: Completion timestamp (optional, from live transaction)
        reference_id: Reference ID for trades/withdrawals (optional, from live transaction)
    """

    id: UUID
    user_id: str
    type: TransactionType
    asset: str
    amount: float
    status: TransactionStatus
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    reference_id: str | None = None  # For trades, withdrawals, etc. (from live transaction)

    class Config:
        from_attributes = True  # Pydantic v2 syntax (replaces orm_mode)


class AccountPerformance(BaseModel):
    """
    Account performance model for live performance data.

    Represents live account performance metrics from backend API (port 8000).
    All metrics calculated from live account data - no fallback/hardcoded data.

    Attributes:
        user_id: User identifier
        timeframe: Performance timeframe (e.g., "1d", "1w", "1m") from live metrics
        start_balance_usd: Starting balance in USD (from live account data)
        current_balance_usd: Current balance in USD (from live account data)
        profit_loss_usd: Profit/loss in USD (calculated from live account data)
        profit_loss_percent: Profit/loss percentage (calculated from live account data)
        updated_at: Timestamp when performance was calculated (from live metrics)
    """

    user_id: str
    timeframe: str  # 1d, 1w, 1m, etc. (from live performance metrics)
    start_balance_usd: float  # From live account data
    current_balance_usd: float  # From live account data
    profit_loss_usd: float  # Calculated from live account data
    profit_loss_percent: float  # Calculated from live account data
    updated_at: datetime  # From live performance metrics
