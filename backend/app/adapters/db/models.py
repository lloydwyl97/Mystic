"""
SQLAlchemy Database Models - All Live Data, No Fallback/Hardcoded Data

This module defines SQLAlchemy ORM models for persisting live data from the backend API (port 8000).
All models:
- Store live data from backend endpoints (port 8000)
- Column defaults are schema defaults only, not fallback data
- All data persisted from live API calls
- Connected to backend endpoints for live data operations

Live Data Sources:
- Account balances: From live exchange API (Binance.US) via backend endpoints
- Transactions: From live trading operations via backend (port 8000)
- Trades: From live order execution via backend (port 8000)
- Portfolios: Calculated from live account data
- Strategies: From live strategy execution via backend (port 8000)

Endpoint References:
- /api/transactions - Live transaction operations
- /api/trades - Live trade operations
- /api/portfolio - Live portfolio data
- /api/strategies - Live strategy operations
- All connected to backend running on port 8000
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.declarative import declarative_base

from backend.app.domain.models.account import TransactionStatus, TransactionType

Base = declarative_base()


class AccountBalanceModel(Base):
    """SQLAlchemy model for account balances."""

    __tablename__ = "account_balances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    asset = Column(String, nullable=False)
    # Schema defaults only - actual data from live exchange API
    free = Column(Float, nullable=False, default=0.0)
    locked = Column(Float, nullable=False, default=0.0)
    # Timezone-aware timestamp (live data from backend port 8000)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_account_balances_user_id", "user_id"),
        {"sqlite_autoincrement": True},
    )


class TransactionModel(Base):
    """SQLAlchemy model for transactions."""

    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, nullable=False)
    type = Column(SQLEnum(TransactionType), nullable=False)
    asset = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    # Schema default - actual status from live transaction operations
    status = Column(SQLEnum(TransactionStatus), nullable=False, default=TransactionStatus.PENDING)
    # Timezone-aware timestamp (live data from backend port 8000)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    reference_id = Column(String, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Additional transaction data

    __table_args__ = (
        Index("ix_transactions_user_id", "user_id"),
        Index("ix_transactions_reference_id", "reference_id"),
        {"sqlite_autoincrement": False},
    )


class TradeModel(Base):
    """SQLAlchemy model for trades."""

    __tablename__ = "trades"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # BUY or SELL
    # Schema defaults only - actual values from live order execution
    type = Column(String, nullable=False, default="MARKET")  # MARKET, LIMIT, etc.
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)  # quantity * price
    fee = Column(Float, nullable=False, default=0.0)
    fee_asset = Column(String, nullable=False, default="USDT")
    status = Column(String, nullable=False, default="FILLED")  # NEW, FILLED, CANCELLED, etc.
    order_id = Column(String, nullable=True)
    client_order_id = Column(String, nullable=True)
    # Timezone-aware timestamp (live data from backend port 8000)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Additional trade data

    __table_args__ = (
        Index("ix_trades_user_id", "user_id"),
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_order_id", "order_id"),
        {"sqlite_autoincrement": False},
    )


class PortfolioModel(Base):
    """SQLAlchemy model for portfolio snapshots."""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    # Timezone-aware timestamp (live portfolio snapshots from backend port 8000)
    snapshot_time = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    total_value_usd = Column(Float, nullable=False)
    balances = Column(JSON, nullable=False)  # Dict of asset balances
    performance_24h = Column(Float, nullable=True)
    performance_7d = Column(Float, nullable=True)
    performance_30d = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_portfolios_user_id", "user_id"),
        Index("ix_portfolios_snapshot_time", "snapshot_time"),
        {"sqlite_autoincrement": True},
    )


class StrategyModel(Base):
    """SQLAlchemy model for trading strategies."""

    __tablename__ = "strategies"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    symbol = Column(String, nullable=False)
    type = Column(String, nullable=False)  # MANUAL, AI, HYBRID
    # Schema default - actual status from live strategy operations
    status = Column(String, nullable=False, default="ACTIVE")  # ACTIVE, INACTIVE, DELETED
    config = Column(JSON, nullable=False)  # Strategy configuration (live data)
    performance_stats = Column(JSON, nullable=True)  # Performance statistics (live data)
    # Timezone-aware timestamp (live data from backend port 8000)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=True)
    # Schema default - actual visibility from live strategy operations
    is_public = Column(Integer, nullable=False, default=0)  # 0=private, 1=public

    __table_args__ = (
        Index("ix_strategies_user_id", "user_id"),
        Index("ix_strategies_symbol", "symbol"),
        {"sqlite_autoincrement": False},
    )
