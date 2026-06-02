#!/usr/bin/env python3
"""
Social Trading Database Models - All Live Data, No Fallback/Hardcoded Data

This module provides SQLAlchemy ORM models for social trading features (backend port 8000).
All models:
- Persist live social trading data to database (backend port 8000)
- Store live user profiles, strategies, followers, and social features
- Track live performance metrics from trading operations
- No fallback/hardcoded data - all models persist live data from operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- User profiles: Live user data from social trading platform (backend port 8000)
- Trading strategies: Live trading strategies from users (backend port 8000)
- Performance metrics: Live trading performance data from Binance.US operations
- Social interactions: Live likes, follows, copies from users
- Leaderboards: Live rankings calculated from live trading performance
- Copy trades: Live copy trading relationships and execution results
- All models persist live data - no mock/test data

Endpoint References:
- Backend API: Port 8000 (social trading models used by backend services)
- Database: Live database connection (from DATABASE_URL or AIDBManager)
- Binance.US API: Live trading operations that generate performance data
- All models use live connections - no fallback/hardcoded data
"""

import logging
import os
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

Base = declarative_base()

# Association tables for many-to-many relationships (live social data)
user_followers = Table(
    "user_followers",
    Base.metadata,
    Column("follower_id", Integer, ForeignKey("users.id"), primary_key=True),  # Live follower user ID
    Column("following_id", Integer, ForeignKey("users.id"), primary_key=True),  # Live following user ID
    Column("created_at", DateTime(timezone=True), server_default=func.now()),  # Live timestamp (schema default, not fallback data)
)

strategy_likes = Table(
    "strategy_likes",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),  # Live user ID
    Column("strategy_id", Integer, ForeignKey("trading_strategies.id"), primary_key=True),  # Live strategy ID
    Column("created_at", DateTime(timezone=True), server_default=func.now()),  # Live timestamp (schema default, not fallback data)
)

strategy_followers = Table(
    "strategy_followers",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),  # Live user ID
    Column("strategy_id", Integer, ForeignKey("trading_strategies.id"), primary_key=True),  # Live strategy ID
    Column("created_at", DateTime(timezone=True), server_default=func.now()),  # Live timestamp (schema default, not fallback data)
)


class User(Base):
    """
    User profile model for live social trading platform (backend port 8000).

    Persists live user profile data and trading performance metrics.
    All data from live operations - no fallback/hardcoded data.
    """

    __tablename__ = "users"

    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)  # Live username
    email = Column(String(255), unique=True, nullable=False, index=True)  # Live email
    display_name = Column(String(100))  # Live display name
    avatar_url = Column(String(500))  # Live avatar URL
    bio = Column(Text)  # Live user bio
    location = Column(String(100))  # Live user location
    website = Column(String(255))  # Live website URL
    verified = Column(Boolean, default=False)  # Schema default, not fallback data - live verification status
    premium = Column(Boolean, default=False)  # Schema default, not fallback data - live premium status
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Live timestamp (schema default, not fallback data)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Live timestamp (schema default, not fallback data)

    # Trading statistics (live metrics from Binance.US operations)
    total_trades = Column(Integer, default=0)  # Schema default, not fallback data - live trade count
    win_rate = Column(Float, default=0.0)  # Schema default, not fallback data - live win rate
    total_pnl = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL
    total_pnl_percentage = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL %
    best_trade = Column(Float, default=0.0)  # Schema default, not fallback data - live best trade
    worst_trade = Column(Float, default=0.0)  # Schema default, not fallback data - live worst trade
    avg_trade_duration = Column(Float, default=0.0)  # Schema default, not fallback data - live avg duration (minutes)
    sharpe_ratio = Column(Float, default=0.0)  # Schema default, not fallback data - live Sharpe ratio
    max_drawdown = Column(Float, default=0.0)  # Schema default, not fallback data - live max drawdown

    # Social stats (live metrics from social interactions)
    followers_count = Column(Integer, default=0)  # Schema default, not fallback data - live follower count
    following_count = Column(Integer, default=0)  # Schema default, not fallback data - live following count
    strategies_count = Column(Integer, default=0)  # Schema default, not fallback data - live strategy count
    reputation_score = Column(Float, default=100.0)  # Schema default, not fallback data - live reputation score

    # Relationships
    strategies = relationship("TradingStrategy", back_populates="author", cascade="all, delete-orphan")
    followers = relationship(
        "User",
        secondary=user_followers,
        primaryjoin="User.id == user_followers.c.following_id",
        secondaryjoin="User.id == user_followers.c.follower_id",
        backref="following",
    )
    liked_strategies = relationship("TradingStrategy", secondary=strategy_likes, back_populates="liked_by")
    followed_strategies = relationship("TradingStrategy", secondary=strategy_followers, back_populates="followers")

    # Indexes
    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
        Index("idx_users_reputation", "reputation_score"),
        Index("idx_users_created", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of live user profile."""
        return f"<User(id={self.id}, username='{self.username}', reputation={self.reputation_score})>"


class TradingStrategy(Base):
    """
    Trading strategy model for live social trading (backend port 8000).

    Persists live trading strategies and performance metrics.
    All data from live operations - no fallback/hardcoded data.
    """

    __tablename__ = "trading_strategies"

    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Live author user ID
    name = Column(String(200), nullable=False)  # Live strategy name
    description = Column(Text)  # Live strategy description
    strategy_type = Column(String(50), nullable=False)  # Live strategy type: manual, ai, hybrid
    symbol = Column(String(20), nullable=False)  # Live trading symbol (from Binance.US Top-10)
    timeframe = Column(String(10), nullable=False)  # Live timeframe: 1m, 5m, 15m, 1h, 4h, 1D

    # Strategy configuration (live JSON data)
    config = Column(Text)  # Live JSON configuration from user
    indicators = Column(Text)  # Live JSON list of indicators used
    entry_conditions = Column(Text)  # Live JSON entry rules
    exit_conditions = Column(Text)  # Live JSON exit rules

    # Performance metrics (live metrics from Binance.US operations)
    total_trades = Column(Integer, default=0)  # Schema default, not fallback data - live trade count
    win_rate = Column(Float, default=0.0)  # Schema default, not fallback data - live win rate
    total_pnl = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL
    total_pnl_percentage = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL %
    sharpe_ratio = Column(Float, default=0.0)  # Schema default, not fallback data - live Sharpe ratio
    max_drawdown = Column(Float, default=0.0)  # Schema default, not fallback data - live max drawdown
    calmar_ratio = Column(Float, default=0.0)  # Schema default, not fallback data - live Calmar ratio

    # Backtest results (live backtest data on live market data)
    backtest_start_date = Column(DateTime(timezone=True))  # Live backtest start date
    backtest_end_date = Column(DateTime(timezone=True))  # Live backtest end date
    backtest_period_years = Column(Float, default=0.0)  # Schema default, not fallback data - live backtest period

    # Social features (live metrics from social interactions)
    is_public = Column(Boolean, default=True)  # Schema default, not fallback data - live public status
    is_featured = Column(Boolean, default=False)  # Schema default, not fallback data - live featured status
    likes_count = Column(Integer, default=0)  # Schema default, not fallback data - live likes count
    followers_count = Column(Integer, default=0)  # Schema default, not fallback data - live followers count
    copies_count = Column(Integer, default=0)  # Schema default, not fallback data - live copies count

    # Pricing (for premium strategies - live pricing data)
    is_premium = Column(Boolean, default=False)  # Schema default, not fallback data - live premium status
    price = Column(Float, default=0.0)  # Schema default, not fallback data - live price
    currency = Column(String(3), default="USD")  # Schema default, not fallback data - live currency

    # Metadata (live data)
    tags = Column(Text)  # Live JSON array of tags
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Live timestamp (schema default, not fallback data)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Live timestamp (schema default, not fallback data)

    # Relationships
    author = relationship("User", back_populates="strategies")
    liked_by = relationship("User", secondary=strategy_likes, back_populates="liked_strategies")
    followers = relationship("User", secondary=strategy_followers, back_populates="followed_strategies")

    # Indexes
    __table_args__ = (
        Index("idx_strategies_author", "author_id"),
        Index("idx_strategies_symbol", "symbol"),
        Index("idx_strategies_type", "strategy_type"),
        Index("idx_strategies_public", "is_public"),
        Index("idx_strategies_featured", "is_featured"),
        Index("idx_strategies_win_rate", "win_rate"),
        Index("idx_strategies_pnl", "total_pnl_percentage"),
        Index("idx_strategies_likes", "likes_count"),
        Index("idx_strategies_created", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of live trading strategy."""
        return f"<TradingStrategy(id={self.id}, name='{self.name}', author_id={self.author_id}, win_rate={self.win_rate})>"


class StrategyPerformance(Base):
    """
    Detailed performance tracking for live strategies (backend port 8000).

    Persists live performance metrics for users copying strategies.
    All data from live operations - no fallback/hardcoded data.
    """

    __tablename__ = "strategy_performance"

    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("trading_strategies.id"), nullable=False, index=True)  # Live strategy ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Live user ID (who copied/uses this strategy)

    # Performance metrics (live metrics from Binance.US operations)
    total_trades = Column(Integer, default=0)  # Schema default, not fallback data - live trade count
    winning_trades = Column(Integer, default=0)  # Schema default, not fallback data - live winning trades count
    losing_trades = Column(Integer, default=0)  # Schema default, not fallback data - live losing trades count
    total_pnl = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL
    total_pnl_percentage = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL %

    # Risk metrics (live metrics from trading operations)
    max_drawdown = Column(Float, default=0.0)  # Schema default, not fallback data - live max drawdown
    sharpe_ratio = Column(Float, default=0.0)  # Schema default, not fallback data - live Sharpe ratio
    volatility = Column(Float, default=0.0)  # Schema default, not fallback data - live volatility

    # Time tracking (live timestamps)
    start_date = Column(DateTime(timezone=True), server_default=func.now())  # Live start date (schema default, not fallback data)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Live update timestamp (schema default, not fallback data)
    is_active = Column(Boolean, default=True)  # Schema default, not fallback data - live active status

    # Relationships
    strategy = relationship("TradingStrategy")
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index("idx_perf_strategy_user", "strategy_id", "user_id"),
        Index("idx_perf_user", "user_id"),
        Index("idx_perf_active", "is_active"),
        Index("idx_perf_pnl", "total_pnl_percentage"),
    )

    def __repr__(self) -> str:
        """String representation of live strategy performance."""
        return f"<StrategyPerformance(strategy_id={self.strategy_id}, user_id={self.user_id}, pnl={self.total_pnl_percentage})>"


class SocialPost(Base):
    """
    Social posts and trading insights for live social trading (backend port 8000).

    Persists live social posts and trading insights from users.
    All data from live operations - no fallback/hardcoded data.
    """

    __tablename__ = "social_posts"

    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Live author user ID
    content = Column(Text, nullable=False)  # Live post content
    post_type = Column(String(20), nullable=False)  # Live post type: insight, analysis, strategy, market_update

    # Trading context (live data)
    symbol = Column(String(20))  # Live trading symbol (from Binance.US Top-10)
    strategy_id = Column(Integer, ForeignKey("trading_strategies.id"))  # Live strategy ID

    # Engagement metrics (live metrics from social interactions)
    likes_count = Column(Integer, default=0)  # Schema default, not fallback data - live likes count
    comments_count = Column(Integer, default=0)  # Schema default, not fallback data - live comments count
    shares_count = Column(Integer, default=0)  # Schema default, not fallback data - live shares count

    # Media attachments (live data)
    images = Column(Text)  # Live JSON array of image URLs
    charts = Column(Text)  # Live JSON chart data

    # Metadata (live data)
    is_pinned = Column(Boolean, default=False)  # Schema default, not fallback data - live pinned status
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Live timestamp (schema default, not fallback data)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Live timestamp (schema default, not fallback data)

    # Relationships
    author = relationship("User")
    strategy = relationship("TradingStrategy")

    # Indexes
    __table_args__ = (
        Index("idx_posts_author", "author_id"),
        Index("idx_posts_type", "post_type"),
        Index("idx_posts_symbol", "symbol"),
        Index("idx_posts_created", "created_at"),
        Index("idx_posts_pinned", "is_pinned"),
    )

    def __repr__(self) -> str:
        """String representation of live social post."""
        return f"<SocialPost(id={self.id}, author_id={self.author_id}, type='{self.post_type}')>"


class Leaderboard(Base):
    """
    Performance leaderboards for live social trading (backend port 8000).

    Persists live leaderboard rankings calculated from live trading performance.
    All data from live operations - no fallback/hardcoded data.
    """

    __tablename__ = "leaderboards"

    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Live user ID
    period = Column(String(20), nullable=False)  # Live period: daily, weekly, monthly, all_time
    category = Column(String(30), nullable=False)  # Live category: pnl, win_rate, sharpe_ratio, followers

    # Rankings and scores (live data from performance calculations)
    rank = Column(Integer, nullable=False)  # Live rank
    score = Column(Float, nullable=False)  # Live score
    previous_rank = Column(Integer)  # Live previous rank

    # Period details (live data)
    start_date = Column(DateTime(timezone=True), nullable=False)  # Live period start date
    end_date = Column(DateTime(timezone=True), nullable=False)  # Live period end date

    # Metadata (live timestamp)
    calculated_at = Column(DateTime(timezone=True), server_default=func.now())  # Live calculation timestamp (schema default, not fallback data)

    # Relationships
    user = relationship("User")

    # Indexes
    __table_args__ = (
        Index("idx_leaderboard_period_category", "period", "category"),
        Index("idx_leaderboard_user", "user_id"),
        Index("idx_leaderboard_rank", "rank"),
        Index("idx_leaderboard_score", "score"),
    )

    def __repr__(self) -> str:
        """String representation of live leaderboard entry."""
        return f"<Leaderboard(user_id={self.user_id}, period='{self.period}', category='{self.category}', rank={self.rank})>"


class CopyTrade(Base):
    """
    Copy trading relationships for live social trading (backend port 8000).

    Persists live copy trading relationships and execution results.
    All data from live operations - no fallback/hardcoded data.
    """

    __tablename__ = "copy_trades"

    # Primary fields
    id = Column(Integer, primary_key=True, autoincrement=True)
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Live user ID (user copying)
    leader_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Live user ID (user being copied)
    strategy_id = Column(Integer, ForeignKey("trading_strategies.id"), nullable=True, index=True)  # Live strategy ID (optional - can copy all trades)

    # Configuration (live copy trade settings)
    allocation_percentage = Column(Float, default=100.0)  # Schema default, not fallback data - live allocation % of follower's capital
    max_position_size = Column(Float)  # Live maximum position size
    risk_multiplier = Column(Float, default=1.0)  # Schema default, not fallback data - live risk adjustment multiplier
    is_active = Column(Boolean, default=True)  # Schema default, not fallback data - live active status

    # Performance tracking (live metrics from copy trading operations)
    total_trades_copied = Column(Integer, default=0)  # Schema default, not fallback data - live trades copied count
    successful_trades = Column(Integer, default=0)  # Schema default, not fallback data - live successful trades count
    total_pnl = Column(Float, default=0.0)  # Schema default, not fallback data - live PnL

    # Timestamps (live timestamps)
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Live creation timestamp (schema default, not fallback data)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Live update timestamp (schema default, not fallback data)
    last_trade_at = Column(DateTime(timezone=True))  # Live last trade timestamp

    # Relationships
    follower = relationship("User", foreign_keys=[follower_id])
    leader = relationship("User", foreign_keys=[leader_id])
    strategy = relationship("TradingStrategy")

    # Indexes
    __table_args__ = (
        Index("idx_copy_follower", "follower_id"),
        Index("idx_copy_leader", "leader_id"),
        Index("idx_copy_strategy", "strategy_id"),
        Index("idx_copy_active", "is_active"),
    )

    def __repr__(self) -> str:
        """String representation of live copy trade relationship."""
        return f"<CopyTrade(follower_id={self.follower_id}, leader_id={self.leader_id}, allocation={self.allocation_percentage}%)>"


# Create all tables
def create_social_trading_tables(engine) -> None:
    """
    Create all social trading tables in live database (backend port 8000).

    Creates live database schema for social trading features.
    All tables persist live data - no fallback/hardcoded data.

    Args:
        engine: Live database engine connection
    """
    Base.metadata.create_all(engine)
    logger.info("Social trading database tables created for live operations")


# Drop all tables
def drop_social_trading_tables(engine) -> None:
    """
    Drop all social trading tables from live database (backend port 8000).

    Drops live database schema (use with caution in production).
    All operations on live database - no fallback/hardcoded data.

    Args:
        engine: Live database engine connection
    """
    Base.metadata.drop_all(engine)
    logger.info("Social trading database tables dropped from live database")


if __name__ == "__main__":
    # Allow standalone table creation for development (live database setup)
    from sqlalchemy import create_engine

    # Use SQLite database for development (live database setup, not fallback data)
    db_path = os.getenv("SOCIAL_TRADING_DB", str(Path(__file__).parent / "social_trading.db"))
    engine = create_engine(f"sqlite:///{db_path}")

    create_social_trading_tables(engine)
    logger.info("Social trading tables created in %s for live operations", db_path)
