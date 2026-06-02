from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Strategy(Base):
    __tablename__ = "strategies"
    __table_args__ = (
        UniqueConstraint("name", name="uq_strategies_name"),
        Index("ix_strategies_is_active", "is_active"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    win_rate = Column(Float, default=0.0, nullable=False)
    avg_profit = Column(Float, default=0.0, nullable=False)
    trades_executed = Column(Integer, default=0, nullable=False)
    total_profit = Column(Float, default=0.0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    trades = relationship("Trade", back_populates="strategy", cascade="all, delete-orphan")
    performances = relationship("StrategyPerformance", back_populates="strategy", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        # Be defensive in case values are None (e.g., when loading legacy rows)
        win = self.win_rate if self.win_rate is not None else 0.0
        avg = self.avg_profit if self.avg_profit is not None else 0.0
        return f"<Strategy id={self.id} name={self.name!r} win_rate={win:.4f} avg_profit={avg:.4f}>"


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_strategy_id", "strategy_id"),
        Index("ix_trades_timestamp", "timestamp"),
        Index("ix_trades_status", "status"),
        Index("ix_trades_coin", "coin"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    coin = Column(String(32), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    strategy_name = Column(String(255))
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, default=1.0, nullable=False)
    profit = Column(Float)
    profit_percentage = Column(Float)
    duration_minutes = Column(Float)
    success = Column(Boolean)
    trade_type = Column(String(32), default="spot", nullable=False)
    status = Column(String(32), default="completed", nullable=False)
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    risk_level = Column(String(32), default="medium", nullable=False)
    tags = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    strategy = relationship("Strategy", back_populates="trades")

    def __repr__(self) -> str:
        return f"<Trade id={self.id} coin={self.coin!r} strategy_id={self.strategy_id} status={self.status!r}>"


class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"
    __table_args__ = (
        Index("ix_strategy_performance_strategy_id", "strategy_id"),
        Index("ix_strategy_performance_date", "date"),
        UniqueConstraint("strategy_id", "date", name="uq_strategy_performance_strategy_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False)
    strategy_name = Column(String(255))
    date = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    win_rate = Column(Float, default=0.0, nullable=False)
    avg_profit = Column(Float, default=0.0, nullable=False)
    total_trades = Column(Integer, default=0, nullable=False)
    total_profit = Column(Float, default=0.0, nullable=False)
    max_drawdown = Column(Float, default=0.0, nullable=False)
    sharpe_ratio = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    strategy = relationship("Strategy", back_populates="performances")

    def __repr__(self) -> str:
        date_str = self.date.isoformat() if self.date is not None else None
        return f"<StrategyPerformance id={self.id} strategy_id={self.strategy_id} date={date_str}>"
