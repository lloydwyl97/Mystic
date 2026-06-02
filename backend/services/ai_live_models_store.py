"""
AI Live DB Models & Utilities

- SQLAlchemy models for live AI strategies, predictions, signals, trade fills, and KPIs
- SQLite-first setup with sensible pragmas (WAL, foreign keys) and rotate-friendly journaling
- Simple session utilities and convenient CRUD helpers
- Designed to work out-of-the-box with SQLite, but supports any SQLAlchemy URL

Environment variables:
  - AI_LIVE_DB_URL      : optional full SQLAlchemy URL (e.g., sqlite:////abs/path/ai_live.sqlite)
  - AI_LIVE_DB_PATH     : optional SQLite file path (overridden by AI_LIVE_DB_URL); default: ai_live.sqlite
  - AI_LIVE_DB_ECHO     : "1" to enable SQL echo (default: "0")

Typical usage:
  from ai_live_db import (
      init_ai_live_db, get_session,
      AILivePrediction, AILiveSignal, AILiveTradeFill, AILiveStrategy, AILiveStrategyKPI,
      record_prediction, enqueue_signal, mark_signal_consumed,
      upsert_trade_fill, update_strategy_kpi, latest_predictions
  )

  init_ai_live_db()
  with get_session() as s:
      record_prediction(s, "BTC/USDT", prob_up=0.62, prob_down=0.28, features={"rsi": 55.0})
      sig = enqueue_signal(s, "BTC/USDT", "buy", "RSI cross up", price=48000.0)
      mark_signal_consumed(s, sig.id)
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# -----------------------------------------------------------------------------
# Database URL & Engine
# -----------------------------------------------------------------------------


def _default_sqlite_url() -> str:
    db_path = os.getenv("AI_LIVE_DB_PATH", "ai_live.sqlite")
    # Make sure we create an absolute path for SQLite to avoid cwd surprises
    if not db_path.startswith(("/", "\\")):
        db_path = str(Path(db_path).resolve())
    return f"sqlite:///{db_path}"


DB_URL: str = os.getenv("AI_LIVE_DB_URL", _default_sqlite_url())
DB_ECHO: bool = os.getenv("AI_LIVE_DB_ECHO", "0") == "1"

# Naming convention improves Alembic migrations
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=naming_convention)

engine: Engine = create_engine(
    DB_URL,
    future=True,
    echo=DB_ECHO,
    pool_pre_ping=True,
)


# SQLite pragmas for durability + concurrency (no-op for other DBs)
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        pass


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
Base = declarative_base(metadata=metadata)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class AILiveStrategy(Base):
    __tablename__ = "ai_live_strategies"
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    symbol = Column(String, index=True)  # ccxt symbol or "MULTI"
    desc = Column(String)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    win_rate = Column(Float, default=0.0)
    trades = Column(Integer, default=0)
    pnl = Column(Float, default=0.0)
    active = Column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<Strategy id={self.id} name={self.name} symbol={self.symbol} active={self.active}>"


class AILivePrediction(Base):
    __tablename__ = "ai_live_predictions"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)  # e.g. "BTC/USDT"
    timeframe = Column(String, default="1m", index=True)
    prob_up = Column(Float, nullable=False)
    prob_down = Column(Float, nullable=False)
    features = Column(JSON)  # store small, compact features dict
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    # Allow multiple per minute; add a composite index for common retrieval
    __table_args__ = (Index("ix_pred_symbol_timeframe_created", "symbol", "timeframe", "created_at"),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "features": self.features,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AILiveSignal(Base):
    __tablename__ = "ai_live_signals"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)  # e.g. "BTC/USDT"
    side = Column(String, index=True)  # "buy" or "sell"
    reason = Column(String)
    price = Column(Float)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    consumed = Column(Boolean, default=False, index=True)

    # Not unique, but add a fast retrieval index for "open" signals
    __table_args__ = (Index("ix_signal_open", "symbol", "consumed", "created_at"),)


class AILiveTradeFill(Base):
    __tablename__ = "ai_live_trade_fills"
    id = Column(Integer, primary_key=True)
    exchange = Column(String, index=True)  # EXCHANGE_ID from trading_universe
    symbol = Column(String, index=True)  # ccxt "BTC/USDT"
    side = Column(String, index=True)  # "buy" or "sell"
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    cost = Column(Float)  # qty * price (approx)
    fee = Column(Float, default=0.0)
    fee_currency = Column(String, default=None)
    ts = Column(DateTime, index=True)  # trade timestamp (UTC)
    trade_id = Column(String, index=True)  # exchange trade id
    order_id = Column(String, default=None, index=True)
    strategy_name = Column(String, default="Live_RSICross", index=True)

    __table_args__ = (
        UniqueConstraint("exchange", "trade_id", name="uq_ai_live_exchange_trade_id"),
        Index("ix_fill_symbol_ts", "symbol", "ts"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "cost": self.cost,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
            "ts": self.ts.isoformat() if self.ts else None,
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "strategy_name": self.strategy_name,
        }


class AILiveStrategyKPI(Base):
    __tablename__ = "ai_live_strategy_kpi"
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String, index=True, unique=True)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    closed_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=dt.datetime.utcnow, index=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "closed_trades": self.closed_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


# -----------------------------------------------------------------------------
# Initialization & Session helpers
# -----------------------------------------------------------------------------


def init_ai_live_db() -> None:
    """Create all tables if they don't exist."""
    # Ensure parent directory exists for SQLite files
    if DB_URL.startswith("sqlite:///"):
        path = DB_URL.replace("sqlite:///", "", 1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterable[Session]:
    """
    Context manager wrapping SessionLocal.
    Example:
        with get_session() as s:
            ...
    """
    session = SessionLocal()  # type: ignore[call-arg]
    try:
        yield session
        session.commit()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        session.rollback()
        raise
    finally:
        session.close()


# -----------------------------------------------------------------------------
# CRUD Helpers
# -----------------------------------------------------------------------------


def record_prediction(
    session: Session,
    symbol: str,
    prob_up: float,
    prob_down: float,
    features: dict[str, Any] | None = None,
    timeframe: str = "1m",
    created_at: dt.datetime | None = None,
) -> AILivePrediction:
    p = AILivePrediction(
        symbol=symbol,
        timeframe=timeframe,
        prob_up=float(prob_up),
        prob_down=float(prob_down),
        features=features or {},
        created_at=created_at or dt.datetime.now(timezone.utc),
    )
    session.add(p)
    return p


def enqueue_signal(
    session: Session,
    symbol: str,
    side: str,
    reason: str,
    price: float,
    created_at: dt.datetime | None = None,
) -> AILiveSignal:
    s = AILiveSignal(
        symbol=symbol,
        side=side.lower().strip(),
        reason=reason,
        price=float(price),
        created_at=created_at or dt.datetime.now(timezone.utc),
        consumed=False,
    )
    session.add(s)
    return s


def mark_signal_consumed(session: Session, signal_id: int) -> bool:
    sig = session.get(AILiveSignal, signal_id)
    if not sig:
        return False
    sig.consumed = True
    return True


def upsert_trade_fill(
    session: Session,
    *,
    exchange: str,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    ts: dt.datetime,
    trade_id: str,
    order_id: str | None = None,
    fee: float = 0.0,
    fee_currency: str | None = None,
    strategy_name: str = "Live_RSICross",
) -> AILiveTradeFill:
    """
    Insert or update a trade fill uniquely identified by (exchange, trade_id).
    Works across SQLite and other RDBMS by first querying for existence.
    """
    existing: AILiveTradeFill | None = session.execute(
        select(AILiveTradeFill).where(
            AILiveTradeFill.exchange == exchange,
            AILiveTradeFill.trade_id == trade_id,
        ),
    ).scalar_one_or_none()

    cost = float(qty) * float(price)
    if existing:
        existing.symbol = symbol
        existing.side = side
        existing.qty = float(qty)
        existing.price = float(price)
        existing.cost = cost
        existing.fee = float(fee or 0.0)
        existing.fee_currency = fee_currency
        existing.ts = ts
        existing.order_id = order_id
        existing.strategy_name = strategy_name
        return existing

    fill = AILiveTradeFill(
        exchange=exchange,
        symbol=symbol,
        side=side,
        qty=float(qty),
        price=float(price),
        cost=cost,
        fee=float(fee or 0.0),
        fee_currency=fee_currency,
        ts=ts,
        trade_id=trade_id,
        order_id=order_id,
        strategy_name=strategy_name,
    )
    session.add(fill)
    try:
        # In case of race in multi-process envs, try to resolve gracefully
        session.flush()
    except IntegrityError:
        session.rollback()
        # Re-fetch and update
        existing = session.execute(
            select(AILiveTradeFill).where(
                AILiveTradeFill.exchange == exchange,
                AILiveTradeFill.trade_id == trade_id,
            ),
        ).scalar_one_or_none()
        if existing:
            return existing
        # If still not found, re-add
        session.add(fill)
    return fill


def update_strategy_kpi(
    session: Session,
    strategy_name: str,
    *,
    realized_delta: float = 0.0,
    unrealized: float | None = None,
    closed_trades_delta: int = 0,
    winning_trades_delta: int = 0,
) -> AILiveStrategyKPI:
    """
    Update (or create) KPI row for a strategy and recompute win rate.
    """
    kpi: AILiveStrategyKPI | None = session.execute(select(AILiveStrategyKPI).where(AILiveStrategyKPI.strategy_name == strategy_name)).scalar_one_or_none()

    if not kpi:
        kpi = AILiveStrategyKPI(strategy_name=strategy_name)
        session.add(kpi)

    kpi.realized_pnl = float(kpi.realized_pnl or 0.0) + float(realized_delta or 0.0)
    if unrealized is not None:
        kpi.unrealized_pnl = float(unrealized)
    kpi.closed_trades = int(kpi.closed_trades or 0) + int(closed_trades_delta or 0)
    kpi.winning_trades = int(kpi.winning_trades or 0) + int(winning_trades_delta or 0)
    kpi.win_rate = (kpi.winning_trades / kpi.closed_trades) if kpi.closed_trades > 0 else 0.0
    kpi.last_updated = dt.datetime.now(timezone.utc)
    return kpi


def latest_predictions(session: Session, symbol: str, limit: int = 50, timeframe: str | None = None) -> list[AILivePrediction]:
    q = select(AILivePrediction).where(AILivePrediction.symbol == symbol)
    if timeframe:
        q = q.where(AILivePrediction.timeframe == timeframe)
    q = q.order_by(AILivePrediction.created_at.desc()).limit(int(limit))
    return list(session.scalars(q))


# Backwards-compatible name kept for callers
# init_ai_live_db is available for import but not called here
