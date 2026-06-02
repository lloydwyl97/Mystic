#!/usr/bin/env python3
"""
Database Initialization and Management for Mystic Trading Platform
Handles database creation, migrations, and connection management

FIXED FOR PRODUCTION RELIABILITY:
- NO HARDCODED DATA: Removed all sample data seeding
- NO PLACEHOLDERS: Eliminated hardcoded symbol fallbacks
- EXCHANGE SYMBOL FORMAT: Standardized to BTCUSDT format (no slashes)
- TOP-10 ENFORCEMENT: Validates symbols against centralized allowlist
- OPTIMIZED QUERIES: Fixed O(n) scans that block UIs
- PROPER SQLITE CONFIG: Fixed StaticPool + WAL contradictions
- SYMBOL VALIDATION: Validates symbols at insert time
- LIVE DATA ONLY: All data must come from live ingest paths

Windows/Python 3.12+ Compatibility:
- Uses SQLAlchemy 2.0+ patterns with modern select() syntax
- SQLite-safe threading configuration for Windows
- Async-safe session management with threadpool execution
- UTC timezone handling with proper column definitions
- ASCII-only logging for Windows PowerShell compatibility
"""

import contextlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    text,
)

# Lazy import for optional Redis service (may not be available in all deployments)
try:
    from backend.services.redis_service import RedisService, get_redis_service
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    RedisService = None  # type: ignore[assignment, misc]
    get_redis_service = None  # type: ignore[assignment, misc]

from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Configure logging
logger = logging.getLogger("mystic.database")

# Database configuration with absolute paths
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mystic_trading.db")
IS_SQLITE = False
DATABASE_PATH = ""

if DATABASE_URL.startswith("sqlite:///"):
    IS_SQLITE = True
    # Convert to absolute path for Windows compatibility
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if not Path(db_path).is_absolute():
        db_path = str(Path(db_path).resolve())
    DATABASE_URL = f"sqlite:///{db_path}"
    DATABASE_PATH = db_path

# Proper SQLite engine configuration for Windows threading
engine = create_engine(
    DATABASE_URL,
    echo=False,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
    },
)

# Session factory
# Note: 'autocommit' param removed in SQLAlchemy 2.0; avoid passing it.
SessionLocal = sessionmaker(autoflush=False, bind=engine, expire_on_commit=False, future=True)


# SQLAlchemy 2.0+ base class
class Base(DeclarativeBase):
    pass


class MarketData(Base):
    """Market data table"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    price = Column(Float)
    volume = Column(Float)
    change_24h = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    exchange = Column(String, default=EXCHANGE_ID)


class TradeLog(Base):
    """Trade execution log"""

    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # buy/sell
    amount = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    strategy = Column(String)
    portfolio_id = Column(String, index=True)
    status = Column(String, default="executed")


class Portfolio(Base):
    """Portfolio management"""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, unique=True, index=True)
    name = Column(String)
    total_value = Column(Float)
    cash = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Position(Base):
    """Portfolio positions"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, index=True)
    symbol = Column(String, index=True)
    amount = Column(Float)
    average_price = Column(Float)
    current_value = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Strategy(Base):
    """Trading strategies"""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    parameters = Column(Text)  # JSON string
    performance_metrics = Column(Text)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class RiskAssessment(Base):
    """Risk assessment records"""

    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, index=True)
    risk_metrics = Column(Text)  # JSON string
    alerts = Column(Text)  # JSON string
    risk_score = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PerformanceMetrics(Base):
    """Performance metrics history"""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, index=True)
    metrics = Column(Text)  # JSON string
    grade = Column(String)
    trend = Column(String)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SystemLog(Base):
    """System operation logs"""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String)  # INFO, WARNING, ERROR, CRITICAL
    component = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    log_metadata = Column(Text)  # Renamed from 'metadata' to avoid collision


# ----------------------------- Symbol Validation -----------------------------


def get_supported_symbols() -> dict[str, Any]:
    """Get supported symbols from trading_universe (single source of truth)"""
    return {
        "status": "available",
        "symbols": list(TRADING_SYMBOLS),  # Exchange format: BTCUSDT
        "source": "trading_universe",
        "count": len(TRADING_SYMBOLS),
    }


def validate_symbol(symbol: str) -> dict[str, Any]:
    """Validate symbol against Top-10 allowlist"""
    try:
        symbols_info = get_supported_symbols()
        if symbols_info["status"] != "available":
            return {
                "valid": False,
                "reason": f"Allowlist unavailable: {symbols_info.get('error', 'Unknown error')}",
                "symbol": symbol,
            }

        # Normalize symbol to exchange format
        normalized = symbol.strip().upper().replace("-", "").replace("/", "").replace("_", "").replace(" ", "")
        if not normalized.endswith("USDT") and len(normalized) <= 5:
            normalized = normalized + "USDT"

        if normalized in symbols_info["symbols"]:
            return {
                "valid": True,
                "reason": "Symbol is in Top-10 Binance.US allowlist",
                "symbol": normalized,
                "original": symbol,
            }
        return {
            "valid": False,
            "reason": f"Symbol '{normalized}' not in Top-10 Binance.US allowlist",
            "symbol": normalized,
            "original": symbol,
            "supported_symbols": symbols_info["symbols"],
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        return {"valid": False, "reason": f"Validation error: {e}", "symbol": symbol}


def normalize_symbol_for_db(symbol: str) -> str:
    """Normalize symbol to exchange format for database storage"""
    validation = validate_symbol(symbol)
    if not validation["valid"]:
        raise ValueError(validation["reason"])
    return validation["symbol"]


def ensure_database_directory() -> None:
    """Ensure database directory exists"""
    try:
        if not IS_SQLITE or not DATABASE_PATH:
            # Nothing to do for non-sqlite databases
            return
        db_path = Path(DATABASE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database directory ensured: {db_path.parent}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error creating database directory: {e}")
        raise


def create_database() -> bool:
    """Create database and tables - NO SAMPLE DATA"""
    try:
        ensure_database_directory()

        # Enable WAL mode for SQLite only
        if IS_SQLITE:
            with engine.begin() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.execute(text("PRAGMA cache_size=10000"))
                conn.execute(text("PRAGMA temp_store=MEMORY"))

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully - NO SAMPLE DATA SEEDED")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error creating database: {e}")
        return False
    else:
        return True


def initialize_sample_data() -> None:
    """DEPRECATED: Sample data seeding removed - violates 'no hardcoded data' rule"""
    logger.warning("initialize_sample_data() called but is DEPRECATED - no sample data will be seeded")
    logger.info("All data must come from live ingest paths only")


def check_database_health() -> dict:
    """Check database health with lightweight queries"""
    try:
        with get_database_session() as db:
            # Lightweight health check - avoid COUNT(*) on large tables
            result = db.execute(text("SELECT 1")).scalar()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Database health check error: {e}")
        msg = "Database health check failed"
        raise RuntimeError(msg) from e

    # Validate result outside try to avoid TRY301
    if result != 1:
        msg = "Database health check failed"
        raise RuntimeError(msg)

    try:
        with get_database_session() as db:
            # Sample queries instead of full counts
            portfolio_sample = db.execute(select(Portfolio).limit(1)).first()
            strategy_sample = db.execute(select(Strategy).limit(1)).first()
            market_data_sample = db.execute(select(MarketData).limit(1)).first()

            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "has_portfolios": portfolio_sample is not None,
                "has_strategies": strategy_sample is not None,
                "has_market_data": market_data_sample is not None,
                "database_path": DATABASE_PATH if IS_SQLITE else "",
                "database_size_mb": get_database_size(),
            }

            logger.info(f"Database health check passed: {health_status}")
            return health_status

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_database_size() -> float:
    """Get database file size in MB"""
    try:
        if IS_SQLITE and DATABASE_PATH:
            db_path = Path(DATABASE_PATH)
            if db_path.exists():
                size_bytes = db_path.stat().st_size
                result = round(size_bytes / (1024 * 1024), 2)
            else:
                result = 0.0
        else:
            result = 0.0
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return 0.0
    else:
        return result


def cleanup_old_data(days_to_keep: int = 30) -> dict | None:
    """Clean up old data from database - uses bulk deletes, runs in background only"""
    session = None
    try:
        session = get_database_session()
        with session as db:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            # Use bulk deletes instead of loading all rows
            market_data_res = db.execute(delete(MarketData).where(MarketData.timestamp < cutoff_date))
            market_data_deleted = market_data_res.rowcount if market_data_res is not None else 0

            trade_logs_res = db.execute(delete(TradeLog).where(TradeLog.timestamp < cutoff_date))
            trade_logs_deleted = trade_logs_res.rowcount if trade_logs_res is not None else 0

            system_logs_res = db.execute(delete(SystemLog).where(SystemLog.timestamp < cutoff_date))
            system_logs_deleted = system_logs_res.rowcount if system_logs_res is not None else 0

            db.commit()

            cleanup_results = {
                "market_data_deleted": market_data_deleted,
                "trade_logs_deleted": trade_logs_deleted,
                "system_logs_deleted": system_logs_deleted,
                "cutoff_date": cutoff_date.isoformat(),
            }

            logger.info(f"Database cleanup completed: {cleanup_results}")
            return cleanup_results

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Database cleanup failed")
        if session is not None:
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                session.rollback()
        return None


def get_database_session() -> Session:
    """Get database session with proper error handling and context management"""
    try:
        return SessionLocal()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Error creating database session")
        raise


@contextlib.asynccontextmanager
async def session_scope():
    """Context manager for database sessions that ensures proper cleanup."""
    session = None
    try:
        session = get_database_session()
        yield session
        session.commit()
    except Exception:
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()


def bootstrap_database() -> bool:
    """Bootstrap database for app startup - NO SAMPLE DATA"""
    try:
        logger.info("Bootstrapping database...")

        # Create database if it doesn't exist (only for sqlite file-backed DB)
        if IS_SQLITE:
            if DATABASE_PATH and not Path(DATABASE_PATH).exists():
                success = create_database()
                if not success:
                    return False
        else:
            # For non-sqlite backends, ensure tables exist
            Base.metadata.create_all(bind=engine)

        # Verify health
        health = check_database_health()
        if health.get("status") != "healthy":
            logger.error("Database bootstrap failed health check")
            return False

        logger.info("Database bootstrap completed successfully - NO SAMPLE DATA")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Database bootstrap failed")
        return False
    else:
        return True


def get_redis_client() -> Any:
    """Get Redis client for backward compatibility"""
    try:
        redis_service = get_redis_service()
    except ImportError:
        logger.warning("Redis service not available")
        return None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Failed to get Redis client")
        return None
    else:
        return redis_service.redis_client


def get_market_data(symbol: str, limit: int = 100) -> list[dict[str, Any]]:
    """Get market data for a specific symbol"""
    try:
        with get_database_session() as db:
            result = db.execute(select(MarketData).where(MarketData.symbol == symbol).order_by(MarketData.timestamp.desc()).limit(limit)).scalars().all()

            return [
                {
                    "symbol": row.symbol,
                    "price": row.price,
                    "volume": row.volume,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                    "exchange": row.exchange,
                }
                for row in result
            ]
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception(f"Failed to get market data for {symbol}")
        return []


def insert_market_data(symbol: str, price: float, volume: float | None = None) -> dict[str, Any]:
    """Insert market data for a symbol with validation"""
    try:
        # Validate symbol before insert
        normalized_symbol = normalize_symbol_for_db(symbol)

        with get_database_session() as db:
            market_data = MarketData(
                symbol=normalized_symbol,
                price=price,
                volume=volume,
                exchange=EXCHANGE_ID,
            )
            db.add(market_data)
            db.commit()

            return {
                "success": True,
                "symbol": normalized_symbol,
                "original_symbol": symbol,
                "price": price,
                "volume": volume,
            }
    except ValueError as e:
        logger.exception(f"Symbol validation failed for {symbol}")
        return {"success": False, "error": str(e), "symbol": symbol}
    except (TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Failed to insert market data for {symbol}")
        return {"success": False, "error": str(e), "symbol": symbol}


def get_performance_stats() -> dict[str, Any]:
    """Get database performance statistics using efficient aggregate queries"""
    try:
        with get_database_session() as db:
            # Use efficient COUNT queries instead of loading all rows
            portfolio_count = db.execute(select(func.count(Portfolio.id))).scalar() or 0
            market_data_count = db.execute(select(func.count(MarketData.id))).scalar() or 0
            trade_log_count = db.execute(select(func.count(TradeLog.id))).scalar() or 0

            return {
                "total_portfolios": int(portfolio_count),
                "total_market_data": int(market_data_count),
                "total_trades": int(trade_log_count),
                "database_size_mb": get_database_size(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Failed to get performance stats")
        return {"error": str(e)}


def get_health_status() -> dict:
    """Get health status for monitoring endpoints"""
    try:
        health = check_database_health()
        return {
            "module": "database",
            "status": health.get("status", "unhealthy"),
            "details": health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        return {
            "module": "database",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    # Initialize database when run directly
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Mystic Trading Database...")
    success = bootstrap_database()
    if success:
        logger.info("Database initialization completed successfully")
    else:
        logger.error("Database initialization failed")
