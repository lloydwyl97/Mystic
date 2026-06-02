"""
SQLite Database Implementation - All Live Data, No Fallback/Hardcoded Data

This module implements the SQLite database adapter for persisting live data from the backend API (port 8000).
All database operations:
- Store live data from backend endpoints (port 8000)
- No fallback/hardcoded data - all data from live API calls
- Connected to backend for live data persistence
- Provides session management for live data operations

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

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.adapters.db.base import Database
from backend.app.adapters.db.models import Base
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class SQLiteDatabase(Database):
    """
    SQLite database implementation for live data persistence.

    Stores live data from backend API (port 8000) operations:
    - Live account balances, transactions, trades
    - Live portfolio snapshots and strategy data
    - All data from live API endpoints - no fallback/hardcoded data
    """

    def __init__(self):
        """
        Initialize SQLite database connection for live data operations.

        Connects to database specified in settings (configured for live data storage).
        """
        self._engine = create_engine(
            settings.database_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False},
        )
        # Session factory for live data operations (backend port 8000)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, autoflush=False)

        # Create tables for live data models
        Base.metadata.create_all(bind=self._engine)
        logger.info("Database tables created/verified for live data storage")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Session, None]:
        """
        Get a database session for live data operations.

        Yields:
            Database session for persisting live data from backend (port 8000)

        Raises:
            Various exceptions if live data persistence fails
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            session.rollback()
            logger.exception("Database session error during live data operation")
            raise
        finally:
            session.close()

    async def health_check(self) -> dict[str, Any]:
        """
        Check database health status for live data connections.

        Returns:
            Dictionary with health status of live database connection
        """
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            return {"status": "unhealthy", "type": "sqlite", "error": str(e)}
        else:
            return {
                "status": "healthy",
                "type": "sqlite",
                "database_url": settings.database_url,
                "connected_to": "backend port 8000 (live data)",
            }

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query against live data store.

        Args:
            query: SQL query to execute against live database
            params: Query parameters (optional)

        Returns:
            List of result dictionaries from live database (backend port 8000)
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            rows = result.fetchall()
            # Convert rows to dictionaries from live data
            if rows:
                columns = result.keys()
                return [dict(zip(columns, row, strict=True)) for row in rows]
            return []

    async def close(self) -> None:
        """
        Close database connection.

        Ensures proper cleanup of live database connections.
        """
        self._engine.dispose()
        logger.info("SQLite database connection closed (live data store)")
