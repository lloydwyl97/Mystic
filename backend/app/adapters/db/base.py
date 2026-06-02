"""
Database Interface - All Live Data, No Fallback/Hardcoded Data

This module defines the abstract database interface for persisting live data from the backend API (port 8000).
All database operations:
- Store live data from backend endpoints (port 8000)
- No fallback/hardcoded data - all data from live API calls
- Connected to backend for live data operations
- Abstract interface allows multiple database implementations

Live Data Sources:
- All database operations persist live data from backend endpoints
- Account balances, transactions, trades from live trading operations
- Portfolio snapshots from live account data
- Strategies from live strategy execution

Endpoint References:
- All database operations connected to backend running on port 8000
- Database sessions used by API endpoints to persist live data
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.orm import Session


class Database(ABC):
    """
    Abstract database interface for live data operations.

    All implementations must:
    - Store live data from backend endpoints (port 8000)
    - Provide session management for live data operations
    - Support health checks for live database connections
    - Execute queries against live data stores
    """

    @abstractmethod
    async def session(self) -> AsyncGenerator[Session, None]:
        """
        Get a database session for live data operations.

        Yields:
            Database session for persisting live data from backend (port 8000)
        """

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check database health status for live connections.

        Returns:
            Dictionary with health status of live database connection
        """

    @abstractmethod
    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query against live data store.

        Args:
            query: SQL query to execute
            params: Query parameters (optional)

        Returns:
            List of result dictionaries from live database
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Close database connection.

        Ensures proper cleanup of live database connections.
        """
