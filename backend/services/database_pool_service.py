#!/usr/bin/env python3
"""
Database Connection Pool Service for High-Performance Data Operations
"""

import asyncio
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    path: str
    max_connections: int = 10
    timeout: float = 30.0
    enable_wal: bool = True
    synchronous: str = "NORMAL"  # FULL, NORMAL, or OFF
    cache_size: int = -64000  # Negative for KB, positive for pages
    journal_mode: str = "WAL"


class DatabaseConnectionPool:
    """Connection pool for SQLite database operations"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=config.max_connections)
        self._lock = asyncio.Lock()
        self._created_connections = 0
        self._stats = {
            "connections_created": 0,
            "connections_active": 0,
            "queries_executed": 0,
            "transactions_committed": 0,
            "errors": 0,
            "avg_query_time": 0.0,
        }

        # Initialize the pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(self.config.max_connections):
            conn = self._create_connection()
            self._pool.put_nowait(conn)
            self._stats["connections_created"] += 1

        logger.info(f"Database connection pool initialized with {self.config.max_connections} connections")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings"""
        conn = sqlite3.connect(
            self.config.path,
            timeout=self.config.timeout,
            isolation_level=None,  # Enable autocommit mode
        )

        # Enable WAL mode for better concurrency
        if self.config.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self.config.synchronous}")

        # Set cache size for better performance
        conn.execute(f"PRAGMA cache_size={self.config.cache_size}")

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")

        # Set busy timeout
        conn.execute(f"PRAGMA busy_timeout={int(self.config.timeout * 1000)}")

        # Use row factory for better data access
        conn.row_factory = sqlite3.Row

        return conn

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        start_time = time.time()

        try:
            # BUG #33 FIX: Monitor pool exhaustion
            available = self._pool.qsize()
            if available == 0:
                logger.warning(f"[POOL EXHAUSTION WARNING] No connections available, waiting... (active: {self._stats['connections_active']})")
            if available < self.config.max_connections // 2:
                logger.warning(f"[POOL LOW] Only {available}/{self.config.max_connections} connections available")

            # Get connection from pool with timeout
            conn = await asyncio.wait_for(self._pool.get(), timeout=10.0)

            self._stats["connections_active"] += 1

            try:
                yield conn
            finally:
                # Return connection to pool
                await self._pool.put(conn)
                self._stats["connections_active"] -= 1

        except asyncio.TimeoutError:
            self._stats["errors"] += 1
            logger.exception(f"[POOL TIMEOUT] Could not acquire connection within 10s (active: {self._stats['connections_active']}, available: {self._pool.qsize()})")
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            self._stats["errors"] += 1
            logger.exception(f"Database connection error: {e}")
            raise
        finally:
            # Update timing stats
            elapsed = time.time() - start_time
            self._update_timing_stats(elapsed)

    def _update_timing_stats(self, elapsed: float):
        """Update timing statistics"""
        # Simple moving average for query time
        alpha = 0.1  # Smoothing factor
        self._stats["avg_query_time"] = alpha * elapsed + (1 - alpha) * self._stats["avg_query_time"]

    async def execute(self, query: str, params: tuple | None = None) -> sqlite3.Cursor:
        """Execute a single query"""
        async with self.get_connection() as conn:
            start_time = time.time()
            try:
                cursor = conn.execute(query, params or ())
                self._stats["queries_executed"] += 1
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self._stats["errors"] += 1
                logger.exception(f"Query execution error: {e} - Query: {query}")
                raise
            else:
                return cursor
            finally:
                elapsed = time.time() - start_time
                self._update_timing_stats(elapsed)

    async def execute_many(self, query: str, param_list: list[tuple]) -> int:
        """Execute multiple queries in batch"""
        async with self.get_connection() as conn:
            start_time = time.time()
            try:
                cursor = conn.executemany(query, param_list)
                affected_rows = cursor.rowcount
                self._stats["queries_executed"] += len(param_list)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self._stats["errors"] += 1
                logger.exception(f"Batch execution error: {e} - Query: {query}")
                raise
            else:
                return affected_rows
            finally:
                elapsed = time.time() - start_time
                self._update_timing_stats(elapsed)

    async def fetch_one(self, query: str, params: tuple | None = None) -> sqlite3.Row | None:
        """Fetch a single row"""
        async with self.get_connection() as conn:
            start_time = time.time()
            try:
                cursor = conn.execute(query, params or ())
                row = cursor.fetchone()
                self._stats["queries_executed"] += 1
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self._stats["errors"] += 1
                logger.exception(f"Fetch one error: {e} - Query: {query}")
                raise
            else:
                return row
            finally:
                elapsed = time.time() - start_time
                self._update_timing_stats(elapsed)

    async def fetch_all(self, query: str, params: tuple | None = None) -> list[sqlite3.Row]:
        """Fetch all rows"""
        async with self.get_connection() as conn:
            start_time = time.time()
            try:
                cursor = conn.execute(query, params or ())
                rows = cursor.fetchall()
                self._stats["queries_executed"] += 1
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                self._stats["errors"] += 1
                logger.exception(f"Fetch all error: {e} - Query: {query}")
                raise
            else:
                return rows
            finally:
                elapsed = time.time() - start_time
                self._update_timing_stats(elapsed)

    async def execute_transaction(self, queries: list[tuple[str, tuple]]) -> bool:
        """Execute multiple queries in a transaction"""
        async with self.get_connection() as conn:
            start_time = time.time()
            try:
                # Begin transaction
                conn.execute("BEGIN TRANSACTION")

                for query, params in queries:
                    conn.execute(query, params or ())

                # Commit transaction
                conn.commit()
                self._stats["transactions_committed"] += 1
                self._stats["queries_executed"] += len(queries)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                # Rollback on error
                with suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    conn.rollback()

                self._stats["errors"] += 1
                logger.exception(f"Transaction error: {e}")
                raise
            else:
                return True
            finally:
                elapsed = time.time() - start_time
                self._update_timing_stats(elapsed)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics"""
        return {
            **self._stats,
            "pool_size": self.config.max_connections,
            "available_connections": self._pool.qsize(),
            "database_path": self.config.path,
        }

    async def close(self):
        """Close all connections in the pool"""
        logger.info("Closing database connection pool...")

        connections = []
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                connections.append(conn)
            except asyncio.QueueEmpty:
                break

        # Close connections in background to avoid blocking
        async def close_connections():
            for conn in connections:
                try:
                    conn.close()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error closing connection: {e}")

        task = await task_manager.create_task(close_connections(), name="database_pool_service:close_connections")
        # Store task reference if class has task tracking
        if hasattr(self, "_tasks"):
            self._tasks.append(task)
        elif not hasattr(self, "_tasks"):
            self._tasks: list[asyncio.Task[Any]] = []
            self._tasks.append(task)
        logger.info("Database connection pool closed")


class BulkOperationManager:
    """Manager for bulk database operations"""

    def __init__(self, pool: DatabaseConnectionPool, batch_size: int = 1000):
        self.pool = pool
        self.batch_size = batch_size
        self._operation_queues: dict[str, list[tuple]] = {}
        self._lock = asyncio.Lock()

    async def queue_operation(self, operation_type: str, query: str, params: tuple):
        """Queue an operation for batch execution"""
        async with self._lock:
            if operation_type not in self._operation_queues:
                self._operation_queues[operation_type] = []

            self._operation_queues[operation_type].append((query, params))

            # Execute if batch size reached
            if len(self._operation_queues[operation_type]) >= self.batch_size:
                await self._execute_batch(operation_type)

    async def flush_all(self):
        """Flush all queued operations"""
        async with self._lock:
            for operation_type in list(self._operation_queues.keys()):
                if self._operation_queues[operation_type]:
                    await self._execute_batch(operation_type)

    async def _execute_batch(self, operation_type: str):
        """Execute a batch of operations"""
        operations = self._operation_queues[operation_type]
        self._operation_queues[operation_type] = []

        if not operations:
            return

        try:
            # Convert to transaction format
            transaction_queries = operations
            await self.pool.execute_transaction(transaction_queries)
            logger.info(f"Executed batch of {len(operations)} {operation_type} operations")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Batch execution failed for {operation_type}: {e}")
            # Re-queue failed operations
            self._operation_queues[operation_type].extend(operations)


class DatabasePoolService:
    """Service managing database connection pools"""

    def __init__(self):
        self.pools: dict[str, DatabaseConnectionPool] = {}
        self.bulk_managers: dict[str, BulkOperationManager] = {}
        self._cleanup_task: asyncio.Task | None = None

    def get_or_create_pool(self, name: str, config: DatabaseConfig) -> DatabaseConnectionPool:
        """Get existing pool or create new one"""
        if name not in self.pools:
            self.pools[name] = DatabaseConnectionPool(config)
            self.bulk_managers[name] = BulkOperationManager(self.pools[name])

        return self.pools[name]

    def get_bulk_manager(self, name: str) -> BulkOperationManager | None:
        """Get bulk operation manager for a database"""
        return self.bulk_managers.get(name)

    async def get_session(self, name: str = "default"):
        """Get database session from pool"""
        pool = self.pools.get(name)
        if not pool:
            # Create default SQLite pool if it doesn't exist
            config = DatabaseConfig(
                path=DATABASE_PATH,
                max_connections=10,
                timeout=30.0,
            )
            pool = self.get_or_create_pool(name, config)
        return await pool.get_connection()

    async def start_cleanup_task(self):
        """Start periodic cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = await task_manager.create_task(self._periodic_cleanup(), name="database_pool_service:periodic_cleanup")
            # Also store in general task list if available
            if hasattr(self, "_tasks"):
                self._tasks.append(self._cleanup_task)

    async def stop_cleanup_task(self):
        """Stop cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    async def _periodic_cleanup(self):
        """Periodic cleanup of resources"""
        while True:
            try:
                # Flush all bulk operations every 30 seconds
                for manager in self.bulk_managers.values():
                    await manager.flush_all()

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                # BUG #46 FIX: Clean exit on cancellation
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Periodic cleanup error: {e}")
                await asyncio.sleep(10)

    async def close_all_pools(self):
        """Close all connection pools"""
        logger.info("Closing all database connection pools...")

        tasks = []
        for _name, pool in self.pools.items():
            task = await task_manager.create_task(pool.close(), name="database_pool_service:close_pool")
            tasks.append(task)
            # Store task reference if class has task tracking
            if hasattr(self, "_tasks"):
                self._tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All database connection pools closed")

    async def cleanup_all(self):
        """Complete cleanup: stop periodic cleanup, close all connections, and clear data structures"""
        logger.info("Starting complete database pool service cleanup...")

        # Stop the periodic cleanup task
        await self.stop_cleanup_task()

        # Close all connection pools
        await self.close_all_pools()

        # Clear the data structures
        self.pools.clear()
        self.bulk_managers.clear()

        logger.info("Database pool service cleanup complete")

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all pools"""
        return {name: pool.get_stats() for name, pool in self.pools.items()}


# Global database pool service instance
database_pool_service = DatabasePoolService()


# Convenience functions for common databases
def get_main_db_pool() -> DatabaseConnectionPool:
    """Get pool for main trading database"""
    config = DatabaseConfig(
        path=DATABASE_PATH,
        max_connections=5,  # Lower for WAL mode
        enable_wal=True,
        synchronous="NORMAL",
        cache_size=-64000,  # 64MB cache
    )
    return database_pool_service.get_or_create_pool("main", config)


def get_ai_db_pool() -> DatabaseConnectionPool:
    """Get pool for AI training database"""
    config = DatabaseConfig(
        path=os.getenv("AI_DB_PATH", "ai_live.sqlite"),
        max_connections=3,  # AI DB has less concurrent access
        enable_wal=True,
        synchronous="NORMAL",
        cache_size=-32000,  # 32MB cache
    )
    return database_pool_service.get_or_create_pool("ai", config)


def get_cache_db_pool() -> DatabaseConnectionPool:
    """Get pool for cache database"""
    config = DatabaseConfig(
        path=os.getenv("CACHE_DB_PATH", "cache.db"),
        max_connections=10,  # Higher for cache operations
        enable_wal=False,  # Cache can use rollback journal
        synchronous="OFF",  # Performance over durability for cache
        cache_size=-16000,  # 16MB cache
    )
    return database_pool_service.get_or_create_pool("cache", config)


def get_database_pool_service() -> DatabasePoolService:
    """Get the global database pool service instance"""
    return database_pool_service
