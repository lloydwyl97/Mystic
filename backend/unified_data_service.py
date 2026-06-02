#!/usr/bin/env python3
"""
Unified Data Service - Live Configuration Only

Provides a single interface to access all trading data from multiple sources.
All configuration values come from live config - no hardcoded values.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

try:
    from backend.services.confidence_normalizer import ConfidenceNormalizer
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    ConfidenceNormalizer = None  # type: ignore[assignment,misc]

# --- Defensive imports for backend hooks ----------------------------------------------------------
try:
    # Expecting these to return iterable collections of trade/strategy records
    from backend.db_logger import (
        get_active_strategies as get_db_logger_strategies,
    )
    from backend.db_logger import (
        get_recent_trades as get_db_logger_trades,
    )
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
    logger.warning("Falling back: backend.db_logger not available (%s). Using no-op providers.", e)

    def get_db_logger_trades(_limit: int | None = None):
        """No-op fallback when db_logger is unavailable."""
        return []  # type: ignore[func-returns-value]

    def get_db_logger_strategies():
        """No-op fallback when db_logger is unavailable."""
        return []  # type: ignore[func-returns-value]


# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_sim_db_path() -> str:
    """Get simulation database path from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "simulation") and hasattr(value.simulation, "db_path"):
                db_path = value.simulation.db_path
                if isinstance(db_path, str) and db_path:
                    return db_path.strip()
            if hasattr(value, "unified_data") and hasattr(value.unified_data, "sim_db_path"):
                db_path = value.unified_data.sim_db_path
                if isinstance(db_path, str) and db_path:
                    return db_path.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    db_path = os.getenv("SIM_DB_PATH", "").strip()
    if db_path:
        return db_path

    # Fallback to default relative path (filename from env)
    fn = os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")
    return str(Path(__file__).resolve().parent.parent / fn)


def _get_db_timeout() -> float:
    """Get SQLite connection timeout from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "unified_data") and hasattr(value.unified_data, "db_timeout"):
                timeout = value.unified_data.db_timeout
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass

    timeout = os.getenv("UNIFIED_DATA_DB_TIMEOUT", "").strip()
    if timeout:
        try:
            return float(timeout)
        except (ValueError, TypeError):
            pass

    return 10.0


def _get_db_busy_timeout() -> int:
    """Get SQLite busy timeout from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "unified_data") and hasattr(value.unified_data, "db_busy_timeout"):
                timeout = value.unified_data.db_busy_timeout
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return int(timeout)
        except (AttributeError, ValueError, TypeError):
            pass

    timeout = os.getenv("UNIFIED_DATA_DB_BUSY_TIMEOUT", "").strip()
    if timeout:
        try:
            return int(timeout)
        except (ValueError, TypeError):
            pass

    return 10000


def _get_default_trades_limit() -> int:
    """Get default trades limit from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "unified_data") and hasattr(value.unified_data, "default_trades_limit"):
                limit = value.unified_data.default_trades_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass

    limit = os.getenv("UNIFIED_DATA_DEFAULT_TRADES_LIMIT", "").strip()
    if limit:
        try:
            return int(limit)
        except (ValueError, TypeError):
            pass

    return 100


def _get_default_stats_limit() -> int:
    """Get default stats limit from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "unified_data") and hasattr(value.unified_data, "stats_limit"):
                limit = value.unified_data.stats_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass

    limit = os.getenv("UNIFIED_DATA_STATS_LIMIT", "").strip()
    if limit:
        try:
            return int(limit)
        except (ValueError, TypeError):
            pass

    return 1000


def _get_trades_multiplier() -> int:
    """Get multiplier for fetching more trades to account for filtering."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "unified_data") and hasattr(value.unified_data, "trades_multiplier"):
                multiplier = value.unified_data.trades_multiplier
                if isinstance(multiplier, int) and multiplier > 0:
                    return multiplier
        except (AttributeError, ValueError, TypeError):
            pass

    multiplier = os.getenv("UNIFIED_DATA_TRADES_MULTIPLIER", "").strip()
    if multiplier:
        try:
            return int(multiplier)
        except (ValueError, TypeError):
            pass

    return 2


# --- Helpers --------------------------------------------------------------------------------------


def _connect_sim_db() -> sqlite3.Connection:
    """
    Get a SQLite connection with row access by name and sensible time parsing.
    Callers MUST use try/finally or `with _connect_sim_db() as conn:` to ensure the connection is closed.
    """
    conn = None
    try:
        db_path = _get_sim_db_path()
        db_timeout = _get_db_timeout()
        busy_timeout = _get_db_busy_timeout()
        conn = sqlite3.connect(db_path, timeout=db_timeout, isolation_level=None)  # autocommit mode
        conn.row_factory = sqlite3.Row
        try:
            # Enable WAL to reduce writer lock contention on Windows
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout};")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        # Ensure schema exists
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulated_trades (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    price REAL,
                    confidence REAL,
                    simulated_profit REAL,
                    strategy TEXT
                )
                """,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Schema check failed for simulated_trades: %s", e)
        return conn
    except Exception as ex:
        if conn is not None:
            conn.close()
        logger.warning("_get_sim_db_connection failed: %s", ex)
        raise


def _parse_ts(ts: Any) -> datetime | None:
    """
    Parse many timestamp shapes into a (naive) datetime for consistent sorting.
    - ISO strings, possibly with 'Z'
    - datetime instances (returned as-is)
    - Unix epoch numbers
    Returns None on failure.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        # Normalize to timezone-aware UTC
        return ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return None
    if isinstance(ts, str):
        try:
            # Handle trailing Z
            s = ts.replace("Z", "+00:00") if "Z" in ts and "+" not in ts else ts
            dt = datetime.fromisoformat(s)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Last resort: try stripping microseconds/oddities
            try:
                return datetime.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return None
    return None


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely retrieve attribute or key from unknown shaped record."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    if is_dataclass(obj):
        try:
            d = asdict(obj)
            return d.get(attr, default)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return default
    # plain object with attributes
    return getattr(obj, attr, default)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


# --- Public API -----------------------------------------------------------------------------------


def get_simulated_trades(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Get trades from the local simulation database (table: simulated_trades).

    Returns unified-like shape:
    - id, timestamp, symbol, side, price, confidence, profit, strategy, exit_price, entry_price
    """
    if limit is None:
        limit = _get_default_trades_limit()
    try:
        with _connect_sim_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, timestamp, symbol, action, price, confidence, simulated_profit, strategy
                FROM simulated_trades
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

        trades: list[dict[str, Any]] = []
        for r in rows:
            price = _to_float(r["price"], 0.0)
            sim_profit = _to_float(r["simulated_profit"], None) if r["simulated_profit"] is not None else None
            ts = r["timestamp"]
            trades.append(
                {
                    "id": r["id"],
                    "timestamp": ts,  # keep raw; normalized in get_unified_trades
                    "symbol": r["symbol"],
                    "side": (r["action"] or "BUY").upper(),
                    "price": price,
                    "confidence": ConfidenceNormalizer.normalize(_to_float(r["confidence"], 0.0)) if ConfidenceNormalizer else _to_float(r["confidence"], 0.0),
                    "profit": sim_profit,
                    "strategy": r["strategy"],
                    # If we only have a single recorded price, reflect it; keep None if truly unknown
                    "exit_price": price if sim_profit is not None else None,
                    "entry_price": price,
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("Could not get simulated trades: %s", e)
        return []
    else:
        return trades


def get_unified_trades(limit: int | None = None, offset: int = 0) -> dict[str, Any]:
    """
    Combine trades from db_logger (real) and local simulation (paper).
    Output is sorted most-recent-first by timestamp and limited to `limit`.
    Returns paginated response with metadata.
    """
    if limit is None:
        limit = _get_default_trades_limit()
    multiplier = _get_trades_multiplier()
    all_trades: list[dict[str, Any]] = []

    # Real trades
    try:
        real_trades = get_db_logger_trades(limit=limit * multiplier) or []  # Get more to account for filtering
        for trade in real_trades:
            entry_price = _safe_get(trade, "entry_price")
            exit_price = _safe_get(trade, "exit_price")
            ts = _safe_get(trade, "timestamp")
            symbol = _safe_get(trade, "pair") or _safe_get(trade, "symbol") or "Unknown"
            side = (_safe_get(trade, "side") or "buy").lower()
            profit = _safe_get(trade, "profit")

            all_trades.append(
                {
                    "id": f"real_{_safe_get(trade, 'id', '')}",
                    "timestamp": ts,
                    "symbol": symbol,
                    "side": side,
                    # Omit price if unknown to avoid downstream rendering errors
                    "price": float(entry_price) if isinstance(entry_price, (int, float)) else None,
                    "profit": profit,
                    "strategy": "Real Trading",
                    "exit_price": exit_price,
                    "entry_price": entry_price,
                    "source": "real",
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("Could not get real trades: %s", e)

    # Simulated trades
    try:
        sim_trades = get_simulated_trades(limit=limit * multiplier)  # Get more to account for filtering
        for t in sim_trades:
            ts = t["timestamp"]
            # Normalize string -> datetime for consistent sorting later in this function
            parsed_ts = _parse_ts(ts)
            all_trades.append(
                {
                    "id": f"sim_{t['id']}",
                    "timestamp": parsed_ts if parsed_ts else ts,  # keep something, normalize in sort
                    "symbol": t["symbol"],
                    "side": (t["side"] or "buy").lower(),
                    "price": float(t["price"]) if isinstance(t["price"], (int, float)) else None,
                    "profit": t["profit"],
                    "strategy": t.get("strategy") or "Paper Trading",
                    # Keep entry/exit as provided; do not force both to same value if unknown
                    "exit_price": t.get("exit_price"),
                    "entry_price": t.get("entry_price"),
                    "source": "simulated",
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning("Could not get simulated trades (merge): %s", e)

    # Sort by timestamp (most recent first). Normalize to aware UTC datetime.
    def _sort_key(tr: dict[str, Any]) -> datetime:
        dt = _parse_ts(tr.get("timestamp"))
        return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

    all_trades.sort(key=_sort_key, reverse=True)

    # Apply pagination
    total_count = len(all_trades)
    paginated_trades = all_trades[offset : offset + limit]

    return {
        "trades": paginated_trades,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total_count,
            "has_more": offset + limit < total_count,
        },
    }


def get_unified_stats() -> dict[str, Any]:
    """Aggregate stats from unified trades and backend strategies."""
    try:
        stats_limit = _get_default_stats_limit()
        trades_response = get_unified_trades(limit=stats_limit)
        all_trades = trades_response["trades"] if isinstance(trades_response, dict) else trades_response
        completed = [t for t in all_trades if t.get("profit") is not None]
        total = len(completed)

        if total == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "active_strategies": 0,
                "paper_trades": 0,
                "real_trades": 0,
            }

        wins = sum(1 for t in completed if _to_float(t.get("profit"), 0.0) > 0.0)
        total_profit = sum(_to_float(t.get("profit"), 0.0) for t in completed)

        paper_trades = sum(1 for t in completed if t.get("source") == "simulated")
        real_trades = sum(1 for t in completed if t.get("source") == "real")

        try:
            active_strategies = len(list(get_db_logger_strategies() or []))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            active_strategies = 0

        return {
            "total_trades": total,
            "win_rate": ((wins / total) * 100.0) if total else 0.0,  # percent 0-100
            "total_profit": total_profit,
            "active_strategies": active_strategies,
            "paper_trades": paper_trades,
            "real_trades": real_trades,
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error getting unified stats: %s", e)
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "active_strategies": 0,
            "paper_trades": 0,
            "real_trades": 0,
        }


def get_unified_profit_history(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Build cumulative PnL history for charting.
    Returns a list sorted by time ascending:
      [{ timestamp, cumulative_profit, trade_profit, symbol, source }, ...]
    """
    if limit is None:
        limit = _get_default_trades_limit()
    try:
        trades_response = get_unified_trades(limit=limit)
        all_trades = trades_response["trades"] if isinstance(trades_response, dict) else trades_response
        completed = [t for t in all_trades if t.get("profit") is not None]

        # Sort oldest -> newest for cumulative sums
        def _asc_key(tr: dict[str, Any]) -> datetime:
            dt = _parse_ts(tr.get("timestamp"))
            return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

        completed.sort(key=_asc_key)

        cum = 0.0
        history: list[dict[str, Any]] = []
        for tr in completed:
            p = _to_float(tr.get("profit"), 0.0)
            cum += p
            ts = _parse_ts(tr.get("timestamp"))
            history.append(
                {
                    "timestamp": (ts.isoformat().replace("+00:00", "Z") if ts else str(tr.get("timestamp"))),
                    "cumulative_profit": cum,
                    "trade_profit": p,
                    "symbol": tr.get("symbol", "Unknown"),
                    "source": tr.get("source", "unknown"),
                },
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("Error getting profit history: %s", e)
        return []
    else:
        return history
