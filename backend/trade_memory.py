from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal

try:
    import redis.asyncio as redis
except (ImportError, ModuleNotFoundError):
    redis = None
from backend.config.redis_config import get_shared_redis_async

logger = logging.getLogger(__name__)

Side = Literal["BUY", "SELL"]
Status = Literal["open", "closed"]

MAX_CLOSED_TRADES_IN_MEMORY = int(os.getenv("MAX_CLOSED_TRADES_IN_MEMORY", "0"))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_time: str = field(default_factory=_utcnow_iso)

    # Closed/optional fields
    status: Status = "open"
    exit_price: float | None = None
    exit_time: str | None = None
    realized_pnl: float | None = None

    # Free-form metadata (strategy, tags, notes, etc.)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Trade:
        # helper to pick the first non-None value from possible keys
        def _pick(keys: tuple[str, ...], default: Any = None) -> Any:
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        trade_id = _pick(("trade_id", "id"), None)
        if trade_id is None:
            msg = "trade_id is required in trade dict"
            raise ValueError(msg)

        symbol = str(_pick(("symbol", "pair"), "UNKNOWN"))
        side_val = _pick(("side",), "BUY")
        side = side_val if side_val in ("BUY", "SELL") else "BUY"

        qty_val = _pick(("qty", "quantity"), 0.0)
        try:
            qty = float(qty_val)
        except (TypeError, ValueError):
            qty = 0.0

        entry_price_val = _pick(("entry_price", "price"), 0.0)
        try:
            entry_price = float(entry_price_val)
        except (TypeError, ValueError):
            entry_price = 0.0

        entry_time = _pick(("entry_time",), _utcnow_iso())
        status_val = _pick(("status",), "open")
        status = status_val if status_val in ("open", "closed") else "open"

        exit_price_raw = _pick(("exit_price",), None)
        exit_price = None
        if exit_price_raw is not None:
            try:
                exit_price = float(exit_price_raw)
            except (TypeError, ValueError):
                exit_price = None

        exit_time = _pick(("exit_time",), None)

        realized_raw = _pick(("realized_pnl", "realizedPnl"), None)
        realized_pnl = None
        if realized_raw is not None:
            try:
                realized_pnl = float(realized_raw)
            except (TypeError, ValueError):
                realized_pnl = None

        meta = _pick(("meta",), {})
        if not isinstance(meta, dict):
            meta = {}

        return Trade(
            trade_id=str(trade_id),
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_time=entry_time,
            status=status,
            exit_price=exit_price,
            exit_time=exit_time,
            realized_pnl=realized_pnl,
            meta=meta,
        )

    def compute_unrealized_pnl(self, mark_price: float | None) -> float | None:
        if self.status == "closed" or mark_price is None:
            return None
        delta = (mark_price - self.entry_price) if self.side == "BUY" else (self.entry_price - mark_price)
        return delta * self.qty

    def close(self, exit_price: float, exit_time: str | None = None) -> None:
        self.exit_price = float(exit_price)
        self.exit_time = exit_time or _utcnow_iso()
        # PnL uses standard convention: BUY profit = (exit - entry) * qty; SELL profit = (entry - exit) * qty
        delta = (self.exit_price - self.entry_price) if self.side == "BUY" else (self.entry_price - self.exit_price)
        self.realized_pnl = delta * self.qty
        self.status = "closed"


class TradeMemory:
    """
    In-memory trade state manager for live trading.
    Stores open trades in memory; persists closed trades to Redis.

    Memory optimization: Closed trades are automatically moved to Redis
    to prevent unbounded memory growth. Only open trades remain in memory.

    Thread-safe via RLock.
    """

    def __init__(self) -> None:
        # trade_id -> Trade (ONLY OPEN TRADES kept in memory)
        self.trades: dict[str, Trade] = {}
        self.open_trades: set[str] = set()
        # closed_trades set kept for stats calculation, but trades dict is pruned
        self.closed_trades: set[str] = set()
        self.by_symbol: dict[str, set[str]] = {}
        self._lock = RLock()

        # Redis for closed trade persistence
        self.redis_client: redis.Redis | None = None
        self.use_redis = redis is not None

    async def _ensure_redis(self) -> bool:
        """Ensure Redis connection is available for closed trade persistence"""
        if not self.use_redis:
            logger.warning("Redis not available - closed trades will not be persisted")
            return False

        if self.redis_client is None:
            try:
                self.redis_client = get_shared_redis_async()
                if self.redis_client is None:
                    logger.warning("Shared Redis client unavailable - closed trades will not be persisted")
                    return False
                await self.redis_client.ping()
                logger.info("Redis connection established for TradeMemory persistence")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return False

        return True

    async def _persist_closed_trade(self, trade: Trade) -> bool:
        """Persist a closed trade to Redis and remove from memory"""
        try:
            if not await self._ensure_redis():
                return False

            trade_data = trade.to_dict()
            key = f"closed_trade:{trade.trade_id}"

            await self.redis_client.hset(
                key,
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in trade_data.items()},
            )

            ttl_days = int(os.getenv("CLOSED_TRADE_TTL_DAYS", "90"))
            await self.redis_client.expire(key, ttl_days * 86400)

            logger.info(f"Persisted closed trade {trade.trade_id} to Redis (TTL: {ttl_days} days)")

        except Exception as e:
            logger.exception(f"Failed to persist closed trade {trade.trade_id}: {e}")
            return False
        else:
            return True

    # --- Original-compatible API ---

    def add_trade(self, trade_id: str, trade_data: dict[str, Any]) -> bool:
        """
        Add a new trade to memory and mark as open.
        Returns True if added, False if trade_id already exists.
        """
        with self._lock:
            if trade_id in self.trades:
                return False
            # normalize and create Trade dataclass
            td = dict(trade_data)
            td["trade_id"] = trade_id
            trade = Trade.from_dict(td)
            trade.status = "open"  # ensure open on add
            self.trades[trade_id] = trade
            self.open_trades.add(trade_id)
            self.closed_trades.discard(trade_id)
            # symbol index
            self.by_symbol.setdefault(trade.symbol, set()).add(trade_id)
            return True

    async def close_trade_async(self, trade_id: str, close_data: dict[str, Any] | None = None) -> bool:
        """
        Mark a trade as closed, persist to Redis, and remove from memory.
        close_data can include exit_price, exit_time, meta updates, etc.
        Returns True if closed; False if not found or already closed.

        Memory optimization: Closed trades are persisted to Redis and removed
        from the in-memory trades dict to prevent memory leaks.
        """
        with self._lock:
            trade = self.trades.get(trade_id)
            if not trade or trade.status == "closed":
                return False

            exit_price = close_data.get("exit_price") if close_data else None
            if exit_price is None:
                exit_price = close_data.get("price") if close_data else None
            if exit_price is None:
                return False

            trade.close(float(exit_price), exit_time=(close_data or {}).get("exit_time"))

            extra_meta = (close_data or {}).get("meta")
            if isinstance(extra_meta, dict) and extra_meta:
                trade.meta.update(extra_meta)

            self.open_trades.discard(trade_id)
            self.closed_trades.add(trade_id)

            await self._persist_closed_trade(trade)

            if MAX_CLOSED_TRADES_IN_MEMORY == 0:
                del self.trades[trade_id]
                logger.debug(f"Removed closed trade {trade_id} from memory")

            return True

    def close_trade(self, trade_id: str, close_data: dict[str, Any] | None = None) -> bool:
        """
        Synchronous wrapper for close_trade_async.
        Mark a trade as closed and update its data (e.g., exit_price).
        close_data can include exit_price, exit_time, meta updates, etc.
        Returns True if closed; False if not found or already closed.

        Note: This is a compatibility wrapper. Closed trades are kept in memory
        in sync mode. Use close_trade_async for memory-optimized persistence.
        """
        with self._lock:
            trade = self.trades.get(trade_id)
            if not trade or trade.status == "closed":
                return False

            exit_price = close_data.get("exit_price") if close_data else None
            if exit_price is None:
                exit_price = close_data.get("price") if close_data else None
            if exit_price is None:
                return False

            trade.close(float(exit_price), exit_time=(close_data or {}).get("exit_time"))

            extra_meta = (close_data or {}).get("meta")
            if isinstance(extra_meta, dict) and extra_meta:
                trade.meta.update(extra_meta)

            self.open_trades.discard(trade_id)
            self.closed_trades.add(trade_id)
            return True

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Return a list of open trade dicts."""
        with self._lock:
            return [self.trades[tid].to_dict() for tid in self.open_trades if tid in self.trades]

    def reset_memory(self) -> None:
        """Clear all trade memory (use with caution)."""
        with self._lock:
            self.trades.clear()
            self.open_trades.clear()
            self.closed_trades.clear()
            self.by_symbol.clear()

    # --- Extended API ---

    def get_trade(self, trade_id: str) -> dict[str, Any] | None:
        with self._lock:
            t = self.trades.get(trade_id)
            return t.to_dict() if t else None

    async def get_closed_trades_async(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get closed trades from memory and Redis.
        If closed trades were removed from memory, fetch from Redis.
        """
        with self._lock:
            in_memory = [self.trades[tid].to_dict() for tid in self.closed_trades if tid in self.trades]

        if len(in_memory) >= len(self.closed_trades):
            return in_memory[:limit] if limit else in_memory

        try:
            if await self._ensure_redis():
                missing_ids = self.closed_trades - set(self.trades.keys())
                for tid in missing_ids:
                    key = f"closed_trade:{tid}"
                    trade_data = await self.redis_client.hgetall(key)
                    if trade_data:
                        parsed = {k: json.loads(v) if k in ("meta",) and v else v for k, v in trade_data.items()}
                        in_memory.append(parsed)
        except Exception as e:
            logger.exception(f"Error fetching closed trades from Redis: {e}")

        return in_memory[:limit] if limit else in_memory

    def get_closed_trades(self) -> list[dict[str, Any]]:
        """Synchronous version - returns only closed trades still in memory"""
        with self._lock:
            return [self.trades[tid].to_dict() for tid in self.closed_trades if tid in self.trades]

    def list_by_symbol(self, symbol: str, include_closed: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            ids = self.by_symbol.get(symbol, set())
            result: list[dict[str, Any]] = []
            for tid in ids:
                t = self.trades.get(tid)
                if not t:
                    continue
                if include_closed or t.status == "open":
                    result.append(t.to_dict())
            return result

    def update_trade(self, trade_id: str, **updates: Any) -> bool:
        """
        Partially update a trade (qty, entry_price, meta, etc.).
        Returns True if updated; False if trade not found.
        """
        with self._lock:
            t = self.trades.get(trade_id)
            if not t:
                return False

            # symbol change requires index maintenance
            new_symbol = updates.pop("symbol", None)
            if new_symbol and new_symbol != t.symbol:
                # remove old
                sset = self.by_symbol.get(t.symbol)
                if sset:
                    sset.discard(trade_id)
                # add new
                t.symbol = str(new_symbol)
                self.by_symbol.setdefault(t.symbol, set()).add(trade_id)

            # direct numeric fields (optional)
            if "qty" in updates and updates["qty"] is not None:
                with contextlib.suppress(TypeError, ValueError):
                    t.qty = float(updates["qty"])
            if "entry_price" in updates and updates["entry_price"] is not None:
                with contextlib.suppress(TypeError, ValueError):
                    t.entry_price = float(updates["entry_price"])
            if "side" in updates and updates["side"] in ("BUY", "SELL"):
                t.side = updates["side"]

            # meta merge
            if "meta" in updates and isinstance(updates["meta"], dict):
                t.meta.update(updates["meta"])

            # allow status update only if valid transition
            if updates.get("status") == "closed" and t.status == "open" and "exit_price" in updates:
                try:
                    t.close(float(updates["exit_price"]), exit_time=updates.get("exit_time"))
                    self.open_trades.discard(trade_id)
                    self.closed_trades.add(trade_id)
                    logger.info(f"Trade {trade_id} closed via update_trade (synchronous - not persisted to Redis)")
                except (TypeError, ValueError):
                    pass

            return True

    def remove_trade(self, trade_id: str) -> bool:
        """Remove a trade entirely. Returns True if removed."""
        with self._lock:
            t = self.trades.pop(trade_id, None)
            if not t:
                return False
            self.open_trades.discard(trade_id)
            self.closed_trades.discard(trade_id)
            sset = self.by_symbol.get(t.symbol)
            if sset:
                sset.discard(trade_id)
                if not sset:
                    self.by_symbol.pop(t.symbol, None)
            return True

    async def stats_async(self, mark_prices: dict[str, float] | None = None) -> dict[str, Any]:
        """
        Returns aggregate stats (async version with Redis support):
            open_count, closed_count, realized_pnl, unrealized_pnl, total_pnl
        If mark_prices is provided (e.g., {"BTCUSDT": 50000.0}), unrealized PnL is computed for open trades.

        Fetches closed trades from Redis if not in memory.
        """
        with self._lock:
            realized = 0.0
            unrealized = 0.0

            for tid in self.closed_trades:
                t = self.trades.get(tid)
                if t and t.realized_pnl is not None:
                    realized += t.realized_pnl

            if mark_prices:
                for tid in self.open_trades:
                    t = self.trades.get(tid)
                    if not t:
                        continue
                    keys = (t.symbol, t.symbol.replace("/", ""))
                    mp = next((mark_prices.get(k) for k in keys if k in mark_prices), None)
                    upnl = t.compute_unrealized_pnl(mp)
                    if upnl is not None:
                        unrealized += upnl

            missing_closed = self.closed_trades - set(self.trades.keys())

        if missing_closed and await self._ensure_redis():
            try:
                for tid in missing_closed:
                    key = f"closed_trade:{tid}"
                    pnl_str = await self.redis_client.hget(key, "realized_pnl")
                    if pnl_str:
                        with contextlib.suppress(TypeError, ValueError):
                            realized += float(pnl_str)
            except Exception as e:
                logger.warning(f"Error fetching closed trade stats from Redis: {e}")

        return {
            "open_count": len(self.open_trades),
            "closed_count": len(self.closed_trades),
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
        }

    def stats(self, mark_prices: dict[str, float] | None = None) -> dict[str, Any]:
        """
        Returns aggregate stats (synchronous version - in-memory only):
            open_count, closed_count, realized_pnl, unrealized_pnl, total_pnl
        If mark_prices is provided (e.g., {"BTCUSDT": 50000.0}), unrealized PnL is computed for open trades.

        Note: Only includes closed trades still in memory. Use stats_async for complete stats.
        """
        with self._lock:
            realized = 0.0
            unrealized = 0.0

            for tid in self.closed_trades:
                t = self.trades.get(tid)
                if t and t.realized_pnl is not None:
                    realized += t.realized_pnl

            if mark_prices:
                for tid in self.open_trades:
                    t = self.trades.get(tid)
                    if not t:
                        continue
                    keys = (t.symbol, t.symbol.replace("/", ""))
                    mp = next((mark_prices.get(k) for k in keys if k in mark_prices), None)
                    upnl = t.compute_unrealized_pnl(mp)
                    if upnl is not None:
                        unrealized += upnl

            return {
                "open_count": len(self.open_trades),
                "closed_count": len(self.closed_trades),
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "total_pnl": realized + unrealized,
            }

    # --- Persistence ---

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trades": {tid: t.to_dict() for tid, t in self.trades.items()},
                "open_trades": list(self.open_trades),
                "closed_trades": list(self.closed_trades),
            }

    def load_json(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.trades.clear()
            self.open_trades.clear()
            self.closed_trades.clear()
            self.by_symbol.clear()

            trades_raw = payload.get("trades", {})
            for tid, td in trades_raw.items():
                t = Trade.from_dict(td)
                self.trades[tid] = t
                if t.status == "open":
                    self.open_trades.add(tid)
                else:
                    self.closed_trades.add(tid)
                self.by_symbol.setdefault(t.symbol, set()).add(tid)

    # --- Python niceties ---

    def __len__(self) -> int:
        with self._lock:
            return len(self.trades)

    def __contains__(self, trade_id: str) -> bool:
        with self._lock:
            return trade_id in self.trades
