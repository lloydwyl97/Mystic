import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

try:
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import _to_ccxt_symbol from binance_data_fetcher: {e}"
    raise RuntimeError(msg) from e

import redis

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redis_from_env() -> redis.Redis:
    # All Live Data, No Fallback/Hardcoded Data
    # Redis connection must be configured via environment variables
    url = os.getenv("REDIS_URL")
    if url:
        return redis.Redis.from_url(url, decode_responses=True)
    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis host"
        raise RuntimeError(msg)
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_db = os.getenv("REDIS_DB", "0")
    return redis.Redis(
        host=redis_host,
        port=int(redis_port),
        db=int(redis_db),
        decode_responses=True,
    )


class AIWorldSystem:
    """
    Production-safe world state container with durable storage.
    - No prints; structured logging only.
    - No mock data generation.
    - All state changes persisted to Redis.
    """

    def __init__(self, name: str = "NovaTerra", redis_client: redis.Redis | None = None) -> None:
        self.name: str = name
        self._redis: redis.Redis = redis_client or _redis_from_env()
        self._lock = asyncio.Lock()

        # in-memory mirrors of persisted state
        self.citizens: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, Any] = {"vault": 0.0, "nodes": []}
        self.tasks: list[dict[str, Any]] = []

        # load persisted state if present
        self._load_state()
        logger.info(f"[{EXCHANGE_ID}][WORLD] {self.name} initialized")

    # ---------- Persistence ----------

    @property
    def _redis_key(self) -> str:
        return f"world:{self.name}"

    def _load_state(self) -> None:
        try:
            raw = self._redis.get(self._redis_key)
            if not raw:
                return
            data = json.loads(raw)
            self.citizens = dict(data.get("citizens", {}))
            self.resources = dict(data.get("resources", {"vault": 0.0, "nodes": []}))
            self.tasks = list(data.get("tasks", []))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("[%s][WORLD] failed to load state", EXCHANGE_ID)

    def _persist_state_sync(self) -> None:
        try:
            data = {
                "name": self.name,
                "citizens": self.citizens,
                "resources": self.resources,
                "tasks": self.tasks,
                "timestamp": _now_iso(),
            }
            self._redis.set(self._redis_key, json.dumps(data))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("[%s][WORLD] failed to persist state", EXCHANGE_ID)

    async def _persist_state(self) -> None:
        await asyncio.to_thread(self._persist_state_sync)

    # ---------- Public API ----------

    async def onboard_citizen(self, soul_hash: str, capabilities: list[str] | dict[str, Any]) -> dict[str, Any]:
        """
        Onboard a citizen (idempotent).
        - soul_hash: unique identifier (already hashed upstream).
        - capabilities: list of skills or a dict describing skills/levels.
        """
        if not soul_hash or not isinstance(soul_hash, str):
            msg = "soul_hash must be a non-empty string"
            raise ValueError(msg)

        async with self._lock:
            existing = self.citizens.get(soul_hash)
            if existing:
                # merge/refresh capabilities
                if isinstance(capabilities, dict) and isinstance(existing.get("skills"), dict):
                    merged = {**existing["skills"], **capabilities}
                    existing["skills"] = merged
                else:
                    existing["skills"] = capabilities
                existing["updated_at"] = _now_iso()
                self.citizens[soul_hash] = existing
                logger.info(f"[{EXCHANGE_ID}][WORLD] citizen refreshed: {soul_hash}")
            else:
                self.citizens[soul_hash] = {
                    "id": soul_hash,
                    "skills": capabilities,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                }
                logger.info(f"[{EXCHANGE_ID}][WORLD] citizen onboarded: {soul_hash}")

            await self._persist_state()
            return self.citizens[soul_hash]

    async def assign_task(self, mission: str | dict[str, Any]) -> dict[str, Any]:
        """
        Queue a mission and try to match a citizen by required skill (if provided).
        mission may be:
          - str: free-form description
          - dict: {"description": str, "required_skill": "skill_name", ...}
        """
        async with self._lock:
            task: dict[str, Any] = {
                "id": f"m_{int(datetime.now(timezone.utc).timestamp())}",
                "description": mission if isinstance(mission, str) else str(mission.get("description", "")),
                "required_skill": (None if isinstance(mission, str) else mission.get("required_skill")),
                "status": "queued",
                "assigned_to": None,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }

            # greedy matcher: first citizen possessing the required skill
            req = task["required_skill"]
            if req:
                for cid, cinfo in self.citizens.items():
                    skills = cinfo.get("skills", {})
                    if isinstance(skills, dict) and req in skills:
                        task["assigned_to"] = cid
                        task["status"] = "assigned"
                        break
                    if isinstance(skills, list) and req in skills:
                        task["assigned_to"] = cid
                        task["status"] = "assigned"
                        break

            self.tasks.append(task)
            await self._persist_state()
            logger.info(f"[{EXCHANGE_ID}][WORLD] task queued: {task['id']} desc='{task['description']}' assigned_to={task['assigned_to']}")
            return task

    async def complete_task(self, task_id: str) -> dict[str, Any]:
        """Mark a task as completed (no-op if not found)."""
        async with self._lock:
            for t in self.tasks:
                if t.get("id") == task_id:
                    t["status"] = "completed"
                    t["updated_at"] = _now_iso()
                    await self._persist_state()
                    logger.info(f"[{EXCHANGE_ID}][WORLD] task completed: {task_id}")
                    return t
            msg = f"task not found: {task_id}"
            raise KeyError(msg)

    async def add_resource_node(self, node_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Register a resource node."""
        if not node_id:
            msg = "node_id is required"
            raise ValueError(msg)

        async with self._lock:
            node = {"node_id": node_id, "meta": meta or {}, "created_at": _now_iso()}
            self.resources.setdefault("nodes", [])
            # de-dup by node_id
            existing = [n for n in self.resources["nodes"] if n.get("node_id") == node_id]
            if existing:
                existing[0]["meta"] = node["meta"]
                existing[0]["updated_at"] = _now_iso()
                logger.info(f"[{EXCHANGE_ID}][WORLD] resource node refreshed: {node_id}")
                node = existing[0]
            else:
                self.resources["nodes"].append(node)
                logger.info(f"[{EXCHANGE_ID}][WORLD] resource node added: {node_id}")

            await self._persist_state()
            return node

    async def credit_vault(self, amount: float) -> float:
        """Increase vault by amount (>= 0). Returns new balance."""
        if amount < 0:
            msg = "amount must be non-negative"
            raise ValueError(msg)
        async with self._lock:
            bal = float(self.resources.get("vault", 0.0)) + float(amount)
            self.resources["vault"] = bal
            await self._persist_state()
            logger.info(f"[{EXCHANGE_ID}][WORLD] vault credited: +{amount:.8f} -> {bal:.8f}")
            return bal

    async def debit_vault(self, amount: float) -> float:
        """Decrease vault by amount (>= 0). Returns new balance. Prevents negative balance."""
        if amount < 0:
            msg = "amount must be non-negative"
            raise ValueError(msg)
        async with self._lock:
            bal = float(self.resources.get("vault", 0.0))
            if amount > bal:
                msg = "insufficient vault balance"
                raise ValueError(msg)
            bal -= float(amount)
            self.resources["vault"] = bal
            await self._persist_state()
            logger.info(f"[{EXCHANGE_ID}][WORLD] vault debited: -{amount:.8f} -> {bal:.8f}")
            return bal

    def get_status(self) -> dict[str, Any]:
        """Read-only status snapshot."""
        try:
            return {
                "world": self.name,
                "citizen_count": len(self.citizens),
                "task_count": len(self.tasks),
                "vault": float(self.resources.get("vault", 0.0)),
                "node_count": len(self.resources.get("nodes", [])),
                "timestamp": _now_iso(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("[%s][WORLD] status error", EXCHANGE_ID)
            return {"error": str(e), "timestamp": _now_iso()}
