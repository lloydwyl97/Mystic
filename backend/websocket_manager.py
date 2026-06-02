"""
WebSocket Connection Manager for Mystic Trading - Live Configuration Only

Manages WebSocket connections with safe broadcasting helpers.
All configuration values come from live config - no hardcoded values.
"""

from __future__ import annotations

import asyncio
import contextlib
import decimal
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger("mystic.websocket")

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None


class WebSocketConnectionManager:
    """Manages WebSocket connections and provides broadcasting capabilities."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        # Observability/state
        self._meta: dict[WebSocket, dict[str, Any]] = {}
        self._messages_sent: int = 0
        self._messages_dropped: int = 0
        # Config - reload from live config
        self._reload_config()

    def _reload_config(self) -> None:
        """Reload configuration values from live config or environment variables."""
        # Send timeout from live configuration
        self._send_timeout = _get_send_timeout()
        # Broadcast concurrency from live configuration
        self._broadcast_concurrency = _get_broadcast_concurrency()
        # Heartbeat interval from live configuration
        self._heartbeat_interval = _get_heartbeat_interval()
        # Allowed subprotocols from live configuration
        allowed = _get_allowed_subprotocols()
        self._allowed_subprotocols: set[str] = {s for s in (p.strip() for p in allowed.split(",")) if s}

    # -------------------------
    # Introspection
    # -------------------------
    @property
    def active_count(self) -> int:
        return len(self._connections)

    def snapshot(self) -> tuple[WebSocket, ...]:
        """Immutable snapshot to iterate safely while the set may change."""
        return tuple(self._connections)

    # -------------------------
    # Lifecycle
    # -------------------------
    async def connect(self, websocket: WebSocket, *, subprotocols: list[str] | None = None) -> None:
        """Accept and register a new WebSocket connection."""
        chosen: str | None = None
        if subprotocols:
            # Server-side validation: pick first allowed match; else accept without subprotocol
            for proto in subprotocols:
                if proto in self._allowed_subprotocols:
                    chosen = proto
                    break
        await websocket.accept(subprotocol=chosen)
        async with self._lock:
            self._connections.add(websocket)
            self._meta[websocket] = {
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "last_error": None,
            }
        logger.info("WebSocket client connected. Total connections: %d", self.active_count)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.discard(websocket)
            self._meta.pop(websocket, None)
        logger.info("WebSocket client disconnected. Total connections: %d", self.active_count)

    async def close_all(self, code: int | None = None, reason: str = "Server shutdown") -> None:
        """Close all connections gracefully."""
        # Get close code from live configuration if not provided
        if code is None:
            code = _get_websocket_close_code()
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                await ws.close(code=code, reason=reason)
        async with self._lock:
            self._connections.clear()
            self._meta.clear()
        logger.info("Closed all WebSocket connections")

    # -------------------------
    # Send helpers
    # -------------------------
    async def _send_text(self, ws: WebSocket, message: str) -> bool:
        try:
            await asyncio.wait_for(ws.send_text(message), timeout=self._send_timeout)
            self._messages_sent += 1
            meta = self._meta.get(ws)
            if meta is not None:
                meta.update({"last_activity": datetime.now(timezone.utc).isoformat()})
        except (
            WebSocketDisconnect,
            RuntimeError,
            ConnectionResetError,
            asyncio.CancelledError,
            asyncio.TimeoutError,
        ) as e:
            logger.warning("WebSocket send_text failed; removing connection: %s", e)
            self._messages_dropped += 1
            await self.disconnect(ws)
            return False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            logger.exception("WebSocket send_text unexpected error: %s", e)
            self._messages_dropped += 1
            meta = self._meta.get(ws)
            if meta is not None:
                meta["last_error"] = str(e)
            await self.disconnect(ws)
            return False

    def _sanitize_json(self, obj: Any) -> Any:
        # Recursively convert non-serializable types
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._sanitize_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_json(v) for v in obj]
        return str(obj)

    async def _send_json(self, ws: WebSocket, payload: dict[str, Any]) -> bool:
        try:
            safe_payload = self._sanitize_json(payload)
            # Validate
            json.dumps(safe_payload, default=str)
            await asyncio.wait_for(ws.send_json(safe_payload), timeout=self._send_timeout)
            self._messages_sent += 1
            meta = self._meta.get(ws)
            if meta is not None:
                meta.update({"last_activity": datetime.now(timezone.utc).isoformat()})
        except (
            WebSocketDisconnect,
            RuntimeError,
            ConnectionResetError,
            asyncio.CancelledError,
            asyncio.TimeoutError,
        ) as e:
            logger.warning("WebSocket send_json failed; removing connection: %s", e)
            self._messages_dropped += 1
            await self.disconnect(ws)
            return False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            logger.exception("WebSocket send_json unexpected error: %s", e)
            self._messages_dropped += 1
            meta = self._meta.get(ws)
            if meta is not None:
                meta["last_error"] = str(e)
            await self.disconnect(ws)
            return False

    # -------------------------
    # Public API
    # -------------------------
    async def broadcast_text(self, message: str) -> int:
        """Broadcast a text message to all connected clients. Returns count sent."""
        conns = self.snapshot()
        sem = asyncio.Semaphore(self._broadcast_concurrency)
        sent = 0

        async def _one(ws: WebSocket) -> bool:
            async with sem:
                return await self._send_text(ws, message)

        results = await asyncio.gather(*[_one(ws) for ws in conns], return_exceptions=False)
        for ok in results:
            if ok:
                sent += 1
        return sent

    async def broadcast_json(self, payload: dict[str, Any]) -> int:
        """Broadcast a JSON message to all connected clients. Returns count sent."""
        conns = self.snapshot()
        sem = asyncio.Semaphore(self._broadcast_concurrency)
        sent = 0

        async def _one(ws: WebSocket) -> bool:
            async with sem:
                return await self._send_json(ws, payload)

        results = await asyncio.gather(*[_one(ws) for ws in conns], return_exceptions=False)
        for ok in results:
            if ok:
                sent += 1
        return sent

    async def send_to(self, websocket: WebSocket, message: str | dict[str, Any]) -> bool:
        """Send to a specific client (text or JSON)."""
        if isinstance(message, str):
            return await self._send_text(websocket, message)
        if isinstance(message, dict):
            return await self._send_json(websocket, message)
        logger.error("Unsupported message type for send_to: %s", type(message).__name__)
        return False

    # -------------------------
    # Introspection / Health
    # -------------------------
    def status(self) -> dict[str, Any]:
        # Reload config to ensure live values are shown
        self._reload_config()
        return {
            "connections": self.active_count,
            "messages_sent": self._messages_sent,
            "messages_dropped": self._messages_dropped,
            "allowed_subprotocols": sorted(self._allowed_subprotocols),
            "last_activities": {id(ws): self._meta.get(ws, {}).get("last_activity") for ws in self._connections},
            "send_timeout": self._send_timeout,
            "broadcast_concurrency": self._broadcast_concurrency,
            "heartbeat_interval": self._heartbeat_interval,
        }


# Global instance + accessor will be defined after helper functions


def _get_send_timeout() -> float:
    """Get send timeout from live configuration."""
    if _mystic_config is not None:
        try:
            # Try to get from config if available
            value = getattr(_mystic_config, "websocket", None)
            if value and hasattr(value, "send_timeout_sec"):
                timeout = value.send_timeout_sec
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WS_SEND_TIMEOUT_SEC", "3.0"))
        return max(0.1, value)
    except (ValueError, TypeError):
        return 3.0


def _get_broadcast_concurrency() -> int:
    """Get broadcast concurrency from live configuration."""
    if _mystic_config is not None:
        try:
            # Try to get from config if available
            value = getattr(_mystic_config, "websocket", None)
            if value and hasattr(value, "broadcast_concurrency"):
                concurrency = value.broadcast_concurrency
                if isinstance(concurrency, int) and concurrency > 0:
                    return concurrency
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("WS_BROADCAST_CONCURRENCY", "25"))
        return max(1, value)
    except (ValueError, TypeError):
        return 25


def _get_heartbeat_interval() -> float:
    """Get heartbeat interval from live configuration."""
    if _mystic_config is not None:
        try:
            # Try to get from config if available
            value = getattr(_mystic_config, "websocket", None)
            if value and hasattr(value, "heartbeat_interval_sec"):
                interval = value.heartbeat_interval_sec
                if isinstance(interval, (int, float)) and interval > 0:
                    return float(interval)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = float(os.getenv("WS_HEARTBEAT_INTERVAL_SEC", "30"))
        return max(1.0, value)
    except (ValueError, TypeError):
        return 30.0


def _get_allowed_subprotocols() -> str:
    """Get allowed subprotocols from live configuration."""
    if _mystic_config is not None:
        try:
            # Try to get from config if available
            value = getattr(_mystic_config, "websocket", None)
            if value and hasattr(value, "allowed_subprotocols"):
                protocols = value.allowed_subprotocols
                if isinstance(protocols, str):
                    return protocols.strip()
                if isinstance(protocols, (list, tuple)):
                    return ",".join(str(p) for p in protocols if p)
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    return os.getenv("WEBSOCKET_ALLOWED_SUBPROTOCOLS", "").strip()


def _get_websocket_close_code() -> int:
    """Get WebSocket close code from live configuration."""
    if _mystic_config is not None:
        try:
            # Try to get from config if available
            value = getattr(_mystic_config, "websocket", None)
            if value and hasattr(value, "close_code"):
                code = value.close_code
                if isinstance(code, int) and 1000 <= code <= 4999:
                    return code
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    try:
        value = int(os.getenv("WS_CLOSE_CODE", "1000"))
        if 1000 <= value <= 4999:
            return value
    except (ValueError, TypeError):
        pass
    return 1000


# Global instance + accessor
websocket_manager = WebSocketConnectionManager()


def get_websocket_manager() -> WebSocketConnectionManager:
    return websocket_manager


# Backward-compat alias (your old code referenced WebSocketManager)
WebSocketManager = WebSocketConnectionManager
