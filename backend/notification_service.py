"""
Notification Service for Mystic Trading

Handles notifications for signal failures, recoveries, and system events.
Supports multiple notification channels: email, Slack, webhook, and in-app.
"""

from __future__ import annotations

import json
import logging
import smtplib
from collections.abc import Iterable
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

import redis

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
    from backend.modules.market.binance_data_fetcher import _to_ccxt_symbol
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe or _to_ccxt_symbol: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _to_str_list(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for x in items:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        elif isinstance(x, str):
            out.append(x)
        else:
            out.append(str(x))
    return out


def _safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else None
    except (TypeError, json.JSONDecodeError, ValueError):
        return None


def _parse_iso(ts: str) -> datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        # Accept ISO with or without Z
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


class NotificationService:
    def __init__(
        self,
        redis_client: redis.Redis | Any,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.http_client: httpx.AsyncClient | None = http_client
        self._using_shared_client = http_client is not None
        self.notifications: list[dict[str, Any]] = []
        self.notification_id_counter = 1
        self.config: dict[str, dict[str, Any]] = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "",
                "to_emails": [],
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "#trading-alerts",
            },
            "webhook": {
                "enabled": False,
                "url": "",
                "headers": {"Content-Type": "application/json"},
            },
            "in_app": {
                "enabled": True,
                "max_notifications": 100,
            },
        }
        self._load_config()

    async def close(self) -> None:
        try:
            # Only close if we created our own client (not using shared client)
            if self.http_client and not self._using_shared_client:
                await self.http_client.aclose()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Error closing HTTP client: {e}")

    def _load_config(self) -> None:
        try:
            raw = self.redis_client.get("notification_config")
            if not raw:
                return
            cfg_json = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            stored_config = _safe_json_loads(cfg_json) or {}
            for channel, cfg in stored_config.items():
                if channel in self.config and isinstance(cfg, dict):
                    self.config[channel].update(cfg)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning(f"Could not load notification config: {e}")

    def _save_config(self) -> None:
        try:
            self.redis_client.setex("notification_config", 3600, json.dumps(self.config))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Could not save notification config: {e}")

    async def _ensure_client(self) -> httpx.AsyncClient:
        if not self.http_client:
            # Create our own if none provided
            self._using_shared_client = False
            timeout = httpx.Timeout(15)
            self.http_client = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "mystic-notifier/1.0"})
        return self.http_client

    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info",
        channels: list[str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if channels is None:
            channels = ["in_app"]

        results: dict[str, Any] = {
            "timestamp": _utc_iso(),
            "title": title,
            "message": message,
            "level": level,
            "channels": {},
            "success": True,
        }

        for channel in channels:
            enabled = bool(self.config.get(channel, {}).get("enabled", False))
            if not enabled:
                results["channels"][channel] = {
                    "success": False,
                    "error": "channel_disabled",
                }
                results["success"] = False
                continue

            try:
                if channel == "email":
                    result = await self._send_email(title, message, level, data)
                elif channel == "slack":
                    result = await self._send_slack(title, message, level, data)
                elif channel == "webhook":
                    result = await self._send_webhook(title, message, level, data)
                elif channel == "in_app":
                    result = await self._send_in_app(title, message, level, data)
                else:
                    result = {"success": False, "error": f"unknown_channel:{channel}"}
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error sending {channel} notification: {e}")
                result = {"success": False, "error": str(e)}

            results["channels"][channel] = result
            if not result.get("success", False):
                results["success"] = False

        logger.info(f"Notification sent [{level}]: {title}")
        return results

    async def _send_email(
        self,
        title: str,
        message: str,
        level: str,
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            cfg = self.config["email"]
            msg = MIMEMultipart()
            msg["From"] = cfg["from_email"]
            # ensure to_emails are strings
            to_emails = [str(e) for e in cfg.get("to_emails", [])]
            msg["To"] = ", ".join(to_emails)
            msg["Subject"] = f"[Mystic Trading] {title}"

            html = [
                "<html><body>",
                f"<h2>{title}</h2>",
                f"<p><strong>Level:</strong> {level.upper()}</p>",
                f"<p><strong>Time:</strong> {_utc_human()}</p>",
                f"<p>{message}</p>",
            ]
            if data:
                html.append("<h3>Additional Data:</h3><pre>")
                html.append(json.dumps(data, indent=2))
                html.append("</pre>")
            html.append("</body></html>")

            msg.attach(MIMEText("".join(html), "html"))

            with smtplib.SMTP(cfg["smtp_server"], int(cfg["smtp_port"])) as server:
                server.starttls()
                server.login(cfg["username"], cfg["password"])
                server.send_message(msg)

            return {"success": True, "recipients": len(to_emails)}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Email notification failed: {e}")
            return {"success": False, "error": str(e)}

    async def _send_slack(
        self,
        title: str,
        message: str,
        level: str,
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            cfg = self.config["slack"]
            color_map = {
                "info": "#36a64f",
                "warning": "#ff9500",
                "error": "#ff0000",
                "critical": "#8b0000",
            }
            slack_message: dict[str, Any] = {
                "channel": cfg.get("channel", "#trading-alerts"),
                "attachments": [
                    {
                        "color": color_map.get(level, "#36a64f"),
                        "title": title,
                        "text": message,
                        "fields": [
                            {"title": "Level", "value": level.upper(), "short": True},
                            {"title": "Time", "value": _utc_human(), "short": True},
                        ],
                        "footer": "Mystic Trading",
                    },
                ],
            }
            if data:
                slack_message["attachments"][0]["fields"].append(
                    {
                        "title": "Additional Data",
                        "value": f"```{json.dumps(data, indent=2)}```",
                        "short": False,
                    }
                )

            client = await self._ensure_client()
            response = await client.post(cfg["webhook_url"], json=slack_message)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Slack notification failed: {e}")
            return {"success": False, "error": str(e)}
        else:
            if response.status_code == 200:
                return {"success": True}
            return {"success": False, "error": f"http_{response.status_code}"}

    async def _send_webhook(
        self,
        title: str,
        message: str,
        level: str,
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            cfg = self.config["webhook"]
            payload: dict[str, Any] = {
                "title": title,
                "message": message,
                "level": level,
                "timestamp": _utc_iso(),
                "source": "mystic_trading_bot",
            }
            if data:
                payload["data"] = data

            client = await self._ensure_client()
            response = await client.post(cfg["url"], json=payload, headers=cfg.get("headers", {}))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Webhook notification failed: {e}")
            return {"success": False, "error": str(e)}
        else:
            if response.status_code in (200, 201, 202):
                return {"success": True}
            return {"success": False, "error": f"http_{response.status_code}"}

    async def _send_in_app(
        self,
        title: str,
        message: str,
        level: str,
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            notification = {
                "id": f"notif_{self.notification_id_counter}",
                "title": title,
                "message": message,
                "level": level,
                "timestamp": _utc_iso(),
                "read": False,
                "data": data or {},
            }
            self.redis_client.lpush("in_app_notifications", json.dumps(notification))
            max_keep = int(self.config.get("in_app", {}).get("max_notifications", 100))
            self.redis_client.ltrim("in_app_notifications", 0, max_keep - 1)
            self.notifications.append(notification)
            self.notification_id_counter += 1
            return {"success": True, "notification_id": notification["id"]}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"In-app notification failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_notifications(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            raw_list = self.redis_client.lrange("in_app_notifications", 0, max(0, limit - 1))
            data_list = _to_str_list(raw_list)
            out: list[dict[str, Any]] = []
            for s in data_list:
                obj = _safe_json_loads(s)
                if obj:
                    out.append(obj)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting notifications: {e}")
            return []
        else:
            return out

    async def mark_read(self, notification_id: str) -> dict[str, Any]:
        try:
            raw_list = self.redis_client.lrange("in_app_notifications", 0, -1)
            data_list = _to_str_list(raw_list)
            for idx, s in enumerate(data_list):
                obj = _safe_json_loads(s)
                if not obj:
                    continue
                if obj.get("id") == notification_id:
                    obj["read"] = True
                    self.redis_client.lset("in_app_notifications", idx, json.dumps(obj))
                    return {
                        "status": "success",
                        "message": f"Notification {notification_id} marked as read",
                        "notification_id": notification_id,
                        "timestamp": obj.get("timestamp", _utc_iso()),
                    }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error marking notification read: {e}")
            return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "error",
                "message": f"Notification {notification_id} not found",
            }

    async def clear_all(self) -> dict[str, Any]:
        try:
            raw_list = self.redis_client.lrange("in_app_notifications", 0, -1)
            data_list = _to_str_list(raw_list)
            cleared = 0
            now = datetime.now(timezone.utc)
            for s in data_list:
                obj = _safe_json_loads(s)
                if not obj:
                    continue
                ts = obj.get("timestamp")
                dt = _parse_iso(ts) if isinstance(ts, str) else None
                if dt:
                    # Ensure dt is timezone-aware for comparison
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    # Remove any notification older than now (i.e., all historical entries)
                    if dt < now:
                        # lrem expects the exact stored value; stored are strings
                        self.redis_client.lrem("in_app_notifications", 1, json.dumps(obj))
                        cleared += 1
            logger.info(f"Cleared {cleared} old notifications")
            return {
                "status": "success",
                "message": "Notifications cleared",
                "cleared_count": cleared,
                "timestamp": _utc_iso(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error clearing notifications: {e}")
            return {"status": "error", "message": str(e)}

    async def create_notification(self, notification_data: dict[str, Any]) -> dict[str, Any]:
        try:
            notification = {
                "id": f"notif_{self.notification_id_counter}",
                "type": notification_data.get("type", "info"),
                "title": notification_data.get("title", "Notification"),
                "message": notification_data.get("message", ""),
                "read": False,
                "timestamp": _utc_iso(),
                "priority": notification_data.get("priority", "medium"),
            }
            self.notifications.append(notification)
            self.notification_id_counter += 1
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error creating notification: {e}")
            return {}
        else:
            return notification

    async def update_config(self, channel: str, config: dict[str, Any]) -> bool:
        try:
            if channel in self.config and isinstance(config, dict):
                self.config[channel].update(config)
                self._save_config()
                logger.info(f"Updated notification config for channel: {channel}")
                result = True
            else:
                logger.error(f"Unknown notification channel: {channel}")
                result = False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating notification config: {e}")
            return False
        else:
            return result


# Notification service state - using dict to avoid global keyword
_notification_service_state: dict[str, NotificationService | None] = {"instance": None}


def get_notification_service(redis_client: redis.Redis | Any, http_client: httpx.AsyncClient | None = None) -> NotificationService:
    if _notification_service_state["instance"] is None:
        _notification_service_state["instance"] = NotificationService(redis_client, http_client)
    # Update the HTTP client if it wasn't set before
    elif http_client and not _notification_service_state["instance"].http_client:
        _notification_service_state["instance"].http_client = http_client
        _notification_service_state["instance"]._using_shared_client = True
    return _notification_service_state["instance"]
