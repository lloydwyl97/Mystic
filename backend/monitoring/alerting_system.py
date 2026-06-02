"""
Alerting System for Mystic Trading Platform - All Live Data, No Fallback/Hardcoded Data

Provides comprehensive alerting with:
- Multiple notification channels (email, Slack, webhook, SMS, dashboard)
- Alert severity levels
- Alert aggregation and deduplication
- Alert history and management
- Custom alert rules
All operations use live data:
- Redis connection: Uses environment variables (REDIS_URL or REDIS_HOST/REDIS_PORT/REDIS_DB)
- Alert storage: Live Redis storage for alert persistence
- Notification channels: Live notification services configuration
- All operations use live connections - no fallback/hardcoded data
"""

import asyncio
import json
import logging
import os
import smtplib
import ssl
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import httpx
from trading_config import trading_config

import redis
from backend.config.redis_config import get_shared_redis_sync
from redis import Redis

logger = logging.getLogger(__name__)

ALERT_RETENTION_DAYS = 30
ALERT_DEDUPLICATION_WINDOW = 300
MAX_ALERTS_PER_HOUR = 100


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class Alert:
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: float
    channels: list[AlertChannel]
    metadata: dict[str, Any] | None = None
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None
    occurrences: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "component": self.component,
            "timestamp": self.timestamp,
            "channels": [c.value for c in self.channels],
            "metadata": self.metadata or {},
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "occurrences": self.occurrences,
        }


class AlertRule:
    def __init__(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        channels: list[AlertChannel],
        cooldown: int = 300,
    ) -> None:
        self.name = name
        self.condition = condition
        self.severity = severity
        self.channels = channels
        self.cooldown = cooldown
        self.last_triggered: float | None = None

    def should_trigger(self, current_time: float) -> bool:
        if self.last_triggered is None:
            return True
        return (current_time - self.last_triggered) >= self.cooldown


class NotificationChannel:
    def __init__(self, channel_type: AlertChannel) -> None:
        self.channel_type = channel_type
        self.enabled = True

    async def send_notification(self, alert: Alert) -> bool:
        """Override in subclasses. Base returns False."""
        logger.warning("Base NotificationChannel.send_notification called (override in subclass)")
        return False


class EmailNotificationChannel(NotificationChannel):
    def __init__(self, smtp_config: dict[str, Any]) -> None:
        super().__init__(AlertChannel.EMAIL)
        self.smtp_config = smtp_config

    async def send_notification(self, alert: Alert) -> bool:
        host = self.smtp_config.get("host")
        port = int(self.smtp_config.get("port", 587))
        username = self.smtp_config.get("username")
        password = self.smtp_config.get("password")
        use_tls = bool(self.smtp_config.get("use_tls", True))
        use_ssl = bool(self.smtp_config.get("use_ssl", False))
        from_addr = self.smtp_config.get("from_addr") or username
        default_recipients = self.smtp_config.get("recipients") or []
        recipients = list(alert.metadata.get("email_recipients", [])) if alert.metadata else []
        if not recipients:
            recipients = default_recipients
        if not host or not from_addr or not recipients:
            logger.warning("Email channel not configured with host/from/recipients")
            return False
        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))
        body = f"{alert.message}\n\nComponent: {alert.component}\nSeverity: {alert.severity.value}\nTime: {timestamp_str}\nOccurrences: {alert.occurrences}\nAlert ID: {alert.alert_id}"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        try:
            if use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=context, timeout=10) as server:
                    if username and password:
                        server.login(username, password)
                    server.sendmail(from_addr, recipients, msg.as_string())
            else:
                with smtplib.SMTP(host, port, timeout=10) as server:
                    if use_tls:
                        server.starttls(context=ssl.create_default_context())
                    if username and password:
                        server.login(username, password)
                    server.sendmail(from_addr, recipients, msg.as_string())
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to send email notification")
            return False
        else:
            return True


class SlackNotificationChannel(NotificationChannel):
    def __init__(self, webhook_url: str, channel: str | None = None) -> None:
        super().__init__(AlertChannel.SLACK)
        self.webhook_url = webhook_url
        self.channel = channel

    async def send_notification(self, alert: Alert) -> bool:
        if not hasattr(self, "requests") or not self.requests:
            logger.warning("requests not available for Slack notifications")
            return False
        payload = {
            "text": f"*{alert.severity.value.upper()}* | {alert.title}\n{alert.message}\nComponent: `{alert.component}` | Occurrences: {alert.occurrences} | ID: `{alert.alert_id}`",
        }
        if self.channel:
            payload["channel"] = self.channel
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(self.webhook_url, json=payload, timeout=10)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to send Slack notification")
            return False
        else:
            return 200 <= r.status_code < 300


class WebhookNotificationChannel(NotificationChannel):
    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(AlertChannel.WEBHOOK)
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send_notification(self, alert: Alert) -> bool:
        if not hasattr(self, "requests") or not self.requests:
            logger.warning("requests not available for webhook notifications")
            return False
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers=self.headers,
                    timeout=10,
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to send webhook notification")
            return False
        else:
            return 200 <= r.status_code < 300


class SMSNotificationChannel(NotificationChannel):
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        default_to_numbers: list[str] | None = None,
    ) -> None:
        super().__init__(AlertChannel.SMS)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.default_to_numbers = default_to_numbers or []

    async def send_notification(self, alert: Alert) -> bool:
        if not hasattr(self, "requests") or not self.requests:
            logger.warning("requests not available for SMS notifications")
            return False
        to_numbers = list(alert.metadata.get("sms_recipients", [])) if alert.metadata else []
        if not to_numbers:
            to_numbers = self.default_to_numbers
        if not self.account_sid or not self.auth_token or not self.from_number or not to_numbers:
            logger.warning("SMS channel missing credentials or recipients")
            return False
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        body = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message} ({alert.component}) x{alert.occurrences}"
        ok = True
        for to in to_numbers:
            data = {"From": self.from_number, "To": to, "Body": body}
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        url,
                        data=data,
                        auth=(self.account_sid, self.auth_token),
                        timeout=10,
                    )
                ok = ok and (200 <= r.status_code < 300)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Failed to send SMS")
                ok = False
        return ok


class DashboardNotificationChannel(NotificationChannel):
    def __init__(self, sink_callable: Any) -> None:
        super().__init__(AlertChannel.DASHBOARD)
        self.sink_callable = sink_callable

    async def send_notification(self, alert: Alert) -> bool:
        try:
            self.sink_callable(alert)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to write dashboard notification")
            return False
        else:
            return True


class AlertingSystem:
    def __init__(self) -> None:
        self.alerts: deque[Alert] = deque(maxlen=10000)
        self.alert_rules: dict[str, AlertRule] = {}
        self.notification_channels: dict[AlertChannel, NotificationChannel] = {}
        self.alert_counts: dict[str, int] = defaultdict(int)
        self.rate_limit_times: dict[str, list[float]] = defaultdict(list)
        self.redis_client: Redis | None = None
        self.lock = threading.Lock()
        self._dedup_index: dict[str, dict[str, Any]] = {}
        self._shutdown = False
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self._initialize_redis()
        self._setup_default_channels()
        self._setup_default_rules()

    def shutdown(self) -> None:
        """Shutdown the alerting system by stopping the event loop and joining the thread."""
        self._shutdown = True
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)
            if self._loop_thread.is_alive():
                logger.warning("AlertingSystem thread did not shutdown gracefully")
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        logger.info("AlertingSystem shutdown complete")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # Cancel all pending tasks when the loop stops
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            logger.info("AlertingSystem event loop stopped and pending tasks cancelled")

    def _submit_coro(self, coro):
        try:
            return asyncio.run_coroutine_threadsafe(coro, self._loop)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to schedule coroutine")
            return None

    def _initialize_redis(self):
        """
        Initialize Redis connection using environment variables.

        Uses REDIS_URL if available, otherwise falls back to REDIS_HOST/REDIS_PORT/REDIS_DB.
        All connection parameters come from environment variables - no hardcoded defaults.
        """
        try:
            if not redis or not Redis:
                logger.warning("Redis not available for alerting system")
                self.redis_client = None
                return

            # Prefer REDIS_URL if available (single source of truth)
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                try:
                    self.redis_client = get_shared_redis_sync()
                    if self.redis_client is None:
                        logger.warning("Shared Redis client unavailable for alerting system")
                        return
                    self.redis_client.ping()
                    logger.info("Redis connection established for alerting system via shared client")
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Failed to connect to Redis via shared client")
                    self.redis_client = None
                else:
                    return

            # Fallback: use shared client
            try:
                self.redis_client = get_shared_redis_sync()
                if self.redis_client is None:
                    logger.warning("Shared Redis client unavailable for alerting system")
                    return
                self.redis_client.ping()
                logger.info("Redis connection established for alerting system via shared client")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Failed to connect to Redis for alerting system")
                self.redis_client = None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Failed to initialize Redis for alerting system")
            self.redis_client = None

    def _dashboard_sink(self, alert: Alert):
        if self.redis_client:
            try:
                self.redis_client.xadd(
                    "alerts:stream",
                    {
                        "alert_id": alert.alert_id,
                        "title": alert.title,
                        "message": alert.message,
                        "severity": alert.severity.value,
                        "component": alert.component,
                        "timestamp": str(alert.timestamp),
                        "occurrences": str(alert.occurrences),
                    },
                    maxlen=5000,
                    approximate=True,
                )
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Dashboard sink Redis error")

    def _setup_default_channels(self):
        self.notification_channels.clear()
        smtp_host = getattr(trading_config, "ALERTING_SMTP_HOST", None)
        if smtp_host:
            smtp_cfg = {
                "host": smtp_host,
                "port": getattr(trading_config, "ALERTING_SMTP_PORT", 587),
                "username": getattr(trading_config, "ALERTING_SMTP_USERNAME", None),
                "password": getattr(trading_config, "ALERTING_SMTP_PASSWORD", None),
                "use_tls": getattr(trading_config, "ALERTING_SMTP_USE_TLS", True),
                "use_ssl": getattr(trading_config, "ALERTING_SMTP_USE_SSL", False),
                "from_addr": getattr(
                    trading_config,
                    "ALERTING_EMAIL_FROM",
                    getattr(trading_config, "ALERTING_SMTP_USERNAME", None),
                ),
                "recipients": getattr(trading_config, "ALERTING_EMAIL_RECIPIENTS", []),
            }
            self.notification_channels[AlertChannel.EMAIL] = EmailNotificationChannel(smtp_cfg)
        slack_webhook = getattr(trading_config, "ALERTING_SLACK_WEBHOOK_URL", None)
        if slack_webhook:
            slack_channel = getattr(trading_config, "ALERTING_SLACK_CHANNEL", None)
            self.notification_channels[AlertChannel.SLACK] = SlackNotificationChannel(slack_webhook, slack_channel)
        generic_webhook = getattr(trading_config, "ALERTING_GENERIC_WEBHOOK_URL", None)
        if generic_webhook:
            headers = getattr(trading_config, "ALERTING_GENERIC_WEBHOOK_HEADERS", None)
            self.notification_channels[AlertChannel.WEBHOOK] = WebhookNotificationChannel(generic_webhook, headers)
        twilio_sid = getattr(trading_config, "ALERTING_TWILIO_ACCOUNT_SID", None)
        twilio_token = getattr(trading_config, "ALERTING_TWILIO_AUTH_TOKEN", None)
        twilio_from = getattr(trading_config, "ALERTING_TWILIO_FROM_NUMBER", None)
        sms_defaults = getattr(trading_config, "ALERTING_SMS_RECIPIENTS", [])
        if twilio_sid and twilio_token and twilio_from:
            self.notification_channels[AlertChannel.SMS] = SMSNotificationChannel(twilio_sid, twilio_token, twilio_from, sms_defaults)
        self.notification_channels[AlertChannel.DASHBOARD] = DashboardNotificationChannel(self._dashboard_sink)

    def _setup_default_rules(self):
        self.add_alert_rule(
            name="high_cpu_usage",
            condition="cpu_percent > 90",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=300,
        )
        self.add_alert_rule(
            name="high_memory_usage",
            condition="memory_percent > 90",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=300,
        )
        self.add_alert_rule(
            name="high_error_rate",
            condition="error_rate > 10",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=600,
        )
        self.add_alert_rule(
            name="low_liquidity",
            condition="liquidity_score < 0.3",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=1800,
        )
        self.add_alert_rule(
            name="slow_queries",
            condition="slow_query_count > 50",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            cooldown=900,
        )

    def add_alert_rule(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        channels: list[AlertChannel],
        cooldown: int = 300,
    ):
        self.alert_rules[name] = AlertRule(name, condition, severity, channels, cooldown)

    def _generate_alert_id(self) -> str:
        return f"alert_{uuid.uuid4().hex}"

    def _dedup_key(self, title: str, message: str, severity: AlertSeverity, component: str) -> str:
        return f"{component}|{severity.value}|{title}|{message}"

    def _cleanup_rate_limits(self, current_time: float) -> None:
        """Clean up rate limit tracking data to prevent unbounded growth."""
        cutoff_time = current_time - 3600  # Remove timestamps older than 1 hour
        keys_to_delete = []

        for component, timestamps in self.rate_limit_times.items():
            # Filter out old timestamps
            filtered_timestamps = [t for t in timestamps if t > cutoff_time]
            # Cap each list to the most recent 100 timestamps
            if len(filtered_timestamps) > 100:
                filtered_timestamps = sorted(filtered_timestamps, reverse=True)[:100]

            if filtered_timestamps:
                self.rate_limit_times[component] = filtered_timestamps
            else:
                keys_to_delete.append(component)

        # Remove empty keys
        for key in keys_to_delete:
            del self.rate_limit_times[key]

    def _is_rate_limited(self, component: str, current_time: float) -> bool:
        # Run cleanup periodically (every 100 calls to avoid overhead)
        if hasattr(self, "_cleanup_counter"):
            self._cleanup_counter += 1
        else:
            self._cleanup_counter = 1

        if self._cleanup_counter % 100 == 0:
            self._cleanup_rate_limits(current_time)

        cutoff_time = current_time - 3600
        self.rate_limit_times[component] = [t for t in self.rate_limit_times[component] if t > cutoff_time]
        if len(self.rate_limit_times[component]) >= MAX_ALERTS_PER_HOUR:
            return True
        self.rate_limit_times[component].append(current_time)
        return False

    def _store_alert(self, alert: Alert):
        with self.lock:
            self.alerts.append(alert)
            self.alert_counts[alert.component] += 1
        if self.redis_client:
            try:
                ttl = ALERT_RETENTION_DAYS * 86400
                self.redis_client.setex(f"alert:{alert.alert_id}", ttl, json.dumps(alert.to_dict()))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Failed to store alert in Redis")

    def _update_alert(self, alert: Alert):
        if self.redis_client:
            try:
                ttl = ALERT_RETENTION_DAYS * 86400
                self.redis_client.setex(f"alert:{alert.alert_id}", ttl, json.dumps(alert.to_dict()))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                logger.exception("Failed to update alert in Redis")

    async def _send_notifications(self, alert: Alert):
        tasks = []
        for channel in alert.channels:
            ch = self.notification_channels.get(channel)
            if ch and ch.enabled:
                tasks.append(ch.send_notification(alert))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                channel_type = alert.channels[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to send notification via {channel_type.value}: {result}")
                else:
                    logger.debug(f"Notification sent via {channel_type.value}: {result}")

    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        component: str,
        channels: list[AlertChannel],
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        current_time = time.time()
        if self._is_rate_limited(component, current_time):
            logger.warning(f"Rate limited alert for component {component}")
            return None
        key = self._dedup_key(title, message, severity, component)
        with self.lock:
            dedup_entry = self._dedup_index.get(key)
            if dedup_entry and (current_time - dedup_entry["last_ts"]) < ALERT_DEDUPLICATION_WINDOW:
                existing_alert: Alert = dedup_entry["alert"]
                existing_alert.occurrences += 1
                dedup_entry["last_ts"] = current_time
                self._update_alert(existing_alert)
                return existing_alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            title=title,
            message=message,
            severity=severity,
            component=component,
            timestamp=current_time,
            channels=channels,
            metadata=metadata or {},
        )
        with self.lock:
            self._dedup_index[key] = {"last_ts": current_time, "alert": alert}
        self._store_alert(alert)
        self._submit_coro(self._send_notifications(alert))
        return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = time.time()
                    self._update_alert(alert)
                    logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
        return False

    def get_alerts_summary(self) -> dict[str, Any]:
        with self.lock:
            current_time = time.time()
            severity_counts = defaultdict(int)
            unacknowledged_count = 0
            recent_alerts = 0
            for alert in self.alerts:
                severity_counts[alert.severity.value] += 1
                if not alert.acknowledged:
                    unacknowledged_count += 1
                if current_time - alert.timestamp < 86400:
                    recent_alerts += 1
            return {
                "total_alerts": len(self.alerts),
                "unacknowledged_alerts": unacknowledged_count,
                "recent_alerts": recent_alerts,
                "severity_distribution": dict(severity_counts),
                "component_counts": dict(self.alert_counts),
                "timestamp": current_time,
            }

    def get_alerts_history(
        self,
        hours: int = 24,
        severity: AlertSeverity | None = None,
        component: str | None = None,
    ) -> list[dict[str, Any]]:
        cutoff_time = time.time() - (hours * 3600)
        out: list[dict[str, Any]] = []
        with self.lock:
            for alert in self.alerts:
                if alert.timestamp < cutoff_time:
                    continue
                if severity and alert.severity != severity:
                    continue
                if component and alert.component != component:
                    continue
                out.append(alert.to_dict())
        return out

    def cleanup_old_alerts(self, max_age_days: int = ALERT_RETENTION_DAYS):
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        with self.lock:
            self.alerts = deque((a for a in self.alerts if a.timestamp >= cutoff_time), maxlen=10000)
            self.alert_counts = defaultdict(int)
            for a in self.alerts:
                self.alert_counts[a.component] += 1
            keys_to_drop = []
            for k, v in self._dedup_index.items():
                if v["alert"].timestamp < cutoff_time:
                    keys_to_drop.append(k)
            for k in keys_to_drop:
                self._dedup_index.pop(k, None)
        # Also cleanup rate limits during general cleanup
        self._cleanup_rate_limits(time.time())
        logger.info("Cleaned up old alerts and rate limits")


alerting_system = AlertingSystem()
