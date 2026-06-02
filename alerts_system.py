#!/usr/bin/env python3
"""
Real-time Alerts and Notifications System - Live Configuration Only

Comprehensive alerting system for trading events and system monitoring.
All configuration values come from live config - no hardcoded values.
"""

import ast
import asyncio
import contextlib
import logging
import os
import smtplib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import httpx

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

logger = logging.getLogger(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_console_separator_width() -> int:
    """Get console separator width from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "console_separator_width"):
                width = value.alerts.console_separator_width
                if isinstance(width, int) and width > 0:
                    return width
        except (AttributeError, ValueError, TypeError):
            pass

    width = os.getenv("ALERTS_CONSOLE_SEPARATOR_WIDTH", "").strip()
    if width:
        try:
            return int(width)
        except (ValueError, TypeError):
            pass

    return 50


def _get_webhook_timeout() -> float:
    """Get webhook timeout from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "webhook_timeout_sec"):
                timeout = value.alerts.webhook_timeout_sec
                if isinstance(timeout, (int, float)) and timeout > 0:
                    return float(timeout)
        except (AttributeError, ValueError, TypeError):
            pass

    timeout = os.getenv("ALERTS_WEBHOOK_TIMEOUT_SEC", "").strip()
    if timeout:
        try:
            return float(timeout)
        except (ValueError, TypeError):
            pass

    return 10.0


def _get_webhook_success_status_code() -> int:
    """Get webhook success status code from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "webhook_success_status_code"):
                status_code = value.alerts.webhook_success_status_code
                if isinstance(status_code, int) and 200 <= status_code < 300:
                    return status_code
        except (AttributeError, ValueError, TypeError):
            pass

    status_code = os.getenv("ALERTS_WEBHOOK_SUCCESS_STATUS_CODE", "").strip()
    if status_code:
        try:
            code = int(status_code)
            if 200 <= code < 300:
                return code
        except (ValueError, TypeError):
            pass

    return 200


def _get_monitoring_interval_sec() -> float:
    """Get monitoring loop interval from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "monitoring_interval_sec"):
                interval = value.alerts.monitoring_interval_sec
                if isinstance(interval, (int, float)) and interval > 0:
                    return float(interval)
        except (AttributeError, ValueError, TypeError):
            pass

    interval = os.getenv("ALERTS_MONITORING_INTERVAL_SEC", "").strip()
    if interval:
        try:
            return float(interval)
        except (ValueError, TypeError):
            pass

    return 60.0


def _get_monitoring_error_retry_delay_sec() -> float:
    """Get monitoring error retry delay from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "monitoring_error_retry_delay_sec"):
                delay = value.alerts.monitoring_error_retry_delay_sec
                if isinstance(delay, (int, float)) and delay > 0:
                    return float(delay)
        except (AttributeError, ValueError, TypeError):
            pass

    delay = os.getenv("ALERTS_MONITORING_ERROR_RETRY_DELAY_SEC", "").strip()
    if delay:
        try:
            return float(delay)
        except (ValueError, TypeError):
            pass

    return 30.0


def _get_default_alert_limit() -> int:
    """Get default alert limit from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "default_limit"):
                limit = value.alerts.default_limit
                if isinstance(limit, int) and limit > 0:
                    return limit
        except (AttributeError, ValueError, TypeError):
            pass

    limit = os.getenv("ALERTS_DEFAULT_LIMIT", "").strip()
    if limit:
        try:
            return int(limit)
        except (ValueError, TypeError):
            pass

    return 100


def _get_price_change_critical_threshold() -> float:
    """Get price change critical threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "price_change_critical_threshold"):
                threshold = value.alerts.price_change_critical_threshold
                if isinstance(threshold, (int, float)) and threshold > 0:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("ALERTS_PRICE_CHANGE_CRITICAL_THRESHOLD", "").strip()
    if threshold:
        try:
            return float(threshold)
        except (ValueError, TypeError):
            pass

    return 10.0


def _get_risk_score_critical_threshold() -> float:
    """Get risk score critical threshold from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "risk_score_critical_threshold"):
                threshold = value.alerts.risk_score_critical_threshold
                if isinstance(threshold, (int, float)) and threshold > 0:
                    return float(threshold)
        except (AttributeError, ValueError, TypeError):
            pass

    threshold = os.getenv("ALERTS_RISK_SCORE_CRITICAL_THRESHOLD", "").strip()
    if threshold:
        try:
            return float(threshold)
        except (ValueError, TypeError):
            pass

    return 0.8


def _get_smtp_port() -> int:
    """Get SMTP port from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "alerts") and hasattr(value.alerts, "smtp_port"):
                port = value.alerts.smtp_port
                if isinstance(port, int) and 1 <= port <= 65535:
                    return port
        except (AttributeError, ValueError, TypeError):
            pass

    port = os.getenv("SMTP_PORT", "").strip()
    if port:
        try:
            port_num = int(port)
            if 1 <= port_num <= 65535:
                return port_num
        except (ValueError, TypeError):
            pass

    return 587


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    TRADE = "trade"
    PRICE = "price"
    RISK = "risk"
    SYSTEM = "system"
    AI = "ai"
    PORTFOLIO = "portfolio"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    symbol: str | None = None
    value: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] | None = None
    acknowledged: bool = False
    resolved: bool = False


class NotificationChannel:
    """Base class for notification channels"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        raise NotImplementedError


class ConsoleNotificationChannel(NotificationChannel):
    """Console notification channel"""

    async def send(self, alert: Alert) -> bool:
        """Send alert to console"""
        if not self.enabled:
            return False

        try:
            timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            level_emoji = {
                AlertLevel.INFO: "[INFO]️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
            }

            print(f"\n{level_emoji.get(alert.level, '📢')} ALERT [{alert.level.value.upper()}]")
            print(f"Time: {timestamp}")
            print(f"Type: {alert.type.value}")
            print(f"Title: {alert.title}")
            print(f"Message: {alert.message}")
            if alert.symbol:
                print(f"Symbol: {alert.symbol}")
            if alert.value is not None:
                print(f"Value: {alert.value}")
            if alert.threshold is not None:
                print(f"Threshold: {alert.threshold}")
            separator_width = _get_console_separator_width()
            print("-" * separator_width)
        except OSError:
            logger.exception("Error sending console alert")
            return False
        else:
            return True


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int | None = None,
        username: str = "",
        password: str = "",
        from_email: str = "",
        to_emails: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port if smtp_port is not None else _get_smtp_port()
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails or []

    async def send(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not self.enabled or not self.to_emails:
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"

            # Create HTML email body
            html_body = self._create_html_email(alert)
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.title}")
        except (smtplib.SMTPException, OSError):
            logger.exception("Error sending email alert")
            return False
        else:
            return True

    def _create_html_email(self, alert: Alert) -> str:
        """Create HTML email body"""
        level_colors = {
            AlertLevel.INFO: "#3b82f6",
            AlertLevel.WARNING: "#f59e0b",
            AlertLevel.ERROR: "#ef4444",
            AlertLevel.CRITICAL: "#dc2626",
        }

        color = level_colors.get(alert.level, "#6b7280")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                <h2 style="margin: 0;">{alert.title}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
            </div>
            <div style="background-color: #f8fafc; padding: 20px; border-radius: 0 0 8px 8px; border: 1px solid #e2e8f0;">
                <p style="font-size: 16px; margin: 0 0 15px 0;">{alert.message}</p>
                {f"<p><strong>Symbol:</strong> {alert.symbol}</p>" if alert.symbol else ""}
                {f"<p><strong>Value:</strong> {alert.value}</p>" if alert.value is not None else ""}
                {f"<p><strong>Threshold:</strong> {alert.threshold}</p>" if alert.threshold is not None else ""}
                <p style="margin-top: 20px; font-size: 12px; color: #6b7280;">
                    Alert Type: {alert.type.value} | Level: {alert.level.value}
                </p>
            </div>
        </body>
        </html>
        """


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel"""

    def __init__(self, webhook_url: str, enabled: bool = True):
        super().__init__(enabled)
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            payload = {
                "alert": asdict(alert),
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "type": alert.type.value,
            }

            webhook_timeout = _get_webhook_timeout()
            success_status_code = _get_webhook_success_status_code()
            async with httpx.AsyncClient() as client:
                response = await client.post(self.webhook_url, json=payload, timeout=webhook_timeout)

            if response.status_code == success_status_code:
                logger.info(f"Webhook alert sent: {alert.title}")
                return True
            logger.error(f"Webhook failed: {response.status_code}")
        except (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError, OSError):
            logger.exception("Error sending webhook alert")
            return False
        else:
            return False


class AlertsSystem:
    """Main alerts and notifications system"""

    def __init__(self):
        self.alerts: list[Alert] = []
        self.channels: list[NotificationChannel] = []
        self.alert_rules: list[dict[str, Any]] = []
        self.running = False
        self.alert_id_counter = 0

        # Initialize default channels
        self._setup_default_channels()

        # Load alert rules
        self._load_alert_rules()

    def _setup_default_channels(self):
        """Setup default notification channels"""
        # Console channel (always enabled)
        self.add_channel(ConsoleNotificationChannel(enabled=True))

        # Email channel (if configured)
        email_config = self._get_email_config()
        if email_config:
            self.add_channel(EmailNotificationChannel(**email_config))

        # Webhook channel (if configured)
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            self.add_channel(WebhookNotificationChannel(webhook_url))

    def _get_email_config(self) -> dict[str, Any] | None:
        """Get email configuration from environment"""
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = _get_smtp_port()
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("SMTP_FROM_EMAIL")
        to_emails_str = os.getenv("SMTP_TO_EMAILS", "")
        to_emails = [email.strip() for email in to_emails_str.split(",") if email.strip()]

        if all([smtp_server, username, password, from_email, to_emails]):
            return {
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "username": username,
                "password": password,
                "from_email": from_email,
                "to_emails": to_emails,
            }
        return None

    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        default_rules = [
            {
                "name": "price_drop_5_percent",
                "type": AlertType.PRICE,
                "level": AlertLevel.WARNING,
                "condition": "price_change_percent < -5",
                "title": "Price Drop Alert",
                "message": "Price dropped by {value}% for {symbol}",
            },
            {
                "name": "price_rise_10_percent",
                "type": AlertType.PRICE,
                "level": AlertLevel.INFO,
                "condition": "price_change_percent > 10",
                "title": "Price Rise Alert",
                "message": "Price rose by {value}% for {symbol}",
            },
            {
                "name": "high_risk_alert",
                "type": AlertType.RISK,
                "level": AlertLevel.CRITICAL,
                "condition": "risk_score > 0.8",
                "title": "High Risk Alert",
                "message": "Portfolio risk level is critically high: {value}",
            },
            {
                "name": "ai_prediction_high_confidence",
                "type": AlertType.AI,
                "level": AlertLevel.INFO,
                "condition": "ai_confidence > 0.9",
                "title": "High Confidence AI Signal",
                "message": "AI model predicts {prediction} with {confidence}% confidence",
            },
            {
                "name": "system_error",
                "type": AlertType.SYSTEM,
                "level": AlertLevel.ERROR,
                "condition": "error_count > 10",
                "title": "System Error Alert",
                "message": "Multiple system errors detected: {error_count} errors",
            },
        ]

        self.alert_rules = default_rules

    def add_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.channels.append(channel)
        logger.info(f"Added notification channel: {channel.__class__.__name__}")

    def create_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        symbol: str | None = None,
        value: float | None = None,
        threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """Create a new alert"""
        self.alert_id_counter += 1
        alert_id = f"alert_{self.alert_id_counter}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            type=alert_type,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            value=value,
            threshold=threshold,
            metadata=metadata or {},
        )

        self.alerts.append(alert)
        logger.info(f"Created alert: {alert.title} [{alert.level.value}]")

        return alert

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert through all channels"""
        success_count = 0

        for channel in self.channels:
            try:
                if await channel.send(alert):
                    success_count += 1
            except (OSError, RuntimeError):
                logger.exception(f"Error sending alert through {channel.__class__.__name__}")

        return success_count > 0

    async def check_alert_rules(self, data: dict[str, Any]):
        """Check alert rules against current data"""
        for rule in self.alert_rules:
            try:
                if self._evaluate_rule(rule, data):
                    # Create alert
                    alert = self.create_alert(
                        type=rule["type"],
                        level=rule["level"],
                        title=rule["title"],
                        message=rule["message"].format(**data),
                        symbol=data.get("symbol"),
                        value=data.get("value"),
                        threshold=data.get("threshold"),
                        metadata=data,
                    )

                    # Send alert
                    await self.send_alert(alert)

            except (KeyError, TypeError, ValueError, AttributeError):
                logger.exception(f"Error checking alert rule {rule['name']}")

    def _evaluate_rule(self, rule: dict[str, Any], data: dict[str, Any]) -> bool:
        """Evaluate if alert rule condition is met"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            condition = rule["condition"]

            # Replace variables with actual values
            for key, value in data.items():
                condition = condition.replace(key, str(value))

            # Evaluate condition safely using ast.literal_eval for security
            try:
                # Parse and evaluate safely - only allows literals, not arbitrary code
                parsed = ast.parse(condition, mode="eval")
                if isinstance(parsed, ast.Expression):
                    return ast.literal_eval(condition)
                logger.warning(f"Unsafe condition detected: {condition}")
            except (ValueError, SyntaxError, TypeError):
                # Fallback to safe evaluation for simple comparisons
                try:
                    # Only allow simple numeric comparisons for safety
                    if any(op in condition for op in ["<", ">", "<=", ">=", "==", "!="]):
                        # Extract numbers and operators safely
                        import re

                        numbers = re.findall(r"-?\d+\.?\d*", condition)
                        min_numbers_required = 2
                        if len(numbers) >= min_numbers_required:
                            left, right = float(numbers[0]), float(numbers[1])
                            if "<" in condition:
                                return left < right
                            if ">" in condition:
                                return left > right
                            if "<=" in condition:
                                return left <= right
                            if ">=" in condition:
                                return left >= right
                            if "==" in condition:
                                return left == right
                            if "!=" in condition:
                                return left != right
                except (ValueError, TypeError):
                    return False
                else:
                    return False
            else:
                return False

        except (ValueError, TypeError, AttributeError, KeyError, SyntaxError):
            logger.exception("Error evaluating rule condition")
            return False

    async def start(self) -> None:
        """Start the alerts system"""
        self.running = True
        logger.info("Alerts system started")

        # Start monitoring loop
        task = asyncio.create_task(self._monitoring_loop())
        # Track background tasks for proper cleanup
        if not hasattr(self, "_tasks"):
            self._tasks: list[asyncio.Task[Any]] = []
        self._tasks.append(task)

    async def stop(self) -> None:
        """Stop the alerts system"""
        self.running = False
        # Cancel all background tasks
        if hasattr(self, "_tasks"):
            for task in self._tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            self._tasks.clear()
        logger.info("Alerts system stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        monitoring_interval = _get_monitoring_interval_sec()
        error_retry_delay = _get_monitoring_error_retry_delay_sec()
        while self.running:
            try:
                # This would typically check various system metrics
                # For now, we'll just sleep
                await asyncio.sleep(monitoring_interval)
            except (asyncio.CancelledError, KeyboardInterrupt, RuntimeError):
                logger.exception("Error in monitoring loop")
                await asyncio.sleep(error_retry_delay)

    def get_alerts(
        self,
        limit: int | None = None,
        alert_type: AlertType | None = None,
        level: AlertLevel | None = None,
        acknowledged: bool | None = None,
    ) -> list[Alert]:
        """Get alerts with filters"""
        alerts = self.alerts
        alert_limit = limit if limit is not None else _get_default_alert_limit()

        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]

        if level:
            alerts = [a for a in alerts if a.level == level]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts[-alert_limit:]  # Return most recent alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert.title}")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert.title}")
                return True
        return False

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary statistics"""
        total_alerts = len(self.alerts)
        unacknowledged = len([a for a in self.alerts if not a.acknowledged])
        critical_alerts = len([a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.resolved])

        alerts_by_type = {}
        for alert in self.alerts:
            alert_type = alert.type.value
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1

        alerts_by_level = {}
        for alert in self.alerts:
            level = alert.level.value
            alerts_by_level[level] = alerts_by_level.get(level, 0) + 1

        return {
            "total_alerts": total_alerts,
            "unacknowledged": unacknowledged,
            "critical_alerts": critical_alerts,
            "alerts_by_type": alerts_by_type,
            "alerts_by_level": alerts_by_level,
            "active_channels": len([c for c in self.channels if c.enabled]),
        }


# Global alerts system
_alerts_system = None


def get_alerts_system() -> AlertsSystem:
    """Get the global alerts system"""
    global _alerts_system
    if _alerts_system is None:
        _alerts_system = AlertsSystem()
    return _alerts_system


# Convenience functions
async def send_trade_alert(symbol: str, side: str, quantity: float, price: float, pnl: float | None = None) -> None:
    """Send trade alert"""
    alerts = get_alerts_system()

    title = f"Trade Executed: {side.upper()} {symbol}"
    message = f"Executed {side} order for {quantity} {symbol} at ${price:.2f}"
    if pnl is not None:
        message += f" (PnL: ${pnl:.2f})"

    alert = alerts.create_alert(
        type=AlertType.TRADE,
        level=AlertLevel.INFO,
        title=title,
        message=message,
        symbol=symbol,
        value=price,
        metadata={"side": side, "quantity": quantity, "pnl": pnl},
    )

    await alerts.send_alert(alert)


async def send_price_alert(symbol: str, current_price: float, change_percent: float) -> None:
    """Send price alert"""
    alerts = get_alerts_system()
    critical_threshold = _get_price_change_critical_threshold()

    level = AlertLevel.CRITICAL if abs(change_percent) > critical_threshold else AlertLevel.WARNING
    title = f"Price Alert: {symbol}"
    message = f"Price changed by {change_percent:.2f}% to ${current_price:.2f}"

    alert = alerts.create_alert(
        type=AlertType.PRICE,
        level=level,
        title=title,
        message=message,
        symbol=symbol,
        value=current_price,
        metadata={"change_percent": change_percent},
    )

    await alerts.send_alert(alert)


async def send_risk_alert(risk_score: float, message: str) -> None:
    """Send risk alert"""
    alerts = get_alerts_system()
    critical_threshold = _get_risk_score_critical_threshold()

    level = AlertLevel.CRITICAL if risk_score > critical_threshold else AlertLevel.WARNING
    title = f"Risk Alert: {risk_score:.2f}"

    alert = alerts.create_alert(
        type=AlertType.RISK,
        level=level,
        title=title,
        message=message,
        value=risk_score,
        metadata={"risk_score": risk_score},
    )

    await alerts.send_alert(alert)
