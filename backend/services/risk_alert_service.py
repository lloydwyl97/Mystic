"""
Risk Alert Service for Mystic AI Trading Platform
Provides real-time risk monitoring and alerting for trading operations.

Quick Test Checklist:
- ASCII-only logging and prints; UTC timestamps.
- No external exchange IDs or non-ASCII symbols.
- Accepts symbol formats like BTCUSDT or BTC/USDT; compares robustly.
- Uses canonical_cache only; network calls limited to optional Discord webhook (httpx, 10s timeout).
- No unreachable code.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

# Add backend to path for imports
from backend.services.canonical_cache import canonical_cache

logger = logging.getLogger(__name__)

# Optional symbol normalization helper
try:
    from backend.utils.symbols import to_ccxt_symbol as _to_ccxt_symbol  # type: ignore[import-not-found]
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):

    def _to_ccxt_symbol(symbol: str) -> str:
        """Fallback symbol normalization if utils.symbols not available"""
        s = str(symbol).strip().upper()
        # Normalize to ccxt format
        if "/" not in s:
            if s.endswith("USDT"):
                s = f"{s[:-4]}/USDT"
            elif s.endswith("USD"):
                s = f"{s[:-3]}/USDT"
            else:
                s = f"{s}/USDT"
        return s


def _iso_now() -> str:
    """Get current UTC timestamp in ISO-8601 format with Z suffix"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(s: str) -> datetime | None:
    try:
        if not s:
            return None
        # handle Z suffix
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _symbol_aliases(symbol: str) -> list[str]:
    """Return a small set of equivalent forms for robust matching."""
    s = str(symbol).strip().upper()
    if not s:
        return []
    if "/" in s:
        base, quote = s.split("/", 1)
    elif "-" in s:
        base, quote = s.split("-", 1)
    elif s.endswith("USDT"):
        base, quote = s[:-4], "USDT"
    else:
        # default to USDT-quoted for aliasing purposes
        base, quote = s, "USDT"
    ccxt = _to_ccxt_symbol(f"{base}/{quote}")
    collapsed = f"{base}{quote}"
    dashed = f"{base}-{quote}"
    return list({s, ccxt, collapsed, dashed})


class RiskAlertService:
    def __init__(self) -> None:
        """Initialize risk alert service with monitoring parameters."""
        self.cache = canonical_cache

        # Risk thresholds (env overrides allowed)
        self.risk_thresholds: dict[str, float] = {
            "drawdown_percentage": float(os.getenv("RISK_DRAWDOWN_PCT", "10.0")),  # 10%
            "volatility_spike_percentage": float(os.getenv("RISK_VOL_SPIKE_PCT", "5.0")),  # 5% avg pct move over window
            "max_exposure_percentage": float(os.getenv("RISK_MAX_EXPOSURE_PCT", "20.0")),  # 20% of portfolio
            "api_delay_threshold_seconds": float(os.getenv("RISK_API_DELAY_SEC", "30")),  # 30s
            "missing_data_threshold_minutes": float(os.getenv("RISK_MISSING_MIN", "5")),  # 5m
        }

        # Alert level tags for console and Discord titles
        self.alert_levels = {
            "LOW": "[LOW]",
            "MEDIUM": "[MED]",
            "HIGH": "[HIGH]",
            "CRITICAL": "[CRIT]",
        }

        # Discord webhook support
        self.discord_webhook_url = os.getenv("RISK_WEBHOOK_URL", "").strip() or None
        self.discord_enabled = bool(self.discord_webhook_url)

        # Risk monitoring state
        self.last_check_time: datetime = datetime.now(timezone.utc)
        self.active_alerts: list[dict[str, Any]] = []
        self.alert_history: list[dict[str, Any]] = []

        # Portfolio tracking
        self.portfolio_exposure: dict[str, float] = {}
        self.last_trade_prices: dict[str, float] = {}

        logger.info("RiskAlertService initialized")

    async def start(self):
        """Start risk alert service (placeholder for lifecycle compatibility)"""
        logger.info("RiskAlertService started")
        return True

    # ----------------------- Calculations -----------------------

    def _calculate_drawdown(self, current_price: float, last_trade_price: float) -> float:
        """Calculate drawdown percentage from last trade."""
        try:
            if last_trade_price <= 0:
                return 0.0
            return max(0.0, (last_trade_price - current_price) / last_trade_price * 100.0)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Drawdown calc failed: %s", e)
            return 0.0

    def _calculate_volatility(self, prices: list[float]) -> float:
        """Average absolute percentage change over the given sequence."""
        try:
            if len(prices) < 2:
                return 0.0
            changes: list[float] = []
            for i in range(1, len(prices)):
                p0 = float(prices[i - 1])
                p1 = float(prices[i])
                if p0 > 0.0:
                    changes.append(abs((p1 - p0) / p0 * 100.0))
            return sum(changes) / len(changes) if changes else 0.0
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Volatility calc failed: %s", e)
            return 0.0

    # ----------------------- Data access -----------------------

    def _get_recent_prices(self, symbol: str, minutes: int = 5) -> list[float]:
        """Get recent price data for volatility calculation."""
        try:
            signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=200)
            aliases = set(_symbol_aliases(symbol))
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            out: list[float] = []
            for sig in signals:
                sym = str(sig.get("symbol", "")).upper()
                if sym not in aliases:
                    continue
                ts = _parse_iso(sig.get("timestamp", ""))
                if not ts or ts < cutoff:
                    continue
                price = float(sig.get("metadata", {}).get("price", 0.0) or 0.0)
                if price > 0.0:
                    out.append(price)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Recent prices fetch failed for %s: %s", symbol, e)
            return []
        else:
            return out

    def _get_portfolio_exposure(self) -> dict[str, float]:
        """Aggregate USD exposure per symbol from trade signals."""
        try:
            signals = self.cache.get_signals_by_type("TRADE_EXECUTED", limit=500)
            exposure: dict[str, float] = {}
            for sig in signals:
                symbol = str(sig.get("symbol", "")).upper()
                data = sig.get("metadata", {}) or {}
                ttype = str(data.get("trade_type", "")).upper()
                amt = float(data.get("amount_usd", 0.0) or 0.0)
                if not symbol or amt <= 0.0:
                    continue
                if ttype == "BUY":
                    exposure[symbol] = exposure.get(symbol, 0.0) + amt
                elif ttype == "SELL":
                    exposure[symbol] = exposure.get(symbol, 0.0) - amt
            # Only keep positive net exposures
            return {k: v for k, v in exposure.items() if v > 0.0}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Portfolio exposure fetch failed: %s", e)
            return {}

    def _get_last_trade_prices(self) -> dict[str, float]:
        """Most recent trade prices per symbol based on timestamp."""
        try:
            signals = self.cache.get_signals_by_type("TRADE_EXECUTED", limit=500)
            latest: dict[str, tuple[datetime, float]] = {}
            for sig in signals:
                symbol = str(sig.get("symbol", "")).upper()
                data = sig.get("metadata", {}) or {}
                price = float(data.get("price", 0.0) or 0.0)
                ts = _parse_iso(sig.get("timestamp", ""))
                if not symbol or price <= 0.0 or ts is None:
                    continue
                prev = latest.get(symbol)
                if prev is None or ts > prev[0]:
                    latest[symbol] = (ts, price)
            return {k: v for k, (_, v) in latest.items()}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Last trade prices fetch failed: %s", e)
            return {}

    # ----------------------- Risk checks -----------------------

    def _check_drawdown_risk(self, symbol: str, current_price: float) -> dict[str, Any] | None:
        try:
            last_price = float(self.last_trade_prices.get(symbol, 0.0) or 0.0)
            if last_price <= 0.0:
                return None
            dd = self._calculate_drawdown(current_price, last_price)
            th = self.risk_thresholds["drawdown_percentage"]
            if dd >= th:
                level = "HIGH" if dd >= max(15.0, th + 5.0) else "MEDIUM"
                return {
                    "risk_type": "DRAWDOWN",
                    "symbol": symbol,
                    "current_price": current_price,
                    "last_trade_price": last_price,
                    "drawdown_percentage": round(dd, 4),
                    "threshold": th,
                    "level": level,
                    "message": f"Drawdown alert: {symbol} down {dd:.2f}% from last trade",
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Drawdown risk check failed for %s: %s", symbol, e)
            return None
        else:
            return None

    def _check_volatility_risk(self, symbol: str) -> dict[str, Any] | None:
        try:
            recent_prices = self._get_recent_prices(
                symbol,
                minutes=int(self.risk_thresholds["missing_data_threshold_minutes"]),
            )
            if len(recent_prices) < 2:
                return None
            vol = self._calculate_volatility(recent_prices)
            th = self.risk_thresholds["volatility_spike_percentage"]
            if vol >= th:
                level = "HIGH" if vol >= max(10.0, th + 5.0) else "MEDIUM"
                return {
                    "risk_type": "VOLATILITY_SPIKE",
                    "symbol": symbol,
                    "volatility_percentage": round(vol, 4),
                    "threshold": th,
                    "price_count": len(recent_prices),
                    "level": level,
                    "message": f"Volatility spike: {symbol} showing {vol:.2f}% average movement",
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Volatility risk check failed for %s: %s", symbol, e)
            return None
        else:
            return None

    def _check_exposure_risk(self, symbol: str, current_exposure: float) -> dict[str, Any] | None:
        try:
            total = sum(self.portfolio_exposure.values())
            if total <= 0.0:
                return None
            pct = current_exposure / total * 100.0
            th = self.risk_thresholds["max_exposure_percentage"]
            if pct >= th:
                level = "CRITICAL" if pct >= max(30.0, th + 10.0) else "HIGH"
                return {
                    "risk_type": "EXPOSURE_LIMIT",
                    "symbol": symbol,
                    "exposure_usd": round(current_exposure, 2),
                    "exposure_percentage": round(pct, 4),
                    "total_portfolio": round(total, 2),
                    "threshold": th,
                    "level": level,
                    "message": f"Exposure limit: {symbol} at {pct:.2f}% of portfolio",
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Exposure risk check failed for %s: %s", symbol, e)
            return None
        else:
            return None

    def _check_api_health(self) -> list[dict[str, Any]]:
        """Check global API health and data freshness."""
        try:
            alerts: list[dict[str, Any]] = []
            signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=10)
            if not signals:
                alerts.append(
                    {
                        "risk_type": "API_HEALTH",
                        "issue": "NO_RECENT_DATA",
                        "level": "CRITICAL",
                        "message": "No recent price data available; API may be down",
                    },
                )
                return alerts

            latest = signals[0]
            ts = _parse_iso(latest.get("timestamp", ""))
            if ts is None:
                alerts.append(
                    {
                        "risk_type": "API_HEALTH",
                        "issue": "INVALID_TIMESTAMP",
                        "level": "MEDIUM",
                        "message": "Latest price update has invalid timestamp",
                    },
                )
                return alerts

            age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
            delay_th = self.risk_thresholds["api_delay_threshold_seconds"]
            if age_sec > delay_th:
                alerts.append(
                    {
                        "risk_type": "API_HEALTH",
                        "issue": "DATA_DELAY",
                        "delay_seconds": int(age_sec),
                        "threshold": delay_th,
                        "level": "HIGH" if age_sec > max(60.0, delay_th * 2) else "MEDIUM",
                        "message": f"API data delay: {age_sec:.0f}s old (threshold: {delay_th:.0f}s)",
                    },
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("API health check failed: %s", e)
            return [
                {
                    "risk_type": "API_HEALTH",
                    "issue": "CHECK_FAILED",
                    "level": "MEDIUM",
                    "message": f"API health check failed: {e!s}",
                },
            ]
        else:
            return alerts

    def _check_missing_data_per_symbol(self, symbol: str) -> dict[str, Any] | None:
        """Emit alert if no price updates for this symbol within threshold minutes."""
        try:
            minutes = int(self.risk_thresholds["missing_data_threshold_minutes"])
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=100)
            aliases = set(_symbol_aliases(symbol))
            latest_ts: datetime | None = None
            for sig in signals:
                sym = str(sig.get("symbol", "")).upper()
                if sym not in aliases:
                    continue
                ts = _parse_iso(sig.get("timestamp", ""))
                if ts and (latest_ts is None or ts > latest_ts):
                    latest_ts = ts
            if latest_ts is None or latest_ts < cutoff:
                age_s = int((datetime.now(timezone.utc) - (latest_ts or cutoff)).total_seconds())
                return {
                    "risk_type": "MISSING_DATA",
                    "symbol": symbol,
                    "level": "MEDIUM",
                    "message": f"No price updates for {symbol} in the last {minutes} minutes",
                    "last_seen_seconds": age_s,
                    "threshold_minutes": minutes,
                }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Missing data check failed for %s: %s", symbol, e)
            return None
        else:
            return None

    # ----------------------- Alert I/O -----------------------

    async def _send_discord_alert(self, alert: dict[str, Any]) -> bool:
        """Send alert to Discord webhook if configured."""
        try:
            if not self.discord_enabled or not self.discord_webhook_url:
                return False

            level_tag = self.alert_levels.get(alert.get("level", "MEDIUM"), "[MED]")
            title = f"{level_tag} Risk Alert: {alert.get('risk_type', 'UNKNOWN')}"
            embed = {
                "title": title,
                "description": alert.get("message", "Risk alert triggered"),
                "color": {
                    "LOW": 0xFFFF00,  # Yellow
                    "MEDIUM": 0xFFA500,  # Orange
                    "HIGH": 0xFF0000,  # Red
                    "CRITICAL": 0x8B0000,  # Dark Red
                }.get(str(alert.get("level", "MEDIUM")).upper(), 0xFFA500),
                "timestamp": _iso_now(),
                "fields": [],
            }

            for key, value in alert.items():
                if key not in {"risk_type", "level", "message"} and value is not None:
                    embed["fields"].append(
                        {
                            "name": key.replace("_", " ").title(),
                            "value": str(value),
                            "inline": True,
                        }
                    )

            async with httpx.AsyncClient() as client:
                resp = await client.post(self.discord_webhook_url, json={"embeds": [embed]}, timeout=10)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Discord alert send failed: %s", e)
            return False
        else:
            return resp.status_code == 204

    def _store_alert(self, alert: dict[str, Any]) -> None:
        """Store alert in cache and keep short active/history windows."""
        try:
            alert_id = f"risk_alert_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            payload = {
                **alert,
                "alert_id": alert_id,
                "timestamp": _iso_now(),
            }
            self.cache.store_signal(
                signal_id=alert_id,
                symbol=alert.get("symbol", "RISK_ALERT"),
                signal_type="RISK_ALERT",
                confidence=1.0,
                strategy="risk_monitoring",
                metadata=payload,
            )
            self.active_alerts.append(payload)
            self.alert_history.append(payload)
            # Trim lists
            if len(self.active_alerts) > 50:
                self.active_alerts = self.active_alerts[-50:]
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Alert store failed: %s", e)

    def _print_alert(self, alert: dict[str, Any]) -> None:
        """Print alert to stdout in ASCII-only."""
        try:
            tag = self.alert_levels.get(alert.get("level", "MEDIUM"), "[MED]")
            level = alert.get("level", "MEDIUM")
            logger.info(f"{tag} [{level}] {alert.get('message', 'Risk alert')}")
            if level in ("HIGH", "CRITICAL"):
                for key, value in alert.items():
                    if key not in {"level", "message"} and value is not None:
                        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
                logger.info("")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Alert print failed: %s", e)

    # ----------------------- Public API -----------------------

    def check_risks(self) -> dict[str, Any]:
        """Check all risk conditions and trigger alerts."""
        try:
            logger.info("Checking risk conditions")

            # Update exposure and last trade prices
            self.portfolio_exposure = self._get_portfolio_exposure()
            self.last_trade_prices = self._get_last_trade_prices()

            all_alerts: list[dict[str, Any]] = []

            # Global API health
            all_alerts.extend(self._check_api_health())

            # Per-symbol checks (only symbols with net positive exposure)
            for symbol, exposure in self.portfolio_exposure.items():
                # Current price: most recent price_update for this symbol
                current_price = 0.0
                aliases = set(_symbol_aliases(symbol))
                recent_signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=50)
                latest_ts: datetime | None = None
                for sig in recent_signals:
                    sym = str(sig.get("symbol", "")).upper()
                    if sym not in aliases:
                        continue
                    ts = _parse_iso(sig.get("timestamp", ""))
                    price = float(sig.get("metadata", {}).get("price", 0.0) or 0.0)
                    if ts and price > 0.0 and (latest_ts is None or ts > latest_ts):
                        latest_ts = ts
                        current_price = price

                # Missing data alert
                missing_alert = self._check_missing_data_per_symbol(symbol)
                if missing_alert:
                    all_alerts.append(missing_alert)

                if current_price <= 0.0:
                    continue

                # Drawdown
                dd_alert = self._check_drawdown_risk(symbol, current_price)
                if dd_alert:
                    all_alerts.append(dd_alert)

                # Volatility
                vol_alert = self._check_volatility_risk(symbol)
                if vol_alert:
                    all_alerts.append(vol_alert)

                # Exposure
                exp_alert = self._check_exposure_risk(symbol, exposure)
                if exp_alert:
                    all_alerts.append(exp_alert)

            # Process alerts
            for alert in all_alerts:
                self._store_alert(alert)
                self._print_alert(alert)
                if self.discord_enabled:
                    self._send_discord_alert(alert)

            # Update last check time
            self.last_check_time = datetime.now(timezone.utc)

            result = {
                "timestamp": self.last_check_time.isoformat(),
                "alerts_generated": len(all_alerts),
                "active_alerts": len(self.active_alerts),
                "portfolio_exposure": self.portfolio_exposure,
                "risk_levels": {level: len([a for a in all_alerts if a.get("level") == level]) for level in self.alert_levels},
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Risk checks failed: %s", e)
            return {"error": str(e), "timestamp": _iso_now()}
        else:
            logger.info("Risk check complete: %d alerts generated", len(all_alerts))
            return result

    def get_latest_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get latest risk alerts from cache."""
        try:
            signals = self.cache.get_signals_by_type("RISK_ALERT", limit=max(1, int(limit)))
            alerts: list[dict[str, Any]] = []
            for sig in signals:
                meta = sig.get("metadata", {}) or {}
                alerts.append(
                    {
                        "alert_id": meta.get("alert_id"),
                        "timestamp": sig.get("timestamp"),
                        "symbol": sig.get("symbol"),
                        "risk_type": meta.get("risk_type"),
                        "level": meta.get("level"),
                        "message": meta.get("message"),
                        "details": {
                            k: v
                            for k, v in meta.items()
                            if k
                            not in {
                                "alert_id",
                                "risk_type",
                                "level",
                                "message",
                                "timestamp",
                            }
                        },
                    },
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Latest alerts fetch failed: %s", e)
            return []
        else:
            return alerts

    def get_risk_status(self) -> dict[str, Any]:
        """Return service status snapshot."""
        try:
            return {
                "service": "RiskAlertService",
                "status": "active",
                "last_check": self.last_check_time.isoformat(),
                "active_alerts": len(self.active_alerts),
                "total_alerts": len(self.alert_history),
                "discord_enabled": self.discord_enabled,
                "risk_thresholds": self.risk_thresholds,
                "portfolio_exposure": self.portfolio_exposure,
                "alert_levels": self.alert_levels,
                "timestamp": _iso_now(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Risk status failed: %s", e)
            return {"success": False, "error": str(e)}


# Global risk alert service instance
risk_alert_service = RiskAlertService()


def get_risk_alert_service() -> RiskAlertService:
    """Get the global risk alert service instance."""
    return risk_alert_service


if __name__ == "__main__":
    # Simple smoke test
    service = RiskAlertService()
    logger.info("RiskAlertService initialized: %s", service is not None)

    result = service.check_risks()
    logger.info("Risk check result: %s", result)

    alerts = service.get_latest_alerts()
    logger.info("Latest alerts: %s", alerts[:3])

    status = service.get_risk_status()
    logger.info("Service status: %s", status.get("status", "unknown"))
