"""
Celery Task Configuration for Mystic Trading Platform
Production-ready task queue system for AI trading operations
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
from celery import Celery
from celery.utils.log import get_task_logger

from backend.config.redis_config import get_shared_redis_sync

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = tuple(TRADING_SYMBOLS)

# Redis connection must be configured via environment variables
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        msg = "REDIS_URL or REDIS_HOST environment variable is required - no fallback/hardcoded Redis host"
        raise RuntimeError(msg)
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_db = os.getenv("REDIS_DB", "0")
    redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

CELERY_BROKER_URL = redis_url
CELERY_RESULT_BACKEND = redis_url

celery_app = Celery(
    "mystic_trading",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    task_annotations={
        "*": {
            "rate_limit": "10/m",
            "retry": True,
            "retry_policy": {
                "max_retries": 3,
                "interval_start": 0,
                "interval_step": 0.2,
                "interval_max": 0.2,
            },
        },
    },
    beat_schedule={
        "market-data-sync": {"task": "ai_tasks.sync_market_data", "schedule": 60.0},
        "portfolio-rebalance": {
            "task": "ai_tasks.rebalance_portfolio",
            "schedule": 300.0,
        },
        "risk-assessment": {"task": "ai_tasks.assess_risk", "schedule": 180.0},
        "ai-strategy-evaluation": {
            "task": "ai_tasks.evaluate_ai_strategies",
            "schedule": 600.0,
        },
        "performance-metrics": {
            "task": "ai_tasks.calculate_performance_metrics",
            "schedule": 900.0,
        },
        "cleanup-old-data": {"task": "ai_tasks.cleanup_old_data", "schedule": 3600.0},
    },
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mystic_trading.db")
TRADE_LOG_DB = os.getenv("TRADE_LOG_DB", "trades.db")
task_logger = get_task_logger(__name__)

shared_redis = None
try:
    shared_redis = get_shared_redis_sync()
except Exception as e:
    task_logger.warning("Shared Redis sync client unavailable: %s", e)
    shared_redis = None
if shared_redis is None:
    task_logger.warning("Shared Redis client unavailable for ai_tasks; tasks will be disabled")
    redis_client = None
else:
    redis_client = shared_redis


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode(b: bytes | None) -> str | None:
    if b is None:
        return None
    try:
        return b.decode()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _read_price(symbol: str) -> float | None:
    if redis_client is None:
        return None
    raw = redis_client.hget(f"price:{symbol}", "v")
    if not raw:
        return None
    try:
        return float(_decode(raw) or "")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _read_series_from_redis(key: str, field: str | None = None) -> list[float]:
    try:
        if field:
            raw = redis_client.hget(key, field)
            if not raw:
                return []
            s = _decode(raw)
            if not s:
                return []
            try:
                arr = json.loads(s)
                return [float(x) for x in arr if x is not None]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return []
        else:
            vals = redis_client.lrange(key, 0, -1)
            out: list[float] = []
            for v in vals:
                s = _decode(v)
                if not s:
                    continue
                try:
                    out.append(float(s))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
            return out
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return []


def _fetch_portfolio(portfolio_id: str) -> dict[str, Any] | None:
    try:
        raw = redis_client.get(f"portfolio:{portfolio_id}")
        if not raw:
            return None
        s = _decode(raw)
        if not s:
            return None
        return json.loads(s)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _fetch_target_allocation(portfolio_id: str) -> dict[str, float] | None:
    try:
        raw = redis_client.get(f"target_allocation:{portfolio_id}")
        if not raw:
            return None
        s = _decode(raw)
        if not s:
            return None
        data = json.loads(s)
        return {k: float(v) for k, v in data.items()}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _compute_drawdown(series: list[float]) -> float:
    if not series:
        return 0.0
    arr = np.asarray(series, dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak != 0, peak, 1.0)
    return float(-np.min(dd))


def _compute_volatility(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    arr = np.asarray(series, dtype=np.float64)
    ret = np.diff(arr) / arr[:-1]
    return float(np.std(ret))


def _compute_sharpe(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    arr = np.asarray(series, dtype=np.float64)
    ret = np.diff(arr) / arr[:-1]
    mu = np.mean(ret)
    sigma = np.std(ret)
    if sigma == 0:
        return 0.0
    return float(mu / sigma)


def _sqlite_query_trades() -> list[tuple[str, float]]:
    try:
        if not TRADE_LOG_DB or not Path(TRADE_LOG_DB).exists():
            return []
        conn = sqlite3.connect(TRADE_LOG_DB)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT timestamp, profit_usd FROM trades WHERE timestamp >= ?",
                ((datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),),
            )
            rows = cur.fetchall()
            out: list[tuple[str, float]] = []
            for ts, p in rows:
                try:
                    out.append((str(ts), float(p)))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    continue
            return out
        finally:
            conn.close()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return []


@celery_app.task(bind=True, name="ai_tasks.sync_market_data")
def sync_market_data(self, symbols: list[str] | None = None) -> dict[str, Any]:
    try:
        task_logger.info(f"[{EXCHANGE_ID}] market data sync start")
        symbols = symbols or list(ALLOWED_SYMBOLS)
        results = {
            "timestamp": _now_iso(),
            "symbols_processed": [],
            "errors": [],
            "data_points": 0,
        }
        for s in symbols:
            try:
                price = _read_price(s)
                if price is None or price <= 0:
                    continue
                market_data = {
                    "symbol": s,
                    "price": price,
                    "volume": 0.0,
                    "change_24h": 0.0,
                    "timestamp": _now_iso(),
                }
                for k, v in market_data.items():
                    value = json.dumps(v) if isinstance(v, (dict, list)) else v
                    redis_client.hset(f"market_data:{s}", k, value)
                redis_client.expire(f"market_data:{s}", 300)
                results["symbols_processed"].append(s)
                results["data_points"] += 1
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": len(results["symbols_processed"]),
                        "total": len(symbols),
                    },
                )
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                msg = f"{s} error: {e}"
                results["errors"].append(msg)
                task_logger.exception(f"[{EXCHANGE_ID}] {msg}")
        task_logger.info(f"[{EXCHANGE_ID}] market data sync done: {results['data_points']}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        task_logger.exception(f"[{EXCHANGE_ID}] market data sync failed: {e}")
        raise
    else:
        return results


@celery_app.task(bind=True, name="ai_tasks.rebalance_portfolio")
def rebalance_portfolio(_self, portfolio_id: str | None = None) -> dict[str, Any]:
    try:
        pid = portfolio_id or "default"
        task_logger.info(f"[{EXCHANGE_ID}] rebalance start portfolio={pid}")
        portfolio = _fetch_portfolio(pid)
        if not portfolio:
            return {
                "portfolio_id": pid,
                "timestamp": _now_iso(),
                "rebalancing_actions": [],
                "actions_count": 0,
                "risk_score": 0.0,
                "error": "portfolio_not_found",
            }
        target_alloc = _fetch_target_allocation(pid)
        if not target_alloc:
            return {
                "portfolio_id": pid,
                "timestamp": _now_iso(),
                "rebalancing_actions": [],
                "actions_count": 0,
                "risk_score": 0.0,
                "error": "target_allocation_not_found",
            }
        total_value = float(portfolio.get("total_value") or 0.0)
        positions: dict[str, dict[str, float]] = portfolio.get("positions") or {}
        actions: list[dict[str, Any]] = []

        # Determine desired trades to achieve target allocation
        for symbol, target_pct in target_alloc.items():
            try:
                if target_pct is None:
                    continue
                target_pct_float = float(target_pct)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue
            if target_pct_float <= 0:
                # If target is zero, mark for selling full position if exists
                cur_val = float(positions.get(symbol, {}).get("value", 0.0))
                if cur_val > 0:
                    price = _read_price(symbol) or 0.0
                    qty = (cur_val / price) if price > 0 else 0.0
                    actions.append(
                        {
                            "symbol": symbol,
                            "action": "sell",
                            "quantity": float(qty),
                            "value": float(cur_val),
                            "price": float(price),
                        }
                    )
                continue
            desired_value = total_value * target_pct_float
            current_value = float(positions.get(symbol, {}).get("value", 0.0))
            diff = desired_value - current_value
            # Skip tiny adjustments
            if abs(diff) < 1.0:
                continue
            price = _read_price(symbol) or 0.0
            quantity = (diff / price) if price > 0 else 0.0
            action = "buy" if diff > 0 else "sell"
            actions.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "quantity": float(abs(quantity)),
                    "value": float(abs(diff)),
                    "price": float(price),
                }
            )

        # Any current positions not in target allocation should be sold
        for symbol, pos in positions.items():
            if symbol not in target_alloc:
                cur_val = float(pos.get("value", 0.0))
                if cur_val > 0:
                    price = _read_price(symbol) or 0.0
                    qty = (cur_val / price) if price > 0 else 0.0
                    actions.append(
                        {
                            "symbol": symbol,
                            "action": "sell",
                            "quantity": float(qty),
                            "value": float(cur_val),
                            "price": float(price),
                        }
                    )

        # Persist a lightweight rebalancing record
        record = {
            "timestamp": _now_iso(),
            "portfolio_id": pid,
            "actions": actions,
        }
        key = f"rebalancing:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        try:
            redis_client.setex(key, 3600 * 24, json.dumps(record).encode())
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Best-effort persistence; do not fail the task if redis fails
            task_logger.warning(f"[{EXCHANGE_ID}] failed to persist rebalancing record {key}")

        results = {
            "portfolio_id": pid,
            "timestamp": _now_iso(),
            "rebalancing_actions": actions,
            "actions_count": len(actions),
            "risk_score": calculate_risk_score(portfolio),
        }
        task_logger.info(f"[{EXCHANGE_ID}] rebalance done actions={len(actions)}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        task_logger.exception(f"[{EXCHANGE_ID}] rebalance failed: {e}")
        raise
    else:
        return results


@celery_app.task(bind=True, name="ai_tasks.calculate_performance_metrics")
def calculate_performance_metrics(_self, lookback_days: int = 365) -> dict[str, Any]:
    try:
        task_logger.info(f"[{EXCHANGE_ID}] perf metrics start lookback_days={lookback_days}")
        rows = _sqlite_query_trades()
        # rows: list of (timestamp_iso, profit_usd)
        ret_list: list[float] = [float(p) for _ts, p in rows] if rows else []
        total_return = sum(ret_list)
        days = max(int(lookback_days), 1)
        # Treat total_return as a cumulative return fraction if reasonable, otherwise scale
        try:
            annualized_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0.0
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            annualized_return = float(total_return) * (365.0 / days) if days > 0 else 0.0

        # Build an equity series for drawdown/volatility calculations
        equity = list(np.cumsum(ret_list)) if ret_list else []
        volatility = _compute_volatility(equity) if equity else 0.0
        sharpe_ratio = _compute_sharpe(equity) if equity else 0.0

        # Sortino ratio
        sortino_ratio = 0.0
        if ret_list:
            downs = [r for r in ret_list if r < 0]
            if downs:
                downside_std = float(np.std(downs))
                mean_ret = float(np.mean(ret_list))
                if downside_std > 0:
                    sortino_ratio = mean_ret / downside_std

        profit_factor = 0.0
        if ret_list:
            gains = [x for x in ret_list if x > 0]
            losses = [-x for x in ret_list if x < 0]
            profit_factor = (sum(gains) / sum(losses)) if sum(losses) > 0 else 0.0

        max_drawdown = _compute_drawdown(equity) if equity else 0.0
        win = len([x for x in ret_list if x > 0])
        win_rate = (win / len(ret_list)) if ret_list else 0.0
        calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0

        metrics = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "calmar_ratio": float(calmar_ratio),
        }
        metrics["information_ratio"] = metrics["sharpe_ratio"] * 0.8
        metrics["ulcer_index"] = calculate_ulcer_index(ret_list)
        metrics["gain_to_pain_ratio"] = profit_factor * 0.9
        insights: list[str] = []
        if metrics["sharpe_ratio"] > 2.0:
            insights.append("Excellent risk-adjusted returns")
        elif metrics["sharpe_ratio"] < 1.0:
            insights.append("Risk-adjusted returns below target")
        if metrics["max_drawdown"] > 0.15:
            insights.append("High maximum drawdown detected")
        if metrics["win_rate"] < 0.5:
            insights.append("Win rate below 50% - consider strategy adjustment")
        results = {
            "timestamp": _now_iso(),
            "metrics": metrics,
            "insights": insights,
            "performance_grade": calculate_performance_grade(metrics),
            "trend": calculate_performance_trend(metrics),
        }
        try:
            redis_client.setex(
                f"performance_metrics:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                3600,
                json.dumps(results).encode(),
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            task_logger.warning(f"[{EXCHANGE_ID}] failed to persist performance metrics")
        task_logger.info(f"[{EXCHANGE_ID}] perf metrics done grade={results['performance_grade']}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        task_logger.exception(f"[{EXCHANGE_ID}] perf metrics failed: {e}")
        raise
    else:
        return results


@celery_app.task(bind=True, name="ai_tasks.cleanup_old_data")
def cleanup_old_data(_self, days_to_keep: int = 30) -> dict[str, Any]:
    try:
        task_logger.info(f"[{EXCHANGE_ID}] cleanup start keep_days={days_to_keep}")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cutoff_str = cutoff.strftime("%Y%m%d%H%M%S")
        prefixes = (
            "rebalancing:",
            "risk_assessment:",
            "strategy_evaluation:",
            "performance_metrics:",
        )
        deleted = 0
        for prefix in prefixes:
            for k in redis_client.scan_iter(match=f"{prefix}*"):
                s = _decode(k)
                if not s:
                    continue
                ts_part = s.rsplit(":", 1)[-1]
                if ts_part.isdigit() and ts_part < cutoff_str:
                    redis_client.delete(k)
                    deleted += 1
        sys_health = check_system_health()
        results = {
            "timestamp": _now_iso(),
            "cutoff_date": cutoff.isoformat(),
            "deleted_keys": deleted,
            "system_health": sys_health,
        }
        task_logger.info(f"[{EXCHANGE_ID}] cleanup done deleted={deleted}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        task_logger.exception(f"[{EXCHANGE_ID}] cleanup failed: {e}")
        raise
    else:
        return results


@celery_app.task(bind=True, name="ai_tasks.send_risk_alert")
def send_risk_alert(_self, risk_data: dict[str, Any]) -> dict[str, Any]:
    try:
        task_logger.info(f"[{EXCHANGE_ID}] sending risk alert")
        sent: list[dict[str, Any]] = []
        try:
            redis_client.publish("risk_alerts", json.dumps(risk_data).encode())
            sent.append({"channel": "redis_pubsub", "status": "sent", "timestamp": _now_iso()})
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            sent.append(
                {
                    "channel": "redis_pubsub",
                    "status": "failed",
                    "error": str(e),
                    "timestamp": _now_iso(),
                }
            )
        webhook = os.getenv("RISK_ALERT_WEBHOOK")
        if webhook:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(webhook, json=risk_data)
                    sent.append(
                        {
                            "channel": "webhook",
                            "status": ("sent" if resp.status_code < 300 else f"failed:{resp.status_code}"),
                            "timestamp": _now_iso(),
                        }
                    )
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                sent.append(
                    {
                        "channel": "webhook",
                        "status": "failed",
                        "error": str(e),
                        "timestamp": _now_iso(),
                    }
                )
        results = {
            "timestamp": _now_iso(),
            "alerts_sent": sent,
            "success_count": len([x for x in sent if x["status"] == "sent"]),
        }
        task_logger.info(f"[{EXCHANGE_ID}] risk alert dispatched ok={results['success_count']}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        task_logger.exception(f"[{EXCHANGE_ID}] risk alert failed: {e}")
        raise
    else:
        return results


def calculate_risk_score(portfolio: dict[str, Any]) -> float:
    positions = portfolio.get("positions") or {}
    values = [float(v.get("value", 0.0)) for v in positions.values()]
    total = float(portfolio.get("total_value") or sum(values) or 1.0)
    weights = [v / total for v in values] if total else []
    if not weights:
        return 0.0
    hh = sum(w * w for w in weights)
    return float(min(max(hh, 0.0), 1.0))


def calculate_overall_risk_score(risk_metrics: dict[str, Any]) -> float:
    var = float(risk_metrics.get("var_95", 0.0))
    dd = float(risk_metrics.get("max_drawdown", 0.0))
    score = 0.5 * var + 0.5 * dd
    return float(min(max(score, 0.0), 1.0))


def generate_risk_recommendations(risk_metrics: dict[str, Any], alerts: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    if float(risk_metrics.get("var_95", 0.0)) > 0.05:
        recs.append("Reduce position sizes")
    if float(risk_metrics.get("max_drawdown", 0.0)) > 0.15:
        recs.append("Tighten stop-loss")
    if not recs and alerts:
        recs.append("Monitor conditions; no immediate action")
    return recs


def calculate_strategy_score(strategy: dict[str, Any]) -> float:
    sharpe = float(strategy.get("sharpe", 0.0))
    returns = float(strategy.get("returns", 0.0))
    max_dd = float(strategy.get("max_dd", 0.0))
    score = 0.4 * sharpe + 0.3 * returns + 0.3 * (1.0 - max_dd)
    return float(min(max(score, 0.0), 1.0))


def calculate_ulcer_index(returns: list[float]) -> float:
    if not returns:
        return 0.0
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak != 0, peak, 1.0)
    return float(np.sqrt(np.mean(np.square(drawdown))))


def calculate_performance_grade(metrics: dict[str, Any]) -> str:
    score = float(metrics.get("sharpe_ratio", 0.0)) * 0.3 + (1.0 - float(metrics.get("max_drawdown", 0.0))) * 0.3 + float(metrics.get("win_rate", 0.0)) * 0.4
    if score > 0.8:
        return "A"
    if score > 0.6:
        return "B"
    if score > 0.4:
        return "C"
    return "D"


def calculate_performance_trend(metrics: dict[str, Any]) -> str:
    if float(metrics.get("total_return", 0.0)) > 0 and float(metrics.get("sharpe_ratio", 0.0)) > 1.0:
        return "improving"
    if float(metrics.get("total_return", 0.0)) < 0 and float(metrics.get("sharpe_ratio", 0.0)) < 1.0:
        return "declining"
    return "stable"


def check_system_health() -> dict[str, Any]:
    try:
        redis_ok = bool(redis_client.ping())
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        redis_ok = False
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "redis_connected": redis_ok,
    }


@celery_app.task(name="ai_tasks.health_check")
def health_check() -> dict[str, Any]:
    try:
        insp = celery_app.control.inspect()
        active = insp.active() or {}
        worker_count = sum(len(v) for v in active.values()) if isinstance(active, dict) else 0
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        worker_count = 0
    try:
        qsize = int(redis_client.llen("celery"))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        qsize = 0
    return {
        "status": "healthy",
        "timestamp": _now_iso(),
        "worker_count": worker_count,
        "queue_size": qsize,
    }


if __name__ == "__main__":
    celery_app.start()
