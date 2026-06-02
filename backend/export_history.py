from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_self_rating import get_ai_health_report  # type: ignore[import-not-found]
from daily_summary import get_performance_metrics  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    Path(tmp_path).replace(path)
    path.chmod(0o644)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    Path(tmp_path).replace(path)
    path.chmod(0o644)


def _open_db(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def export_trade_history(
    output_file: str | os.PathLike[str] = "trade_history.json",
    db_path: str | os.PathLike[str] | None = None,
    table: str = "simulated_trades",
) -> str:
    if db_path is None:
        db_path = os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")
    conn = None
    try:
        conn = _open_db(db_path)
        cur = conn.execute(
            f"""
            SELECT id, symbol, side, amount, price, timestamp, strategy, portfolio_id, status
            FROM {table}
            ORDER BY timestamp DESC
            """,
        )
        rows = [dict(row) for row in cur.fetchall()]
        payload = json.dumps(rows, indent=2, ensure_ascii=False)
        out = Path(output_file).expanduser().resolve()
        _atomic_write_text(out, payload)
        logger.info(f"[Export] Trade history exported to {out}")
        return str(out)
    except sqlite3.Error as e:
        logger.exception(f"[Export] Database error: {e}")
        return ""
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[Export] Error exporting trade history: {e}")
        return ""
    finally:
        if conn:
            conn.close()


def export_performance_report(
    output_file: str | os.PathLike[str] = "performance_report.json",
) -> str:
    metrics: dict[str, Any]
    health: dict[str, Any]
    try:
        metrics = get_performance_metrics() or {}
        health = get_ai_health_report() or {}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.info(f"[Export] Metrics sources unavailable: {e}")
        metrics, health = {}, {}

    summary = {
        "total_trades": (metrics.get("summary", {})).get("total_trades", 0),
        "total_profit": (metrics.get("summary", {})).get("total_profit", 0.0),
        "avg_profit": (metrics.get("summary", {})).get("avg_profit", 0.0),
        "ai_score": (health.get("rating", {})).get("ai_score", 0.0),
        "ai_rating": (health.get("rating", {})).get("rating", "unknown"),
    }

    report = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_metrics": metrics,
        "ai_health": health,
        "summary": summary,
    }

    try:
        out = Path(output_file).expanduser().resolve()
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        _atomic_write_text(out, payload)
        logger.info(f"[Export] Performance report exported to {out}")
        return str(out)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[Export] Error exporting performance report: {e}")
        return ""


def export_csv_trades(
    output_file: str | os.PathLike[str] = "trades.csv",
    db_path: str | os.PathLike[str] | None = None,
    table: str = "simulated_trades",
) -> str:
    if db_path is None:
        db_path = os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")
    conn = None
    try:
        conn = _open_db(db_path)
        cur = conn.execute(
            f"""
            SELECT id, symbol, side, amount, price, timestamp, strategy, portfolio_id, status
            FROM {table}
            ORDER BY timestamp DESC
            """,
        )
        rows = cur.fetchall()
        out_path = Path(output_file).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            _atomic_write_bytes(out_path, b"")
            logger.info(f"[Export] No rows found. Created empty file at {out_path}")
            return str(out_path)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(out_path.parent)) as tmp:
            writer = csv.writer(tmp)
            headers = list(rows[0].keys())
            writer.writerow(headers)
            for r in rows:
                writer.writerow([r[h] for h in headers])
            tmp_path = Path(tmp.name)

        Path(tmp_path).replace(out_path)
        out_path.chmod(0o644)
        logger.info(f"[Export] CSV trades exported to {out_path}")
        return str(out_path)
    except sqlite3.Error as e:
        logger.exception(f"[Export] Database error: {e}")
        return ""
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[Export] Error exporting CSV trades: {e}")
        return ""
    finally:
        if conn:
            conn.close()
