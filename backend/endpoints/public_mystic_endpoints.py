"""Read-only public-safe Mystic summary endpoints for MarketLens bridge."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/public", tags=["public"])


def _pgrep(pattern: str) -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def _redis_ping() -> bool:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        client = get_shared_redis_sync()
        if client is None:
            return False
        return client.ping()
    except Exception:
        return False


def _system_state() -> str:
    core = [
        _pgrep("uvicorn"),
        _pgrep("start_portfolio_engine_integration.py"),
        _pgrep("backend.services.binance_scalp.runner"),
    ]
    if all(core):
        return "online"
    if any(core):
        return "degraded"
    return "offline"


@router.get("/mystic-summary")
async def mystic_summary() -> dict[str, Any]:
    """Public-safe Mystic operator summary — no secrets, no account balances."""
    state = _system_state()
    day_running = _pgrep("start_portfolio_engine_integration.py")
    scalp_running = _pgrep("backend.services.binance_scalp.runner")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_state": state,
        "engines": {
            "day": "running" if day_running else "stopped",
            "scalp": "running" if scalp_running else "stopped",
        },
        "redis": "ok" if _redis_ping() else "unavailable",
        "note": "Read-only public feed — no operator controls or private balances.",
    }


@router.get("/mystic-regime")
async def mystic_regime() -> dict[str, Any]:
    """Market regime summary (public-safe)."""
    regime_label = "unknown"
    try:
        from backend.services.fear_greed import get_fear_greed_index

        fg = get_fear_greed_index()
        if fg and fg.get("value") is not None:
            regime_label = fg.get("classification") or fg.get("value_classification") or "neutral"
    except Exception:
        pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fear_greed_label": regime_label,
        "note": "Crypto Fear & Greed index — not ML price-trend regime.",
    }


@router.get("/mystic-decisions")
async def mystic_decisions() -> dict[str, Any]:
    """Latest AI decision explanations (public-safe, no account data)."""
    day_decision = None
    scalp_decision = None
    try:
        from backend.services.day_position_health import load_health

        health = load_health()
        if health:
            day_decision = {
                "idle_reason": health.get("capital_idle_reason"),
                "open_positions": health.get("open_positions_count"),
            }
    except Exception:
        pass
    try:
        from backend.services.binance_scalp.scalp_status_cache import get_cached_scalp_status

        if _pgrep("backend.services.binance_scalp.runner"):
            st = get_cached_scalp_status(warm_rounds=0)
            scalp_decision = {
                "overall_decision": st.get("overall_decision"),
                "top_blocker": st.get("top_blocker"),
            }
    except Exception:
        pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "day": day_decision,
        "scalp": scalp_decision,
    }


@router.get("/mystic-learning-summary")
async def mystic_learning_summary() -> dict[str, Any]:
    """Learning ingestion summary (counts only, no trade details)."""
    day_closed = 0
    scalp_closed = 0
    try:
        import sqlite3
        from pathlib import Path

        db = Path(__file__).resolve().parents[2] / "mystic_trading.db"
        if db.exists():
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT COUNT(*) FROM trade_learning_outcomes").fetchone()
                day_closed = int(row[0]) if row else 0
    except Exception:
        pass
    try:
        from backend.services.binance_scalp.config import get_scalp_config

        path = get_scalp_config().database_path
        import sqlite3
        from pathlib import Path

        sp = Path(path)
        if sp.exists():
            with sqlite3.connect(f"file:{sp}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM scalp_paper_trades WHERE side='SELL'"
                ).fetchone()
                scalp_closed = int(row[0]) if row else 0
    except Exception:
        pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "day_learning_rows": day_closed,
        "scalp_closed_roundtrips": scalp_closed,
        "scalp_learning_ready": scalp_closed > 0,
    }


@router.get("/mystic-marketlens-feed")
async def mystic_marketlens_feed() -> dict[str, Any]:
    """Combined read-only feed for MarketLens integration."""
    summary = await mystic_summary()
    regime = await mystic_regime()
    decisions = await mystic_decisions()
    learning = await mystic_learning_summary()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "regime": regime,
        "decisions": decisions,
        "learning": learning,
    }
