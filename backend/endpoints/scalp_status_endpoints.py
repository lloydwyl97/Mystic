"""Read-only scalp status API — isolated from Mystic DAY."""

from __future__ import annotations

from fastapi import APIRouter

from backend.services.binance_scalp.pnl_summary import build_scalp_pnl_summary
from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.status_snapshot import build_scalp_status
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies

router = APIRouter(prefix="/api/scalp", tags=["scalp"])


def _scalp_runner_active() -> bool:
    """True iff the scalp paper runner process is actually running."""
    import subprocess

    try:
        res = subprocess.run(
            ["pgrep", "-f", "backend.services.binance_scalp.runner"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


@router.get("/status")
def scalp_status() -> dict:
    """Read-only scalp engine status — isolated from DAY top-four.

    NEAR_PASS / SPREAD_TOO_WIDE / MOMENTUM_* blockers are scalp diagnostics only.
    They are not DAY faults and must not appear on the DAY scoreboard.
    """
    active = _scalp_runner_active()
    pnl = build_scalp_pnl_summary()
    if not active:
        return {
            "runner_active": False,
            "engine": "scalp",
            "scalp_engaged": False,
            "pnl_summary": pnl,
            "note": "Scalp paper runner is not running. Start with './start_mystic.sh core' or 'scalp'.",
        }
    # Use a short warm so the diagnostic view (the table the operator sees) reflects
    # realistic momentum_confirmed / would_enter after the runner has been up.
    # The live paper engine does its own full 60s warm on start for actual decisions.
    # 6 rounds (~30s) is a practical compromise for /api/scalp/status freshness.
    return {"runner_active": True, "engine": "scalp", "pnl_summary": pnl, **build_scalp_status(warm_rounds=6)}


@router.get("/strategies")
def scalp_strategies() -> dict:
    """Enabled/disabled strategy inventory (no market fetch)."""
    config = get_scalp_config()
    return {
        "all": list(STRATEGY_NAMES),
        "enabled": [s.name for s in enabled_strategies(config)],
        "disabled": sorted(config.disabled_strategies),
        "disabled_env": "SCALP_DISABLED_STRATEGIES",
    }
