"""Read-only scalp status API — isolated from Mystic DAY."""

from __future__ import annotations

from fastapi import APIRouter

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.pnl_summary import build_scalp_pnl_summary
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
def scalp_status(*, warm: int = 0) -> dict:
    """Read-only scalp engine status — isolated from DAY top-four.

    NEAR_PASS / SPREAD_TOO_WIDE / MOMENTUM_* blockers are scalp diagnostics only.
    They are not DAY faults and must not appear on the DAY scoreboard.

    Query param warm=6 runs a slow momentum warm (diagnostics only). Default warm=0
    uses a cached snapshot refreshed every SCALP_STATUS_CACHE_TTL_SEC (default 20s).
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
    from backend.services.binance_scalp.scalp_status_cache import get_cached_scalp_status

    warm_rounds = max(0, min(int(warm), 12))
    return {
        "runner_active": True,
        "engine": "scalp",
        "pnl_summary": pnl,
        **get_cached_scalp_status(warm_rounds=warm_rounds),
    }


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
