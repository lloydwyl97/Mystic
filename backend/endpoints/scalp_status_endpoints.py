"""Read-only scalp status API — isolated from Mystic DAY."""

from __future__ import annotations

from fastapi import APIRouter

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.status_snapshot import build_scalp_status
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies

router = APIRouter(prefix="/api/scalp", tags=["scalp"])


@router.get("/status")
def scalp_status() -> dict:
    """Read-only scalp readiness — legacy preflight + multi-strategy router."""
    return build_scalp_status(warm_rounds=0)


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
