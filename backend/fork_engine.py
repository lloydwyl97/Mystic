from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def fork_agent(strategy_profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(strategy_profile, dict):
        msg = "strategy_profile must be a dict"
        raise TypeError(msg)

    base = deepcopy(strategy_profile)
    parent_id = str(base.get("id") or "agent")
    mutation_rate = float(base.get("mutation_rate", 0.10))
    risk_factor = float(base.get("risk_factor", 1.00))

    child = deepcopy(base)
    child["id"] = f"{parent_id}_FORK"
    child["parent_id"] = parent_id
    child["mutation_rate"] = max(0.0, round(mutation_rate * 1.2, 6))
    child["risk_factor"] = max(0.0, round(risk_factor * 0.95, 6))
    child["forked_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"[FORKED] {child['id']} from {parent_id} | mutation_rate={child['mutation_rate']} risk_factor={child['risk_factor']}")
    return child
