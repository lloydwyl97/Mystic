"""One fail-open CLOCK-V2 v5 research cycle. Never touches trading.

Registers the partition contract, persists the planned v5 experiment (once, with
its real insertion timestamp), labels matured DEVELOPMENT groups at the frozen 3h
horizon, and snapshots readiness. Does not train, promote, or read the sealed 4H
lock. Every failure is swallowed so research can never disturb the trading loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_clock_v2_v5_cycle(db_path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "groups_scanned": 0,
        "labels_written": 0,
        "valid": 0,
        "immature": 0,
        "errors": 0,
        "planned_inserted": False,
        "readiness": None,
    }
    if not db_path:
        return out
    try:
        from backend.services.day_clock_v2_labels import run_v5_label_batch
        from backend.services.day_clock_v2_partition import register_partition_contract
        from backend.services.day_path_clock_v2_readiness import (
            evaluate_clock_v2_v5_readiness,
            record_planned_clock_v2_v5,
        )
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("DAY_CLOCK_V2_V5 import failed: %s", exc)
        out["errors"] += 1
        return out
    try:
        register_partition_contract(db_path)
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_V5 partition registration failed: %s", exc)
        out["errors"] += 1
    try:
        planned = record_planned_clock_v2_v5(db_path)
        out["planned_inserted"] = bool(planned.get("inserted"))
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_V5 experiment persistence failed: %s", exc)
        out["errors"] += 1
    try:
        summary = run_v5_label_batch(db_path)
        out.update(
            {
                "groups_scanned": summary.get("groups_scanned", 0),
                "labels_written": summary.get("labels_written", 0),
                "valid": summary.get("valid", 0),
                "immature": summary.get("immature", 0),
            }
        )
        out["errors"] += int(summary.get("errors") or 0)
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_V5 label batch failed: %s", exc)
        out["errors"] += 1
    try:
        snap = evaluate_clock_v2_v5_readiness(db_path)
        out["readiness"] = snap.get("DATA_READINESS")
        if snap.get("train") or snap.get("promoted"):
            raise RuntimeError("clock-v2 v5 readiness must never set train or promoted")
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_V5 readiness failed: %s", exc)
        out["errors"] += 1
    return out


__all__ = ["run_clock_v2_v5_cycle"]
