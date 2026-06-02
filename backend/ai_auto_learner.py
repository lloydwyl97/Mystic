from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulation_logger import SimulationLogger

logger = logging.getLogger(__name__)

MODEL_STATE_FILE = os.getenv("MODEL_STATE_PATH") or "ai_model_state.json"

DEFAULT_STATE: dict[str, Any] = {
    "version": 1,
    "confidence_threshold": 0.75,
    "avg_profit_threshold": 0.5,
    "adjustment_count": 0,
    "last_update": None,
}


class AIAutoLearner:
    def __init__(self) -> None:
        self.logger = SimulationLogger()
        self.state: dict[str, Any] = self._load_state()

    def _ensure_parent_dir(self, path: str) -> None:
        parent = str(Path(path).resolve().parent)
        if parent and not Path(parent).exists():
            Path(parent).mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict[str, Any]:
        try:
            if MODEL_STATE_FILE and Path(MODEL_STATE_FILE).exists():
                model_state_path = Path(MODEL_STATE_FILE)
                with model_state_path.open(encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Merge with defaults to tolerate missing keys
                        merged = {**DEFAULT_STATE, **data}
                        return merged.copy()
            return DEFAULT_STATE.copy()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.warning("Failed to load model state (%s). Using defaults.", e)
            return DEFAULT_STATE.copy()

    def _atomic_write_json(self, path: str, obj: dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        # Ensure parent directory exists for the temporary file as well
        self._ensure_parent_dir(tmp_path)
        tmp_path_obj = Path(tmp_path)
        with tmp_path_obj.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_path).replace(Path(path))

    def _save_state(self) -> None:
        if not MODEL_STATE_FILE:
            msg = "MODEL_STATE_FILE is not set"
            raise ValueError(msg)

        try:
            self._ensure_parent_dir(MODEL_STATE_FILE)
            self._atomic_write_json(MODEL_STATE_FILE, self.state)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to save model state: %s", e)

    def evaluate_and_adapt(self) -> None:
        try:
            summary = self.logger.get_summary() or {}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Failed to fetch simulation summary: %s", e)
            return

        if not isinstance(summary, dict):
            summary = {}

        avg_profit = 0.0
        try:
            avg_profit = float(summary.get("avg_profit", 0.0))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            avg_profit = 0.0

        improved = False
        th = float(self.state.get("confidence_threshold", DEFAULT_STATE["confidence_threshold"]))
        profit_threshold = float(self.state.get("avg_profit_threshold", DEFAULT_STATE["avg_profit_threshold"]))

        if avg_profit > profit_threshold:
            th = min(0.95, th + 0.01)
            improved = True
        elif avg_profit < 0:
            th = max(0.50, th - 0.01)
            improved = True

        if improved:
            self.state["confidence_threshold"] = th
            self.state["adjustment_count"] = int(self.state.get("adjustment_count", 0)) + 1
            self.state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save_state()
            logger.info(
                "AI strategy adjusted | threshold=%.4f | avg_profit=%.4f | adjustments=%d",
                self.state["confidence_threshold"],
                avg_profit,
                self.state["adjustment_count"],
            )
        else:
            logger.debug(
                "No strategy change. Performance stable | avg_profit=%.4f | threshold=%.4f",
                avg_profit,
                th,
            )

    def get_current_threshold(self) -> float:
        try:
            return float(self.state.get("confidence_threshold", DEFAULT_STATE["confidence_threshold"]))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return float(DEFAULT_STATE["confidence_threshold"])
