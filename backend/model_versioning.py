import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

STATE_FILE = "ai_model_state.json"
VERSIONS_DIR = "model_versions"
INVALID_VERSION_CHARS = set('/\\:*?"<>|')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_versioning")


def _safe_read_json(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        path_obj = Path(path)
        with path_obj.open(encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, json.JSONDecodeError) as e:
        return None, f"read_json_failed: {e}"


def _safe_write_json(path: str, data: dict[str, Any]) -> str | None:
    try:
        directory = Path(path).parent
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name
        Path(temp_name).replace(Path(path))
    except OSError as e:
        return f"write_json_failed: {e}"
    else:
        return None


def _validate_state(state: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(state, dict):
        return False, "invalid_state_type"
    if "confidence_threshold" not in state or "adjustment_count" not in state:
        return False, "missing_required_keys"
    try:
        float(state["confidence_threshold"])
        int(state["adjustment_count"])
    except (TypeError, ValueError):
        return False, "invalid_key_types"
    return True, None


def rollback_model() -> bool:
    if not Path(STATE_FILE).exists():
        logger.info("Rollback skipped: state file not found")
        return False

    state, err = _safe_read_json(STATE_FILE)
    if err:
        logger.warning("Rollback aborted: %s", err)
        return False

    ok, why = _validate_state(state)  # type: ignore[arg-type]
    if not ok:
        logger.warning("Rollback aborted: invalid state (%s)", why)
        return False

    adj_count = int(state["adjustment_count"])  # type: ignore[index]
    if adj_count <= 0:
        logger.info("Rollback skipped: no previous versions to revert")
        return False

    try:
        new_threshold = round(float(state["confidence_threshold"]) - 0.01, 2)  # type: ignore[index]
        state["confidence_threshold"] = new_threshold
        state["adjustment_count"] = adj_count - 1
    except (TypeError, ValueError) as e:
        logger.warning("Rollback aborted: %s", e)
        return False

    err = _safe_write_json(STATE_FILE, state)  # type: ignore[arg-type]
    if err:
        logger.error("Rollback write failed: %s", err)
        return False

    logger.info(
        "Rollback applied: threshold=%.2f, adjustment_count=%d",
        state["confidence_threshold"],
        state["adjustment_count"],
    )  # type: ignore[index]
    return True


def save_model_version(version_name: str) -> bool:
    if not version_name or any(ch in INVALID_VERSION_CHARS for ch in version_name):
        logger.warning("Versioning aborted: invalid version name")
        return False

    if not Path(STATE_FILE).exists():
        logger.info("Versioning skipped: state file not found")
        return False

    state, err = _safe_read_json(STATE_FILE)
    if err:
        logger.warning("Versioning aborted: %s", err)
        return False

    ok, why = _validate_state(state)  # type: ignore[arg-type]
    if not ok:
        logger.warning("Versioning aborted: invalid state (%s)", why)
        return False

    dest_dir = Path(VERSIONS_DIR) / version_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = str(dest_dir / Path(STATE_FILE).name)

    try:
        shutil.copy2(STATE_FILE, dest_path)
    except OSError as e:
        logger.exception("Versioning copy failed: %s", e)
        return False

    logger.info("Model version saved: %s", version_name)
    return True


def load_model_version(version_name: str) -> bool:
    if not version_name or any(ch in INVALID_VERSION_CHARS for ch in version_name):
        logger.warning("Load aborted: invalid version name")
        return False

    version_path = str(Path(VERSIONS_DIR) / version_name / Path(STATE_FILE).name)
    if not Path(version_path).exists():
        logger.info("Load skipped: version not found (%s)", version_name)
        return False

    state, err = _safe_read_json(version_path)
    if err:
        logger.warning("Load aborted: %s", err)
        return False

    ok, why = _validate_state(state)  # type: ignore[arg-type]
    if not ok:
        logger.warning("Load aborted: invalid version state (%s)", why)
        return False

    try:
        directory = Path(STATE_FILE).parent
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(version_path, STATE_FILE)
    except OSError as e:
        logger.exception("Load copy failed: %s", e)
        return False

    logger.info("Model version loaded: %s", version_name)
    return True
