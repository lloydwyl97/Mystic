"""
Strategy Versioning System for AI Trading
Manages strategy versions, saves/loads configurations, and tracks evolution.
Built for Windows 11 Home + PowerShell.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STRATEGY_VERSIONS_DIR = "strategy_versions"
Path(STRATEGY_VERSIONS_DIR).mkdir(parents=True, exist_ok=True)


def generate_version_id(config: dict[str, Any], strategy_type: str) -> str:
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{strategy_type}_v{timestamp}_{config_hash}"


def save_strategy_version(
    config: dict[str, Any],
    strategy_type: str,
    performance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    try:
        version_id = generate_version_id(config, strategy_type)
        version_data = {
            "version_id": version_id,
            "strategy_type": strategy_type,
            "config": config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "performance": performance or {},
            "metadata": metadata or {},
        }
        filename = Path(STRATEGY_VERSIONS_DIR) / f"{version_id}.json"
        with filename.open("w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2)
        logger.info(f"Saved strategy version: {version_id}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error saving strategy version: {e}")
        return ""
    else:
        return version_id


def load_strategy_version(version_id: str) -> dict[str, Any] | None:
    try:
        filename = Path(STRATEGY_VERSIONS_DIR) / f"{version_id}.json"
        if not filename.exists():
            logger.warning(f"Strategy version not found: {version_id}")
            return None
        with filename.open(encoding="utf-8") as f:
            version_data = json.load(f)
        logger.info(f"Loaded strategy version: {version_id}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error loading strategy version {version_id}: {e}")
        return None
    else:
        return version_data


def list_strategy_versions(strategy_type: str | None = None) -> list[dict[str, Any]]:
    try:
        versions: list[dict[str, Any]] = []
        if not Path(STRATEGY_VERSIONS_DIR).exists():
            return versions
        for path in Path(STRATEGY_VERSIONS_DIR).iterdir():
            filename = path.name
            if filename.endswith(".json"):
                try:
                    filepath = Path(STRATEGY_VERSIONS_DIR) / filename
                    with filepath.open(encoding="utf-8") as f:
                        version_data = json.load(f)
                    if strategy_type and version_data.get("strategy_type") != strategy_type:
                        continue
                    summary = {
                        "version_id": version_data.get("version_id"),
                        "strategy_type": version_data.get("strategy_type"),
                        "created_at": version_data.get("created_at"),
                        "performance": version_data.get("performance", {}),
                        "config_summary": {k: v for k, v in version_data.get("config", {}).items() if isinstance(v, (int, float, str))},
                    }
                    versions.append(summary)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.warning(f"Error reading version file {filename}: {e}")
                    continue
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error listing strategy versions: {e}")
        return []
    else:
        return versions


def get_best_performing_version(strategy_type: str, metric: str = "total_profit") -> dict[str, Any] | None:
    try:
        versions = list_strategy_versions(strategy_type)
        if not versions:
            return None
        versions_with_performance = [v for v in versions if v.get("performance") and metric in v["performance"]]
        if not versions_with_performance:
            return None
        best_version = max(
            versions_with_performance,
            key=lambda x: x["performance"].get(metric, float("-inf")),
        )
        return load_strategy_version(best_version["version_id"])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting best performing version: {e}")
        return None


def delete_strategy_version(version_id: str) -> bool:
    try:
        filename = str(Path(STRATEGY_VERSIONS_DIR) / f"{version_id}.json")
        if not Path(filename).exists():
            logger.warning(f"Strategy version not found for deletion: {version_id}")
            return False
        Path(filename).unlink()
        logger.info(f"Deleted strategy version: {version_id}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error deleting strategy version {version_id}: {e}")
        return False
    else:
        return True


def compare_versions(version_id1: str, version_id2: str) -> dict[str, Any]:
    try:
        version1 = load_strategy_version(version_id1)
        version2 = load_strategy_version(version_id2)
        if not version1 or not version2:
            return {"error": "One or both versions not found"}
        comparison: dict[str, Any] = {
            "version1": {
                "version_id": version1.get("version_id"),
                "strategy_type": version1.get("strategy_type"),
                "created_at": version1.get("created_at"),
                "performance": version1.get("performance", {}),
                "config": version1.get("config", {}),
            },
            "version2": {
                "version_id": version2.get("version_id"),
                "strategy_type": version2.get("strategy_type"),
                "created_at": version2.get("created_at"),
                "performance": version2.get("performance", {}),
                "config": version2.get("config", {}),
            },
            "differences": {
                "config_differences": {},
                "performance_differences": {},
            },
        }
        config1 = version1.get("config", {})
        config2 = version2.get("config", {})
        all_keys = set(config1.keys()) | set(config2.keys())
        for key in all_keys:
            if config1.get(key) != config2.get(key):
                comparison["differences"]["config_differences"][key] = {
                    "version1": config1.get(key),
                    "version2": config2.get(key),
                }
        perf1 = version1.get("performance", {})
        perf2 = version2.get("performance", {})
        all_perf_keys = set(perf1.keys()) | set(perf2.keys())
        for key in all_perf_keys:
            if perf1.get(key) != perf2.get(key):
                comparison["differences"]["performance_differences"][key] = {
                    "version1": perf1.get(key),
                    "version2": perf2.get(key),
                }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error comparing versions: {e}")
        return {"error": str(e)}
    else:
        return comparison


def export_strategy_version(version_id: str, export_path: str) -> bool:
    try:
        version_data = load_strategy_version(version_id)
        if not version_data:
            return False
        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with export_path_obj.open("w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2)
        logger.info(f"Exported strategy version {version_id} to {export_path}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error exporting strategy version: {e}")
        return False
    else:
        return True


def import_strategy_version(import_path: str) -> str:
    try:
        import_path_obj = Path(import_path)
        with import_path_obj.open(encoding="utf-8") as f:
            version_data = json.load(f)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error loading strategy version from {import_path}: {e}")
        return ""

    required_fields = ["strategy_type", "config"]
    for field in required_fields:
        if field not in version_data:
            msg = f"Missing required field: {field}"
            raise ValueError(msg)

    if "version_id" not in version_data:
        version_data["version_id"] = generate_version_id(version_data["config"], version_data["strategy_type"])
    try:
        return save_strategy_version(
            version_data["config"],
            version_data["strategy_type"],
            version_data.get("performance"),
            version_data.get("metadata"),
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error importing strategy version: {e}")
        return ""


def save_optimized_config(config: dict[str, Any], strategy_type: str, performance: dict[str, Any]) -> str:
    metadata = {
        "source": "hyperparameter_optimization",
        "optimization_method": "auto_tuned",
    }
    return save_strategy_version(config, strategy_type, performance, metadata)


def load_latest_version(strategy_type: str) -> dict[str, Any] | None:
    versions = list_strategy_versions(strategy_type)
    if not versions:
        return None
    latest_version_id = versions[0]["version_id"]
    return load_strategy_version(latest_version_id)


if __name__ == "__main__":
    logger.info("Testing Strategy Versioning System...")
    test_config = {
        "rsi_period": 14,
        "ema_fast": 12,
        "ema_slow": 26,
        "stop_loss_pct": 0.02,
    }
    version_id = save_strategy_version(
        test_config,
        "rsi_ema_breakout",
        {"total_profit": 150.0, "win_rate": 0.65},
    )
    logger.info(f"Saved test version: {version_id}")
    loaded_version = load_strategy_version(version_id)
    logger.info(f"Loaded version: {loaded_version is not None}")
    versions = list_strategy_versions("rsi_ema_breakout")
    logger.info(f"Found {len(versions)} versions")
    logger.info("Strategy versioning system test completed!")
