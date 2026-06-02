import gzip
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path("./data/agent_memory")
DEFAULT_SUFFIX = ".pgz"


def _memory_path(agent_id: str, directory: Path = DEFAULT_DIR) -> Path:
    safe_id = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
    return directory / f"{safe_id}_memory{DEFAULT_SUFFIX}"


def save_agent_memory(agent_id: str, memory: Any, directory: Path = DEFAULT_DIR) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = _memory_path(agent_id, directory)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=9) as f:
            pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
        Path(tmp).replace(target)
        logger.info("[MEMORY] Saved agent_id=%s to %s", agent_id, str(target))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("save_agent_memory failed for agent_id=%s: %s", agent_id, e)
        try:
            if "tmp" in locals() and tmp.exists():
                tmp.unlink()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        return False
    else:
        return True


def load_agent_memory(agent_id: str, directory: Path = DEFAULT_DIR) -> Any | None:
    path = _memory_path(agent_id, directory)
    if not path.exists():
        logger.info("[MEMORY] No memory file found for agent_id=%s", agent_id)
        return None
    try:
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("[MEMORY] Loaded agent_id=%s from %s", agent_id, str(path))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("load_agent_memory failed for agent_id=%s: %s", agent_id, e)
        return None
    else:
        return data


def delete_agent_memory(agent_id: str, directory: Path = DEFAULT_DIR) -> bool:
    path = _memory_path(agent_id, directory)
    try:
        if path.exists():
            path.unlink()
            logger.info("[MEMORY] Deleted agent_id=%s file %s", agent_id, str(path))
            result = True
        else:
            logger.info("[MEMORY] No file to delete for agent_id=%s", agent_id)
            result = False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("delete_agent_memory failed for agent_id=%s: %s", agent_id, e)
        return False
    else:
        return result


def list_agent_memory_ids(directory: Path = DEFAULT_DIR) -> list[str]:
    try:
        if not directory.exists():
            return []
        ids: list[str] = []
        for p in directory.glob(f"*{DEFAULT_SUFFIX}"):
            name = p.name
            if name.endswith(f"_memory{DEFAULT_SUFFIX}"):
                agent_id = name[: -len(f"_memory{DEFAULT_SUFFIX}")]
                ids.append(agent_id)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception("list_agent_memory_ids failed: %s", e)
        return []
    else:
        return ids
