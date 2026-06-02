import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def replicate_to(path: str = "./replica", dna: str = "core") -> str | None:
    dst_base = Path(path)
    src = Path.cwd().resolve()
    dst_base.mkdir(parents=True, exist_ok=True)

    # Resolve the base so we can create an absolute destination without resolving a non-existent child
    try:
        dst_base_resolved = dst_base.resolve()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # Fallback to the original dst_base if resolve() fails for some reason
        dst_base_resolved = dst_base

    dst = dst_base_resolved / dna

    ignore_names = {".git", "__pycache__", ".venv", "env", "node_modules"}
    try:
        rel = dst.relative_to(src)
        if rel.parts:
            ignore_names.add(rel.parts[0])
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # ValueError if not relative, or other exceptions if paths are incompatible; ignore in either case
        pass

    def _ignore(_dirname: str, names: list[str]) -> set[str]:
        return {n for n in names if n in ignore_names}

    try:
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore)
        logger.info(f"[REPLICATOR] {dna} replicated to {dst}")
        return str(dst)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[REPLICATOR] Replication failed: {e}")
        backup = dst_base / f"{dna}_backup.txt"
        try:
            with backup.open("w", encoding="utf-8") as f:
                f.write(f"System backup created at {backup}")
            logger.info("[REPLICATOR] Created backup file instead")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as ee:
            logger.exception(f"[REPLICATOR] Backup failed: {ee}")
        return None
