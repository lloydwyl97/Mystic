import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def auto_mirror_system(destination=r".\fallback_mirror"):
    logger.info("[DEFENSE] Activating protocol mirror...")
    src = Path(__file__).resolve().parent
    dest = Path(destination).resolve()

    # Prevent copying into itself or into a subdirectory of the source.
    if dest == src:
        msg = "Destination cannot be the source or inside it"
        raise ValueError(msg)

    try:
        # Available on Python 3.9+
        if dest.is_relative_to(src):
            msg = "Destination cannot be the source or inside it"
            raise ValueError(msg)
    except AttributeError:
        # Fallback for older Python versions: use relative_to which raises ValueError
        try:
            dest.relative_to(src)
        except ValueError:
            pass
        else:
            msg = "Destination cannot be the source or inside it"
            raise ValueError(msg)

    shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
    logger.info(f"[DEFENSE] AI system mirrored to {dest}")
