import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def deploy_protocol(name="MysticVault"):
    """Deploy a protocol by creating a spec file.

    Logs progress and creates a file named '<name>_spec.md' containing
    a simple header and a note that it was auto-deployed.
    """
    logger.info(f"[PROTOCOL] Deploying protocol: {name}")
    filename = Path(f"{name}_spec.md")
    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(f"# {name}\nAuto-deployed by AI.")
        logger.info(f"[PROTOCOL] {filename} created.")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as exc:
        logger.exception(f"[PROTOCOL] Failed to create {filename}: {exc}")
        raise
