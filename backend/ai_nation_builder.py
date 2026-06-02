import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_ai_nation(name, constitution, citizens):
    """
    Create a nation file containing the constitution.

    Args:
        name (str): Nation name used as filename prefix.
        constitution (str): Text content to write to the constitution file.
        citizens: Iterable/collection of citizens; used only to report count.

    Returns:
        str: The path to the created constitution file.
    """
    try:
        count = len(citizens)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # If citizens is an int, use it; otherwise fall back to 0
        count = citizens if isinstance(citizens, int) else 0

    logger.info(f"[NATION] Creating {name} with {count} founding AIs.")

    filename = Path(f"{name}_constitution.txt")
    text = constitution if isinstance(constitution, str) else str(constitution)
    # Write using a context manager to ensure the file is properly closed
    with filename.open("w", encoding="utf-8") as f:
        f.write(text)

    return filename
