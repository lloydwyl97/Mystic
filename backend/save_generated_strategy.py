import logging
import re
from pathlib import Path

from strategy_prompt_builder import generate_strategy_code

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


def _sanitize(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if base.lower().endswith(".py"):
        base = base[:-3]
    base = _SAFE.sub("_", base).strip("_")
    return base or "strategy"


def create_strategy_from_prompt(prompt: str, file_name: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        msg = "prompt required"
        raise ValueError(msg)
    base = _sanitize(file_name)
    out_dir = Path("strategies")
    out_dir.mkdir(parents=True, exist_ok=True)
    code = generate_strategy_code(prompt)
    if not isinstance(code, str) or not code.strip():
        msg = "empty strategy code"
        raise ValueError(msg)
    path = out_dir / f"{base}.py"
    with path.open("w", encoding="utf-8") as f:
        f.write(code)
    logger.info(f"[LLM] Created {path.as_posix()}")
    return str(path)
