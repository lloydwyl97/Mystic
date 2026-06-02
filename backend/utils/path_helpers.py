"""
Path helper utilities for AI model management
"""

from __future__ import annotations

import os
from pathlib import Path


def _base_dir() -> Path:
    base = os.getenv("MODEL_BASE_DIR", "models")
    p = Path(base).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_segment(seg: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in seg.strip())


def get_model_base_path() -> str:
    """Return absolute base path for all model artifacts."""
    return str(_base_dir())


def get_model_subdir(subdir: str) -> str:
    """Return absolute subdirectory path under the model base."""
    safe = _safe_segment(subdir) or "misc"
    p = _base_dir() / safe
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def validate_model_path(path: str) -> bool:
    """Return True if the path exists and is readable/writable."""
    try:
        p = Path(path)
        return p.exists() and os.access(p, os.R_OK | os.W_OK)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False


def ensure_model_directories() -> dict[str, str]:
    """Create and return standard subdirectories."""
    return {
        "base": get_model_base_path(),
        "versions": get_model_subdir("versions"),
        "backups": get_model_subdir("backups"),
        "performance": get_model_subdir("performance"),
        "training_data": get_model_subdir("training_data"),
        "active": get_model_subdir("active"),
        "scalers": get_model_subdir("scalers"),
    }


def get_model_file_path(model_type: str, symbol: str, suffix: str = "") -> str:
    """Return absolute path for a model file (.pth) in the active directory."""
    mt = _safe_segment(model_type) or "model"
    sym = _safe_segment(symbol.replace("/", "_"))
    suf = _safe_segment(suffix) if suffix else ""
    filename = f"{mt}_{sym}{suf}.pth"
    return str(Path(get_model_subdir("active")) / filename)


def get_scaler_file_path(model_type: str, symbol: str, suffix: str = "") -> str:
    """Return absolute path for a scaler file (.pkl)."""
    mt = _safe_segment(model_type) or "model"
    sym = _safe_segment(symbol.replace("/", "_"))
    suf = _safe_segment(suffix) if suffix else ""
    filename = f"{mt}_{sym}{suf}.pkl"
    return str(Path(get_model_subdir("scalers")) / filename)
