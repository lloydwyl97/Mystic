"""Per-symbol spread caps for paper/calibration only — never used when SCALP_LIVE=true."""

from __future__ import annotations

import json
import os

DEFAULT_PAPER_SPREAD_CAPS: dict[str, float] = {
    "BTCUSDT": 0.0008,
    "ETHUSDT": 0.0006,
    "SOLUSDT": 0.0005,
    "XRPUSDT": 0.0008,
}


def _repair_bash_stripped_json_object(text: str) -> str:
    """Recover `{BTCUSDT:0.0008}` after bash `source` strips JSON key quotes."""
    import re

    s = str(text or "").strip()
    if not s.startswith("{") or '"' in s:
        return s
    return re.sub(r"([A-Za-z0-9_/]+)\s*:", r'"\1":', s)


def parse_paper_spread_caps_json(raw: str | None = None) -> dict[str, float]:
    """Parse SCALP_PAPER_SPREAD_CAPS_JSON; fall back to Phase 4 defaults."""
    text = raw if raw is not None else os.getenv("SCALP_PAPER_SPREAD_CAPS_JSON", "")
    if not text or not str(text).strip():
        return dict(DEFAULT_PAPER_SPREAD_CAPS)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = json.loads(_repair_bash_stripped_json_object(text))
        except json.JSONDecodeError:
            return dict(DEFAULT_PAPER_SPREAD_CAPS)
    if not isinstance(data, dict):
        raise TypeError("SCALP_PAPER_SPREAD_CAPS_JSON must be a JSON object")
    out: dict[str, float] = {}
    for key, val in data.items():
        sym = str(key).strip().upper()
        if sym:
            out[sym] = float(val)
    return out or dict(DEFAULT_PAPER_SPREAD_CAPS)


def uses_paper_spread_caps(*, scalp_live: bool, calibration_mode: bool, scalp_paper_enabled: bool) -> bool:
    """Paper caps apply only when not live and paper/calibration is active."""
    if scalp_live:
        return False
    return calibration_mode or scalp_paper_enabled
