"""Paper-only scalp calibration profiles — never applied when SCALP_LIVE=true."""

from __future__ import annotations

import os
from dataclasses import replace

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.paper_spread_caps import (
    parse_paper_spread_caps_json,
    uses_paper_spread_caps,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

CALIBRATION_PROFILES: dict[str, dict[str, float]] = {
    # A — current strict production paper defaults
    "strict": {
        "net_profit_target_pct": 0.0025,
        "entry_edge_buffer_pct": 0.0010,
        "min_projected_surplus_pct": 0.0005,
    },
    # B — moderate scalp (paper calibration only)
    "moderate": {
        "net_profit_target_pct": 0.0015,
        "entry_edge_buffer_pct": 0.0005,
        "min_projected_surplus_pct": 0.0003,
    },
    # C — fast scalp (paper calibration only)
    "fast": {
        "net_profit_target_pct": 0.0010,
        "entry_edge_buffer_pct": 0.0003,
        "min_projected_surplus_pct": 0.0002,
    },
}


def apply_profile(base: ScalpEconomics, profile: str) -> ScalpEconomics:
    key = (profile or "strict").strip().lower()
    overrides = CALIBRATION_PROFILES.get(key)
    if overrides is None:
        raise ValueError(f"unknown SCALP_CALIBRATION_PROFILE: {profile!r}")
    return replace(base, **overrides)


def _attach_paper_spread_caps(econ: ScalpEconomics, config: ScalpConfig) -> ScalpEconomics:
    if not uses_paper_spread_caps(
        scalp_live=config.scalp_live,
        calibration_mode=config.calibration_mode,
        scalp_paper_enabled=config.scalp_paper_enabled,
    ):
        return econ
    caps = parse_paper_spread_caps_json()
    return replace(econ, paper_spread_caps=caps)


def economics_for_config(config: ScalpConfig) -> ScalpEconomics:
    """Paper economics follow realized MFE (moderate 0.15% target) by default.

    Closed-trade audit: most MAX_HOLD exits never reached 0.25% MFE; the two
    NET_PROFIT winners cleared ~0.25–0.79%. Moderate target aligns hold/exit.
    Live path never uses calibration profiles.
    """
    base = ScalpEconomics.from_env()
    econ = base
    if config.scalp_live:
        econ = _attach_paper_spread_caps(econ, config)
        return econ
    if config.calibration_mode:
        econ = apply_profile(base, config.calibration_profile)
    elif config.scalp_paper_enabled and _env_bool("SCALP_PAPER_ECON_AUTO", True):
        # Auto-apply profile for paper even when SCALP_CALIBRATION_MODE=false.
        profile = (config.calibration_profile or "moderate").strip().lower()
        if profile not in CALIBRATION_PROFILES:
            profile = "moderate"
        econ = apply_profile(base, profile)
    econ = _attach_paper_spread_caps(econ, config)
    return econ


def economics_for_profile_name(profile: str) -> ScalpEconomics:
    return apply_profile(ScalpEconomics.from_env(), profile)
