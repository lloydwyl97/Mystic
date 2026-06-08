"""Paper-only scalp calibration profiles — never applied when SCALP_LIVE=true."""

from __future__ import annotations

from dataclasses import replace

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.paper_spread_caps import (
    parse_paper_spread_caps_json,
    uses_paper_spread_caps,
)

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
    base = ScalpEconomics.from_env()
    econ = base
    if config.calibration_mode:
        if config.scalp_live:
            raise RuntimeError("SCALP_CALIBRATION_MODE cannot be enabled with SCALP_LIVE=true")
        econ = apply_profile(base, config.calibration_profile)
    econ = _attach_paper_spread_caps(econ, config)
    return econ


def economics_for_profile_name(profile: str) -> ScalpEconomics:
    return apply_profile(ScalpEconomics.from_env(), profile)
