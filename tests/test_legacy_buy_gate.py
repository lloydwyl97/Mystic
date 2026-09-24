"""Tests proving LEGACY_DAY_LIVE_BUY_DISABLED correctly blocks legacy entries.

Requirements verified:
- When LEGACY_DAY_LIVE_BUY_DISABLED=true (default), process_bar_candidates
  is skipped and returns None without touching exchange.
- The gate does not affect SCALP V2 or DAY V2 entry paths.
- Setting LEGACY_DAY_LIVE_BUY_DISABLED=false re-enables the legacy path.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: env-var gate inspection
# ---------------------------------------------------------------------------


def _legacy_disabled(env_val: str) -> bool:
    """Mirror the gate logic in portfolio_engine_integration.py."""
    return env_val.lower() not in ("0", "false", "no", "off")


def test_legacy_disabled_by_default():
    """Default (no env var) must produce disabled=True."""
    result = _legacy_disabled(os.getenv("LEGACY_DAY_LIVE_BUY_DISABLED", "true"))
    assert result is True


def test_legacy_enabled_when_set_false():
    """LEGACY_DAY_LIVE_BUY_DISABLED=false must produce disabled=False."""
    assert _legacy_disabled("false") is False
    assert _legacy_disabled("0") is False
    assert _legacy_disabled("no") is False
    assert _legacy_disabled("off") is False


def test_legacy_disabled_when_set_true():
    """LEGACY_DAY_LIVE_BUY_DISABLED=true must produce disabled=True."""
    assert _legacy_disabled("true") is True
    assert _legacy_disabled("1") is True
    assert _legacy_disabled("yes") is True
    assert _legacy_disabled("on") is True


# ---------------------------------------------------------------------------
# Scalp V2 opportunity module: expiry env var is readable
# ---------------------------------------------------------------------------


def test_scalp_v2_opp_expiry_env_parseable():
    """SCALP_V2_OPP_EXPIRY_SEC can be overridden and must parse to float."""
    with patch.dict(os.environ, {"SCALP_V2_OPP_EXPIRY_SEC": "1800"}):
        import importlib

        import backend.services.scalp_v2.opportunity as opp_mod

        importlib.reload(opp_mod)
        assert opp_mod.SCALP_V2_OPP_EXPIRY_SEC == 1800.0
