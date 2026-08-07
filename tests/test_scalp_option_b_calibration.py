"""Option B: SCALP paper calibration — focus setups + moderate target, no live."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.paper_proof import compute_paper_proof
from backend.services.binance_scalp.strategies import enabled_strategies


def test_default_disables_never_filled_setups(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SCALP_DISABLED_STRATEGIES", raising=False)
    monkeypatch.setenv("SCALP_PAPER_ENABLED", "true")
    monkeypatch.setenv("SCALP_LIVE", "false")
    cfg = ScalpConfig.from_env()
    names = {s.name for s in enabled_strategies(cfg)}
    assert names == {"range_bounce_scalp", "vwap_ema_reclaim"}
    assert "trend_pullback_micro" in cfg.disabled_strategies
    assert "compression_breakout" in cfg.disabled_strategies


def test_paper_auto_moderate_target(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALP_PAPER_ENABLED", "true")
    monkeypatch.setenv("SCALP_LIVE", "false")
    monkeypatch.setenv("SCALP_CALIBRATION_MODE", "false")
    monkeypatch.setenv("SCALP_CALIBRATION_PROFILE", "moderate")
    monkeypatch.setenv("SCALP_PAPER_ECON_AUTO", "true")
    monkeypatch.setenv("SCALP_NET_PROFIT_TARGET_PCT", "0.0025")  # raw env overridden by profile
    cfg = ScalpConfig.from_env()
    econ = economics_for_config(cfg)
    assert econ.net_profit_target_pct == pytest.approx(0.0015)
    assert cfg.scalp_live is False


def test_live_never_gets_calibration_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SCALP_PAPER_ENABLED", "true")
    monkeypatch.setenv("SCALP_LIVE", "true")
    monkeypatch.setenv("SCALP_CALIBRATION_MODE", "false")
    monkeypatch.setenv("SCALP_NET_PROFIT_TARGET_PCT", "0.0025")
    cfg = ScalpConfig.from_env()
    # Live + calibration_mode forbidden at assert; economics path skips profile.
    econ = economics_for_config(cfg)
    assert econ.net_profit_target_pct == pytest.approx(0.0025)


def test_paper_proof_not_ready_on_tiny_history(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE scalp_paper_trades (
                id INTEGER PRIMARY KEY,
                side TEXT,
                pnl_usd REAL,
                exit_reason TEXT,
                diagnostics_json TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO scalp_paper_trades VALUES (1,'SELL',1.0,'NET_PROFIT_TARGET',?)",
            (json.dumps({"setup_name": "range_bounce_scalp"}),),
        )
        conn.commit()
    proof = compute_paper_proof(db)
    assert proof["closed_sells"] == 1
    assert proof["ready_for_live_discussion"] is False
    assert proof["live_blocked"] is True
