"""Completion-pass contracts: DAY bandit isolation, live mode vs hard safety, buy ownership gate."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from types import SimpleNamespace

import pytest

from backend.services.learning_provenance import day_strategy_learning_allowed
from backend.services.live_readiness_service import configured_live_contract
from backend.services.ownership_repair import _backup
from tests.test_engine_lot_ownership import _engine, _lot


@pytest.mark.parametrize(
    ("engine_id", "reason", "dust", "allowed"),
    [
        ("DAY_V2", "NET_PROFIT_EXIT", False, True),
        ("LEGACY_DAY_LIVE", "STOP_LOSS_EXIT", False, True),
        ("SCALP_V2", "STOP_LOSS_EXIT", False, False),
        ("DAY_V2", "NET_PROFIT_EXIT", True, False),
        ("DAY_V2", "MANUAL_UNMATCHED", False, False),
        ("DAY_V2", "HUMAN_MANUAL_SELL", False, False),
        ("SCALP_V2", "HUMAN_MANUAL_SELL", False, False),
    ],
)
def test_day_bandit_learning_is_day_family_only(engine_id, reason, dust, allowed):
    assert day_strategy_learning_allowed(engine_id, reason, is_dust=dust) is allowed


def test_scalp_close_does_not_write_a_day_bandit_arm(tmp_path):
    eng = _engine(tmp_path)
    pos = _lot("SOL/USDT", "SCALP_V2", qty=0.2, price=120.0, trade_id="sc_bandit")
    eng._record_learning_outcome(
        symbol="SOL/USDT",
        position=pos,
        close_reason="STOP_LOSS_EXIT",
        manual_sell=False,
        source="engine",
        exit_price=117.5,
        realized_profit=-0.5,
        cooldown_until=0.0,
        fill_found=True,
    )
    with sqlite3.connect(eng.db_path) as conn:
        exists = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='day_outcome_bandit_arms'").fetchone()[0]
        arms = 0 if not exists else conn.execute("SELECT COUNT(*) FROM day_outcome_bandit_arms").fetchone()[0]
    assert arms == 0


def test_day_close_still_writes_a_day_bandit_arm(tmp_path):
    eng = _engine(tmp_path)
    pos = _lot("BTC/USDT", "DAY_V2", qty=0.001, price=80000.0, trade_id="day_bandit")
    eng._record_learning_outcome(
        symbol="BTC/USDT",
        position=pos,
        close_reason="NET_PROFIT_EXIT",
        manual_sell=False,
        source="engine",
        exit_price=81000.0,
        realized_profit=1.0,
        cooldown_until=0.0,
        fill_found=True,
    )
    with sqlite3.connect(eng.db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM day_outcome_bandit_arms").fetchone()[0]
    assert n == 1


def test_latest_bandit_update_inverts_exactly_after_decay():
    from backend.services.day_outcome_bandit import PRIOR_ALPHA, PRIOR_BETA, _is_win, _weight, invert_latest_bandit_update

    alpha, beta, wins, losses, total, n_obs = 4.5, 11.25, 8, 40, -3.2, 48
    pnl, reason = -0.8891778613742858, "STOP_LOSS_EXIT"
    assert _is_win(pnl, reason) is False
    weight = _weight(pnl)
    beta_after = beta + weight
    n_after = n_obs + 1
    alpha_stored = PRIOR_ALPHA + (alpha - PRIOR_ALPHA) * 0.92
    beta_stored = PRIOR_BETA + (beta_after - PRIOR_BETA) * 0.92
    restored = invert_latest_bandit_update(
        alpha=alpha_stored,
        beta=beta_stored,
        wins=wins,
        losses=losses + 1,
        total_pnl=total + pnl,
        n_obs=n_after,
        pnl=pnl,
        exit_reason=reason,
    )
    assert restored is not None
    assert restored["n_obs"] == n_obs
    assert restored["losses"] == losses
    assert restored["alpha"] == pytest.approx(alpha)
    assert restored["beta"] == pytest.approx(beta)
    assert restored["total_pnl"] == pytest.approx(total)


def test_legacy_structure_break_label_replaces_generic_manual(tmp_path):
    from backend.services.ownership_repair import relabel_generic_manual_exit

    db = tmp_path / "m.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, side TEXT, exit_reason TEXT, exit_type TEXT, explainability_json TEXT)")
        conn.execute(
            "INSERT INTO paper_trades VALUES (1, 'SELL', 'MANUAL_EXIT', 'MANUAL', ?)",
            (json.dumps({"raw_exit_reason": "DAY_4H_STRUCTURE_BREAK_EXIT"}),),
        )
        assert relabel_generic_manual_exit(conn, 1) == "DAY_4H_STRUCTURE_BREAK_EXIT"
        assert tuple(conn.execute("SELECT exit_reason, exit_type FROM paper_trades").fetchone()) == ("DAY_4H_STRUCTURE_BREAK_EXIT", "DAY_4H_STRUCTURE_BREAK_EXIT")


def test_hard_safety_block_does_not_rewrite_configured_live_mode(monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    status = configured_live_contract(
        entry_permitted=False,
        hard_safety_block="stale_market_data",
        kill_switch="RESUME",
        account_health="healthy",
    )
    assert status["configured_mode"] == "LIVE"
    assert status["execution_mode"] == "LIVE"
    assert status["day_live"] is True
    assert status["scalp_live"] is True
    assert status["entry_permitted"] is False
    assert status["hard_safety_block"] == "stale_market_data"


async def test_live_buy_waits_for_ownership_reconcile(tmp_path):
    eng = _engine(tmp_path)
    eng._ownership_state_known = False
    result = await eng.execute_buy_fifo(
        "BTC/USDT",
        0.001,
        80000.0,
        0.0,
        1.0,
        0.5,
        1,
        explainability=None,
        entry_authority="DAY_V2_CONFIRMED",
    )
    assert result is None
    assert eng.last_buy_reject_reason == "OWNERSHIP_STATE_UNKNOWN"


async def test_ownership_cap_warning_is_not_repeated_for_the_same_lot(tmp_path, caplog):
    eng = _engine(tmp_path)
    pos = _lot("BTC/USDT", "DAY_V2", qty=0.01, price=80000.0, trade_id="caplog")
    eng._fill_owned_qty = lambda _p: 0.001
    with caplog.at_level(logging.WARNING):
        assert eng._ownership_capped_qty(pos, 0.01) == pytest.approx(0.001)
        assert eng._ownership_capped_qty(pos, 0.01) == pytest.approx(0.001)
        assert eng._ownership_capped_qty(pos, 0.02) == pytest.approx(0.001)
    warnings = [r.message for r in caplog.records if "LOT_QTY_OWNERSHIP_CAPPED" in r.message]
    assert len(warnings) == 2


def test_status_reads_a_fresh_reconcile_file_as_ownership_known(tmp_path, monkeypatch):
    from backend.services.portfolio_engine import _configured_live_status

    path = tmp_path / "reconcile.json"
    path.write_text(json.dumps({"time_epoch": time.time(), "actions": "periodic"}))
    monkeypatch.setenv("MYSTIC_RECONCILE_STATE_FILE", str(path))
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    engine = SimpleNamespace(_ownership_state_known=False)
    status = _configured_live_status(engine, {"effective_entry_permitted": True, "accounting_healthy": True, "kill_switch_mode": "RESUME"})
    assert status["configured_mode"] == "LIVE"
    assert status["entry_permitted"] is True
    assert status["hard_safety_block"] == "none"


def test_weight_backup_uses_the_real_table_name(tmp_path):
    db = tmp_path / "w.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE ai_strategy_score_weights (strategy_id TEXT, symbol TEXT, regime TEXT, component_name TEXT, weight REAL)")
        conn.execute("INSERT INTO ai_strategy_score_weights VALUES ('day','BTCUSDT','neutral::RANGE','trend',1.2)")
        conn.execute("CREATE TEMP VIEW _w AS SELECT strategy_id||'|'||symbol||'|'||regime||'|'||component_name AS k, * FROM ai_strategy_score_weights")
        _backup(conn, "_w", "k", "day|BTCUSDT|neutral::RANGE|trend", "restore_adaptive_weight", "test", table_label="ai_strategy_score_weights")
        label = conn.execute("SELECT table_name FROM ownership_repair_backup").fetchone()[0]
    assert label == "ai_strategy_score_weights"
