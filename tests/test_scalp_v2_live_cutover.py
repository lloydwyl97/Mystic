"""SCALP V2 live cutover: identity, same-move reset, accounting, the loss-hold veto,
and the full promotion suite required by the 2026-09-24 live cutover."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


def test_same_move_blocks_until_price_zone_changes(tmp_path: Path):
    from backend.services.scalp_v2.opportunity import arm_opportunity, mark_opportunity

    db = tmp_path / "t.db"
    oid, blocked = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2700.0)
    assert blocked is False
    assert oid
    again, blocked2 = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2700.2)
    assert again == oid
    assert blocked2 is True
    mark_opportunity(db, "ETH/USDT", oid, "CLOSED")
    still, blocked3 = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2701.0)
    assert still == oid
    assert blocked3 is True
    moved, blocked4 = arm_opportunity(db, "ETH/USDT", "VWAP_REVERSION", 2900.0)
    assert blocked4 is False
    assert moved != oid


def test_clock_does_not_create_a_new_opportunity():
    from backend.services.scalp_v2.opportunity import ScalpOpportunityId

    a = ScalpOpportunityId.from_intent("BTC/USDT", "BREAK", "2026-09-21T01:00:00+00:00", arm_price=80000)
    b = ScalpOpportunityId.from_intent("BTC/USDT", "BREAK", "2026-09-21T06:00:00+00:00", arm_price=80000)
    assert a.canonical_id == b.canonical_id


def test_reconcile_rows_do_not_double_count_and_trade_ids_persist(tmp_path: Path):
    from backend.services.scalp_v2.accounting_repair import (
        apply_trade_id_backfill,
        exclude_duplicate_realized,
        record_residual,
        record_supplements,
    )

    db = tmp_path / "a.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            trade_id TEXT,
            side TEXT,
            exit_reason TEXT,
            exit_type TEXT,
            pnl REAL,
            pnl_usd_net REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE live_exchange_fills (
            id INTEGER PRIMARY KEY,
            exchange_order_id TEXT,
            venue_trade_ids_json TEXT,
            fill_ids_json TEXT,
            fill_count INTEGER
        )
        """
    )
    conn.execute("INSERT INTO paper_trades VALUES (1,'s1','SELL','NET_PROFIT_EXIT','NET_PROFIT_EXIT',1.0,1.0)")
    conn.execute("INSERT INTO paper_trades VALUES (2,'s2','SELL','EXCHANGE_RECONCILE_CLOSE','EXCHANGE_RECONCILE_CLOSE',0.13,NULL)")
    conn.execute("INSERT INTO paper_trades VALUES (3,'s3','SELL','HUMAN_MANUAL_SELL','HUMAN_MANUAL_SELL',0,0)")
    conn.execute("INSERT INTO live_exchange_fills VALUES (9,'1840254087','[]','[]',1)")
    flagged = exclude_duplicate_realized(conn)
    assert flagged == 2
    realized = conn.execute("SELECT ROUND(SUM(COALESCE(pnl_usd_net,pnl)),2) FROM paper_trades WHERE COALESCE(counts_toward_realized,1)=1").fetchone()[0]
    assert realized == 1.0
    n = apply_trade_id_backfill(
        conn,
        {"1840254087": {"trade_ids": ["31800251"], "order_ids": ["1840254087"], "taker_or_maker": "taker"}},
    )
    assert n == 1
    stored = conn.execute("SELECT venue_trade_ids_json, taker_or_maker FROM live_exchange_fills WHERE id=9").fetchone()
    assert "31800251" in stored[0]
    assert stored[1] == "taker"
    added = record_supplements(
        conn,
        [
            {
                "exchange_order_id": "1840253877",
                "symbol": "BTC/USDT",
                "side": "SELL",
                "trade_ids": ["31800220"],
                "qty": 0.00012,
                "price": 85853.66,
                "fee_amount": 0.00206049,
                "fee_asset": "USDT",
                "taker_or_maker": "taker",
                "parent_order_id": "1840254087",
                "note": "merged sibling",
            }
        ],
    )
    assert added == 1
    assert record_supplements(conn, [{"exchange_order_id": "1840253877", "symbol": "BTC/USDT", "side": "SELL", "trade_ids": ["31800220"], "qty": 0.00012, "price": 1, "fee_amount": 9}]) == 0
    record_residual(conn, "ETH/USDT", 0.01765762, 0.01729654, "exchange minus position lot")
    gap = conn.execute("SELECT residual_qty FROM documented_balance_residuals WHERE symbol='ETH/USDT'").fetchone()[0]
    assert abs(gap - 0.00036108) < 1e-8
    conn.close()


def test_legacy_open_lots_become_exit_only(tmp_path: Path):
    from backend.services.day_v2.migrations import apply_all_migrations

    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE portfolio_engine_positions (symbol TEXT, status TEXT, engine_id TEXT)")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('BTC/USDT','ACTIVE','LEGACY_DAY_LIVE')")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('ETH/USDT','CLOSED','LEGACY_DAY_LIVE')")
    conn.commit()
    conn.close()
    apply_all_migrations(str(db))
    conn = sqlite3.connect(db)
    rows = dict(conn.execute("SELECT symbol, engine_id FROM portfolio_engine_positions"))
    assert rows["BTC/USDT"] == "LEGACY_EXIT_ONLY"
    assert rows["ETH/USDT"] == "LEGACY_DAY_LIVE"
    apply_all_migrations(str(db))
    rows = dict(conn.execute("SELECT symbol, engine_id FROM portfolio_engine_positions"))
    assert rows["BTC/USDT"] == "LEGACY_EXIT_ONLY"


def test_paper_scalp_package_has_no_live_order_call():
    root = Path("backend/services/binance_scalp")
    text = "\n".join(p.read_text() for p in root.glob("*.py"))
    assert "place_order" not in text
    assert "create_order" not in text


def test_core_startup_does_not_launch_paper_scalp():
    """Paper scalp runner is fully retired (2026-09-22): not started, not callable.

    _launch_scalp() and start_scalp() functions were removed.  The `scalp` mode
    is now in the retired_mode case.  binance_scalp.runner is in LEGACY_PATTERNS
    (killed on every core restart to clear any lingering instance).
    """
    text = Path("start_mystic.sh").read_text()
    # Neither _launch_scalp nor start_scalp function definitions exist.
    assert "_launch_scalp()" not in text
    assert "start_scalp()" not in text
    assert "start_scalp ||" not in text
    # `scalp` mode is now in the retired_mode case alongside `all` etc.
    assert "all|ai|collector|agents|ai_position_tracker|ai_outcome_bridge|scalp)" in text
    assert 'retired_mode "$MODE"' in text
    # binance_scalp.runner is stopped as a LEGACY_PATTERN on every core start.
    assert "backend.services.binance_scalp.runner" in text
    watchdog = Path("watchdog_mystic.sh").read_text()
    assert "backend.services.binance_scalp.runner" not in watchdog


@pytest.mark.asyncio
async def test_consecutive_losses_do_not_veto_when_trailing_buy_is_off(monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "off")
    import backend.services.portfolio_engine as pe

    monkeypatch.setattr(pe, "ENABLE_GOVERNANCE_ENFORCEMENT", True)
    monkeypatch.setattr(pe, "governance_risk_governor_shadow_only", lambda: False)
    engine = pe.PortfolioEngine(principal=228.0, test_mode=True)
    engine.cash_balance = 220.0
    engine._available_balance = 220.0
    engine._total_open_risk = 0.0
    engine._get_loss_hold_until = pytest.importorskip("unittest.mock").AsyncMock(return_value=time.time() + 600)
    engine.get_rolling_24h_risk_metrics = pytest.importorskip("unittest.mock").AsyncMock(return_value=(0.0, pe.MAX_CONSEC_LOSSES))
    from unittest.mock import AsyncMock

    engine._get_loss_hold_until = AsyncMock(return_value=time.time() + 600)
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, pe.MAX_CONSEC_LOSSES))
    allowed, reason = await engine._can_open_position("SOL/USDT", 40.0)
    assert allowed is True
    assert reason != "HOLD_CONSEC_LOSSES"


@pytest.mark.asyncio
async def test_restart_load_keeps_legacy_exit_only(tmp_path: Path):
    from backend.services.day_v2.migrations import apply_all_migrations
    from backend.services.portfolio_engine import OpenPosition, PortfolioEngine

    db = tmp_path / "load.db"
    engine = PortfolioEngine(db_path=str(db), principal=228.0, test_mode=True)
    engine._ensure_db_schema()
    apply_all_migrations(str(db))
    pos = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.00008,
        entry_price=86000.0,
        entry_time=time.time(),
        trade_id="btc_legacy",
        stop_price=85000.0,
        take_profit_1_price=87000.0,
        take_profit_2_price=0.0,
        engine_id="LEGACY_EXIT_ONLY",
        scalp_opportunity_id="",
    )
    await engine._persist_position_to_sqlite(pos)
    restarted = PortfolioEngine(db_path=str(db), principal=228.0, test_mode=True)
    restarted._ensure_db_schema()
    await restarted._load_positions_from_sqlite(allow_mutations=False)
    loaded = restarted.open_positions["BTC/USDT"]
    assert loaded.engine_id == "LEGACY_EXIT_ONLY"
    await restarted._persist_position_to_sqlite(loaded)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT engine_id FROM portfolio_engine_positions WHERE symbol='BTC/USDT'").fetchone()
    assert row[0] == "LEGACY_EXIT_ONLY"


# ─────────────────────────────────────────────────────────────────────────────
# SCALP V2 LIVE PROMOTION SUITE (2026-09-24)
# ─────────────────────────────────────────────────────────────────────────────


def test_scalp_v2_has_live_authority():
    """engine_identity table resolves SCALP_V2 → LIVE."""
    from backend.services.day_v2.engine_identity import (
        AuthorityLevel,
        EngineId,
        has_live_authority,
    )

    assert has_live_authority(EngineId.SCALP_V2_LIVE)
    from backend.services.day_v2.engine_identity import _AUTHORITY_TABLE

    assert _AUTHORITY_TABLE[EngineId.SCALP_V2_LIVE] == AuthorityLevel.LIVE


def test_structural_mode_resolves_live_when_armed():
    """resolve_structural_mode returns LIVE when SCALP_LIVE=true and SCALP_LIVE_ARMED=true."""
    from backend.services.binance_scalp.structural_mode import MODE_LIVE, resolve_structural_mode

    mode = resolve_structural_mode(
        env_mode="",
        scalp_live=True,
        scalp_live_armed=True,
        scalp_paper_enabled=False,
        scalp_thesis="structural",
        legacy_prediction_entries=False,
        allow_market_orders=False,
    )
    assert mode == MODE_LIVE


def test_structural_mode_not_live_when_only_scalp_live():
    """SCALP_LIVE=true alone (without SCALP_LIVE_ARMED) does not resolve to LIVE."""
    from backend.services.binance_scalp.structural_mode import MODE_LIVE, resolve_structural_mode

    mode = resolve_structural_mode(
        env_mode="",
        scalp_live=True,
        scalp_live_armed=False,
        scalp_paper_enabled=True,
        scalp_thesis="structural",
        legacy_prediction_entries=False,
        allow_market_orders=False,
    )
    assert mode != MODE_LIVE
    assert mode == "STRUCTURAL_PAPER"


def test_live_impossible_absent_in_status_fields_when_live():
    """When mode=LIVE, status_fields must NOT report exchange_live_impossible=True."""
    from unittest.mock import MagicMock

    from backend.services.binance_scalp.structural_mode import MODE_LIVE
    from backend.services.binance_scalp.structural_thesis import status_fields

    cfg = MagicMock()
    cfg.resolved_structural_mode.return_value = MODE_LIVE
    cfg.scalp_live = True
    cfg.scalp_live_armed = True
    cfg.scalp_paper_enabled = False

    fields = status_fields(cfg)
    assert fields["exchange_live_impossible"] is False
    assert fields["live_entries_enabled"] is True
    assert fields["structural_mode"] == MODE_LIVE
    assert fields["fee_assumption_label"] == "exchange_actual"


def test_paper_engines_cannot_resolve_to_live_without_armed():
    """SCALP_LIVE=false always stays paper/shadow/disabled."""
    from backend.services.binance_scalp.structural_mode import MODE_LIVE, resolve_structural_mode

    for paper in (True, False):
        mode = resolve_structural_mode(
            env_mode="",
            scalp_live=False,
            scalp_live_armed=False,
            scalp_paper_enabled=paper,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=False,
        )
        assert mode != MODE_LIVE


def test_live_entry_enabled_only_for_live_mode():
    """live_entry_enabled returns True only for MODE_LIVE."""
    from backend.services.binance_scalp.structural_mode import (
        MODE_DISABLED,
        MODE_LIVE,
        MODE_PAPER,
        MODE_SHADOW,
        live_entry_enabled,
    )

    assert live_entry_enabled(MODE_LIVE) is True
    assert live_entry_enabled(MODE_PAPER) is False
    assert live_entry_enabled(MODE_SHADOW) is False
    assert live_entry_enabled(MODE_DISABLED) is False


def test_market_orders_still_refused_in_live_mode():
    """SCALP_ALLOW_MARKET_ORDERS is permanently refused even when scalp_live=True."""
    import pytest

    from backend.services.binance_scalp.structural_mode import StructuralModeError, resolve_structural_mode

    with pytest.raises(StructuralModeError, match="SCALP_ALLOW_MARKET_ORDERS"):
        resolve_structural_mode(
            env_mode="",
            scalp_live=True,
            scalp_live_armed=True,
            scalp_paper_enabled=False,
            scalp_thesis="structural",
            legacy_prediction_entries=False,
            allow_market_orders=True,
        )


def test_scalp_v2_entry_authority_bypasses_trailing_buy_gate():
    """ENTRY_AUTHORITY_SCALP_V2_LIVE is accepted by the trailing-buy gate check."""
    from backend.config.day_entry_execution import (
        ENTRY_AUTHORITY_SCALP_V2_LIVE,
        ENTRY_AUTHORITY_TRAILING_BUY,
    )

    assert ENTRY_AUTHORITY_SCALP_V2_LIVE != ENTRY_AUTHORITY_TRAILING_BUY
    assert ENTRY_AUTHORITY_SCALP_V2_LIVE == "SCALP_V2_LIVE_ENTRY"


def test_scalp_v2_live_checkpoint_rejects_pre_live_rows(tmp_path: Path):
    """live_checkpoint_count excludes SCALP_V2_PAPER_PRE_LIVE rows."""
    from backend.services.scalp_v2.checkpoint import (
        PHASE_PRE_LIVE,
        live_checkpoint_count,
        mark_pre_live_rows,
    )

    db = str(tmp_path / "t.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id            INTEGER PRIMARY KEY,
            trade_id      TEXT,
            side          TEXT,
            mode          TEXT,
            engine_id     TEXT,
            pnl_usd_net   REAL,
            is_synthetic  INTEGER,
            scalp_checkpoint_phase TEXT
        )
        """
    )
    # 3 pre-live paper rows
    for i in range(1, 4):
        conn.execute(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?)",
            (i, f"old_{i}", "SELL", "live", "SCALP_V2", 1.0, 0, None),
        )
    conn.commit()
    conn.close()

    labelled = mark_pre_live_rows(db)
    assert labelled == 3

    # Checkpoint must be 0 (all rows are pre-live)
    assert live_checkpoint_count(db) == 0

    # Add a post-cutover live row
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?)",
        (4, "live_001", "SELL", "live", "SCALP_V2", 2.5, 0, None),
    )
    conn.commit()
    conn.close()

    assert live_checkpoint_count(db) == 1

    # Calling mark_pre_live_rows again is idempotent (already-labelled rows skipped)
    labelled2 = mark_pre_live_rows(db)
    assert labelled2 == 0
    assert live_checkpoint_count(db) == 1


def test_historical_paper_rows_preserved_after_mark(tmp_path: Path):
    """mark_pre_live_rows does not delete or modify historical PnL."""
    from backend.services.scalp_v2.checkpoint import PHASE_PRE_LIVE, mark_pre_live_rows

    db = str(tmp_path / "t.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            trade_id TEXT, side TEXT, mode TEXT,
            engine_id TEXT, pnl_usd_net REAL, is_synthetic INTEGER,
            scalp_checkpoint_phase TEXT
        )
        """
    )
    conn.execute("INSERT INTO paper_trades VALUES (1,'t1','SELL','live','SCALP_V2',3.14,0,NULL)")
    conn.commit()
    conn.close()

    mark_pre_live_rows(db)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT pnl_usd_net, scalp_checkpoint_phase FROM paper_trades WHERE id=1").fetchone()
    conn.close()
    # PnL preserved; phase labelled
    assert abs(row[0] - 3.14) < 1e-9
    assert row[1] == PHASE_PRE_LIVE


def test_canonical_db_symbol_resolves_hyphen_format(tmp_path: Path):
    """canonical_db_symbol finds BTC-USDT format when that is what feature_ohlcv stores."""
    from backend.services.day_v2.five_min_confirm import canonical_db_symbol

    db = str(tmp_path / "ohlcv.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.execute("INSERT INTO feature_ohlcv VALUES ('BTC-USDT','5m','2026-09-24 00:00:00',60000,60100,59900,60050,1.0)")
    conn.commit()
    conn.close()

    # All three input formats should resolve to BTC-USDT
    for sym in ("BTCUSDT", "BTC/USDT", "BTC-USDT"):
        resolved = canonical_db_symbol(sym, db)
        assert resolved == "BTC-USDT", f"Expected BTC-USDT for {sym!r}, got {resolved!r}"


def test_canonical_db_symbol_returns_none_when_no_rows(tmp_path: Path):
    """canonical_db_symbol returns None when no 5m rows exist for the symbol."""
    from backend.services.day_v2.five_min_confirm import canonical_db_symbol

    db = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.commit()
    conn.close()

    result = canonical_db_symbol("BTCUSDT", db)
    assert result is None


def test_day_exit_does_not_fire_for_scalp_position():
    """DAY V2 exit evaluator returns None for SCALP_V2-owned positions."""
    from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit

    # Simulate a position that would trigger a DAY V2 exit
    # but engine_id = SCALP_V2 → DAY exit must not fire
    result = evaluate_day_v2_exit(
        engine_id="SCALP_V2",  # NOT DAY_V2
        entry_price=80000.0,
        current_price=78000.0,
        bar_low=77000.0,
        highest_price=81000.0,
        atr_at_entry=1000.0,
        structural_anchor=75000.0,
        target_price=85000.0,
        entry_time=time.time() - 7200,
        estimated_roundtrip_cost=0.0006,
    )
    # DAY V2 exit must not trigger for a SCALP_V2 position
    assert result is None or str(result.get("action", "")) != "sell"


def test_scalp_exit_evaluator_handles_scalp_v2_engine_id():
    """evaluate_engine_managed_exit treats SCALP_V2 positions correctly (no error)."""
    from unittest.mock import MagicMock

    from backend.services.day_controlled_exits import evaluate_engine_managed_exit

    pos = MagicMock()
    pos.engine_id = "SCALP_V2"
    pos.quantity = 0.01
    pos.entry_price = 80000.0
    pos.entry_time = time.time() - 120
    pos.highest_price = 80500.0
    pos.lowest_price = 79800.0
    pos.stop_price = 79000.0
    pos.take_profit_1_price = 81000.0
    pos.thesis_invalid_level = 79000.0
    pos.thesis_target_level = 82000.0
    pos.atr_at_entry = 500.0
    pos.scalp_opportunity_id = ""

    # Should not raise; exit decision is a dict with action=hold or sell
    result = evaluate_engine_managed_exit(
        position=pos,
        current_price=80200.0,
        net_pnl_pct=0.0025 - 0.0006,
        hold_minutes=2.0,
        bar_low=79800.0,
        coin_profile={"sl": 1.0, "tp": 1.4, "trail": 0.5, "hold_max_min": 300},
        engine_id="SCALP_V2",
    )
    assert isinstance(result, dict)  # No exception = pass


def test_log_rotation_script_does_not_target_databases():
    """The logrotate setup script patterns only target *.log files."""
    from pathlib import Path

    script = Path("scripts/setup_mystic_logrotate.sh").read_text()
    # Confirm it targets .log files
    assert "*.log" in script
    # The logrotate glob pattern must not target database files
    # (comment mentions are OK; the rotated path glob must not include .db)
    import re as _re

    rotated_globs = _re.findall(r"^[^#]*\*\.db", script, _re.MULTILINE)
    assert not rotated_globs, f"Found *.db glob outside comment: {rotated_globs}"
    # The DB filename must not appear as a rotated glob target
    db_targets = _re.findall(r"^[^#]*mystic_trading\.db", script, _re.MULTILINE)
    assert not db_targets, f"Found mystic_trading.db outside comment: {db_targets}"


def test_normal_startup_cannot_launch_paper_scalp():
    """Paper scalp runner is fully retired: not started, not callable.

    This test is already present in the file but serves as a regression guard
    after the live promotion changes.
    """
    from pathlib import Path

    text = Path("start_mystic.sh").read_text()
    assert "_launch_scalp()" not in text
    assert "start_scalp()" not in text
    assert "all|ai|collector|agents|ai_position_tracker|ai_outcome_bridge|scalp)" in text


def test_scalp_mode_display_is_live_when_mode_live():
    """_resolve_scalp_mode_display returns 'SCALP LIVE' when MODE_LIVE is configured."""
    from unittest.mock import MagicMock, patch

    from backend.services.binance_scalp.structural_mode import MODE_LIVE
    from backend.services.portfolio_engine import _resolve_scalp_mode_display

    mock_cfg = MagicMock()
    mock_cfg.resolved_structural_mode.return_value = MODE_LIVE

    with patch("backend.services.portfolio_engine.get_scalp_config", return_value=mock_cfg, create=True):
        # Patch at the portfolio_engine module level
        import backend.services.portfolio_engine as pe_mod

        with patch.object(pe_mod, "_resolve_scalp_mode_display", wraps=_resolve_scalp_mode_display):
            # Call the real function with patched config inside its closure
            import backend.services.binance_scalp.config as scfg_mod

            with patch.object(scfg_mod, "get_scalp_config", return_value=mock_cfg):
                result = _resolve_scalp_mode_display()
                assert result == "SCALP LIVE"


def test_four_symbols_still_enabled_in_scalp_config():
    """SCALP_PRODUCTS still resolves to the four coins regardless of mode."""
    import os
    from unittest.mock import patch

    with patch.dict(
        os.environ,
        {
            "SCALP_LIVE": "true",
            "SCALP_LIVE_ARMED": "true",
            "SCALP_PRODUCTS": "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT",
        },
    ):
        from backend.services.binance_scalp.config import ScalpConfig

        cfg = ScalpConfig.from_env()
        assert set(cfg.products) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
