"""Candidate scoring reuse, inference index, and markout non-finite guard."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.services import symbol_setup_outcome_penalty as ssop
from backend.services.adaptive_learning import record_candidate, resolve_markouts
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.portfolio_engine import PortfolioEngine


def _paper_db(tmp_path: Path) -> str:
    db = str(tmp_path / "perf.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                pnl REAL,
                exit_reason TEXT,
                timestamp TEXT,
                updated_at TEXT,
                entry_price REAL,
                price REAL,
                explainability_json TEXT,
                hold_time_seconds REAL,
                is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        for i in range(ssop.CLEAN_INFRA_MIN_SELL_ID, ssop.CLEAN_INFRA_MIN_SELL_ID + 12):
            sym = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")[i % 4]
            setup = ("BREAKOUT_CONTINUATION", "RANGE_BOUNCE", "HTF_TREND_PULLBACK")[i % 3]
            explain = {
                "setup_type_canonical": setup,
                "setup_type": setup,
                "day_route_regime": "range",
                "selected_net_expected_value": 0.002 if i % 2 else -0.003,
                "mfe_pct": 0.0004 if i % 2 == 0 else 0.004,
                "mae_pct": 0.004 if i % 2 == 0 else 0.001,
                "raw_exit_reason": "STALL_EXIT_DEAD_NO_MFE" if i % 2 == 0 else "AI_NET_PROFIT_SELL",
            }
            conn.execute(
                """
                INSERT INTO paper_trades (
                    id, symbol, side, pnl, exit_reason, timestamp, updated_at,
                    entry_price, price, explainability_json, hold_time_seconds, is_synthetic
                ) VALUES (?, ?, 'SELL', ?, ?, '2026-11-01T00:00:00Z', '2026-11-01T00:00:00Z', 100.0, 99.5, ?, 900, 0)
                """,
                (
                    i,
                    sym,
                    -0.45 if i % 2 == 0 else 0.55,
                    explain["raw_exit_reason"],
                    json.dumps(explain),
                ),
            )
        conn.commit()
    return db


def _uncached_closed_trades(db_path: str, min_sell_id: int) -> list[ssop.ClosedTradeRow]:
    rows: list[ssop.ClosedTradeRow] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id, symbol, pnl, exit_reason, timestamp, entry_price, price,
                   explainability_json, hold_time_seconds
            FROM paper_trades
            WHERE side = 'SELL'
              AND id >= ?
              AND COALESCE(is_synthetic, 0) = 0
            ORDER BY id ASC
            """,
            (min_sell_id,),
        )
        for r in cur.fetchall():
            explain = ssop._parse_explain(r["explainability_json"])
            setup, regime, selected_ev = ssop._extract_setup_regime_ev(explain)
            hold_sec = float(r["hold_time_seconds"] or explain.get("hold_time_seconds") or explain.get("time_in_trade_sec") or 0)
            hold_min = hold_sec / 60.0 if hold_sec > 0 else 75.0
            mfe, mae = ssop._extract_mfe_mae(explain)
            raw_exit = str(explain.get("raw_exit_reason") or explain.get("exit_reason_raw") or explain.get("exit_trigger") or r["exit_reason"] or "")
            rows.append(
                ssop.ClosedTradeRow(
                    sell_id=int(r["id"]),
                    symbol=ssop.normalize_symbol(r["symbol"]),
                    setup=setup,
                    regime=regime,
                    exit_reason=str(r["exit_reason"] or ""),
                    pnl=float(r["pnl"] or 0.0),
                    hold_min=hold_min,
                    selected_ev=selected_ev,
                    timestamp=str(r["timestamp"] or ""),
                    mfe_pct=mfe,
                    mae_pct=mae,
                    raw_exit_reason=raw_exit,
                    worth_taking=None,
                )
            )
    return rows


def test_closed_trade_cache_matches_uncached_and_invalidates_on_in_place_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = _paper_db(tmp_path)
    ssop._CLOSED_TRADES_CACHE.clear()
    ssop._EXPLAIN_PARSE_CACHE.clear()

    first = ssop._load_closed_trades(db)
    assert first == _uncached_closed_trades(db, ssop.CLEAN_INFRA_MIN_SELL_ID)

    decodes = 0
    orig = ssop._parse_explain

    def _counted(raw):
        nonlocal decodes
        decodes += 1
        return orig(raw)

    monkeypatch.setattr(ssop, "_parse_explain", _counted)
    second = ssop._load_closed_trades(db)
    assert second == first
    assert decodes == 0

    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
        for setup in ("BREAKOUT_CONTINUATION", "RANGE_BOUNCE", "HTF_TREND_PULLBACK"):
            dd = {"setup_type": setup, "day_route_regime": "range", "selected_net_expected_value": 0.0018}
            cached_out = ssop.apply_v3_outcome_ranking_to_decision_data(dd, sym, raw_rank_score=0.42, buy_margin=0.18, db_path=db)
            ssop._CLOSED_TRADES_CACHE.clear()
            fresh_out = ssop.apply_v3_outcome_ranking_to_decision_data(dd, sym, raw_rank_score=0.42, buy_margin=0.18, db_path=db)
            assert cached_out == fresh_out

    decodes = 0
    target_id = ssop.CLEAN_INFRA_MIN_SELL_ID + 1
    with sqlite3.connect(db) as conn:
        raw = json.loads(conn.execute("SELECT explainability_json FROM paper_trades WHERE id=?", (target_id,)).fetchone()[0])
        raw["mfe_pct"] = 0.0001
        raw["mae_pct"] = 0.009
        raw["raw_exit_reason"] = "STALL_EXIT_DEAD_NO_MFE"
        conn.execute(
            "UPDATE paper_trades SET explainability_json=?, pnl=-1.25 WHERE id=?",
            (json.dumps(raw), target_id),
        )
        conn.commit()

    after_edit = ssop._load_closed_trades(db)
    assert decodes == 1
    assert after_edit == _uncached_closed_trades(db, ssop.CLEAN_INFRA_MIN_SELL_ID)
    assert after_edit != first


def test_adaptive_weights_refresh_avoids_separate_count_table_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import backend.services.ai_strategy_score_weight_writer as wwriter

    db = str(tmp_path / "weights.db")
    eng = PortfolioEngine(db_path=db, principal=500.0, test_mode=True)
    eng._ensure_db_schema()
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_strategy_score_weights (
                strategy_id TEXT, symbol TEXT, regime TEXT, component_name TEXT, weight REAL
            )
            """
        )
        conn.execute("INSERT INTO ai_strategy_score_weights VALUES ('day', 'BTCUSDT', 'range', 'buy_margin', 120.0)")
        conn.commit()

    count_calls = 0
    orig = wwriter.adaptive_weights_row_count

    def _spy(db_path: str = db) -> int:
        nonlocal count_calls
        count_calls += 1
        return orig(db_path)

    monkeypatch.setattr(wwriter, "adaptive_weights_row_count", _spy)
    for _ in range(5):
        weights = eng._load_adaptive_weights("day", "BTC/USDT", "range")
        assert weights == {"buy_margin": pytest.approx(120.0)}
    assert count_calls == 0


def test_inference_log_decision_id_uses_index(tmp_path: Path):
    db = str(tmp_path / "infer.db")
    ensure_ai_canonical_tables(db)
    with sqlite3.connect(db) as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN SELECT decision_id, features_json FROM ai_inference_log WHERE decision_id='d-1'").fetchall()
    detail = " ".join(str(r[-1]) for r in plan)
    assert "ix_ai_infer_decision_id" in detail


def test_missing_or_nonfinite_quote_is_retried_until_valid(tmp_path: Path):
    db = str(tmp_path / "nan.db")
    record_candidate(
        db,
        engine="DAY_V2",
        symbol="ETHUSDT",
        setup="RANGE_BOUNCE",
        regime="range",
        ref_price=100.0,
        roundtrip_cost=0.0006,
        signaled=True,
        evaluated_at=1_000.0,
    )
    with sqlite3.connect(db) as conn:
        horizon = float(conn.execute("SELECT label_horizon FROM adaptive_candidate_markouts").fetchone()[0])

    due = 1_000.0 + horizon * 60.0
    assert resolve_markouts(db, lambda _s, _t: float("nan"), now=due + 60.0) == 0
    with sqlite3.connect(db) as conn:
        learned = conn.execute("SELECT learned FROM adaptive_candidate_markouts").fetchone()[0]
    assert learned == 0

    assert resolve_markouts(db, lambda _s, _t: 101.0, now=due + 120.0) == 1
    assert resolve_markouts(db, lambda _s, _t: 101.0, now=due + 180.0) == 0
    with sqlite3.connect(db) as conn:
        n, ewma = conn.execute("SELECT n, ewma FROM adaptive_metric_state WHERE metric='markout_forward'").fetchone()
    assert n == pytest.approx(1.0)
    assert ewma == pytest.approx((101.0 - 100.0) / 100.0 - 0.0006)
