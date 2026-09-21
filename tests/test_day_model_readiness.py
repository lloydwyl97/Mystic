"""Data-readiness gate, corrected scorecard denominators, and label-integrity guarantees.

Everything here is offline analytics. The golden test at the bottom proves the audit layer
cannot reach ranking, sizing, exits or the book.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from backend.services.day_4h_entry_scorecard import (
    comparable_coin_markouts,
    decision_role,
    load_scorecard_rows,
    ranked_buy_rows,
    summarize,
    traded_rows,
)
from backend.services.day_4h_label_runner import persist_labels
from backend.services.day_4h_outcome_labeler import label_candidate
from backend.services.day_decision_label_contract import ensure_label_schema
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS, _ensure_schema
from backend.services.day_experiment_registry import ensure_registry_schema, seed_historical
from backend.services.day_forward_lock import register_lock
from backend.services.day_model_readiness import (
    MIN_EVENTS_PER_FEATURE,
    acceptance_standard,
    check_accounting,
    check_counterfactual_integrity,
    check_feature_coverage,
    check_forward_span,
    check_locked_test_protection,
    check_production_label_integrity,
    check_time_authority,
    evaluate_readiness,
    feature_availability_start,
    fifo_residual_report,
    is_residual_writeoff,
    load_state,
)

NOW = 1_788_500_000.0
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _telemetry() -> dict[str, object]:
    return {"distance_to_4h_break_bps": 42.0, "4h_structure_state": "intact", "production_4h_break_true_at_decision": False}


def _build_db(path, *, groups: int = 3, traded_every: int = 1, with_telemetry: bool = True, start: float = NOW - 86_400) -> str:
    db = str(path / "readiness.db")
    _ensure_schema(db)
    ensure_label_schema(db)
    ensure_registry_schema(db)
    seed_historical(db)
    register_lock(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS position_close_ledger (id INTEGER PRIMARY KEY, symbol TEXT, closed_at TEXT, "
        "closed_at_epoch REAL, close_reason TEXT, manual_sell INT, realized_profit REAL, realized_profit_unknown INT, "
        "cooldown_until REAL, quantity REAL, entry_price REAL, exit_price REAL, sell_trade_id TEXT, detail TEXT)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS paper_trades (trade_id TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL, remaining_position REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS portfolio_engine_positions (symbol TEXT, quantity REAL, entry_price REAL, status TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS portfolio_engine_ledger (id INTEGER PRIMARY KEY, total_equity REAL)")
    conn.execute("INSERT OR REPLACE INTO portfolio_engine_ledger (id,total_equity) VALUES (1, 1000.0)")

    for index in range(groups):
        gid = f"daygrp_{index}"
        created = start + index * 900
        iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(created))
        traded = index % traded_every == 0
        symbol = COINS[index % len(COINS)]
        buy_id = f"buy_{index}"
        contract = {
            "candidates": [
                {
                    "symbol": s,
                    "eligible": True,
                    "spread_bps": 1.0,
                    "expected_slippage": 0.5,
                    "quote_timestamp": iso,
                    "4h_entry_telemetry": _telemetry() if with_telemetry else {},
                }
                for s in COINS
            ],
            "4h_entry_telemetry": {s: (_telemetry() if with_telemetry else {}) for s in COINS},
        }
        conn.execute(
            f"INSERT INTO {TABLE_GROUPS} (decision_group_id, created_at, account_execution_mode, schema_version, feature_schema, "
            "selected_action, selected_symbol, contract_json, execute_authorized, fill_trade_id, lifecycle_state) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                gid,
                iso,
                "paper",
                1,
                "day_v1",
                f"BUY_{symbol}",
                symbol,
                json.dumps(contract),
                1 if traded else 0,
                buy_id if traded else None,
                "filled" if traded else "blocked_after_ranking",
            ),
        )
        for s in COINS:
            conn.execute(
                f"INSERT INTO {TABLE_CANDIDATES} (decision_group_id, symbol, created_at, eligible, p_buy, path_ev, final_rank_score, feature_json) VALUES (?,?,?,?,?,?,?,?)",
                (gid, s, iso, 1, 0.6, 0.001, 0.7, json.dumps({"4h_entry_telemetry": _telemetry() if with_telemetry else {}})),
            )
        if traded:
            conn.execute(
                "INSERT INTO position_close_ledger (symbol, closed_at_epoch, close_reason, realized_profit, quantity, entry_price, exit_price, sell_trade_id, detail) VALUES (?,?,?,?,?,?,?,?,?)",
                (symbol, created + 3600, "TRAILING_STOP_EXIT", 1.0, 1.0, 100.0, 101.0, f"sell_{index}", f"buy_trade_id={buy_id}"),
            )
    conn.commit()
    conn.close()
    return db


def _write_labels(db: str, *, authoritative_for_traded: bool = True) -> None:
    conn = sqlite3.connect(db)
    rows = conn.execute(f"SELECT decision_group_id, selected_symbol, execute_authorized, fill_trade_id FROM {TABLE_GROUPS}").fetchall()
    conn.close()
    batch = []
    for gid, selected, auth, fill in rows:
        for symbol in (*COINS, "HOLD"):
            is_fill = bool(auth == 1 and fill and symbol == selected and authoritative_for_traded)
            if symbol == "HOLD":
                payload = {
                    "decision_group_id": gid,
                    "symbol": "HOLD",
                    "provenance": "authoritative",
                    "counterfactual": False,
                    "markouts": {"15m": 0.0, "30m": 0.0, "1h": 0.0, "2h": 0.0, "4h": 0.0},
                    "production_exit_net_bps": 0.0,
                }
            else:
                payload = {
                    "decision_group_id": gid,
                    "symbol": symbol,
                    "provenance": "authoritative" if is_fill else "reconstructed",
                    "counterfactual": not is_fill,
                    "markouts": {"15m": 1.0, "30m": 2.0, "1h": 3.0, "2h": 4.0, "4h": 5.0 + COINS.index(symbol)},
                    "mfe_bps": 60.0,
                    "mae_bps": -10.0,
                    "covered_genuine_cost": True,
                    "reached_production_BE_level": True,
                    "reached_production_trail_level": True,
                    "production_exit_net_bps": 100.0 if is_fill else None,
                    "exit_reason": "TRAILING_STOP_EXIT" if is_fill else None,
                    "regret_vs_hold_bps": 100.0 if is_fill else None,
                }
            batch.append(payload)
    persist_labels(db, batch)


# --------------------------------------------------------------------------------------
# corrected scorecard denominators
# --------------------------------------------------------------------------------------
def test_decision_role_separates_ranking_from_trading():
    assert decision_role({"selected_action": "HOLD"}) == "HOLD"
    assert decision_role({"selected_action": "BUY_BTCUSDT", "execute_authorized": 1, "fill_trade_id": "t1"}) == "traded"
    assert decision_role({"selected_action": "BUY_BTCUSDT", "execute_authorized": 1, "fill_trade_id": ""}) == "ranking_only"
    assert decision_role({"selected_action": "BUY_BTCUSDT", "execute_authorized": 0}) == "blocked_after_ranking"
    assert decision_role({"selected_action": "BUY_BTCUSDT"}) == "ranking_only"


def test_ranked_but_unfilled_groups_are_not_counted_as_trades(tmp_path):
    db = _build_db(tmp_path, groups=4, traded_every=2)
    _write_labels(db)
    rows = load_scorecard_rows(db)
    assert len(ranked_buy_rows(rows)) == 4
    assert len(traded_rows(rows)) == 2
    out = summarize(rows)
    assert out["ranked_buy_count"] == 4
    assert out["selected_trade_count"] == 2
    assert out["blocked_after_ranking_count"] == 2


def test_missing_labels_do_not_deflate_outcome_rates(tmp_path):
    """A group with no authoritative label must be excluded, not scored as a loss."""
    db = _build_db(tmp_path, groups=4, traded_every=1)
    _write_labels(db)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM day_decision_outcome_labels WHERE decision_group_id='daygrp_3'")
    conn.commit()
    conn.close()
    out = summarize(load_scorecard_rows(db))
    assert out["selected_trade_count"] == 4
    assert out["authoritative_label_count"] == 3
    # every surviving label is a winner, so the rate is 1.0 rather than 3/4
    assert out["positive_net_rate"] == pytest.approx(1.0)
    assert out["cost_cover_rate"] == pytest.approx(1.0)


def test_best_coin_uses_one_measure_and_flags_incomparable_groups(tmp_path):
    db = _build_db(tmp_path, groups=2, traded_every=1)
    _write_labels(db)
    rows = load_scorecard_rows(db)
    assert comparable_coin_markouts(rows[0]) is not None
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM day_decision_outcome_labels WHERE decision_group_id='daygrp_1' AND symbol='SOLUSDT'")
    conn.commit()
    conn.close()
    rows = load_scorecard_rows(db)
    incomplete = next(r for r in rows if r["decision_group_id"] == "daygrp_1")
    assert comparable_coin_markouts(incomplete) is None
    out = summarize(rows)
    assert out["best_coin_not_comparable"] == 1
    assert out["best_coin_selection_measure"] == "markout_4h"
    assert out["best_coin_chance_baseline"] == pytest.approx(0.25)


def test_production_net_is_never_compared_against_a_markout(tmp_path):
    """The filled leg has a production exit; the losers do not. Both must be scored alike."""
    db = _build_db(tmp_path, groups=1, traded_every=1)
    _write_labels(db)
    row = load_scorecard_rows(db)[0]
    nets = comparable_coin_markouts(row)
    assert nets is not None
    selected_label = row["labels"][row["selected_symbol"]]
    assert selected_label["production_exit_net_bps"] == 100.0
    # the selected coin is scored on its markout, not on the 100 bps production exit
    assert nets[row["selected_symbol"]] != 100.0


# --------------------------------------------------------------------------------------
# label integrity
# --------------------------------------------------------------------------------------
def test_authoritative_label_requires_a_real_fill(tmp_path):
    db = _build_db(tmp_path, groups=3, traded_every=1)
    _write_labels(db)
    result = check_production_label_integrity(load_state(db), now=NOW)
    assert result["pass"] is True
    assert result["authoritative_without_fill"] == []


def test_authoritative_label_on_an_unfilled_group_is_a_violation(tmp_path):
    db = _build_db(tmp_path, groups=2, traded_every=2)
    _write_labels(db)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE day_decision_outcome_labels SET provenance='authoritative' WHERE decision_group_id='daygrp_1'")
    conn.commit()
    conn.close()
    result = check_production_label_integrity(load_state(db), now=NOW)
    assert result["pass"] is False
    assert "daygrp_1" in result["authoritative_without_fill"]


def test_counterfactual_may_not_carry_production_fill_fields(tmp_path):
    db = _build_db(tmp_path, groups=2, traded_every=1)
    _write_labels(db)
    assert check_counterfactual_integrity(load_state(db))["pass"] is True
    conn = sqlite3.connect(db)
    conn.execute("UPDATE day_decision_outcome_labels SET exit_reason='TRAILING_STOP_EXIT', production_exit_net_bps=12.0 WHERE decision_group_id='daygrp_0' AND symbol='ETHUSDT'")
    conn.commit()
    conn.close()
    result = check_counterfactual_integrity(load_state(db))
    assert result["pass"] is False
    assert result["violations"]["counterfactual_has_exit_reason"] >= 1


def test_hold_is_authoritative_at_zero_and_excluded_from_fill_stats(tmp_path):
    db = _build_db(tmp_path, groups=2, traded_every=1)
    _write_labels(db)
    state = load_state(db)
    hold = state["labels"]["daygrp_0"]["HOLD"]
    assert hold["provenance"] == "authoritative"
    assert hold["production_exit_net_bps"] == 0.0
    # HOLD never appears in the counterfactual population nor in the fill population
    assert check_counterfactual_integrity(state)["pass"] is True
    assert check_production_label_integrity(state, now=NOW)["authoritative_without_fill"] == []


# --------------------------------------------------------------------------------------
# dust, accounting, FIFO
# --------------------------------------------------------------------------------------
def test_residual_writeoff_rule_separates_dust_from_a_real_loss():
    assert is_residual_writeoff("DUST_WRITEOFF", -0.001, 100.0, 0.00001) is True
    # a sliver booked at minus its own notional under a normal exit reason is still dust
    assert is_residual_writeoff("TRAILING_STOP_EXIT", -0.772918, 77291.8, 0.00001) is True
    # a genuine 207 bps loss must never be treated as dust
    assert is_residual_writeoff("DAY_4H_STRUCTURE_BREAK_EXIT", -0.9633, 1.4487, 32.1) is False
    assert is_residual_writeoff("TRAILING_STOP_EXIT", -50.0, 100.0, 1.0) is False


def test_accounting_check_reconciles_labels_to_the_close_ledger(tmp_path):
    db = _build_db(tmp_path, groups=2, traded_every=1)
    _write_labels(db)
    result = check_accounting(load_state(db))
    assert "fifo_residual" in result
    assert result["labels_reconciled"] == 2


def test_fifo_residual_reports_unreconciled_lots(tmp_path):
    db = _build_db(tmp_path, groups=1, traded_every=1)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO paper_trades VALUES ('t1','SOL/USDT','BUY',0.47,100.0,0.0009)")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('SOL/USDT',0.0,100.0,'DUST_PENDING')")
    conn.commit()
    conn.close()
    report = fifo_residual_report(load_state(db))
    assert report["per_symbol"]["SOL/USDT"]["unreconciled_qty"] == pytest.approx(0.0009)
    assert report["bps_of_equity"] is not None


# --------------------------------------------------------------------------------------
# time authority and horizon isolation
# --------------------------------------------------------------------------------------
def test_timezone_stability_is_verified_by_the_gate(tmp_path):
    db = _build_db(tmp_path, groups=1, traded_every=1)
    _write_labels(db)
    result = check_time_authority(load_state(db))
    assert result["timezone_stable"] is True
    assert len(set(result["timezone_readings"])) == 1


def test_markout_horizons_are_isolated_from_later_bars():
    """Mutating bars after decision + h must not move the markout at h."""
    decision = 1_788_300_000.0
    bars = [(int(decision + 60 * i), 100.0, 100.0 + i * 0.01, 100.0, 100.0 + i * 0.01, 1.0) for i in range(300)]
    base = label_candidate(
        decision_group_id="g",
        symbol="BTCUSDT",
        decision_epoch=decision,
        entry_px=100.0,
        bars=bars,
        now_epoch=decision + 4 * 3600,
        break_seconds=None,
    )
    mutated = [b if b[0] <= decision + 1800 else (b[0], 9e5, 9e5, 9e5, 9e5, 1.0) for b in bars]
    after = label_candidate(
        decision_group_id="g",
        symbol="BTCUSDT",
        decision_epoch=decision,
        entry_px=100.0,
        bars=mutated,
        now_epoch=decision + 4 * 3600,
        break_seconds=None,
    )
    assert after["markouts"]["15m"] == base["markouts"]["15m"]
    assert after["markouts"]["30m"] == base["markouts"]["30m"]
    # the mutation must actually have bitten, otherwise the test proves nothing
    assert after["markouts"]["1h"] != base["markouts"]["1h"]


def test_market_data_cutoff_covers_the_furthest_markout_consumed():
    decision = 1_788_300_000.0
    bars = [(int(decision + 60 * i), 100.0, 100.5, 99.5, 100.0 + i * 0.01, 1.0) for i in range(300)]
    label = label_candidate(
        decision_group_id="g",
        symbol="BTCUSDT",
        decision_epoch=decision,
        entry_px=100.0,
        bars=bars,
        now_epoch=decision + 4 * 3600,
        break_seconds=None,
        fill={"exit_epoch": decision + 600, "net_bps": 5.0, "gross_bps": 6.0, "exit_reason": "TRAILING_STOP_EXIT", "holding_seconds": 600},
    )
    from datetime import datetime

    cutoff = datetime.fromisoformat(label["market_data_cutoff"]).timestamp()
    furthest = max(sec for name, sec in {"15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400}.items() if label["markouts"].get(name) is not None)
    assert cutoff + 1.0 >= decision + furthest


# --------------------------------------------------------------------------------------
# gate composition
# --------------------------------------------------------------------------------------
def test_feature_availability_start_finds_the_switch_on_point(tmp_path):
    db = _build_db(tmp_path, groups=3, traded_every=1, with_telemetry=False)
    state = load_state(db)
    assert feature_availability_start(state) is None
    (tmp_path / "b").mkdir()
    db2 = _build_db(tmp_path / "b", groups=3, traded_every=1, with_telemetry=True)
    assert feature_availability_start(load_state(db2)) is not None


def test_forward_span_fails_on_thin_data_and_names_the_shortfall(tmp_path):
    db = _build_db(tmp_path, groups=3, traded_every=1)
    _write_labels(db)
    result = check_forward_span(load_state(db), cutoff="2020-01-01T00:00:00+00:00", now=NOW)
    assert result["pass"] is False
    assert result["required_mature_trade_labels"] == result["challenger_feature_count"] * MIN_EVENTS_PER_FEATURE
    assert result["mature_authoritative_trade_labels"] < result["required_mature_trade_labels"]


def test_locked_test_protection_fails_once_inspected(tmp_path):
    db = _build_db(tmp_path, groups=1, traded_every=1)
    assert check_locked_test_protection(load_state(db))["pass"] is True
    conn = sqlite3.connect(db)
    conn.execute("UPDATE day_forward_lock_registry SET inspected=1")
    conn.commit()
    conn.close()
    assert check_locked_test_protection(load_state(db))["pass"] is False


def test_experiment_registry_history_is_preserved(tmp_path):
    db = _build_db(tmp_path, groups=1, traded_every=1)
    state = load_state(db)
    assert len(state["registry"]) >= 12
    assert sum(1 for r in state["registry"] if r.get("promoted")) == 0


def test_evaluate_readiness_reports_reasons_not_a_date(tmp_path):
    db = _build_db(tmp_path, groups=3, traded_every=1)
    _write_labels(db)
    out = evaluate_readiness(db, cutoff="2020-01-01T00:00:00+00:00", now=NOW)
    assert out["ready"] is False
    assert "G_forward_span" in out["reasons_not_ready"]
    assert out["sample_support"]["primary_unit"] == "decision_group"
    assert all(key in out["checks"] for key in ("A_production_label_integrity", "F_accounting", "I_experiment_registry"))


def test_readiness_is_fail_open_on_a_bad_database(tmp_path):
    bad = tmp_path / "missing.db"
    bad.write_text("not a database")
    out = evaluate_readiness(str(bad))
    assert out["ready"] is False
    assert out["reasons_not_ready"]


def test_acceptance_standard_is_documented_and_hold_is_zero():
    standard = acceptance_standard()
    assert standard["hold_value_bps"] == 0.0
    assert "HOLD" in standard["actions"]
    assert any("profit factor" in c for c in standard["criteria"])
    assert any("lock" in c for c in standard["criteria"])


# --------------------------------------------------------------------------------------
# golden behaviour
# --------------------------------------------------------------------------------------
def test_readiness_layer_never_writes_to_the_database(tmp_path):
    """The gate is an auditor. It must not be able to change a single byte."""
    db = _build_db(tmp_path, groups=3, traded_every=1)
    _write_labels(db)
    before = Path(db).stat().st_size
    snapshot = sqlite3.connect(db).execute("SELECT COUNT(*) FROM day_decision_group_records").fetchone()[0]
    evaluate_readiness(db, now=NOW)
    check_accounting(load_state(db))
    after = sqlite3.connect(db).execute("SELECT COUNT(*) FROM day_decision_group_records").fetchone()[0]
    assert Path(db).stat().st_size == before
    assert after == snapshot


def test_readiness_imports_no_trading_module():
    """A trading import here would let an audit failure reach the order path."""
    import backend.services.day_model_readiness as module

    imports = [line for line in Path(module.__file__).read_text().splitlines() if line.startswith(("import ", "from "))]
    joined = "\n".join(imports)
    for forbidden in ("portfolio_engine", "order_executor", "execute_buy", "binance_scalp", "place_order", "day_direct_path_ev"):
        assert forbidden not in joined, f"readiness gate must not import {forbidden}"
