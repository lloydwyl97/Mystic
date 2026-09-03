"""Offline DAY outcome-label runner. Research/observability only.

Covers the pieces that close the forward learning loop: maturity gating, authoritative vs
counterfactual provenance, dust/multi-tranche accounting, timezone-independent bars, and
the linear 4H-break scan that stands in for the quadratic reference implementation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

from backend.services.day_4h_entry_scorecard import breakdowns, build_scorecard, summarize
from backend.services.day_4h_label_runner import (
    RESIDUAL_LOSS_FRACTION,
    authoritative_fill,
    buy_trade_id_from_detail,
    is_residual_writeoff,
    load_1m_bars_utc,
    load_close_ledger,
    parse_epoch_utc,
    pending_groups,
    persist_labels,
    run_label_batch,
    scan_first_4h_break,
    trade_id_epoch,
)
from backend.services.day_4h_outcome_labeler import first_4h_break_seconds, label_candidate, persist_label
from backend.services.day_decision_label_contract import TABLE_LABELS
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS
from backend.services.day_direct_path_ev_authority import select_action

BASE = 1788400000
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _ramp_bars(start: int, count: int, px: float, step: float) -> list[tuple[int, float, float, float, float, float]]:
    bars = []
    for i in range(count):
        close = px + step * i
        bars.append((start + i * 60, close, close + 0.5, close - 0.5, close, 1.0))
    return bars


def _make_db(path, *, groups: int = 2, decision_epoch: int = BASE, with_fill: bool = True) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        f"""
        CREATE TABLE {TABLE_GROUPS} (
            decision_group_id TEXT PRIMARY KEY, created_at TEXT, account_execution_mode TEXT,
            selected_action TEXT, selected_symbol TEXT, selected_ranking_action TEXT,
            execute_authorized INTEGER, lifecycle_state TEXT, schema_version TEXT,
            feature_schema TEXT, model_version TEXT, feature_artifact_ref TEXT,
            slot_count INTEGER, cash_balance REAL, contract_json TEXT, order_id TEXT,
            client_order_id TEXT, fill_trade_id TEXT, maker_taker TEXT, commission REAL,
            commission_asset TEXT
        );
        CREATE TABLE {TABLE_CANDIDATES} (
            decision_group_id TEXT, symbol TEXT, created_at TEXT, eligible INTEGER,
            exclusion_reason TEXT, base_score REAL, p_buy REAL, path_ev REAL,
            rank_deltas_json TEXT, final_rank_score REAL, feature_json TEXT, feature_hash TEXT
        );
        CREATE TABLE feature_ohlcv (
            symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL
        );
        CREATE TABLE position_close_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, closed_at TEXT,
            closed_at_epoch REAL, close_reason TEXT, manual_sell INTEGER,
            realized_profit REAL, realized_profit_unknown INTEGER, cooldown_until TEXT,
            quantity REAL, entry_price REAL, exit_price REAL, sell_trade_id TEXT, detail TEXT
        );
        """
    )
    # 1m tape written the way production writes it: naive, no UTC offset.
    for sym, px in (("BTC-USDT", 80000.0), ("ETH-USDT", 2500.0), ("SOL-USDT", 100.0), ("XRP-USDT", 1.45)):
        for i in range(600):
            ep = decision_epoch - 3600 + i * 60
            close = px * (1.0 + 0.00002 * i)
            conn.execute(
                "INSERT INTO feature_ohlcv(symbol,interval,ts,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)",
                (
                    sym,
                    "1m",
                    datetime.fromtimestamp(ep, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
                    close,
                    close * 1.001,
                    close * 0.999,
                    close,
                    1.0,
                ),
            )
    for g in range(groups):
        gid = f"daygrp_{decision_epoch + g * 900}"
        created = _iso(decision_epoch + g * 900)
        trade_id = f"mystic_XRP/USDT_{(decision_epoch + g * 900) * 1000}"
        contract = {
            "candidates": [{"symbol": s, "p_buy": 0.6, "path_ev": 0.001, "final_rank_score": 0.7, "spread_bps": 0.6} for s in (*COINS, "HOLD")],
            "4h_peer_structure": {"selected_already_broken_at_ranking": False, "healthiest_peer_symbol": "BTCUSDT"},
        }
        if with_fill:
            contract.update({"trade_id": trade_id, "fill_price": 1.45, "filled_qty": 30.0, "commission": 0.01})
        conn.execute(
            f"INSERT INTO {TABLE_GROUPS}(decision_group_id,created_at,selected_action,selected_symbol,lifecycle_state,contract_json) VALUES (?,?,?,?,?,?)",
            (gid, created, "BUY_XRPUSDT", "XRPUSDT", "filled" if with_fill else "ranking_selected", json.dumps(contract)),
        )
        for s in (*COINS, "HOLD"):
            conn.execute(
                f"INSERT INTO {TABLE_CANDIDATES}(decision_group_id,symbol,created_at,p_buy,path_ev,final_rank_score,feature_json) VALUES (?,?,?,?,?,?,?)",
                (gid, s, created, 0.6, 0.001, 0.7, json.dumps({"4h_entry_telemetry": {"distance_to_4h_break_bps": 25.0, "4h_structure_state": "intact"}})),
            )
        if with_fill:
            exit_ep = decision_epoch + g * 900 + 3600
            conn.execute(
                "INSERT INTO position_close_ledger(symbol,closed_at,closed_at_epoch,close_reason,realized_profit,realized_profit_unknown,quantity,entry_price,exit_price,sell_trade_id,detail) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("XRP/USDT", _iso(exit_ep), exit_ep, "TRAILING_STOP_EXIT", 0.30, 0, 30.0, 1.45, 1.46, "sell_1", f"buy_trade_id={trade_id}"),
            )
    conn.commit()
    conn.close()


def test_parse_epoch_utc_is_timezone_independent(monkeypatch):
    """A naive `feature_ohlcv` timestamp must resolve to the same epoch on any host."""
    naive = "2026-09-03 22:36:50.159198"
    expected = datetime(2026, 9, 3, 22, 36, 50, 159198, tzinfo=timezone.utc).timestamp()
    for tz in ("UTC", "America/Chicago", "Asia/Tokyo"):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
        assert parse_epoch_utc(naive) == expected
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


def test_parse_epoch_utc_handles_offsets_and_epochs():
    assert parse_epoch_utc(1788474600) == 1788474600
    assert parse_epoch_utc(1788474600000) == 1788474600
    assert parse_epoch_utc("2026-09-03T22:30:00+00:00") == 1788474600
    assert parse_epoch_utc(None) is None
    assert parse_epoch_utc("not-a-time") is None


def test_load_1m_bars_utc_matches_written_timestamps(tmp_path):
    db = tmp_path / "bars.db"
    _make_db(db, groups=1)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    bars = load_1m_bars_utc(conn, "XRPUSDT")
    conn.close()
    assert bars, "expected XRP bars via the XRP-USDT alias"
    assert bars == sorted(bars, key=lambda b: b[0])
    assert bars[0][0] == BASE - 3600


def test_trade_id_epoch_and_detail_parsing():
    assert trade_id_epoch("mystic_XRP/USDT_1788469212809") == 1788469212.809
    assert trade_id_epoch("no-epoch-here") is None
    assert trade_id_epoch(None) is None
    assert buy_trade_id_from_detail("exit_trigger=X;exit_type=MANUAL;buy_trade_id=abc123") == "abc123"
    assert buy_trade_id_from_detail("no buy id") is None


def test_residual_writeoff_detection():
    """A tranche booked at ~= minus its own notional is dust, not a market outcome."""
    assert is_residual_writeoff("DUST_WRITEOFF", None, None, None) is True
    # -100% of notional, including the fee overshoot seen in production.
    assert is_residual_writeoff("TRAILING_STOP_EXIT", -0.1365409700181179, 1.3654, 0.1) is True
    assert is_residual_writeoff("TRAILING_STOP_EXIT", -0.14535005967090717, 1.4535, 0.1) is True
    # A real, even severe, loss is not a write-off.
    assert is_residual_writeoff("TRAILING_STOP_EXIT", -0.5 * RESIDUAL_LOSS_FRACTION * 43.5, 1.45, 30.0) is False
    assert is_residual_writeoff("NET_PROFIT_EXIT", 0.30, 1.45, 30.0) is False
    assert is_residual_writeoff("X", None, 1.45, 30.0) is False


def test_close_ledger_aggregates_tranches_and_drops_dust(tmp_path):
    db = tmp_path / "closes.db"
    _make_db(db, groups=1)
    conn = sqlite3.connect(str(db))
    tid = f"mystic_XRP/USDT_{BASE * 1000}"
    # Two dust slivers booked at exactly -notional, as production records them.
    for _ in range(2):
        conn.execute(
            "INSERT INTO position_close_ledger(symbol,closed_at,closed_at_epoch,close_reason,realized_profit,realized_profit_unknown,quantity,entry_price,exit_price,sell_trade_id,detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("XRP/USDT", _iso(BASE + 3700), BASE + 3700, "TRAILING_STOP_EXIT", -0.145, 0, 0.1, 1.45, 1.46, "s", f"buy_trade_id={tid}"),
        )
    conn.commit()
    closes = load_close_ledger(conn)
    conn.close()
    agg = closes[tid]
    assert agg["dust_tranches_excluded"] == 2
    assert agg["tranches"] == 1
    assert agg["quantity"] == 30.0
    assert agg["realized_profit"] == 0.30


def test_authoritative_fill_requires_a_real_round_trip(tmp_path):
    db = tmp_path / "fill.db"
    _make_db(db, groups=1)
    conn = sqlite3.connect(str(db))
    closes = load_close_ledger(conn)
    conn.close()
    tid = f"mystic_XRP/USDT_{BASE * 1000}"
    fill = authoritative_fill({"trade_id": tid, "fill_price": 1.45, "commission": 0.01}, closes)
    assert fill is not None
    assert fill["entry_price"] == 1.45
    assert fill["net_bps"] == 0.30 / (1.45 * 30.0) * 1e4
    assert 0 < fill["gross_bps"] < 100
    # No trade id, unknown id, and unknown P&L must never produce an invented fill.
    assert authoritative_fill({}, closes) is None
    assert authoritative_fill({"trade_id": "does-not-exist"}, closes) is None
    closes[tid]["realized_profit_unknown"] = True
    assert authoritative_fill({"trade_id": tid}, closes) is None


def test_scan_first_4h_break_matches_reference_implementation():
    """The linear scan must agree with the quadratic reference on the same tape."""
    for step in (-8.0, 0.0, 12.0):
        bars = _ramp_bars(BASE - 4 * 3600, 400, 80000.0, step)
        decision = bars[300][0]
        end = decision + 3600
        assert scan_first_4h_break(bars, decision_epoch=decision, end_epoch=end) == first_4h_break_seconds(
            bars, decision_epoch=decision, end_epoch=end
        )


def test_scan_first_4h_break_empty_tape():
    assert scan_first_4h_break([], decision_epoch=BASE, end_epoch=BASE + 3600) is None


def test_pending_groups_respects_maturity(tmp_path):
    db = tmp_path / "pending.db"
    _make_db(db, groups=2)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    # Immediately after the decision nothing is mature.
    assert pending_groups(conn, now_epoch=BASE + 60) == []
    mature = pending_groups(conn, now_epoch=BASE + 4 * 3600)
    conn.close()
    assert len(mature) == 2
    assert [g["decision_group_id"] for g in mature] == sorted(g["decision_group_id"] for g in mature)


def test_run_label_batch_labels_every_candidate_and_hold(tmp_path):
    db = tmp_path / "run.db"
    _make_db(db, groups=2)
    summary = run_label_batch(db, now_epoch=BASE + 6 * 3600)
    assert summary["errors"] == 0
    assert summary["groups_scanned"] == 2
    # BTC/ETH/SOL/XRP + HOLD for each group.
    assert summary["labels_written"] == 10
    assert summary["hold"] == 2
    assert summary["authoritative"] == 2

    conn = sqlite3.connect(str(db))
    rows = conn.execute(f"SELECT symbol, provenance FROM {TABLE_LABELS}").fetchall()
    by_symbol = {}
    for sym, prov in rows:
        by_symbol.setdefault(sym, set()).add(prov)
    assert set(by_symbol) == {*COINS, "HOLD"}
    assert by_symbol["HOLD"] == {"authoritative"}
    # Only the executed symbol has a production round trip; the rest are counterfactual.
    assert "authoritative" in by_symbol["XRPUSDT"]
    for loser in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        assert by_symbol[loser] == {"reconstructed"}
    counterfactual = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE_LABELS} WHERE provenance='reconstructed' AND production_exit_net_bps IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert counterfactual == 0, "counterfactual candidates must not report a production exit"


def test_run_label_batch_is_idempotent_and_does_not_touch_decisions(tmp_path):
    db = tmp_path / "idem.db"
    _make_db(db, groups=2)
    conn = sqlite3.connect(str(db))
    before_groups = conn.execute(f"SELECT * FROM {TABLE_GROUPS}").fetchall()
    before_cands = conn.execute(f"SELECT * FROM {TABLE_CANDIDATES}").fetchall()
    conn.close()

    first = run_label_batch(db, now_epoch=BASE + 6 * 3600)
    second = run_label_batch(db, now_epoch=BASE + 6 * 3600)

    conn = sqlite3.connect(str(db))
    assert conn.execute(f"SELECT * FROM {TABLE_GROUPS}").fetchall() == before_groups
    assert conn.execute(f"SELECT * FROM {TABLE_CANDIDATES}").fetchall() == before_cands
    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_LABELS}").fetchone()[0]
    conn.close()
    assert total == first["labels_written"]
    # Settled groups are skipped on the second pass rather than rewritten.
    assert second["labels_written"] == 0
    assert second["errors"] == 0


def test_run_label_batch_is_fail_open(tmp_path):
    """A missing or unreadable database must return a summary, never raise."""
    summary = run_label_batch(tmp_path / "nonexistent.db", now_epoch=BASE + 6 * 3600)
    assert summary["labels_written"] == 0
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    assert run_label_batch(empty, now_epoch=BASE + 6 * 3600)["labels_written"] == 0


def test_immature_group_is_left_alone(tmp_path):
    db = tmp_path / "young.db"
    _make_db(db, groups=1)
    summary = run_label_batch(db, now_epoch=BASE + 60)
    assert summary["groups_scanned"] == 0
    assert summary["labels_written"] == 0


def test_scorecard_reports_new_metrics_and_breakdowns(tmp_path):
    db = tmp_path / "score.db"
    _make_db(db, groups=2)
    run_label_batch(db, now_epoch=BASE + 6 * 3600)
    report = build_scorecard(db, window="30d")
    for key in (
        "cost_cover_rate",
        "BE_rate",
        "trail_rate",
        "MFE",
        "MAE",
        "selected_near_break_distribution",
        "best_coin_selection_rate",
        "regret_vs_best_labeled_candidate",
        "breakdowns",
    ):
        assert key in report, f"missing scorecard metric {key}"
    dims = report["breakdowns"]
    for dim in (
        "symbol",
        "hour_utc",
        "session",
        "4h_structure_state",
        "distance_to_break_bin",
        "path_ev_bin",
        "p_buy_bin",
        "rank_score_bin",
        "spread_state",
        "volatility_state",
        "liquidity_state",
    ):
        assert dim in dims, f"missing breakdown dimension {dim}"
    assert dims["symbol"]["XRPUSDT"]["n"] == 2
    assert report["label_coverage"] == 2
    assert report["label_missing"] == 0


def test_scorecard_handles_no_labels(tmp_path):
    db = tmp_path / "nolabels.db"
    _make_db(db, groups=1)
    report = build_scorecard(db, window="30d")
    assert report["label_coverage"] == 0
    assert report["average_net_bps"] is None
    assert breakdowns([]) != {}


def test_hold_rows_are_never_dropped_from_the_group(tmp_path):
    db = tmp_path / "hold.db"
    _make_db(db, groups=1)
    run_label_batch(db, now_epoch=BASE + 6 * 3600)
    conn = sqlite3.connect(str(db))
    hold = conn.execute(
        f"SELECT production_exit_net_bps, mfe_bps, mae_bps FROM {TABLE_LABELS} WHERE symbol='HOLD'"
    ).fetchone()
    conn.close()
    assert hold == (0.0, 0.0, 0.0), "HOLD must stay explicit with zero economic value"


def test_break_seconds_kwarg_default_is_byte_identical_to_reference():
    """The added `break_seconds` kwarg must not change any label when left unset.

    Default (sentinel) runs the reference scan; passing the precomputed value must give a
    byte-identical payload, so the runner's linear scan is a pure speedup.
    """
    bars = _ramp_bars(BASE - 4 * 3600, 400, 80000.0, -8.0)
    decision = bars[300][0]
    now = decision + 6 * 3600
    default = label_candidate(
        decision_group_id="g1", symbol="BTCUSDT", decision_epoch=decision,
        entry_px=bars[300][4], bars=bars, now_epoch=now,
    )
    precomputed = label_candidate(
        decision_group_id="g1", symbol="BTCUSDT", decision_epoch=decision,
        entry_px=bars[300][4], bars=bars, now_epoch=now,
        break_seconds=scan_first_4h_break(
            bars, decision_epoch=decision, end_epoch=min(now, decision + 8 * 3600)
        ),
    )
    for payload in (default, precomputed):
        payload.pop("label_completed_at", None)
    assert json.dumps(default, sort_keys=True, default=str) == json.dumps(
        precomputed, sort_keys=True, default=str
    )


def test_break_seconds_none_means_no_break_not_unscanned():
    """`None` from a completed scan must be honoured, not treated as 'please rescan'."""
    flat = _ramp_bars(BASE - 4 * 3600, 400, 80000.0, 12.0)
    decision = flat[300][0]
    payload = label_candidate(
        decision_group_id="g", symbol="BTCUSDT", decision_epoch=decision,
        entry_px=flat[300][4], bars=flat, now_epoch=decision + 6 * 3600, break_seconds=None,
    )
    assert payload["time_to_4h_break_sec"] is None
    assert payload["4h_break_within_3m"] is False


def test_golden_ranking_and_exit_unchanged_by_labeling(tmp_path):
    """Running the offline labeler must not perturb ranking or exit outputs."""
    scores = {
        "btc_path_ev": 0.0001, "eth_path_ev": 0.0008, "sol_path_ev": 0.0002,
        "xrp_path_ev": 0.0001, "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
    }
    before = select_action(scores, old_rank_nominee="BTCUSDT", old_rank_score=9.0)
    db = tmp_path / "golden.db"
    _make_db(db, groups=2)
    assert run_label_batch(db, now_epoch=BASE + 6 * 3600)["labels_written"] == 10
    after = select_action(scores, old_rank_nominee="BTCUSDT", old_rank_score=9.0)
    for key in ("selected_action", "selected_symbol", "path_ev_winner", "selected_ev", "why_selected"):
        assert before[key] == after[key]

    def stable(payload: dict) -> str:
        # `select_action` stamps its own wall clock; everything else must match byte for byte.
        return json.dumps(
            {k: v for k, v in payload.items() if "timestamp" not in k and not k.endswith("_at")},
            sort_keys=True,
            default=str,
        )

    assert stable(before) == stable(after)


def test_label_runner_writes_only_the_label_table(tmp_path):
    """Nothing outside `day_decision_outcome_labels` may change."""
    db = tmp_path / "isolation.db"
    _make_db(db, groups=2)
    conn = sqlite3.connect(str(db))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    before = {t: conn.execute(f"SELECT * FROM '{t}'").fetchall() for t in tables}
    conn.close()

    run_label_batch(db, now_epoch=BASE + 6 * 3600)

    conn = sqlite3.connect(str(db))
    after_tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        assert conn.execute(f"SELECT * FROM '{t}'").fetchall() == before[t], f"{t} was modified"
    conn.close()
    new_tables = set(after_tables) - set(tables)
    assert new_tables <= {TABLE_LABELS, "sqlite_sequence"}


def test_batch_persist_matches_persist_label(tmp_path):
    """The batched writer must store exactly what the per-row writer stores."""
    bars = _ramp_bars(BASE - 4 * 3600, 400, 80000.0, -8.0)
    decision = bars[300][0]
    payloads = [
        label_candidate(
            decision_group_id=f"g{i}", symbol=sym, decision_epoch=decision,
            entry_px=bars[300][4], bars=bars, now_epoch=decision + 6 * 3600, break_seconds=None,
        )
        for i, sym in enumerate(COINS)
    ]

    ref = tmp_path / "ref.db"
    for payload in payloads:
        persist_label(ref, payload)
    batch = tmp_path / "batch.db"
    assert persist_labels(batch, payloads) == len(payloads)

    cols = (
        "decision_group_id, symbol, provenance, markout_15m_net_bps, markout_30m_net_bps, "
        "markout_1h_net_bps, markout_2h_net_bps, markout_4h_net_bps, mfe_bps, mae_bps, "
        "time_to_mfe_sec, time_to_mae_sec, cost_cover, production_exit_gross_bps, "
        "commission_bps, spread_bps, slippage_bps, production_exit_net_bps, holding_time_sec, "
        "capture_ratio, exit_reason, regret_vs_best_eligible_bps, regret_vs_hold_bps, label_json"
    )
    query = f"SELECT {cols} FROM {TABLE_LABELS} ORDER BY decision_group_id, symbol"
    ref_rows = sqlite3.connect(str(ref)).execute(query).fetchall()
    batch_rows = sqlite3.connect(str(batch)).execute(query).fetchall()
    assert len(ref_rows) == len(payloads)
    for ref_row, batch_row in zip(ref_rows, batch_rows, strict=True):
        assert ref_row[:-1] == batch_row[:-1]
        assert json.loads(ref_row[-1]) == json.loads(batch_row[-1])


def test_persist_labels_is_fail_open(tmp_path):
    assert persist_labels(tmp_path / "missing_dir" / "x.db", [{"decision_group_id": "g", "symbol": "BTCUSDT"}]) == 0
    assert persist_labels(tmp_path / "ok.db", []) == 0


def test_summarize_empty_is_safe():
    out = summarize([])
    assert out["decision_groups"] == 0
    assert out["selected_trade_count"] == 0
    assert out["average_net_bps"] is None
