import json
import sqlite3
from datetime import datetime, timezone

from backend.services.day_4h_entry_scorecard import build_scorecard, consistency_table, summarize
from backend.services.day_4h_outcome_labeler import (
    LABEL_VERSION,
    clip_bars_asof,
    hold_label,
    horizon_mature,
    label_candidate,
    persist_label,
    provenance_or_unknown,
)
from backend.services.day_decision_label_contract import TABLE_LABELS, write_outcome_label
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS, build_group_contract, record_day_ranking_group
from backend.services.day_direct_path_ev_authority import select_action
from backend.services.day_experiment_registry import SEED_ARMS, registry, seed_historical
from backend.services.day_forward_lock import FORWARD_LOCK_START, HISTORICAL_66_END, challenger_export_schema, register_lock
from backend.services.sqlite_large_table_retention import PROTECTED_TABLES, RETENTION_POLICIES


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def test_horizon_maturity_and_provenance():
    t0 = _ts("2026-09-03T00:00:00+00:00")
    assert horizon_mature(decision_epoch=t0, horizon_sec=900, now_epoch=t0 + 899) is False
    assert horizon_mature(decision_epoch=t0, horizon_sec=900, now_epoch=t0 + 900) is True
    assert provenance_or_unknown("authoritative") == "authoritative"
    assert provenance_or_unknown("magic") == "unknown"


def test_clip_bars_rejects_future():
    bars = [(100, 1, 2, 0.5, 1.5, 1.0), (200, 1.5, 3, 1, 2, 1.0)]
    assert [b[0] for b in clip_bars_asof(bars, 150)] == [100]


def test_hold_label_zeros_and_maturity():
    t0 = _ts("2026-09-03T00:00:00+00:00")
    early = hold_label(decision_group_id="g1", decision_epoch=t0, now_epoch=t0 + 600)
    assert early["symbol"] == "HOLD"
    assert early["production_exit_net_bps"] == 0.0
    assert early["markouts"]["15m"] is None
    late = hold_label(decision_group_id="g1", decision_epoch=t0, now_epoch=t0 + 4 * 3600)
    assert late["markouts"]["15m"] == 0.0
    assert late["label_version"] == LABEL_VERSION


def test_label_candidate_markouts_mfe_and_rapid_break(tmp_path):
    t0 = int(_ts("2026-08-28T04:00:00+00:00"))
    bars = []
    px = 100.0
    for i in range(40):
        ep = t0 + i * 60
        close = 100.0 - i * 0.05
        bars.append((ep, px, max(px, close), min(px, close), close, 1.0))
        px = close
    payload = label_candidate(
        decision_group_id="g2",
        symbol="ETHUSDT",
        decision_epoch=float(t0),
        entry_px=100.0,
        bars=bars,
        now_epoch=float(t0 + 3600),
        fill={
            "exit_epoch": t0 + 120,
            "net_bps": -8.0,
            "net_dollars": -0.4,
            "exit_reason": "DAY_4H_STRUCTURE_BREAK_EXIT",
            "holding_seconds": 120,
            "commission_bps": 4.0,
            "spread_bps": 1.0,
            "slippage_bps": 1.0,
            "gross_bps": -2.0,
        },
    )
    assert payload["provenance"] == "authoritative"
    assert payload["4h_break_within_3m"] in {True, False}
    assert payload["markouts"]["15m"] is not None
    assert payload["mfe_bps"] is not None
    assert payload["mae_bps"] is not None
    assert payload["counterfactual"] is False
    db = tmp_path / "labels.db"
    persist_label(db, payload)
    conn = sqlite3.connect(db)
    row = conn.execute(f"SELECT provenance, production_exit_net_bps FROM {TABLE_LABELS}").fetchone()
    conn.close()
    assert row == ("authoritative", -8.0)


def test_unselected_is_counterfactual_not_a_fill():
    t0 = int(_ts("2026-08-28T04:00:00+00:00"))
    bars = [(t0 + i * 60, 100.0, 100.2, 99.8, 100.1, 1.0) for i in range(20)]
    payload = label_candidate(
        decision_group_id="g3",
        symbol="BTCUSDT",
        decision_epoch=float(t0),
        entry_px=100.0,
        bars=bars,
        now_epoch=float(t0 + 1800),
    )
    assert payload["counterfactual"] is True
    assert payload["provenance"] == "reconstructed"
    assert payload["production_exit_net_bps"] is None


def test_scorecard_consistency_and_regret():
    rows = [
        {
            "selected_action": "BUY_ETHUSDT",
            "selected_symbol": "ETHUSDT",
            "execute_authorized": 1,
            "fill_trade_id": "mystic_ETH/USDT_1",
            "contract": {
                "4h_peer_structure": {
                    "selected_already_broken_at_ranking": True,
                    "selected_broken_peer_intact_flag": True,
                    "all_four_already_broken": False,
                    "healthiest_peer_symbol": "BTCUSDT",
                    "healthiest_peer_distance_bps": 80.0,
                    "selected_vs_best_peer_distance_bps": 75.0,
                },
                "4h_entry_telemetry": {"ETHUSDT": {"distance_to_4h_break_bps": 5.0}},
            },
            "labels": {
                "ETHUSDT": {
                    "provenance": "authoritative",
                    "production_exit_net_bps": -7.0,
                    "regret_vs_hold_bps": -7.0,
                    "mfe_bps": 0.0,
                    "mae_bps": -4.0,
                    "covered_genuine_cost": False,
                    "reached_production_BE_level": False,
                    "reached_production_trail_level": False,
                    "4h_break_within_3m": True,
                    "4h_break_within_15m": True,
                    "4h_break_within_30m": True,
                }
            },
        },
        {
            "selected_action": "HOLD",
            "selected_symbol": "HOLD",
            "contract": {"4h_peer_structure": {"all_four_already_broken": True}},
            "labels": {"HOLD": {"production_exit_net_bps": 0.0, "regret_vs_hold_bps": 0.0}},
        },
    ]
    summary = summarize(rows)
    assert summary["decision_groups"] == 2
    assert summary["selected_trade_count"] == 1
    assert summary["selected_HOLD_count"] == 1
    assert summary["regret_vs_HOLD"] == -7.0
    table = consistency_table(rows)
    assert table["selected_already_broken_at_decision"]["n"] == 1
    assert table["selected_breaks_within_3m"]["n"] == 1


def test_observability_persists_4h_and_does_not_change_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "true")
    decision = select_action(
        {
            "btc_path_ev": 0.0001,
            "eth_path_ev": 0.0008,
            "sol_path_ev": 0.0002,
            "xrp_path_ev": 0.0001,
            "path_net_status": "predicted",
            "path_net_model_id": "day_path_net_v1",
        }
    )
    before = json.dumps(decision, sort_keys=True, default=str)
    db = tmp_path / "obs.db"
    gid = record_day_ranking_group(str(db), decision=decision, bar_timestamp=99)
    after = json.dumps(decision, sort_keys=True, default=str)
    assert before == after
    assert gid
    conn = sqlite3.connect(db)
    symbols = {r[0] for r in conn.execute(f"SELECT symbol FROM {TABLE_CANDIDATES} WHERE decision_group_id=?", (gid,))}
    contract = json.loads(conn.execute(f"SELECT contract_json FROM {TABLE_GROUPS} WHERE decision_group_id=?", (gid,)).fetchone()[0])
    conn.close()
    assert symbols == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"}
    assert "4h_entry_telemetry" in contract
    assert "selected_already_broken_at_ranking" in contract


def test_registry_and_forward_lock_do_not_reset(tmp_path):
    db = tmp_path / "reg.db"
    n1 = seed_historical(db)
    n2 = seed_historical(db)
    assert n1 == len(SEED_ARMS)
    assert n2 == 0
    info = registry(db)
    assert info["arm_count"] >= 12
    assert info["reset"] is False
    assert info["promoted_count"] == 0
    lock = register_lock(db)
    assert lock["historical_66_excluded"] is True
    assert lock["dataset_cutoff"] == FORWARD_LOCK_START
    assert HISTORICAL_66_END < FORWARD_LOCK_START
    schema = challenger_export_schema()
    assert schema["train"] is False
    assert "production_exit_net_bps" in schema["targets"]


def test_retention_keeps_90_days_for_learning_tables():
    keep = {p.table: p.keep_days for p in RETENTION_POLICIES}
    assert keep["day_decision_outcome_labels"] == 90
    assert keep["day_decision_group_records"] == 90


def test_experiment_and_lock_registries_are_protected_not_expired():
    """Sampled learning rows age out; the record of what was tried must not.

    These two tables describe experiments and locks rather than sampling them, so deleting
    them on a timer would erase the evidence needed to reproduce a sealed result.
    """
    keep = {p.table: p.keep_days for p in RETENTION_POLICIES}
    assert "day_experiment_registry" not in keep
    assert "day_forward_lock_registry" not in keep
    assert "day_clock_v2_partition_registry" not in keep
    assert "day_clock_v2_outcome_labels" not in keep
    assert set(PROTECTED_TABLES) == {
        "day_experiment_registry",
        "day_forward_lock_registry",
        "day_path_clock_feature_snapshots",
        "day_path_clock_readiness_history",
        "day_path_clock_v2_candidate_artifact",
        "day_path_clock_v2_readiness_history",
        "day_clock_v2_partition_registry",
        "day_clock_v2_outcome_labels",
    }


def test_report_window_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    report = build_scorecard(db, window="24h")
    assert report["decision_groups"] == 0
    assert report["window"] == "24h"


def test_write_outcome_label_still_works(tmp_path):
    write_outcome_label(
        tmp_path / "l.db",
        {
            "decision_group_id": "g",
            "symbol": "HOLD",
            "provenance": "authoritative",
            "markouts": {"15m": 0, "30m": 0, "1h": 0, "2h": 0, "4h": 0},
            "production_exit_net_bps": 0.0,
            "regret_vs_hold_bps": 0.0,
        },
    )


def test_build_group_contract_does_not_mutate_scores():
    decision = select_action(
        {
            "btc_path_ev": 0.0004,
            "eth_path_ev": 0.0001,
            "sol_path_ev": 0.0002,
            "xrp_path_ev": 0.00005,
            "path_net_status": "predicted",
        }
    )
    snap = {k: decision[k] for k in ("selected_action", "selected_symbol", "btc_path_ev", "eth_path_ev", "sol_path_ev", "xrp_path_ev")}
    contract = build_group_contract(decision=decision, bar_timestamp=7)
    assert {k: decision[k] for k in snap} == snap
    assert contract["selected_action"] == decision["selected_action"]
    assert len(contract["candidates"]) == 5
