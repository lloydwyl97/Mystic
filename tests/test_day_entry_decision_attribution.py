import json
import sqlite3
import tempfile
from pathlib import Path

from backend.services.day_entry_decision_attribution import backfill_day_buy_attribution
from backend.services.entry_decision_authority import build_day_entry_provenance


def test_provenance_stamps_join_fields():
    prov = build_day_entry_provenance(
        decision_data={
            "path_net_status": "predicted",
            "forward_net_model_version": "day_path_net_v1",
            "selected_net_expected_value": 0.00101,
            "prob_buy": 0.41,
            "prob_sell": 0.33,
            "prob_hold": 0.26,
            "ml_score": -0.348,
            "setup_type": "VWAP_REVERSION",
            "ai_inference_log_id": "inf-12",
            "features_json_hash": "abc123",
            "prior_4h_low": 1.3588,
            "distance_to_prior_4h_low_bps": 6.6,
            "universe_rank": 1,
        },
        symbol="XRP/USDT",
        decision_id="dec-xrp-1945",
    )
    assert prov["decision_id"] == "dec-xrp-1945"
    assert prov["p_buy"] == 0.41
    assert prov["ml_score"] == -0.348
    assert prov["ai_inference_log_id"] == "inf-12"
    assert prov["features_json_hash"] == "abc123"
    assert prov["distance_to_prior_4h_low_bps"] == 6.6
    assert prov["attribution_join_status"] == "live"


def test_backfill_marks_ambiguous_instead_of_guessing():
    path = Path(tempfile.mkdtemp()) / "attr.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                trade_id TEXT, symbol TEXT, side TEXT, timestamp TEXT,
                decision_id TEXT, explainability_json TEXT, strategy_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE day_decision_records (
                decision_id TEXT, created_at TEXT, symbol TEXT, ml_score REAL, detail_json TEXT
            )
            """
        )
        conn.execute("INSERT INTO paper_trades VALUES ('buy1','XRP/USDT','BUY','2026-09-01T19:45:00','','{}','day')")
        conn.execute("INSERT INTO day_decision_records VALUES ('d1','2026-09-01T19:44:00','XRP/USDT',-0.3,'{}')")
        conn.execute("INSERT INTO day_decision_records VALUES ('d2','2026-09-01T19:45:00','XRP/USDT',-0.2,'{}')")
        conn.commit()
    rows = backfill_day_buy_attribution(str(path), trade_date="2026-09-01")
    assert rows[0]["status"] == "ambiguous"
    with sqlite3.connect(str(path)) as conn:
        expl = json.loads(conn.execute("SELECT explainability_json FROM paper_trades").fetchone()[0])
    assert expl["attribution_join_status"] == "ambiguous"
