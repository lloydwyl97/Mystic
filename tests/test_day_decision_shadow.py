import backend.services.decision_book_tape as dbt
from backend.services.decision_book_tape import record_day_bar_authority

# The former shadow-merge test was removed: it imported
# backend.services.day_decision_shadow, which was never committed, so it broke
# collection for the entire suite rather than testing anything real.


def test_record_day_bar_authority_still_does_not_flip_action(monkeypatch, tmp_path):
    path = str(tmp_path / "tape.db")
    monkeypatch.setattr(dbt, "DATABASE_PATH", path)
    monkeypatch.setattr(dbt, "_TABLE_READY", False)
    monkeypatch.setattr(dbt, "_LAST_QUIET", {})
    decision = {
        "selected_action": "HOLD",
        "selected_symbol": "",
        "selected_ev": 0.0,
        "path_ev_winner": "HOLD",
        "why_selected": "HOLD_WINS",
        "btc_path_ev": -0.1,
        "eth_path_ev": -0.2,
        "sol_path_ev": -0.3,
        "xrp_path_ev": -0.4,
        "path_net_model_id": "day_path_net_v1",
        "prediction_timestamp": "2026-09-01T12:00:00+00:00",
    }
    n = record_day_bar_authority(decision, redis_client=None)
    assert n >= 0
    assert decision["selected_action"] == "HOLD"
    assert decision["path_ev_winner"] == "HOLD"
