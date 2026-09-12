import backend.services.decision_book_tape as dbt
from backend.services.day_decision_shadow import champion_shadow_fields, merge_shadow_extras
from backend.services.day_direct_path_ev_authority import HOLD_EV
from backend.services.decision_book_tape import record_day_bar_authority


def test_shadow_does_not_change_selected_action():
    decision = {
        "selected_action": "HOLD",
        "selected_symbol": "",
        "selected_ev": 0.0,
        "path_ev_winner": "HOLD",
        "why_selected": "HOLD_WINS",
        "btc_path_ev": -0.0001,
        "eth_path_ev": -0.0002,
        "sol_path_ev": -0.0003,
        "xrp_path_ev": -0.0004,
        "path_net_model_id": "day_path_net_v1",
        "prob_buy": 0.2,
    }
    extras = {
        "selected_action": "HOLD",
        "path_ev_winner": "HOLD",
        "btc_path_ev": -0.0001,
        "eth_path_ev": -0.0002,
        "sol_path_ev": -0.0003,
        "xrp_path_ev": -0.0004,
        "hold_ev": 0.0,
        "why_selected": "HOLD_WINS",
    }
    merged = merge_shadow_extras(extras, decision)
    assert merged["selected_action"] == "HOLD"
    assert merged["path_ev_winner"] == "HOLD"
    assert merged["challenger_status"] == "not_promoted"
    assert merged["champion_hold_value"] == HOLD_EV
    fields = champion_shadow_fields(decision)
    assert fields["champion_selected_action"] == "HOLD"
    assert decision["selected_action"] == "HOLD"


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
