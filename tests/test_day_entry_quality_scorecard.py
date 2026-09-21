from backend.services.day_entry_quality_scorecard import scorecard_from_labeled_groups


def test_scorecard_is_offline_and_does_not_authorize_trades():
    groups = [
        {
            "selected_symbol": "ETHUSDT",
            "selected_final_score": 0.0008,
            "predicted_net_bps": 4.0,
            "labels": {
                "ETHUSDT": {"net_bps": -12.0, "cost_cover": False},
                "BTCUSDT": {"net_bps": 8.0, "cost_cover": True},
                "HOLD": {"net_bps": 0.0},
            },
        },
        {
            "selected_symbol": "HOLD",
            "labels": {
                "ETHUSDT": {"net_bps": -5.0, "cost_cover": False},
                "HOLD": {"net_bps": 0.0},
            },
        },
    ]
    out = scorecard_from_labeled_groups(groups)
    assert out["live_feed"] is False
    assert out["n_groups"] == 2
    assert out["negative_entry_rate"] == 0.5
    assert "opportunity_rate" in out
    assert "regret_vs_HOLD" in out
    assert out["profit_by_symbol"]["ETHUSDT"] == -12.0


def test_empty_scorecard():
    assert scorecard_from_labeled_groups([])["n_groups"] == 0
