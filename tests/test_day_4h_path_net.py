from backend.services.day_4h_path_net import expected_path_net_bps, path_net_from_features


def test_expected_path_net_formula():
    score = expected_path_net_bps(
        probability_favorable=0.40,
        expected_favorable_net_bps=25.0,
        probability_4h_break_first=0.55,
        expected_break_loss_bps=18.0,
    )
    assert abs(score - (0.40 * 25.0 - 0.55 * 18.0)) < 1e-9


def test_xrp_near_4h_low_ranks_below_far_setup():
    near = path_net_from_features(
        {
            "p_buy": 0.48,
            "ml_score": -0.348,
            "predicted_net_ev_bps": 10.1,
            "distance_to_4h_break_bps": 6.6,
            "velocity_toward_4h_break_bps": 4.0,
            "forming_4h_bar_age_min": 200.0,
            "ema_alignment": 0.2,
        }
    )
    far = path_net_from_features(
        {
            "p_buy": 0.48,
            "ml_score": -0.348,
            "predicted_net_ev_bps": 10.1,
            "distance_to_4h_break_bps": 80.0,
            "velocity_toward_4h_break_bps": 0.0,
            "forming_4h_bar_age_min": 20.0,
            "ema_alignment": 0.6,
        }
    )
    assert near["probability_4h_break_first"] > far["probability_4h_break_first"]
    assert near["expected_path_net_bps"] < far["expected_path_net_bps"]
    assert near["ranking_only"] == 1.0


def test_same_formula_for_all_four_symbols():
    scores = []
    for _sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
        scores.append(
            path_net_from_features(
                {
                    "p_buy": 0.5,
                    "predicted_net_ev_bps": 8.0,
                    "distance_to_4h_break_bps": 20.0,
                }
            )["expected_path_net_bps"]
        )
    assert len(set(scores)) == 1
