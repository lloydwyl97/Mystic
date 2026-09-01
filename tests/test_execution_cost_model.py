from backend.config.execution_cost_model import (
    ARM_B_MIN_PREDICTED_GROSS_PCT,
    bnb_savings_usd,
    expected_exchange_commission_rt_pct,
    honest_all_in_rt_pct,
    named_cost_breakdown,
    stamp_named_costs,
)


def test_named_fields_are_separate_from_legacy_veto():
    br = named_cost_breakdown("BTCUSDT", p_buy=0.20, p_sell=0.0)
    assert br.expected_exchange_commission == expected_exchange_commission_rt_pct()
    assert br.expected_spread > 0
    assert br.expected_slippage > 0
    assert br.predicted_gross_trade_value == 0.20 * 0.012
    assert abs(br.predicted_net_trade_value - (br.predicted_gross_trade_value - (br.expected_exchange_commission + br.expected_spread + br.expected_slippage))) < 1e-12
    assert br.min_executable_net_ev == 0.0
    assert ARM_B_MIN_PREDICTED_GROSS_PCT == 0.0022


def test_stamp_does_not_overwrite_veto_inputs():
    dd = {"estimated_fees_pct": 0.001, "estimated_slippage_pct": 0.0008, "spread_pct": 0.0004, "prob_buy": 0.5}
    stamp_named_costs(dd, "ETHUSDT")
    assert dd["estimated_fees_pct"] == 0.001
    assert dd["expected_exchange_commission"] == expected_exchange_commission_rt_pct()
    assert "predicted_net_trade_value" in dd


def test_honest_cost_is_below_legacy_22bps():
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        assert honest_all_in_rt_pct(sym) < 0.0022


def test_bnb_savings_at_requested_sizes():
    assert abs(bnb_savings_usd(notional_usd=625) - 625 * 0.0004 * 0.05) < 1e-12
    assert abs(bnb_savings_usd(notional_usd=2500) - 2500 * 0.0004 * 0.05) < 1e-12
