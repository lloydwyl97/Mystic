from backend.services.day_liquidity_gate import apply_liquidity_gate_to_decision_data
from backend.services.spread_book_telemetry import shadow_liquidity_compare


def test_shadow_does_not_change_live_liquidity_factor():
    dd = {"spread_pct": 0.0004}
    live = apply_liquidity_gate_to_decision_data(dict(dd), "BTC/USDT")
    before = live["liquidity_quality_size_factor"]
    shadow = shadow_liquidity_compare(
        symbol="BTCUSDT",
        current_decision_data=dd,
        real_spread_bps=0.59,
        current_notional_usd=4000.0,
    )
    after = apply_liquidity_gate_to_decision_data(dict(dd), "BTC/USDT")
    assert after["liquidity_quality_size_factor"] == before
    assert shadow["live_sizing_unchanged"] is True
    assert shadow["proposed_real_spread_liquidity_credit"] >= shadow["current_fallback_liquidity_credit"]
    assert shadow["difference_usd"] != 0.0 or shadow["proposed_position_size_usd"] == 4000.0
