"""A filled close teaches the executable net, not the ticker minus a flat cost."""

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.services.adaptive_learning import close_learning_net_pct


def test_fill_net_is_the_learning_target_when_the_ticker_equals_entry():
    """ETH scalp 2026-10-08 19:15: the ticker mark equalled the entry and the
    learner was taught exactly -flat cost. The fill net is the target."""
    fill_net = -0.0005060746009779401
    learned = close_learning_net_pct(
        fill_net_pct=fill_net,
        exit_price=2451.54,
        entry_price=2451.54,
        flat_cost=float(ESTIMATED_ROUNDTRIP_COST),
    )
    assert learned == fill_net
    assert learned != -float(ESTIMATED_ROUNDTRIP_COST)


def test_flat_zero_fill_is_kept():
    assert close_learning_net_pct(fill_net_pct=0.0, exit_price=100.0, entry_price=99.0, flat_cost=0.00066) == 0.0


def test_missing_fill_falls_back_to_mark_minus_one_flat_cost():
    learned = close_learning_net_pct(fill_net_pct=None, exit_price=100.25, entry_price=100.0, flat_cost=0.00066)
    assert abs(learned - (0.0025 - 0.00066)) < 1e-12


def test_non_finite_fill_does_not_replace_the_fallback():
    learned = close_learning_net_pct(fill_net_pct=float("nan"), exit_price=100.0, entry_price=100.0, flat_cost=0.00066)
    assert abs(learned - (-0.00066)) < 1e-12
