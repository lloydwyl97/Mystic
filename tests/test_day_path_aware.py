"""DAY path-aware exit and HOLD-as-action EV."""

from __future__ import annotations

import pytest

from backend.services.day_controlled_exits import (
    EXIT_NET_PROFIT,
    EXIT_PATH_EXECUTABLE_PROFIT,
    EXIT_STALL_DEAD,
    EXIT_TIME_STOP,
    evaluate_engine_managed_exit,
)
from backend.services.day_path_net import (
    predict_decision_net,
    reset_day_artifact_cache,
    resolve_day_path_ev,
    stamp_day_path_prediction,
)


class _Pos:
    def __init__(self, **kw):
        self.entry_price = kw.get("entry_price", 100.0)
        self.highest_price = kw.get("highest_price", 100.0)
        self.lowest_price = kw.get("lowest_price", 99.50)
        self.stop_price = kw.get("stop_price", 99.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.005)
        self.take_profit_1_price = kw.get("take_profit_1_price", 0.0)
        self.entry_thesis = kw.get("entry_thesis", "HTF_TREND_PULLBACK")
        self.entry_vwap = kw.get("entry_vwap", 100.0)
        self.thesis_invalid_level = kw.get("thesis_invalid_level", 99.0)
        self.thesis_target_level = kw.get("thesis_target_level", 101.0)
        self.thesis_score = kw.get("thesis_score", 0.7)
        self.max_hold_min = kw.get("max_hold_min", 360)
        self.day_route_regime_at_entry = kw.get("day_route_regime_at_entry", "")


@pytest.fixture(autouse=True)
def _path_aware_on(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    monkeypatch.setenv("DAY_PATH_MIN_EXECUTABLE_NET_PCT", "0.0001")


def test_path_aware_takes_first_executable_net():
    out = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=100.08,
        net_pnl_pct=0.0006,
        hold_minutes=12.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_PATH_EXECUTABLE_PROFIT


def test_path_aware_uses_net_profit_label_at_floor():
    out = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=100.50,
        net_pnl_pct=0.0045,
        hold_minutes=20.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_NET_PROFIT


def test_path_aware_does_not_stall_red():
    out = evaluate_engine_managed_exit(
        position=_Pos(highest_price=100.05, lowest_price=99.60, stop_price=0.0, thesis_invalid_level=0.0),
        current_price=99.65,
        net_pnl_pct=-0.0045,
        hold_minutes=130.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "hold"
    assert out["reason"] == "PATH_AWARE_HOLD"
    assert out.get("reason") != EXIT_STALL_DEAD


def test_path_aware_holds_loser_to_horizon():
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, max_hold_min=360),
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=60.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "hold"


def test_path_aware_max_hold_still_exits():
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, max_hold_min=300),
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=300.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP


def test_predict_without_artifact_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("DAY_PATH_NET_ARTIFACT", str(tmp_path / "missing.json"))
    reset_day_artifact_cache()
    assert predict_decision_net({"bars_1m": [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}] * 20}) is None
    ev, stamped = resolve_day_path_ev({})
    assert ev is None
    assert "forward_net_model_version" not in stamped
    reset_day_artifact_cache()


def test_accepted_artifact_without_bars_is_hold_not_invented(monkeypatch):
    reset_day_artifact_cache()
    ev, stamped = resolve_day_path_ev({"buy_margin": 0.20, "confidence": 0.90, "prob_buy": 0.80})
    assert ev == 0.0
    assert stamped["path_net_status"] == "unavailable_hold"
    assert stamped["forward_net_model_version"] == "day_path_net_v1"
    assert stamped["predicted_net_return"] == 0.0
    reset_day_artifact_cache()


def test_accepted_artifact_with_bars_stamps_model_version():
    reset_day_artifact_cache()
    bars = []
    price = 100.0
    for i in range(40):
        price *= 1.0 + (0.0004 if i % 3 == 0 else -0.0002)
        bars.append({"open": price, "high": price * 1.001, "low": price * 0.999, "close": price, "volume": 10.0, "ts": 1786750000 + i * 60})
    ev, stamped = resolve_day_path_ev({"bars_1m": bars, "symbol": "ETHUSDT"})
    assert ev is not None
    assert stamped["path_net_status"] == "predicted"
    assert stamped["forward_net_model_version"] == "day_path_net_v1"
    reset_day_artifact_cache()
