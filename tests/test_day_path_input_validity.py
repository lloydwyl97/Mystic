"""Sparse/stale/gap path-EV input safety. No live OOD or btc_ret_5 change."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.services.day_direct_path_ev_authority import select_action
from backend.services.day_path_input_validity import (
    PATH_INPUT_INVALID_GAP,
    PATH_INPUT_INVALID_SPARSE,
    PATH_INPUT_INVALID_STALE,
    validate_path_bars,
)
from backend.services.day_path_net import (
    predict_decision_net,
    reset_day_artifact_cache,
    resolve_day_path_ev,
)


def _dense_bars(n: int = 40, start: datetime | None = None, close0: float = 100.0) -> list[dict]:
    t0 = start or datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    bars = []
    px = close0
    for i in range(n):
        px *= 1.0001 if i % 4 == 0 else 0.99995
        ts = t0 + timedelta(seconds=60 * i)
        bars.append({"open": px, "high": px * 1.0002, "low": px * 0.9998, "close": px, "volume": 10.0, "ts": ts})
    return bars


def _sparse_hole_bars(close0: float = 1.3995) -> list[dict]:
    t0 = datetime(2026, 9, 4, 8, 25, 15, tzinfo=timezone.utc)
    bars = []
    px = close0 * 1.035
    for i in range(20):
        ts = t0 + timedelta(seconds=30 * i)
        bars.append({"open": px, "high": px, "low": px, "close": px, "volume": 1.0, "ts": ts})
    t1 = datetime(2026, 9, 4, 23, 25, 15, tzinfo=timezone.utc)
    px = close0
    for i in range(20):
        ts = t1 + timedelta(seconds=30 * i)
        bars.append({"open": px, "high": px, "low": px, "close": px, "volume": 1.0, "ts": ts})
    return bars


def test_dense_window_valid():
    bars = _dense_bars()
    tel = validate_path_bars(bars, as_of=bars[-1]["ts"])
    assert tel["path_input_valid"] is True
    assert tel["path_invalid_reason"] is None


def test_sparse_hole_invalid_for_every_top4_symbol():
    for close0 in (80000.0, 2500.0, 100.0, 1.3995):
        bars = _sparse_hole_bars(close0)
        tel = validate_path_bars(bars, as_of=bars[-1]["ts"])
        assert tel["path_input_valid"] is False
        assert tel["path_invalid_reason"] in {PATH_INPUT_INVALID_GAP, PATH_INPUT_INVALID_SPARSE}
        assert float(tel["path_actual_lookback_seconds"]) > 14 * 3600


def test_stale_last_bar_invalid():
    bars = _dense_bars()
    tel = validate_path_bars(bars, as_of=bars[-1]["ts"] + timedelta(seconds=181))
    assert tel["path_input_valid"] is False
    assert tel["path_invalid_reason"] == PATH_INPUT_INVALID_STALE


def test_duplicate_timestamps_invalid():
    bars = _dense_bars()
    bars[10]["ts"] = bars[9]["ts"]
    tel = validate_path_bars(bars, as_of=bars[-1]["ts"])
    assert tel["path_input_valid"] is False
    assert tel["path_invalid_reason"] == "PATH_INPUT_INVALID_SCHEMA"


def test_out_of_order_timestamps_invalid():
    bars = _dense_bars()
    bars[10]["ts"], bars[11]["ts"] = bars[11]["ts"], bars[10]["ts"]
    tel = validate_path_bars(bars, as_of=max(b["ts"] for b in bars))
    assert tel["path_input_valid"] is False
    assert tel["path_invalid_reason"] == "PATH_INPUT_INVALID_SCHEMA"


def test_dense_path_ev_equals_legacy_predict():
    reset_day_artifact_cache()
    bars = _dense_bars()
    dd = {"bars_1m": bars, "symbol": "ETHUSDT", "btc_ret_5": 0.0}
    legacy = predict_decision_net(dd)
    ev, stamped = resolve_day_path_ev(dd, symbol="ETHUSDT")
    assert legacy is not None
    assert ev == legacy
    assert stamped["path_input_valid"] is True
    assert stamped["path_net_status"] == "predicted"
    assert stamped["legacy_btc_ret_5"] == 0.0
    reset_day_artifact_cache()


def test_sparse_cannot_emit_authority_for_btc_eth_sol_xrp():
    reset_day_artifact_cache()
    for symbol, close0 in (("BTCUSDT", 80000.0), ("ETHUSDT", 2500.0), ("SOLUSDT", 100.0), ("XRPUSDT", 1.3995)):
        bars = _sparse_hole_bars(close0)
        legacy = predict_decision_net({"bars_1m": bars, "btc_ret_5": 0.0})
        ev, stamped = resolve_day_path_ev({"bars_1m": bars, "symbol": symbol, "btc_ret_5": 0.0}, symbol=symbol)
        assert legacy is not None
        assert ev == 0.0
        assert stamped["path_input_valid"] is False
        assert stamped["selected_net_expected_value"] == 0.0
        if symbol == "XRPUSDT":
            assert abs(float(stamped.get("legacy_path_ev") or 0)) > 0.001 or abs(float(legacy)) > 0.001
    reset_day_artifact_cache()


def test_all_invalid_holds_path_input_invalid():
    out = select_action(
        {
            "btc_path_ev": 0.021576067272686527,
            "eth_path_ev": 0.0,
            "sol_path_ev": 0.0,
            "xrp_path_ev": 0.021576067272686527,
            "valid": {"btc": False, "eth": False, "sol": False, "xrp": False},
        }
    )
    assert out["selected_action"] == "HOLD"
    assert out["why_selected"] == "PATH_INPUT_INVALID"
    assert out["selected_ev"] == 0.0


def test_one_valid_three_invalid_uses_only_valid_coin():
    out = select_action(
        {
            "btc_path_ev": 0.002,
            "eth_path_ev": 0.01,
            "sol_path_ev": 0.009,
            "xrp_path_ev": 0.0215,
            "valid": {"btc": True, "eth": False, "sol": False, "xrp": False},
        }
    )
    assert out["selected_action"] == "BUY_BTCUSDT"
    assert out["why_selected"] == "PATH_NET_BEATS_HOLD"


def test_shadow_btc_and_ood_do_not_change_winner():
    reset_day_artifact_cache()
    bars = _dense_bars()
    ev, stamped = resolve_day_path_ev({"bars_1m": bars, "symbol": "ETHUSDT", "btc_ret_5": 0.0}, symbol="ETHUSDT")
    assert stamped["path_input_valid"] is True
    assert ev == stamped["selected_net_expected_value"]
    out = select_action(
        {
            "btc_path_ev": ev,
            "eth_path_ev": 0.0,
            "sol_path_ev": 0.0,
            "xrp_path_ev": 0.0,
            "valid": {"btc": True, "eth": False, "sol": False, "xrp": False},
            "shadow_correct_btc_winner": "ETHUSDT",
            "path_input_by_symbol": {
                "BTCUSDT": {
                    "shadow_correct_btc_path_ev": 0.05,
                    "path_max_abs_z": 12.0,
                    "path_ood_feature_count_at_8": 10,
                }
            },
        }
    )
    assert out["selected_action"] == "BUY_BTCUSDT"
    assert out["shadow_correct_btc_winner"] == "ETHUSDT"
    assert out["winner_disagreement"] is True
    reset_day_artifact_cache()
