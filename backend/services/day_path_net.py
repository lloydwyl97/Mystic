"""DAY path-net predictor. HOLD EV = 0. Not a threshold gate.

Uses reconstructable 1m features and an accepted artifact only.
Does not reuse the SCALP 20m artifact.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.binance_scalp.forward_net_predictor import (
    ForwardNetArtifact,
    predict_artifact,
)
from backend.services.binance_scalp.reconstructable_features import reconstructable_features
from backend.services.day_path_input_validity import (
    PATH_FEATURE_SCHEMA_VERSION,
    PATH_INPUT_INVALID_SCHEMA,
    five_bar_return,
    ood_telemetry,
    parse_bar_ts,
    validate_path_bars,
)

DAY_PATH_MODEL_VERSION = "day_path_net_v1"
DAY_HORIZONS_MIN = (60, 120, 180)
LOOKBACK_BARS = 40

_LOADED: ForwardNetArtifact | None = None
_LOAD_ATTEMPTED = False


def reset_day_artifact_cache() -> None:
    global _LOADED, _LOAD_ATTEMPTED
    _LOADED = None
    _LOAD_ATTEMPTED = False


def day_artifact_path() -> Path:
    raw = os.getenv("DAY_PATH_NET_ARTIFACT", "")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "models" / "day_path_net_v1.json"


def load_accepted_day_artifact() -> ForwardNetArtifact | None:
    global _LOADED, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LOADED if _LOADED is not None and _LOADED.accepted else None
    _LOAD_ATTEMPTED = True
    path = day_artifact_path()
    if not path.exists():
        return None
    try:
        art = ForwardNetArtifact.from_dict(json.loads(path.read_text()))
    except Exception:
        return None
    _LOADED = art
    return art if art.accepted else None


def _norm_ohlcv_symbol(symbol: str) -> str:
    s = str(symbol or "").replace("/", "-").replace("_", "-").upper()
    if s.endswith("USDT") and "-" not in s:
        return f"{s[:-4]}-USDT"
    return s


def _symbol_lookups(symbol: str) -> list[str]:
    raw = str(symbol or "").strip()
    if not raw:
        return []
    norm = _norm_ohlcv_symbol(raw)
    slash = norm.replace("-", "/")
    compact = norm.replace("-", "")
    out: list[str] = []
    for item in (norm, slash, compact, raw.upper()):
        if item and item not in out:
            out.append(item)
    return out


def load_recent_bars(db_path: str, symbol: str, *, n: int = LOOKBACK_BARS) -> list[dict[str, Any]]:
    path = Path(db_path) if db_path else Path()
    if not db_path or not path.exists():
        fallback = Path(__file__).resolve().parents[2] / "mystic_trading.db"
        if fallback.exists():
            path = fallback
        else:
            return []
    names = _symbol_lookups(symbol)
    if not names:
        return []
    rows: list[tuple[Any, ...]] = []
    try:
        with sqlite3.connect(str(path)) as conn:
            for want in names:
                rows = conn.execute(
                    """
                    SELECT open, high, low, close, volume, ts
                    FROM feature_ohlcv
                    WHERE interval='1m' AND symbol=?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    (want, int(n)),
                ).fetchall()
                if rows:
                    break
    except Exception:
        return []
    bars: list[dict[str, Any]] = []
    for o, h, low, c, v, ts in reversed(rows):
        bars.append(
            {
                "open": float(o or 0),
                "high": float(h or 0),
                "low": float(low or 0),
                "close": float(c or 0),
                "volume": float(v or 0),
                "ts": ts,
            }
        )
    return bars


def attach_bars(decision_data: dict[str, Any], db_path: str, symbol: str) -> dict[str, Any]:
    dd = dict(decision_data or {})
    existing = dd.get("bars_1m") or []
    if isinstance(existing, list) and len(existing) >= 8:
        return dd
    bars = load_recent_bars(db_path, symbol)
    if bars:
        dd["bars_1m"] = bars
    return dd


def _as_of_from_decision(dd: dict[str, Any], bars: list[Any]) -> datetime | None:
    raw = dd.get("path_as_of") or dd.get("decision_ts")
    if raw:
        parsed = parse_bar_ts(raw)
        if parsed is not None:
            return parsed
    if dd.get("path_as_of_now"):
        return datetime.now(timezone.utc)
    if isinstance(bars, list) and bars:
        last = bars[-1] if isinstance(bars[-1], dict) else {}
        return parse_bar_ts(last.get("ts")) if isinstance(last, dict) else None
    return datetime.now(timezone.utc)


def _shadow_correct_btc_ev(
    dd: dict[str, Any],
    *,
    db_path: str,
    as_of: datetime | None,
) -> tuple[float | None, float | None]:
    """Shadow-only BTC 5-bar relative EV. Never used for live authority."""
    btc_bars = dd.get("btc_bars_1m")
    if not isinstance(btc_bars, list) or len(btc_bars) < 40:
        btc_bars = load_recent_bars(db_path, "BTCUSDT") if db_path else []
    btc_tel = validate_path_bars(btc_bars, as_of=as_of)
    if not btc_tel.get("path_input_valid"):
        return None, None
    ret = five_bar_return(btc_bars)
    if ret is None:
        return None, None
    shadow = predict_decision_net({**dd, "btc_ret_5": float(ret)})
    return float(ret), (float(shadow) if shadow is not None else None)


def _features_from_decision(dd: dict[str, Any]) -> dict[str, float]:
    bars = dd.get("bars_1m") or []
    if not isinstance(bars, list) or len(bars) < 8:
        return {}
    ts = None
    last = bars[-1] if bars else {}
    raw_ts = last.get("ts") if isinstance(last, dict) else None
    if raw_ts is not None:
        try:
            ts = raw_ts if hasattr(raw_ts, "hour") else datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        except Exception:
            try:
                ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except Exception:
                ts = None
    btc_ret = float(dd.get("btc_ret_5") or 0.0)
    return reconstructable_features(
        bars,
        btc_ret_5=btc_ret,
        market_vol_5=abs(btc_ret),
        ts=ts,
        projected_move=float(dd.get("projected_move") or 0.0),
    )


def predict_decision_net(decision_data: dict[str, Any]) -> float | None:
    """Runtime DAY BUY EV from the accepted artifact, or None."""
    art = load_accepted_day_artifact()
    if art is None:
        return None
    feats = _features_from_decision(decision_data or {})
    if not feats:
        return None
    pred = predict_artifact(art, feats)
    return float(pred["predicted_net_ev"])


def stamp_day_path_prediction(decision_data: dict[str, Any]) -> dict[str, Any]:
    _, stamped = resolve_day_path_ev(decision_data)
    return stamped


def _stamp_common(dd: dict[str, Any], art: ForwardNetArtifact, tel: dict[str, Any]) -> None:
    dd.update(tel)
    dd["forward_net_model_version"] = art.version
    dd["path_model_version"] = art.version
    dd["path_feature_schema_version"] = PATH_FEATURE_SCHEMA_VERSION
    dd["day_path_horizon_min"] = int(art.primary_horizon_min)
    dd["hold_action_ev"] = 0.0
    dd["selected_net_expected_value_is_net"] = "1"
    dd["legacy_btc_ret_5"] = float(dd.get("btc_ret_5") or 0.0)


def resolve_day_path_ev(
    decision_data: dict[str, Any] | None,
    *,
    symbol: str = "",
    db_path: str = "",
) -> tuple[float | None, dict[str, Any]]:
    """Accepted artifact only. Sparse/stale/gappy 1m inputs cannot be authority.

    Live EV still uses legacy btc_ret_5 (default 0). Corrected BTC-relative EV
    and OOD z-counts are shadow telemetry only.
    ev is None only when no accepted artifact is loaded.
    """
    dd = dict(decision_data or {})
    art = load_accepted_day_artifact()
    if art is None:
        return None, dd
    supplied_bars = dd.get("bars_1m")
    had_supplied_bars = isinstance(supplied_bars, list) and len(supplied_bars) >= 8
    sym = str(symbol or dd.get("symbol") or dd.get("symbol_bus") or "")
    if sym:
        dd["symbol"] = sym
        dd = attach_bars(dd, db_path, sym)
    bars = dd.get("bars_1m") or []
    if dd.get("path_as_of_now") or (bool(db_path) and not had_supplied_bars):
        as_of = datetime.now(timezone.utc)
    else:
        as_of = _as_of_from_decision(dd, bars if isinstance(bars, list) else [])
    tel = validate_path_bars(bars, as_of=as_of)
    tel["path_model_version"] = art.version
    feats = _features_from_decision(dd)
    dd.update(ood_telemetry(feats, art) if feats else ood_telemetry({}, art))
    correct_btc, shadow_ev = _shadow_correct_btc_ev(dd, db_path=db_path, as_of=as_of)
    dd["correct_btc_ret_5"] = correct_btc
    dd["shadow_correct_btc_path_ev"] = shadow_ev
    if not tel.get("path_input_valid"):
        _stamp_common(dd, art, tel)
        reason = str(tel.get("path_invalid_reason") or PATH_INPUT_INVALID_SCHEMA)
        dd["path_input_valid"] = False
        dd["path_invalid_reason"] = reason
        dd["path_net_status"] = reason
        dd["selected_net_expected_value"] = 0.0
        dd["predicted_net_return"] = 0.0
        if feats:
            blocked = predict_decision_net(dd)
            if blocked is not None:
                dd["legacy_path_ev"] = float(blocked)
        return 0.0, dd
    pred = predict_decision_net(dd)
    _stamp_common(dd, art, tel)
    dd["path_input_valid"] = True
    dd["path_invalid_reason"] = None
    if pred is None:
        dd["selected_net_expected_value"] = 0.0
        dd["predicted_net_return"] = 0.0
        dd["path_net_status"] = "unavailable_hold"
        return 0.0, dd
    pred_f = float(pred)
    dd["selected_net_expected_value_raw"] = pred_f
    dd["legacy_path_ev"] = pred_f
    try:
        from backend.services.symbol_setup_outcome_penalty import evaluate_low_mfe_stall_penalty

        setup = str(dd.get("setup_type") or dd.get("entry_thesis") or "")
        regime = str(dd.get("day_route_regime") or dd.get("regime") or "")
        pen = evaluate_low_mfe_stall_penalty(sym, setup, regime, db_path=db_path or None)
        if pen.get("applied"):
            factor = float(pen.get("ev_factor") or 1.0)
            pred_f = pred_f * factor
            dd["outcome_low_mfe_stall_penalty_applied"] = True
            dd["outcome_low_mfe_stall_ev_factor"] = factor
    except Exception:
        pass
    dd["selected_net_expected_value"] = pred_f
    dd["predicted_net_return"] = pred_f
    dd["path_net_status"] = "predicted"
    return pred_f, dd
