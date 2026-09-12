"""Score DAY coins from the live tape, live signal, and live context.

No frozen path-net file. Spot is long-only: down and sideways are HOLD.
"""

from __future__ import annotations

import math
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.day_path_net import load_recent_bars

LIVE_AUTHORITY_MODE = "live_tape_learn_v1"
_UP = "up"
_DOWN = "down"
_SIDEWAYS = "sideways"
_UNKNOWN = "unknown"


def _api(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def _coin_key(api: str) -> str:
    s = _api(api)
    return s[:-4].lower() if s.endswith("USDT") else s.lower()


def _f(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = payload.get(key)
        if raw in (None, ""):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isnan(val):
            return val
    return None


def classify_tape(bars: list[dict[str, Any]]) -> dict[str, Any]:
    """Up / down / sideways from the latest 1m closes. No stored model."""
    closes = [float(b.get("close") or 0.0) for b in bars if float(b.get("close") or 0.0) > 0]
    if len(closes) < 20:
        return {"state": _UNKNOWN, "ret_15": 0.0, "ret_60": 0.0, "vol": 0.0}
    last = closes[-1]
    ret_15 = last / closes[-15] - 1.0 if len(closes) >= 15 else last / closes[0] - 1.0
    look = min(60, len(closes) - 1)
    ret_60 = last / closes[-1 - look] - 1.0
    rets = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / max(1, len(rets) - 1)
    vol = math.sqrt(max(var, 0.0))
    noise = max(vol * math.sqrt(15.0), 0.0008)
    if abs(ret_15) < noise:
        state = _SIDEWAYS
    elif ret_15 > 0 and ret_60 >= 0:
        state = _UP
    elif ret_15 < 0 and ret_60 <= 0:
        state = _DOWN
    elif ret_15 > 0:
        state = _UP
    else:
        state = _DOWN
    return {"state": state, "ret_15": float(ret_15), "ret_60": float(ret_60), "vol": float(vol)}


def _read_live_hashes(api: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import redis

        r = redis.Redis(decode_responses=True, socket_timeout=1.5)
        sig = r.hgetall(f"ai_signal:day:{api}") or {}
        ctx = r.hgetall(f"ai_context:{api}") or {}
        return dict(sig), dict(ctx)
    except Exception:
        return {}, {}


def _norm_regime(*raw: str) -> str:
    text = " ".join(str(x or "") for x in raw).lower()
    if any(tok in text for tok in ("trending_down", "downtrend", "bear")):
        return _DOWN
    if any(tok in text for tok in ("trending_up", "uptrend", "bull")):
        return _UP
    if any(tok in text for tok in ("chop", "rang", "sideways")):
        return _SIDEWAYS
    return _UNKNOWN


def score_live_coin(*, symbol: str, db_path: str = "", signal: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """One coin: tape + live learning signal + live context. HOLD unless all agree up."""
    api = _api(symbol)
    bars = load_recent_bars(db_path, api)
    tape = classify_tape(bars)
    sig = dict(signal or {})
    ctx = dict(context or {})
    if not sig and not ctx:
        live_sig, live_ctx = _read_live_hashes(api)
        sig = live_sig
        ctx = live_ctx
    ctx_state = _norm_regime(ctx.get("ctx_market_regime"), ctx.get("market_regime"), sig.get("regime"), sig.get("regime_label"))
    p_buy = _f(sig, "prob_buy") or 0.0
    p_hold = _f(sig, "prob_hold") or 0.0
    p_sell = _f(sig, "prob_sell") or 0.0
    buy_margin = _f(sig, "buy_margin")
    side = str(sig.get("side") or sig.get("prediction") or sig.get("argmax_action") or "").strip().lower()
    trained_at = str(sig.get("model_trained_at") or "")
    live_wants_buy = side == "buy" and p_buy > p_hold and p_buy > p_sell and p_hold < 0.55 and (buy_margin is None or buy_margin > 0.0)
    tape_state = str(tape["state"])
    if tape_state in (_DOWN, _SIDEWAYS, _UNKNOWN) or ctx_state == _DOWN or not live_wants_buy:
        ev = 0.0
        why = f"HOLD tape={tape_state} ctx={ctx_state or 'na'} side={side or 'na'} p_hold={p_hold:.3f}"
    else:
        edge = max(0.0, p_buy - p_sell)
        ev = float(tape["ret_15"]) * edge - float(ESTIMATED_ROUNDTRIP_COST)
        why = f"LIVE_UP tape={tape_state} ctx={ctx_state or 'na'} side={side} ev={ev:.6f}"
        if ev <= 0.0:
            ev = 0.0
            why = f"HOLD_COST tape_up but net={ev:.6f}"
    return {
        "symbol": api,
        "ev": float(ev),
        "tape_state": tape_state,
        "ctx_state": ctx_state,
        "ret_15": float(tape["ret_15"]),
        "ret_60": float(tape["ret_60"]),
        "prob_buy": p_buy,
        "prob_hold": p_hold,
        "prob_sell": p_sell,
        "side": side,
        "model_trained_at": trained_at,
        "why": why,
    }


def score_live_four_coins(*, db_path: str = "", candidate_signals: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Live four-coin scores. Missing/unclear tape is 0, not an invented edge."""
    details: dict[str, Any] = {}
    evs: dict[str, float] = {}
    for api in DAY_TRADE_SYMBOLS:
        key = _coin_key(api)
        extra = (candidate_signals or {}).get(api) or (candidate_signals or {}).get(key) or {}
        row = score_live_coin(symbol=api, db_path=db_path, signal=extra.get("signal"), context=extra.get("context"))
        evs[key] = float(row["ev"])
        details[key] = row
    return {
        "btc_path_ev": float(evs.get("btc", 0.0)),
        "eth_path_ev": float(evs.get("eth", 0.0)),
        "sol_path_ev": float(evs.get("sol", 0.0)),
        "xrp_path_ev": float(evs.get("xrp", 0.0)),
        "hold_ev": 0.0,
        "statuses": {k: str((details.get(k) or {}).get("tape_state") or "unknown") for k in ("btc", "eth", "sol", "xrp")},
        "live_details": details,
        "path_net_model_id": LIVE_AUTHORITY_MODE,
        "path_net_status": "live_tape",
        "model_trained_at": "live",
        "horizon_minutes": 15,
        "costs_bps": round(float(ESTIMATED_ROUNDTRIP_COST) * 1e4, 4),
        "day_authority_mode": LIVE_AUTHORITY_MODE,
    }
