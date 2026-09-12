"""DAY paper authority: four-coin path-EV vs HOLD(0).

Old 15m rank does not select. No hybrid. HOLD EV is exactly 0.
Does not change SCALP. Does not change DAY exits.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.day_live_tape_authority import LIVE_AUTHORITY_MODE, score_live_four_coins
from backend.services.day_path_net import DAY_PATH_MODEL_VERSION

DAY_AUTHORITY_MODE = LIVE_AUTHORITY_MODE
DAY_POLICY_ID = "day_live_tape_learn_v1"
HOLD_ACTION = "HOLD"
# Require model to predict at least 10 bps of edge above zero before trading.
# The model is measured at -8.7 bps OOS — marginal positive EVs are noise.
HOLD_EV = float(os.getenv("DAY_MIN_EV_FLOOR", "0.0010"))
OLD_RANK_EXECUTION_AUTHORITY = False

_COIN_KEYS = ("btc", "eth", "sol", "xrp")


def _api_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def _slash_symbol(api: str) -> str:
    s = _api_symbol(api)
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return s


def _coin_key(api: str) -> str:
    s = _api_symbol(api)
    if s.endswith("USDT"):
        s = s[:-4]
    return s.lower()


def _opt_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = payload.get(key)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isnan(val):
            return val
    return None


def post_cost_economics_ev(decision_data: dict[str, Any] | None) -> float | None:
    """Reconstruct post-cost EV from candidate fields. None if not identifiable.

    Used to keep ranking honest: a path-net score cannot beat HOLD when the
    candidate's own expected move after fees/slip/spread is non-positive.
    Does not invent an edge from buy_margin or regime.
    """
    dd = decision_data or {}
    efe = _opt_float(dd, "expected_favorable_excursion", "estimated_mfe", "estimated_win_pct")
    eae = _opt_float(dd, "expected_adverse_excursion", "estimated_mae", "estimated_loss_pct")
    if efe is None or eae is None:
        return None
    p_buy = _opt_float(dd, "prob_buy", "winner_probability")
    p_sell = _opt_float(dd, "prob_sell")
    if p_buy is None:
        p_buy = 0.5
    if p_sell is None:
        p_sell = max(0.0, 1.0 - float(p_buy))
    p_hold = max(0.0, _opt_float(dd, "prob_hold") or 0.0)
    total = float(p_buy) + float(p_sell) + float(p_hold)
    if total > 0:
        p_buy = float(p_buy) / total
        p_sell = float(p_sell) / total
    fees = max(0.0, _opt_float(dd, "estimated_fees_pct") or 0.0)
    slip = max(0.0, _opt_float(dd, "estimated_slippage_pct") or 0.0)
    spread = max(0.0, _opt_float(dd, "spread_cost_pct", "spread_pct") or 0.0)
    if fees == 0.0 and slip == 0.0 and spread == 0.0:
        fees = float(ESTIMATED_ROUNDTRIP_COST)
    return float(p_buy) * float(efe) - float(p_sell) * abs(float(eae)) - fees - slip - spread


def score_four_coins(*, db_path: str = "") -> dict[str, Any]:
    """Score BTC/ETH/SOL/XRP from the live tape and live learning signal."""
    live = score_live_four_coins(db_path=db_path)
    live["hold_ev"] = HOLD_EV
    live["path_net_model_id"] = live.get("path_net_model_id") or DAY_PATH_MODEL_VERSION
    return live


def select_action(
    scores: dict[str, Any],
    *,
    old_rank_nominee: str = "",
    old_rank_score: float | None = None,
) -> dict[str, Any]:
    """Highest EV among four coins and HOLD(0). Old rank is telemetry only."""
    hold_ev = HOLD_EV
    valid_map = scores.get("valid") if isinstance(scores.get("valid"), dict) else None
    pairs: list[tuple[str, float]] = []
    for api, key in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol"), ("XRPUSDT", "xrp")):
        if valid_map is not None and not valid_map.get(key, False):
            continue
        pairs.append((api, float(scores.get(f"{key}_path_ev") or HOLD_EV)))
    pairs.append((HOLD_ACTION, hold_ev))
    winner_name, winner_ev = max(pairs, key=lambda p: (p[1], 0 if p[0] == HOLD_ACTION else 1))
    if winner_ev <= hold_ev:
        selected_action = HOLD_ACTION
        selected_symbol = ""
        selected_ev = hold_ev
        if valid_map is not None and not any(valid_map.values()):
            why = "PATH_INPUT_INVALID"
        else:
            why = "HOLD_WINS"
        path_ev_winner = HOLD_ACTION
    else:
        selected_action = f"BUY_{winner_name}"
        selected_symbol = winner_name
        selected_ev = float(winner_ev)
        why = "PATH_NET_BEATS_HOLD"
        path_ev_winner = winner_name
    now = datetime.now(timezone.utc).isoformat()
    return {
        "day_authority_mode": DAY_AUTHORITY_MODE,
        "old_rank_execution_authority": OLD_RANK_EXECUTION_AUTHORITY,
        "old_rank_nominee": _api_symbol(old_rank_nominee) if old_rank_nominee else "",
        "old_rank_score": old_rank_score,
        "btc_path_ev": float(scores.get("btc_path_ev") or HOLD_EV),
        "eth_path_ev": float(scores.get("eth_path_ev") or HOLD_EV),
        "sol_path_ev": float(scores.get("sol_path_ev") or HOLD_EV),
        "xrp_path_ev": float(scores.get("xrp_path_ev") or HOLD_EV),
        "hold_ev": hold_ev,
        "path_ev_winner": path_ev_winner,
        "selected_action": selected_action,
        "selected_symbol": selected_symbol,
        "selected_ev": selected_ev,
        "path_net_model_id": scores.get("path_net_model_id") or DAY_PATH_MODEL_VERSION,
        "path_aware_policy_id": DAY_POLICY_ID,
        "model_trained_at": scores.get("model_trained_at") or "",
        "model_accuracy": scores.get("model_accuracy"),
        "prediction_timestamp": now,
        "costs_bps": scores.get("costs_bps"),
        "horizon_minutes": scores.get("horizon_minutes"),
        "true_safety_reject_reason": None,
        "why_selected": why,
        "path_net_status": scores.get("path_net_status") or "predicted",
        "forward_net_model_version": scores.get("path_net_model_id") or DAY_PATH_MODEL_VERSION,
        "selected_net_expected_value": selected_ev,
        "predicted_net_return": selected_ev,
        "hold_action_ev": hold_ev,
        "valid": valid_map,
        "path_input_by_symbol": scores.get("path_input_by_symbol") or {},
        "legacy_winner": path_ev_winner,
        "shadow_correct_btc_winner": scores.get("shadow_correct_btc_winner"),
        "winner_disagreement": bool(scores.get("shadow_correct_btc_winner") and scores.get("shadow_correct_btc_winner") != path_ev_winner),
    }


def old_rank_telemetry(candidates: list[Any] | None) -> tuple[str, float | None]:
    """Best old-rank nominee for stamps only. Never used to pick."""
    rows = list(candidates or [])
    if not rows:
        return "", None

    def _score(c: Any) -> float:
        dd = getattr(c, "decision_data", None) or {}
        try:
            return float(dd.get("final_selection_score") or dd.get("selection_score") or c.rank_score())
        except Exception:
            return 0.0

    top = max(rows, key=_score)
    return _api_symbol(getattr(top, "symbol", "") or ""), _score(top)


def ranked_path_ev_buys(decision: dict[str, Any]) -> list[tuple[str, float]]:
    """Coins whose path-EV beats HOLD, highest first. Path-EV remains the authority."""
    hold_ev = float(decision.get("hold_ev") if decision.get("hold_ev") is not None else HOLD_EV)
    rows: list[tuple[str, float]] = []
    for api, key in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol"), ("XRPUSDT", "xrp")):
        try:
            ev = float(decision.get(f"{key}_path_ev") or HOLD_EV)
        except (TypeError, ValueError):
            ev = HOLD_EV
        if ev > hold_ev:
            rows.append((_api_symbol(api), ev))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def next_executable_path_ev_symbol(
    decision: dict[str, Any],
    *,
    executable_symbols: set[str] | list[str],
) -> tuple[str, float] | None:
    """First path-EV winner that both beats HOLD and has an executable candidate.

    Production recorded BUY_SOL / BUY_BTC while execution silently HOLDed when the
    winner had no thesis candidate. That is not HOLD_WINS. Walk the scored list.
    """
    allowed = {_api_symbol(s) for s in executable_symbols if _api_symbol(s)}
    if not allowed:
        return None
    for api, ev in ranked_path_ev_buys(decision):
        if api in allowed:
            return api, ev
    return None

_LEARNING_VETO_CONSEC = int(os.getenv("DAY_LEARNING_VETO_CONSEC_LOSSES", "3"))
_LEARNING_VETO_LOOKBACK = int(os.getenv("DAY_LEARNING_VETO_LOOKBACK", "10"))


def _learning_vetoed_coins(db_path: str) -> set[str]:
    """Check recent trade outcomes per coin. If the last N consecutive trades
    for a coin were all losses, suppress it so the model picks a different
    coin or HOLDs. This gives the learning pipeline real influence on entries.
    """
    if _LEARNING_VETO_CONSEC <= 0 or not db_path:
        return set()
    vetoed: set[str] = set()
    try:
        import sqlite3

        with sqlite3.connect(db_path, timeout=5) as conn:
            for api in DAY_TRADE_SYMBOLS:
                rows = conn.execute(
                    """
                    SELECT net_profit_usd FROM trade_learning_outcomes
                    WHERE symbol = ? AND mode IN ('paper', 'live')
                      AND net_profit_usd IS NOT NULL
                    ORDER BY exit_timestamp DESC LIMIT ?
                    """,
                    (api, _LEARNING_VETO_LOOKBACK),
                ).fetchall()
                if len(rows) >= _LEARNING_VETO_CONSEC:
                    recent = [r[0] for r in rows[:_LEARNING_VETO_CONSEC]]
                    if all(pnl < 0 for pnl in recent):
                        vetoed.add(_coin_key(api))
    except Exception:
        pass
    return vetoed


def decide_day_bar(*, db_path: str = "", candidates: list[Any] | None = None) -> dict[str, Any]:
    nominee, score = old_rank_telemetry(candidates)
    scores = score_four_coins(db_path=db_path)
    vetoed = _learning_vetoed_coins(db_path)
    for coin in vetoed:
        key = f"{coin}_path_ev"
        if key in scores:
            scores[key] = HOLD_EV
    if vetoed:
        scores["learning_vetoed_coins"] = sorted(vetoed)
    return select_action(scores, old_rank_nominee=nominee, old_rank_score=score)
