"""Relative-strength basket ranking across DAY top-four (BTC/ETH/SOL/XRP)."""

from __future__ import annotations

import math
from typing import Any

from backend.config.trading_universe import DAY_TRADE_SYMBOLS


def _norm_symbol(sym: str) -> str:
    return (sym or "").replace("/", "").upper()


def _f(dd: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = dd.get(key)
    try:
        v = float(raw)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _rank_values(values: dict[str, float], *, higher_better: bool = True) -> dict[str, int]:
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_better)
    out: dict[str, int] = {}
    for i, (sym, _v) in enumerate(items):
        out[sym] = i + 1
    return out


def enrich_basket_relative_strength(candidates: list[Any]) -> None:
    """Stamp per-candidate RS ranks and rank delta on decision_data (in-place)."""
    if not candidates:
        return

    by_sym: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        sym = _norm_symbol(getattr(cand, "symbol", "") or "")
        if not sym:
            continue
        dd = dict(getattr(cand, "decision_data", None) or {})
        by_sym[sym] = dd

    if len(by_sym) < 2:
        for cand in candidates:
            dd = dict(getattr(cand, "decision_data", None) or {})
            dd.setdefault("rs_rank", 1)
            dd.setdefault("relative_strength_rank", 1)
            dd["basket_rs_rank_delta"] = 0.0
            setattr(cand, "decision_data", dd)
        return

    rs_vals = {s: _f(d, "ctx_rs_btc", 0.0) + _f(d, "ctx_rs_eth", 0.0) for s, d in by_sym.items()}
    trend_vals = {s: _f(d, "ema_alignment", 0.5) for s, d in by_sym.items()}
    mom_vals = {s: _f(d, "price_momentum", 0.0) for s, d in by_sym.items()}
    vol_vals = {s: _f(d, "ctx_relative_volume", 1.0) for s, d in by_sym.items()}
    volat_vals = {s: _f(d, "atr", 0.0) / max(_f(d, "current_price", 1.0), 1e-9) for s, d in by_sym.items()}
    exec_vals = {s: _f(d, "execution_quality_score", 0.5) for s, d in by_sym.items()}
    fh_vals = {s: _f(d, "feature_health_score", 0.5) for s, d in by_sym.items()}
    setup_vals = {s: _f(d, "setup_score", 0.5) for s, d in by_sym.items()}

    rs_rank = _rank_values(rs_vals)
    trend_rank = _rank_values(trend_vals)
    mom_rank = _rank_values(mom_vals)
    vol_rank = _rank_values(vol_vals)
    volat_rank = _rank_values(volat_vals, higher_better=False)
    exec_rank = _rank_values(exec_vals)
    fh_rank = _rank_values(fh_vals)
    setup_rank = _rank_values(setup_vals)

    n = len(by_sym)
    for cand in candidates:
        sym = _norm_symbol(getattr(cand, "symbol", "") or "")
        dd = dict(getattr(cand, "decision_data", None) or {})
        rr = rs_rank.get(sym, n)
        composite = (
            rs_rank.get(sym, n)
            + trend_rank.get(sym, n)
            + mom_rank.get(sym, n)
            + vol_rank.get(sym, n)
            + exec_rank.get(sym, n)
            + fh_rank.get(sym, n)
            + setup_rank.get(sym, n)
        ) / 7.0
        dd["rs_rank"] = rr
        dd["trend_rank"] = trend_rank.get(sym, n)
        dd["momentum_rank"] = mom_rank.get(sym, n)
        dd["volume_rank"] = vol_rank.get(sym, n)
        dd["volatility_rank"] = volat_rank.get(sym, n)
        dd["execution_rank"] = exec_rank.get(sym, n)
        dd["feature_health_rank"] = fh_rank.get(sym, n)
        dd["setup_quality_rank"] = setup_rank.get(sym, n)
        dd["relative_strength_rank"] = int(round(composite))
        # Rank 1 = best → positive delta; rank n = worst → negative
        dd["basket_rs_rank_delta"] = round(max(-0.06, min(0.06, ((n + 1 - composite) / n - 0.5) * 0.12)), 4)
        setattr(cand, "decision_data", dd)


def leading_lagging_summary(candidates: list[Any]) -> dict[str, str]:
    if not candidates:
        return {"leader": "", "lagger": ""}
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        sym = _norm_symbol(getattr(cand, "symbol", "") or "")
        dd = getattr(cand, "decision_data", None) or {}
        scored.append((sym, float(dd.get("relative_strength_rank") or 99)))
    scored.sort(key=lambda x: x[1])
    return {"leader": scored[0][0] if scored else "", "lagger": scored[-1][0] if scored else ""}


__all__ = ["DAY_TRADE_SYMBOLS", "enrich_basket_relative_strength", "leading_lagging_summary"]
