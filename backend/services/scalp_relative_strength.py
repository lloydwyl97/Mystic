"""SCALP relative strength across top-4 symbols (rank only)."""

from __future__ import annotations

from typing import Any


def _rank(values: dict[str, float], *, higher_better: bool = True) -> dict[str, int]:
    items = sorted(values.items(), key=lambda x: x[1], reverse=higher_better)
    out: dict[str, int] = {}
    for i, (sym, _) in enumerate(items, start=1):
        out[sym] = i
    return out


def enrich_scalp_basket_relative_strength(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    syms = [str(c.get("symbol") or "") for c in candidates]
    mom = {s: float((next(c for c in candidates if c.get("symbol") == s)).get("mid_change_30s") or 0) for s in syms}
    vol = {s: float((next(c for c in candidates if c.get("symbol") == s)).get("kline_volume_ratio") or 0) for s in syms}
    spread = {s: float((next(c for c in candidates if c.get("symbol") == s)).get("spread_pct") or 1) for s in syms}
    depth = {s: abs(float((next(c for c in candidates if c.get("symbol") == s)).get("order_book_imbalance") or 0)) for s in syms}
    exec_q = {s: float((next(c for c in candidates if c.get("symbol") == s)).get("scalp_execution_quality_score") or 0) for s in syms}
    health = {s: float((next(c for c in candidates if c.get("symbol") == s)).get("scalp_feature_health_score") or 0) for s in syms}
    setup = {s: float((next(c for c in candidates if c.get("symbol") == s)).get("setup_score") or 0) for s in syms}

    mom_r = _rank(mom)
    vol_r = _rank(vol)
    spread_r = _rank(spread, higher_better=False)
    depth_r = _rank(depth)
    exec_r = _rank(exec_q)
    health_r = _rank(health)
    setup_r = _rank(setup)

    out: list[dict[str, Any]] = []
    for c in candidates:
        sym = str(c.get("symbol") or "")
        rs = round(
            (
                (5 - mom_r.get(sym, 4)) * 0.22
                + (5 - vol_r.get(sym, 4)) * 0.18
                + (5 - spread_r.get(sym, 4)) * 0.20
                + (5 - depth_r.get(sym, 4)) * 0.12
                + (5 - exec_r.get(sym, 4)) * 0.18
                + (5 - setup_r.get(sym, 4)) * 0.10
            )
            / 4.0,
            4,
        )
        row = dict(c)
        row["micro_momentum_rank"] = mom_r.get(sym, 4)
        row["volume_burst_rank"] = vol_r.get(sym, 4)
        row["spread_rank"] = spread_r.get(sym, 4)
        row["depth_rank"] = depth_r.get(sym, 4)
        row["execution_rank"] = exec_r.get(sym, 4)
        row["feature_health_rank"] = health_r.get(sym, 4)
        row["setup_quality_rank"] = setup_r.get(sym, 4)
        row["scalp_rs_rank"] = rs
        row["relative_strength_rank"] = rs
        out.append(row)
    out.sort(key=lambda x: (-float(x.get("scalp_rs_rank") or 0), float(x.get("spread_pct") or 1)))
    return out


__all__ = ["enrich_scalp_basket_relative_strength"]
