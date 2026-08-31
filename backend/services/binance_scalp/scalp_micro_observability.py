"""Observability-only SCALP peer-EV and size stamps.

Never changes candidate order, eligibility, EV calculation, or sizing.
"""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.scalp_micro_contract import EV_HORIZONS_SEC

PEER_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
EV_KEYS: tuple[str, ...] = tuple(f"EV_{h}s" for h in EV_HORIZONS_SEC)


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _setup_context(row: dict[str, Any]) -> dict[str, Any]:
    sig = row.get("signal")
    if sig is None:
        return {}
    if isinstance(sig, dict):
        ctx = sig.get("setup_context") or {}
        return ctx if isinstance(ctx, dict) else {}
    ctx = getattr(sig, "setup_context", None) or {}
    return ctx if isinstance(ctx, dict) else {}


def _lookup_ev(row: dict[str, Any], key: str) -> float | None:
    meta = row.get("rank_meta") or {}
    ctx = _setup_context(row)
    for src in (row, ctx, meta):
        if isinstance(src, dict) and src.get(key) is not None:
            return _num(src.get(key))
    return None


def extract_peer_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Read-only field extract. Does not mutate ``row``."""
    meta = row.get("rank_meta") or {}
    ctx = _setup_context(row)
    symbol = str(row.get("symbol") or "").upper()
    evs = {k: _lookup_ev(row, k) for k in EV_KEYS}
    static = _num(row.get("static_rank_score"), None)
    if static is None:
        static = _num(meta.get("static_rank_score") or meta.get("raw_rank_score"), None)
    micro_adj = _num(row.get("microstructure_adjustment"), None)
    if micro_adj is None:
        micro_adj = _num(meta.get("microstructure_adjustment"), 0.0)
    learned = _num(row.get("learned_adjustment"), None)
    if learned is None:
        learned = _num(ctx.get("learned_adjustment") or meta.get("learned_adjustment"), 0.0)
    final_rank = _num(row.get("rank_score") or meta.get("best_rank_score"), 0.0)
    strategy_passed = bool(row.get("strategy_passed") if row.get("strategy_passed") is not None else meta.get("strategy_passed"))
    soft_rank = (not strategy_passed) or bool(ctx.get("soft_rank_entry") or row.get("soft_reason") or meta.get("soft_reason"))
    return {
        "symbol": symbol,
        **evs,
        "static_rank_score": static,
        "microstructure_adjustment": micro_adj,
        "learning_adjustment": learned,
        "final_rank_score": final_rank,
        "entry_eligible": bool(row.get("entry_eligible") if row.get("entry_eligible") is not None else meta.get("entry_eligible")),
        "strategy_passed": strategy_passed,
        "soft_rank": bool(soft_rank) and not strategy_passed,
        "soft_reason": row.get("soft_reason") or meta.get("soft_reason"),
        "hard_block": row.get("hard_block") or meta.get("hard_block"),
        "rank_components": row.get("rank_components") or meta.get("rank_components") or ctx.get("rank_components") or {},
        "selection_version": row.get("selection_version") or meta.get("selection_version") or ctx.get("selection_version"),
        "ev_position": None,
    }


def build_peer_micro_snapshot(
    rows: list[dict[str, Any]] | None,
    *,
    open_symbols: set[str] | list[str] | None = None,
    max_open: int = 4,
    open_count: int | None = None,
    selected_symbol: str | None = None,
) -> dict[str, Any]:
    """Four-symbol ranking-time snapshot. Does not sort or mutate ``rows``."""
    open_set = {str(s).upper() for s in (open_symbols or [])}
    src = list(rows or [])
    by_sym = {str(r.get("symbol") or "").upper(): r for r in src}
    extracted = {sym: extract_peer_fields(by_sym[sym]) for sym in by_sym if sym}
    ordered_syms = sorted(
        extracted,
        key=lambda s: (
            -int(bool(extracted[s].get("entry_eligible"))),
            -float(extracted[s].get("final_rank_score") or 0.0),
        ),
    )
    available_syms = [s for s in ordered_syms if extracted[s].get("entry_eligible") and s not in open_set]
    peers: dict[str, Any] = {}
    for pos, sym in enumerate(ordered_syms, start=1):
        rec = dict(extracted[sym])
        rec["rank_position"] = pos
        rec["already_open"] = sym in open_set
        rec["available"] = bool(rec.get("entry_eligible")) and sym not in open_set
        rec["available_rank_position"] = (available_syms.index(sym) + 1) if sym in available_syms else None
        peers[sym] = rec
    for sym in PEER_SYMBOLS:
        if sym not in peers:
            peers[sym] = {
                "symbol": sym,
                **dict.fromkeys(EV_KEYS),
                "static_rank_score": None,
                "microstructure_adjustment": None,
                "learning_adjustment": None,
                "final_rank_score": None,
                "rank_position": None,
                "available_rank_position": None,
                "entry_eligible": False,
                "strategy_passed": False,
                "soft_rank": False,
                "already_open": sym in open_set,
                "available": False,
                "hard_block": "NOT_IN_CYCLE",
            }
    ev_ordered = sorted(
        [s for s in available_syms if extracted[s].get("EV_10s") is not None],
        key=lambda s: -float(extracted[s].get("EV_10s") or 0.0),
    )
    for pos, sym in enumerate(ev_ordered, start=1):
        if sym in peers:
            peers[sym]["ev_position"] = pos
    used_open = int(open_count if open_count is not None else len(open_set))
    slots_free = used_open < int(max_open)
    selected = str(selected_symbol).upper() if selected_symbol else None
    sel_rec = peers.get(selected) if selected else None
    return {
        "version": "scalp_peer_micro_v1",
        "selected_symbol": selected,
        "open_symbols": sorted(open_set),
        "open_count": used_open,
        "max_open": int(max_open),
        "slots_available": bool(slots_free),
        "selected_rank_position": (sel_rec or {}).get("available_rank_position") or (sel_rec or {}).get("rank_position"),
        "selected_ev_position": (sel_rec or {}).get("ev_position"),
        "selected_final_rank": (sel_rec or {}).get("final_rank_score"),
        "peers": peers,
    }


def size_diagnostics(
    sizing: Any,
    *,
    base_notional: float,
    qty: float,
    strategy_passed: bool,
    microstructure_size_factor: float,
    learning_size_multiplier: float,
    soft_rank_multiplier: float,
    calibration_mult: float,
    arm_penalty_mult: float,
    mtf_penalty_mult: float,
) -> dict[str, Any]:
    """Copy already-used sizing inputs. Does not recompute notional or qty."""
    return {
        "base_notional": float(base_notional),
        "confidence_factor": float(getattr(sizing, "confidence_factor", 0.0) or 0.0),
        "microstructure_size_factor": float(microstructure_size_factor),
        "learning_size_multiplier": float(learning_size_multiplier),
        "soft_rank_multiplier": float(soft_rank_multiplier),
        "volatility_adjustment": float(getattr(sizing, "volatility_adjustment", 1.0) or 1.0),
        "liquidity_adjustment": float(getattr(sizing, "liquidity_adjustment", 1.0) or 1.0),
        "calibration_mult": float(calibration_mult),
        "arm_penalty_mult": float(arm_penalty_mult),
        "mtf_penalty_mult": float(mtf_penalty_mult),
        "final_combined_multiplier": float(getattr(sizing, "combined_multiplier", 0.0) or 0.0),
        "capped_final_notional": float(getattr(sizing, "notional", 0.0) or 0.0),
        "actual_selected_qty": float(qty),
        "strategy_passed": bool(strategy_passed),
    }


__all__ = [
    "EV_KEYS",
    "PEER_SYMBOLS",
    "build_peer_micro_snapshot",
    "extract_peer_fields",
    "size_diagnostics",
]
