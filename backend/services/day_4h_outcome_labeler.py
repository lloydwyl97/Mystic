"""Offline DAY outcome labeler. Never participates in live ranking.

Attaches matured markouts / MFE / MAE / 4H-break timing / production exits
to prior ranking candidates. Fail-open. Does not invent fills.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL, asof_bundle_from_1m, drop_bars_after
from backend.services.day_decision_label_contract import TABLE_LABELS, ensure_label_schema, write_outcome_label
from backend.services.day_production_lifecycle_replay import parse_epoch
from backend.services.day_trade_thesis import htf_4h_rise_broken

LABEL_VERSION = "day_4h_outcome_label_v1"
HORIZONS_SEC = {
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
}
BREAK_WINDOWS_SEC = {
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}
BE_TRIGGER_BPS = 30.0
DEFAULT_TRAIL_BPS = 40.0
MAX_LIFECYCLE_SEC = 8 * 3600


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bps(start: float | None, end: float | None) -> float | None:
    if start in (None, 0) or end is None:
        return None
    return (float(end) - float(start)) / float(start) * 1e4


def provenance_or_unknown(value: str | None) -> str:
    if value in {"authoritative", "reconstructed", "estimated", "unknown"}:
        return value
    return "unknown"


def horizon_mature(*, decision_epoch: float, horizon_sec: float, now_epoch: float | None = None) -> bool:
    now = _now_epoch() if now_epoch is None else float(now_epoch)
    return now + 1e-9 >= float(decision_epoch) + float(horizon_sec)


def clip_bars_asof(bars: list[tuple[int, float, ...]], cutoff_epoch: float) -> list[tuple[int, float, ...]]:
    return [b for b in bars if int(b[0]) <= float(cutoff_epoch) + 1e-9]


def markout_at(
    bars: list[tuple[int, float, ...]],
    *,
    entry_px: float,
    decision_epoch: float,
    horizon_sec: float,
) -> tuple[float | None, str]:
    if entry_px <= 0:
        return None, "unknown"
    target = float(decision_epoch) + float(horizon_sec)
    chosen = None
    for bar in bars:
        if int(bar[0]) < decision_epoch - 1e-9:
            continue
        if int(bar[0]) > target + 1e-9:
            break
        chosen = bar
    if chosen is None:
        return None, "unknown"
    return _bps(entry_px, float(chosen[4])), "reconstructed"


def path_extrema(
    bars: list[tuple[int, float, ...]],
    *,
    entry_px: float,
    start_epoch: float,
    end_epoch: float,
) -> dict[str, Any]:
    if entry_px <= 0:
        return {"mfe_bps": None, "mae_bps": None, "time_to_mfe_sec": None, "time_to_mae_sec": None, "provenance": "unknown"}
    hi = lo = entry_px
    t_hi = t_lo = None
    seen = False
    for bar in bars:
        ep = int(bar[0])
        if ep < start_epoch - 1e-9:
            continue
        if ep > end_epoch + 1e-9:
            break
        seen = True
        high = float(bar[2])
        low = float(bar[3])
        if high >= hi:
            hi = high
            t_hi = ep
        if low <= lo:
            lo = low
            t_lo = ep
    if not seen:
        return {"mfe_bps": None, "mae_bps": None, "time_to_mfe_sec": None, "time_to_mae_sec": None, "provenance": "unknown"}
    return {
        "mfe_bps": (hi - entry_px) / entry_px * 1e4,
        "mae_bps": (lo - entry_px) / entry_px * 1e4,
        "time_to_mfe_sec": None if t_hi is None else max(0.0, float(t_hi) - float(start_epoch)),
        "time_to_mae_sec": None if t_lo is None else max(0.0, float(t_lo) - float(start_epoch)),
        "provenance": "reconstructed",
    }


def first_4h_break_seconds(
    bars: list[tuple[int, float, ...]],
    *,
    decision_epoch: float,
    end_epoch: float,
    seed_4h: list[list[float]] | None = None,
) -> float | None:
    clipped = clip_bars_asof(bars, end_epoch)
    if not clipped:
        return None
    last_state = False
    for bar in clipped:
        ep = int(bar[0])
        if ep < decision_epoch - 1e-9:
            continue
        bundle = asof_bundle_from_1m(clip_bars_asof(clipped, ep), float(ep), seed_4h=seed_4h)
        bundle["4h"] = drop_bars_after(bundle.get("4h"), float(ep))
        broken = bool(htf_4h_rise_broken(bundle, current_price=float(bar[4]), now_epoch=float(ep)))
        if broken and not last_state:
            return max(0.0, float(ep) - float(decision_epoch))
        last_state = broken
    return None


def hold_label(*, decision_group_id: str, decision_epoch: float, now_epoch: float) -> dict[str, Any]:
    mature = {name: horizon_mature(decision_epoch=decision_epoch, horizon_sec=sec, now_epoch=now_epoch) for name, sec in HORIZONS_SEC.items()}
    return {
        "decision_group_id": decision_group_id,
        "symbol": HOLD_SYMBOL,
        "provenance": "authoritative",
        "markouts": {name: 0.0 if ready else None for name, ready in mature.items()},
        "mfe_bps": 0.0,
        "mae_bps": 0.0,
        "time_to_mfe_sec": 0.0,
        "time_to_mae_sec": 0.0,
        "cost_cover": False,
        "covered_genuine_cost": False,
        "time_to_cost_cover": None,
        "reached_production_BE_level": False,
        "reached_production_trail_level": False,
        "production_exit_gross_bps": 0.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "production_exit_net_bps": 0.0,
        "production_exit_net_dollars": 0.0,
        "holding_time_sec": 0.0,
        "capture_ratio": 0.0,
        "gross_capture_ratio": 0.0,
        "net_capture_ratio": 0.0,
        "exit_reason": "HOLD",
        "regret_vs_hold_bps": 0.0,
        "counterfactual": False,
        "label_started_at": datetime.fromtimestamp(decision_epoch, tz=timezone.utc).isoformat(),
        "label_completed_at": _now_iso() if all(mature.values()) else None,
        "label_version": LABEL_VERSION,
        "market_data_cutoff": datetime.fromtimestamp(min(now_epoch, decision_epoch + HORIZONS_SEC["4h"]), tz=timezone.utc).isoformat(),
    }


def label_candidate(
    *,
    decision_group_id: str,
    symbol: str,
    decision_epoch: float,
    entry_px: float | None,
    bars: list[tuple[int, float, ...]],
    now_epoch: float | None = None,
    fill: dict[str, Any] | None = None,
    seed_4h: list[list[float]] | None = None,
    trail_bps: float = DEFAULT_TRAIL_BPS,
) -> dict[str, Any]:
    now = _now_epoch() if now_epoch is None else float(now_epoch)
    if symbol == HOLD_SYMBOL:
        return hold_label(decision_group_id=decision_group_id, decision_epoch=decision_epoch, now_epoch=now)
    cost = honest_all_in_rt_pct(symbol) * 1e4
    end = min(now, decision_epoch + MAX_LIFECYCLE_SEC)
    if fill and fill.get("exit_epoch"):
        end = min(end, float(fill["exit_epoch"]))
    extrema = path_extrema(bars, entry_px=float(entry_px or 0.0), start_epoch=decision_epoch, end_epoch=end)
    markouts: dict[str, float | None] = {}
    for name, sec in HORIZONS_SEC.items():
        if not horizon_mature(decision_epoch=decision_epoch, horizon_sec=sec, now_epoch=now):
            markouts[name] = None
            continue
        val, _prov = markout_at(bars, entry_px=float(entry_px or 0.0), decision_epoch=decision_epoch, horizon_sec=sec)
        markouts[name] = val
    break_sec = first_4h_break_seconds(bars, decision_epoch=decision_epoch, end_epoch=end, seed_4h=seed_4h)
    mfe = extrema.get("mfe_bps")
    covered = bool(mfe is not None and mfe + 1e-12 >= cost)
    time_to_cover = None
    if covered and mfe is not None:
        time_to_cover = extrema.get("time_to_mfe_sec")
    fill_net = _num((fill or {}).get("net_bps"))
    fill_usd = _num((fill or {}).get("net_dollars"))
    fill_reason = (fill or {}).get("exit_reason")
    fill_hold = _num((fill or {}).get("holding_seconds"))
    authoritative = bool(fill and fill.get("exit_epoch") and fill_net is not None)
    capture_gross = None
    capture_net = None
    if mfe not in (None, 0) and fill_net is not None:
        capture_net = fill_net / mfe if mfe else None
    if mfe not in (None, 0) and fill and fill.get("gross_bps") is not None:
        capture_gross = float(fill["gross_bps"]) / mfe
    payload = {
        "decision_group_id": decision_group_id,
        "symbol": symbol,
        "provenance": "authoritative" if authoritative else ("reconstructed" if entry_px else "unknown"),
        "markouts": markouts,
        "mfe_bps": extrema.get("mfe_bps"),
        "mae_bps": extrema.get("mae_bps"),
        "time_to_mfe_sec": extrema.get("time_to_mfe_sec"),
        "time_to_mae_sec": extrema.get("time_to_mae_sec"),
        "cost_cover": covered,
        "covered_genuine_cost": covered,
        "time_to_cost_cover": time_to_cover,
        "reached_production_BE_level": bool(mfe is not None and mfe + 1e-12 >= BE_TRIGGER_BPS),
        "reached_production_trail_level": bool(mfe is not None and mfe + 1e-12 >= trail_bps),
        "4h_break_within_3m": None if not horizon_mature(decision_epoch=decision_epoch, horizon_sec=180, now_epoch=now) else bool(break_sec is not None and break_sec < 180),
        "4h_break_within_5m": None if not horizon_mature(decision_epoch=decision_epoch, horizon_sec=300, now_epoch=now) else bool(break_sec is not None and break_sec < 300),
        "4h_break_within_15m": None if not horizon_mature(decision_epoch=decision_epoch, horizon_sec=900, now_epoch=now) else bool(break_sec is not None and break_sec < 900),
        "4h_break_within_30m": None if not horizon_mature(decision_epoch=decision_epoch, horizon_sec=1800, now_epoch=now) else bool(break_sec is not None and break_sec < 1800),
        "4h_break_within_1h": None if not horizon_mature(decision_epoch=decision_epoch, horizon_sec=3600, now_epoch=now) else bool(break_sec is not None and break_sec < 3600),
        "production_exit_gross_bps": (fill or {}).get("gross_bps") if authoritative else None,
        "commission_bps": (fill or {}).get("commission_bps") if authoritative else None,
        "spread_bps": (fill or {}).get("spread_bps") if authoritative else None,
        "slippage_bps": (fill or {}).get("slippage_bps") if authoritative else None,
        "production_exit_net_bps": fill_net if authoritative else None,
        "production_exit_net_dollars": fill_usd if authoritative else None,
        "holding_time_sec": fill_hold if authoritative else None,
        "capture_ratio": capture_net,
        "gross_capture_ratio": capture_gross,
        "net_capture_ratio": capture_net,
        "exit_reason": fill_reason if authoritative else None,
        "regret_vs_hold_bps": fill_net if authoritative else None,
        "counterfactual": not authoritative,
        "label_started_at": datetime.fromtimestamp(decision_epoch, tz=timezone.utc).isoformat(),
        "label_completed_at": _now_iso() if horizon_mature(decision_epoch=decision_epoch, horizon_sec=HORIZONS_SEC["4h"], now_epoch=now) else None,
        "label_version": LABEL_VERSION,
        "market_data_cutoff": datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
        "estimated_all_in_cost_bps": cost,
        "time_to_4h_break_sec": break_sec,
    }
    if not authoritative:
        payload["provenance"] = "reconstructed" if entry_px else "unknown"
    return payload


def persist_label(db_path: str | Path, payload: dict[str, Any]) -> None:
    """Write contract columns plus extended 4H fields in label_json. Offline only."""
    try:
        write_outcome_label(db_path, payload)
        ensure_label_schema(db_path)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            raw = conn.execute(
                f"SELECT label_json FROM {TABLE_LABELS} WHERE decision_group_id=? AND symbol=?",
                (payload.get("decision_group_id"), payload.get("symbol")),
            ).fetchone()
            stored = json.loads(raw[0]) if raw and raw[0] else {}
            stored.update(payload)
            stored["label_version"] = LABEL_VERSION
            conn.execute(
                f"UPDATE {TABLE_LABELS} SET label_json=? WHERE decision_group_id=? AND symbol=?",
                (json.dumps(stored, default=str)[:32000], payload.get("decision_group_id"), payload.get("symbol")),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return


def load_1m_bars(conn: sqlite3.Connection, symbol: str) -> list[tuple[int, float, float, float, float, float]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(feature_ohlcv)")}
    if "ts" not in present:
        return []
    names = (symbol, f"{symbol[:-4]}-USDT", f"{symbol[:-4]}/USDT")
    for name in names:
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE interval='1m' AND symbol=? ORDER BY ts ASC",
            (name,),
        ).fetchall()
        if not rows:
            continue
        out = []
        for ts, o, h, low, c, v in rows:
            ep = parse_epoch(ts)
            if ep is None:
                continue
            out.append((int(ep), float(o or 0), float(h or 0), float(low or 0), float(c or 0), float(v or 0)))
        return out
    return []


__all__ = [
    "BE_TRIGGER_BPS",
    "COINS",
    "HOLD_SYMBOL",
    "HORIZONS_SEC",
    "LABEL_VERSION",
    "clip_bars_asof",
    "first_4h_break_seconds",
    "hold_label",
    "horizon_mature",
    "label_candidate",
    "persist_label",
    "provenance_or_unknown",
]
