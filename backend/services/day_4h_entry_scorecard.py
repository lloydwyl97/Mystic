"""Offline entry-quality / ranking-regret scorecard. Research only."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_decision_label_contract import TABLE_LABELS
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS

DISTANCE_BINS = (
    ("<=0", -1e18, 0.0),
    ("0-5", 0.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-50", 30.0, 50.0),
    ("50+", 50.0, 1e18),
)


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(xs: list[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return None if not vals else sum(vals) / len(vals)


def _median(xs: list[float | None]) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _rate(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return sum(1 for f in flags if f) / len(flags)


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def load_scorecard_rows(db_path: str | Path, *, since: datetime | None = None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    try:
        groups = conn.execute(
            f"SELECT decision_group_id, created_at, selected_action, selected_symbol, contract_json FROM {TABLE_GROUPS}"
        ).fetchall()
        cands = conn.execute(
            f"SELECT decision_group_id, symbol, path_ev, p_buy, final_rank_score, feature_json FROM {TABLE_CANDIDATES}"
        ).fetchall()
        labels = conn.execute(
            f"SELECT decision_group_id, symbol, provenance, production_exit_net_bps, mfe_bps, mae_bps, cost_cover, exit_reason, regret_vs_hold_bps, label_json FROM {TABLE_LABELS}"
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return []
    conn.close()
    by_group: dict[str, dict[str, Any]] = {}
    for gid, created, action, symbol, contract in groups:
        created_dt = _parse_iso(created)
        if since and created_dt and created_dt < since:
            continue
        payload = _loads(contract)
        by_group[str(gid)] = {
            "decision_group_id": gid,
            "created_at": created,
            "selected_action": action,
            "selected_symbol": symbol,
            "contract": payload,
            "candidates": {},
            "labels": {},
        }
    for gid, symbol, path_ev, p_buy, final, feats in cands:
        if gid not in by_group:
            continue
        by_group[gid]["candidates"][symbol] = {
            "path_ev": path_ev,
            "p_buy": p_buy,
            "final_rank_score": final,
            "feature_json": _loads(feats),
        }
    for gid, symbol, prov, net, mfe, mae, cover, reason, regret, raw in labels:
        if gid not in by_group:
            continue
        extra = _loads(raw)
        by_group[gid]["labels"][symbol] = {
            "provenance": prov,
            "production_exit_net_bps": net if net is not None else extra.get("production_exit_net_bps"),
            "mfe_bps": mfe if mfe is not None else extra.get("mfe_bps"),
            "mae_bps": mae if mae is not None else extra.get("mae_bps"),
            "cost_cover": bool(cover) if cover is not None else bool(extra.get("covered_genuine_cost")),
            "exit_reason": reason or extra.get("exit_reason"),
            "regret_vs_hold_bps": regret if regret is not None else extra.get("regret_vs_hold_bps"),
            **extra,
        }
    return list(by_group.values())


def _selected_4h(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("contract") or {}
    selected = str(row.get("selected_symbol") or HOLD_SYMBOL)
    tel = (contract.get("4h_entry_telemetry") or {}).get(selected) or {}
    if not tel:
        feat = ((row.get("candidates") or {}).get(selected) or {}).get("feature_json") or {}
        tel = feat.get("4h_entry_telemetry") or {}
    return tel


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nets = []
    mfes = []
    maes = []
    covers = []
    bes = []
    trails = []
    profits = []
    for row in rows:
        selected = str(row.get("selected_symbol") or HOLD_SYMBOL)
        lab = (row.get("labels") or {}).get(selected) or {}
        nets.append(_num(lab.get("production_exit_net_bps")))
        mfes.append(_num(lab.get("mfe_bps")))
        maes.append(_num(lab.get("mae_bps")))
        covers.append(bool(lab.get("covered_genuine_cost") or lab.get("cost_cover")))
        bes.append(bool(lab.get("reached_production_BE_level")))
        trails.append(bool(lab.get("reached_production_trail_level")))
        net = _num(lab.get("production_exit_net_bps"))
        profits.append(bool(net is not None and net > 0))
    return {
        "n": len(rows),
        "mean_net_bps": _mean(nets),
        "median_net_bps": _median(nets),
        "profit_rate": _rate(profits),
        "cost_cover_rate": _rate(covers),
        "BE_rate": _rate(bes),
        "trail_rate": _rate(trails),
        "mean_MFE": _mean(mfes),
        "mean_MAE": _mean(maes),
    }


def consistency_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def pick(name: str, pred) -> dict[str, Any]:
        chosen = [r for r in rows if pred(r)]
        return {"name": name, **_metrics(chosen)}

    def selected_exec(row: dict[str, Any]) -> bool:
        return str(row.get("selected_action") or "").upper().startswith("BUY")

    def lab(row: dict[str, Any]) -> dict[str, Any]:
        return (row.get("labels") or {}).get(str(row.get("selected_symbol") or "")) or {}

    def tel(row: dict[str, Any]) -> dict[str, Any]:
        return _selected_4h(row)

    def peer(row: dict[str, Any]) -> dict[str, Any]:
        return (row.get("contract") or {}).get("4h_peer_structure") or row.get("contract") or {}

    executed = [r for r in rows if selected_exec(r)]
    return {
        "selected_already_broken_at_decision": pick(
            "selected already broken at decision",
            lambda r: selected_exec(r) and peer(r).get("selected_already_broken_at_ranking") is True,
        ),
        "selected_not_broken_at_decision": pick(
            "selected not broken at decision",
            lambda r: selected_exec(r) and peer(r).get("selected_already_broken_at_ranking") is False,
        ),
        "selected_broken_while_peer_intact": pick(
            "selected broken while at least one peer intact",
            lambda r: selected_exec(r) and peer(r).get("selected_broken_peer_intact_flag") is True,
        ),
        "all_four_broken": pick(
            "all four broken",
            lambda r: peer(r).get("all_four_already_broken") is True,
        ),
        "selected_closest_to_invalidation": pick(
            "selected closest to invalidation",
            lambda r: selected_exec(r)
            and tel(r).get("distance_to_4h_break_bps") is not None
            and peer(r).get("healthiest_peer_distance_bps") is not None
            and float(tel(r)["distance_to_4h_break_bps"]) <= float(peer(r)["healthiest_peer_distance_bps"])
            and str(r.get("selected_symbol")) != str(peer(r).get("healthiest_peer_symbol")),
        ),
        "selected_healthiest_structure": pick(
            "selected healthiest structure",
            lambda r: selected_exec(r) and str(r.get("selected_symbol")) == str(peer(r).get("healthiest_peer_symbol")),
        ),
        "selected_breaks_within_3m": pick(
            "selected breaks within 3 minutes",
            lambda r: selected_exec(r) and lab(r).get("4h_break_within_3m") is True,
        ),
        "selected_breaks_within_15m": pick(
            "selected breaks within 15 minutes",
            lambda r: selected_exec(r) and lab(r).get("4h_break_within_15m") is True,
        ),
        "selected_breaks_within_30m": pick(
            "selected breaks within 30 minutes",
            lambda r: selected_exec(r) and lab(r).get("4h_break_within_30m") is True,
        ),
        "n_executed": len(executed),
    }


PATH_EV_BINS = (
    ("<=0", -1e18, 0.0),
    ("0-5bps", 0.0, 0.0005),
    ("5-15bps", 0.0005, 0.0015),
    ("15-30bps", 0.0015, 0.0030),
    ("30bps+", 0.0030, 1e18),
)
P_BUY_BINS = (
    ("<0.50", -1e18, 0.50),
    ("0.50-0.60", 0.50, 0.60),
    ("0.60-0.75", 0.60, 0.75),
    ("0.75-0.90", 0.75, 0.90),
    ("0.90+", 0.90, 1e18),
)
RANK_SCORE_BINS = (
    ("<0.60", -1e18, 0.60),
    ("0.60-0.70", 0.60, 0.70),
    ("0.70-0.80", 0.70, 0.80),
    ("0.80+", 0.80, 1e18),
)
SPREAD_BINS = (
    ("<=0.5bps", -1e18, 0.5),
    ("0.5-1bps", 0.5, 1.0),
    ("1-2bps", 1.0, 2.0),
    ("2bps+", 2.0, 1e18),
)
VOLATILITY_BINS = (
    ("<50bps", -1e18, 50.0),
    ("50-100bps", 50.0, 100.0),
    ("100-200bps", 100.0, 200.0),
    ("200bps+", 200.0, 1e18),
)
SESSIONS = (("asia", 0, 8), ("europe", 8, 14), ("us", 14, 22), ("late", 22, 24))


def _bin_label(value: float | None, bins: tuple[tuple[str, float, float], ...]) -> str:
    if value is None:
        return "unknown"
    for name, low, high in bins:
        if low <= float(value) < high:
            return name
    return "unknown"


def _selected_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Candidate payload for the selected symbol, preferring the stored contract copy."""
    selected = str(row.get("selected_symbol") or HOLD_SYMBOL)
    for cand in (row.get("contract") or {}).get("candidates") or []:
        if str(cand.get("symbol")) == selected:
            return cand
    return (row.get("candidates") or {}).get(selected) or {}


def _session_of(created_at: Any) -> str:
    parsed = _parse_iso(created_at if isinstance(created_at, str) else None)
    if parsed is None:
        return "unknown"
    hour = parsed.astimezone(timezone.utc).hour
    for name, low, high in SESSIONS:
        if low <= hour < high:
            return name
    return "unknown"


def _hour_of(created_at: Any) -> str:
    parsed = _parse_iso(created_at if isinstance(created_at, str) else None)
    return "unknown" if parsed is None else f"{parsed.astimezone(timezone.utc).hour:02d}"


def _volatility_bps(row: dict[str, Any]) -> float | None:
    """Prior completed 4H range in bps. The only volatility measure stored per decision."""
    tel = _selected_4h(row)
    high = _num(tel.get("prior_4h_high"))
    low = _num(tel.get("prior_4h_low"))
    close = _num(tel.get("prior_4h_close"))
    if high is None or low is None or not close:
        return None
    return (high - low) / close * 1e4


def _labeled_coin_nets(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, lab in (row.get("labels") or {}).items():
        if symbol == HOLD_SYMBOL or symbol not in COINS:
            continue
        net = _num(lab.get("production_exit_net_bps"))
        if net is None:
            net = _num((lab.get("markouts") or {}).get("4h")) if isinstance(lab.get("markouts"), dict) else None
        if net is not None:
            out[str(symbol)] = float(net)
    return out


def breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Entry-quality metrics sliced by the dimensions stored on each decision.

    Dimensions with no stored field resolve to an `unknown` bucket rather than being
    dropped, so a missing feed is visible instead of silently absent.
    """
    executed = [r for r in rows if str(r.get("selected_action") or "").upper().startswith("BUY")]

    def group_by(name: str, key) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in executed:
            try:
                bucket = str(key(row))
            except Exception:
                bucket = "unknown"
            buckets.setdefault(bucket, []).append(row)
        return {name: {b: _metrics(rs) for b, rs in sorted(buckets.items())}}

    def cand_num(row: dict[str, Any], field: str) -> float | None:
        return _num(_selected_candidate(row).get(field))

    out: dict[str, Any] = {}
    out.update(group_by("symbol", lambda r: r.get("selected_symbol") or HOLD_SYMBOL))
    out.update(group_by("hour_utc", lambda r: _hour_of(r.get("created_at"))))
    out.update(group_by("session", lambda r: _session_of(r.get("created_at"))))
    out.update(group_by("4h_structure_state", lambda r: _selected_4h(r).get("4h_structure_state") or "unknown"))
    out.update(
        group_by(
            "distance_to_break_bin",
            lambda r: _bin_label(_num(_selected_4h(r).get("distance_to_4h_break_bps")), DISTANCE_BINS),
        )
    )
    out.update(group_by("path_ev_bin", lambda r: _bin_label(cand_num(r, "path_ev"), PATH_EV_BINS)))
    out.update(group_by("p_buy_bin", lambda r: _bin_label(cand_num(r, "p_buy"), P_BUY_BINS)))
    out.update(group_by("rank_score_bin", lambda r: _bin_label(cand_num(r, "final_rank_score"), RANK_SCORE_BINS)))
    out.update(group_by("spread_state", lambda r: _bin_label(cand_num(r, "spread_bps"), SPREAD_BINS)))
    out.update(group_by("volatility_state", lambda r: _bin_label(_volatility_bps(r), VOLATILITY_BINS)))
    out.update(
        group_by(
            "liquidity_state",
            lambda r: _bin_label(cand_num(r, "expected_slippage"), SPREAD_BINS),
        )
    )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    executed = [r for r in rows if str(r.get("selected_action") or "").upper().startswith("BUY")]
    holds = [r for r in rows if not str(r.get("selected_action") or "").upper().startswith("BUY")]
    already = []
    rapid = []
    regrets = []
    peer_healthier = []
    all_weak = []
    broken_peer_intact = []
    labeled = 0
    missing = 0
    for row in executed:
        selected = str(row.get("selected_symbol") or "")
        lab = (row.get("labels") or {}).get(selected) or {}
        peer = (row.get("contract") or {}).get("4h_peer_structure") or row.get("contract") or {}
        if lab:
            labeled += 1
        else:
            missing += 1
        already.append(bool(peer.get("selected_already_broken_at_ranking")))
        rapid.append(bool(lab.get("4h_break_within_3m")))
        regrets.append(_num(lab.get("regret_vs_hold_bps") if lab.get("regret_vs_hold_bps") is not None else lab.get("production_exit_net_bps")))
        if peer.get("selected_vs_best_peer_distance_bps") not in (None, "") and float(peer.get("selected_vs_best_peer_distance_bps") or 0) > 0:
            peer_healthier.append(True)
        all_weak.append(bool(peer.get("all_four_already_broken")))
        broken_peer_intact.append(bool(peer.get("selected_broken_peer_intact_flag")))
    nets = [_num(((r.get("labels") or {}).get(str(r.get("selected_symbol") or "")) or {}).get("production_exit_net_bps")) for r in executed]
    selected_metrics = _metrics(executed)
    near_break: dict[str, int] = {name: 0 for name, _lo, _hi in DISTANCE_BINS}
    near_break["unknown"] = 0
    best_coin_hits: list[bool] = []
    regret_vs_best: list[float | None] = []
    for row in executed:
        near_break[_bin_label(_num(_selected_4h(row).get("distance_to_4h_break_bps")), DISTANCE_BINS)] += 1
        coin_nets = _labeled_coin_nets(row)
        selected = str(row.get("selected_symbol") or "")
        if len(coin_nets) < 2 or selected not in coin_nets:
            continue
        best_symbol = max(coin_nets, key=lambda s: coin_nets[s])
        best_coin_hits.append(coin_nets[selected] >= coin_nets[best_symbol] - 1e-9)
        regret_vs_best.append(coin_nets[best_symbol] - coin_nets[selected])
    return {
        "decision_groups": len(rows),
        "selected_trade_count": len(executed),
        "selected_HOLD_count": len(holds),
        "selected_already_4h_broken_rate": _rate(already),
        "selected_near_break_distribution": near_break,
        "rapid_4h_break_rate": _rate(rapid),
        "cost_cover_rate": selected_metrics["cost_cover_rate"],
        "BE_rate": selected_metrics["BE_rate"],
        "trail_rate": selected_metrics["trail_rate"],
        "positive_net_rate": _rate([bool(n is not None and n > 0) for n in nets]),
        "average_net_bps": _mean(nets),
        "median_net_bps": _median(nets),
        "MFE": selected_metrics["mean_MFE"],
        "MAE": selected_metrics["mean_MAE"],
        "regret_vs_HOLD": _mean(regrets),
        "regret_vs_best_labeled_candidate": _mean(regret_vs_best),
        "best_coin_selection_rate": _rate(best_coin_hits),
        "selected_negative_rate": _rate([bool(n is not None and n < 0) for n in nets]),
        "all_four_weak_rate": _rate(all_weak),
        "selected_broken_while_peer_intact_rate": _rate(broken_peer_intact),
        "another_coin_healthier_structure_rate": _rate(peer_healthier) if peer_healthier or executed else None,
        "label_coverage": labeled,
        "label_missing": missing,
        "consistency": consistency_table(rows),
        "breakdowns": breakdowns(rows),
    }


def window_since(window: str) -> datetime | None:
    raw = str(window or "").strip().lower()
    now = datetime.now(timezone.utc)
    if raw in {"24h", "1d"}:
        return now - timedelta(hours=24)
    if raw in {"7d", "7day", "7days"}:
        return now - timedelta(days=7)
    if raw in {"30d", "30day", "30days"}:
        return now - timedelta(days=30)
    return None


def build_scorecard(db_path: str | Path, *, window: str = "24h") -> dict[str, Any]:
    rows = load_scorecard_rows(db_path, since=window_since(window))
    return {"window": window, "generated_at": datetime.now(timezone.utc).isoformat(), **summarize(rows)}


__all__ = [
    "DISTANCE_BINS",
    "PATH_EV_BINS",
    "P_BUY_BINS",
    "RANK_SCORE_BINS",
    "SPREAD_BINS",
    "VOLATILITY_BINS",
    "breakdowns",
    "build_scorecard",
    "consistency_table",
    "load_scorecard_rows",
    "summarize",
]
