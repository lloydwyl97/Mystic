"""Versioned DAY research rows: one per current-version DAY candidate.

Each row joins, by symbol and time only, the latest stored 145-dim vector with
``ts_utc <= decision_ts`` and the last 15m order-flow bar that had closed by
``decision_ts``. ``decision_ts`` is the candidate's ``evaluated_at`` (the bar
boundary the decision was keyed to), which is never later than when the ask
was read, so every joined input was observable at the decision.

Market alpha and realized policy value stay separate:

* ``market_label`` is the lifecycle replay of the live exit contract from the
  decision ask, present for every candidate once final.
* ``realized_net`` is only set for a filled candidate with a recorded close;
  unfilled candidates carry ``None``, never an imputed policy result.
* ``policy_effect = realized_net - market_label`` only when both exist, so
  ``realized_net == market_label + policy_effect`` holds by construction.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from backend.services.day_feature_catalog import DAY_FEATURE_CATALOG_VERSION, DAY_VECTOR_DIM, causal_features

DAY_RESEARCH_ROWS_VERSION = "DAY_RESEARCH_ROWS_V1"
DAY_ENGINE = "DAY_V2"
VECTOR_MAX_AGE_SEC = 300.0
FLOW_BAR_SEC = 900


def _epoch(ts: Any) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, int | float):
        return float(ts)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _json(raw: Any) -> dict[str, Any]:
    try:
        out = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class _VectorIndex:
    """Per-symbol vectors sorted by stamp; lookup is strictly as-of."""

    def __init__(self, conn: sqlite3.Connection, since: float, until: float) -> None:
        self.ts: dict[str, list[float]] = defaultdict(list)
        self.rows: dict[str, list[tuple]] = defaultdict(list)
        lo = datetime.fromtimestamp(since - VECTOR_MAX_AGE_SEC, tz=UTC).isoformat()
        hi = datetime.fromtimestamp(until, tz=UTC).isoformat()
        cur = conn.execute(
            "SELECT symbol, ts_utc, features_json, prob_buy, feature_version, feature_dim, model_artifact FROM ai_inference_log "
            "WHERE strategy_id='day' AND feature_dim=? AND ts_utc>=? AND ts_utc<=? ORDER BY symbol, ts_utc",
            (DAY_VECTOR_DIM, lo, hi),
        )
        for sym, ts, fj, pb, fv, fd, art in cur:
            e = _epoch(ts)
            if e is None:
                continue
            self.ts[str(sym)].append(e)
            self.rows[str(sym)].append((e, fj, pb, fv, fd, art))

    def asof(self, symbol: str, t: float) -> tuple | None:
        ts = self.ts.get(symbol, [])
        i = bisect.bisect_right(ts, t) - 1
        return self.rows[symbol][i] if i >= 0 else None


def _flow_index(conn: sqlite3.Connection, since: float, until: float) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    try:
        cur = conn.execute(
            "SELECT symbol, bar_open_epoch, trade_count, imbalance, notional_imbalance, cvd_notional, buy_notional, sell_notional, coverage_sec "
            "FROM day_order_flow_bars WHERE bar_sec=? AND bar_open_epoch>=? AND bar_open_epoch<?",
            (FLOW_BAR_SEC, int(since) - 4 * FLOW_BAR_SEC, int(until)),
        )
    except sqlite3.Error:
        return out
    for sym, bo, n, imb, nimb, cvd, bn, sn, cov in cur:
        out[str(sym)][int(bo)] = {
            "flow_trades": float(n or 0),
            "flow_imbalance": float(imb or 0),
            "flow_notional_imbalance": float(nimb or 0),
            "flow_cvd_notional": float(cvd or 0),
            "flow_log_notional": math.log1p(float(bn or 0) + float(sn or 0)),
            "flow_coverage_sec": float(cov or 0),
        }
    return out


def _closed_flow(flow: dict[int, dict[str, float]], decision_ts: float) -> tuple[int, dict[str, float]]:
    """Last bar with ``open + 900 <= decision_ts``. A bar with no prints is a real zero."""
    bar = int(math.floor(decision_ts / FLOW_BAR_SEC) * FLOW_BAR_SEC) - FLOW_BAR_SEC
    got = flow.get(bar)
    return bar, dict(got) if got else {"flow_trades": 0.0, "flow_imbalance": 0.0, "flow_notional_imbalance": 0.0, "flow_cvd_notional": 0.0, "flow_log_notional": 0.0, "flow_coverage_sec": 0.0}


def build_day_research_rows(db_path: str, *, economic_version: str | None = None, since: float = 0.0, until: float | None = None, read_only: bool = True) -> list[dict[str, Any]]:
    """Research rows for DAY candidates of ``economic_version`` (default: current)."""
    if economic_version is None:
        from backend.services.adaptive_learning import current_economic_version

        economic_version = current_economic_version(DAY_ENGINE)
    uri = f"file:{db_path}?mode=ro" if read_only else db_path
    conn = sqlite3.connect(uri, uri=read_only)
    conn.row_factory = sqlite3.Row
    try:
        cands = conn.execute(
            "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND economic_version=? AND evaluated_at>=? AND evaluated_at<? ORDER BY evaluated_at, id",
            (DAY_ENGINE, economic_version, float(since), float(until if until is not None else 1e12)),
        ).fetchall()
        if not cands:
            return []
        lo, hi = float(cands[0]["evaluated_at"]), float(cands[-1]["evaluated_at"])
        vectors = _VectorIndex(conn, lo, hi)
        flow = _flow_index(conn, lo, hi + FLOW_BAR_SEC)
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for r in cands:
        decision_ts = float(r["evaluated_at"])
        sym = str(r["symbol"])
        econ = _json(r["economic_json"])
        marks = _json(r["markouts_json"])
        life = marks.get("lifecycle") if isinstance(marks.get("lifecycle"), dict) else {}
        vec = vectors.asof(sym, decision_ts)
        features: dict[str, float] = {}
        feature_ts = prob_buy = None
        versions: dict[str, Any] = {
            "research": DAY_RESEARCH_ROWS_VERSION,
            "catalog": DAY_FEATURE_CATALOG_VERSION,
            "economic_version": str(r["economic_version"] or ""),
            "strategy_version": str(r["strategy_version"] or ""),
        }
        if vec is not None and decision_ts - vec[0] <= VECTOR_MAX_AGE_SEC:
            try:
                raw = json.loads(vec[1])
                features = causal_features([float(x) for x in raw])
                feature_ts, prob_buy = vec[0], _finite(vec[2])
                versions.update({"feature_version": vec[3], "feature_dim": vec[4], "model_artifact": str(vec[5] or "")})
            except (TypeError, ValueError):
                features = {}
        flow_bar, flow_feats = _closed_flow(flow.get(sym, {}), decision_ts)
        market = _finite(life.get("net")) if life else None
        market_final = market is not None and life.get("reason") is not None
        filled = bool(int(r["filled"] or 0))
        realized = _finite(r["realized_net"]) if filled else None
        out.append(
            {
                "candidate_id": int(r["id"]),
                "decision_ts": decision_ts,
                "decision_point": int(decision_ts // FLOW_BAR_SEC),
                "symbol": sym,
                "setup": str(r["setup"] or ""),
                "regime": str(r["regime"] or ""),
                "candidate_state": str(r["candidate_state"] or ""),
                "signaled": bool(int(r["signaled"] or 0)),
                "opportunity_id": str(r["opportunity_id"] or ""),
                "feature_ts": feature_ts,
                "feature_age_sec": (decision_ts - feature_ts) if feature_ts is not None else None,
                "features": features,
                "state_features": _json(r["features_json"]),
                "flow_bar_open": flow_bar,
                "flow": flow_feats,
                "prob_buy": prob_buy,
                "ref_price": float(r["ref_price"]),
                "roundtrip_cost": float(r["roundtrip_cost"] or 0),
                "market_alpha": _finite(econ.get("market_alpha")),
                "policy_gap": _finite(econ.get("policy_gap")),
                "policy_value": _finite(econ.get("policy_value")),
                "rank_score": _finite(econ.get("rank_score")),
                "rank_position": econ.get("rank_position"),
                "rank_of": econ.get("rank_of"),
                "size_mult": _finite(econ.get("size_mult")),
                "selected": filled,
                "market_label": market if market_final else None,
                "market_label_reason": str(life.get("reason") or "") if life else "",
                "market_label_censored": bool(life.get("censored")) if life else False,
                "market_mfe": _finite(life.get("mfe")) if life else None,
                "market_mae": _finite(life.get("mae")) if life else None,
                "realized_net": realized,
                "policy_effect": (realized - market) if realized is not None and market_final and market is not None else None,
                "versions": versions,
            }
        )
    return out


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Timestamp and lineage audit of research rows; every count except the totals must be 0."""
    return {
        "rows": len(rows),
        "with_vector": sum(1 for r in rows if r["features"]),
        "feature_after_decision": sum(1 for r in rows if r["feature_ts"] is not None and r["feature_ts"] > r["decision_ts"]),
        "flow_bar_unclosed": sum(1 for r in rows if r["flow_bar_open"] + FLOW_BAR_SEC > r["decision_ts"]),
        "policy_on_unfilled": sum(1 for r in rows if not r["selected"] and r["realized_net"] is not None),
        "untagged": sum(1 for r in rows if not r["versions"].get("economic_version") or not r["versions"].get("research")),
        "policy_identity_broken": sum(1 for r in rows if r["policy_effect"] is not None and abs(r["market_label"] + r["policy_effect"] - r["realized_net"]) > 1e-12),
    }


__all__ = ["DAY_RESEARCH_ROWS_VERSION", "VECTOR_MAX_AGE_SEC", "audit_rows", "build_day_research_rows"]
