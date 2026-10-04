"""Rebuild the 1d/1w-derived dims of stored DAY 145-feature vectors.

From 2026-09-18 the live DAY bundle served one frozen 1d copy and one frozen 1w
copy (canonical store held < 60 daily / < 24 weekly bars and the bundle reused
its prior copy without an age bound). Every DAY vector built in that window
carries the same 1d/1w-derived values. This module recomputes exactly those
dims from completed exchange bars as of each row's timestamp, using the current
live contract (completed bars only, DAY bundle depth), and leaves every other
dim untouched.

Dim 136 (mean EMA alignment) averages one term per non-4h DAY timeframe. Only
its 1d and 1w terms are swapped: the frozen terms are recovered from an exact
reconstruction of the frozen copies, validated against the frozen dims.
"""

from __future__ import annotations

from typing import Any

from backend.config.canonical_candle_intervals import align_open_ms, interval_ms
from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, fetch_limit_for_day_tf
from backend.services.ai_feature_v2 import _clip, _safe_float, _slope_norm
from backend.services.ai_market_context import _summarize_tf
from backend.services.day_active_market_bundle import month_context_four_from_daily
from backend.services.feature_builder import daily_calendar_changes

REBUILD_VERSION = "day_htf_rebuild_v1"
DIM_CHANGE_24H = 5
DIM_CHANGE_7D = 6
DIM_CHANGE_30D = 7
DIM_SLOPE_4H = 129
DIM_SLOPE_1D = 132
DIM_SLOPE_1W = 133
DIM_MONTH_LOG_RET = 134
DIM_MONTH_VOL = 135
DIM_MEAN_EMA = 136
# Identical on every vector built from the same frozen 1d copy.
FROZEN_1D_SIGNATURE_DIMS: tuple[int, ...] = (DIM_CHANGE_24H, DIM_CHANGE_7D, DIM_CHANGE_30D, DIM_SLOPE_1D, DIM_MONTH_LOG_RET, DIM_MONTH_VOL)
HTF_DERIVED_DIMS: tuple[int, ...] = (*FROZEN_1D_SIGNATURE_DIMS[:4], DIM_SLOPE_1W, DIM_MONTH_LOG_RET, DIM_MONTH_VOL, DIM_MEAN_EMA)
EMA_TERM_COUNT = sum(1 for tf in DAY_ACTIVE_TIMEFRAMES if tf != "4h")
DEPTH_1D = fetch_limit_for_day_tf("1d")
DEPTH_1W = fetch_limit_for_day_tf("1w")
DEPTH_4H = fetch_limit_for_day_tf("4h")


def completed_asof(rows: list[list], tf: str, t_ms: int, depth: int) -> list[list]:
    """Last ``depth`` bars of ascending ``rows`` fully closed at ``t_ms``."""
    width = interval_ms(tf)
    sel = [r for r in rows if int(r[0]) + width <= int(t_ms)]
    return sel[-int(depth) :]


def ema_term(rows: list[list] | None) -> float:
    """The per-TF term ``context_vector_day_full_mtf`` averages into dim 136."""
    snap = _summarize_tf(rows)
    return max(0.0, min(1.0, float(snap.get("ema_align", 0.5) or 0.5)))


def htf_dim_values(rows_1d: list[list], rows_1w: list[list]) -> dict[int, float] | None:
    """1d/1w-derived dims (except 136) exactly as the live builder computes them."""
    month, _err = month_context_four_from_daily(rows_1d)
    changes = daily_calendar_changes(rows_1d)
    if month is None or len(changes) < 3:
        return None
    return {
        DIM_CHANGE_24H: changes["change_24h"],
        DIM_CHANGE_7D: changes["change_7d"],
        DIM_CHANGE_30D: changes["change_30d"],
        DIM_SLOPE_1D: _slope_norm(_summarize_tf(rows_1d).get("slope", 0.0)),
        DIM_SLOPE_1W: _slope_norm(_summarize_tf(rows_1w).get("slope", 0.0)),
        DIM_MONTH_LOG_RET: _clip(_safe_float(month[0], 0.0), -6.0, 6.0),
        DIM_MONTH_VOL: _clip(_safe_float(month[1], 0.0), -6.0, 6.0),
    }


def reconstruct_frozen_daily(completed_1d: list[list], forming_open_ms: int, frozen_change_24h: float, depth: int = DEPTH_1D) -> list[list]:
    """Frozen REST copy: ``depth - 1`` completed bars + the bar forming at fetch time.

    Only closes enter the derived dims, and change_24h pins the forming close.
    """
    prior = [r for r in completed_1d if int(r[0]) < int(forming_open_ms)][-(int(depth) - 1) :]
    if len(prior) < 2:
        return []
    close = float(prior[-1][4]) * (1.0 + float(frozen_change_24h))
    return [*prior, [int(forming_open_ms), close, close, close, close, 0.0]]


def reconstruct_frozen_weekly(completed_1w: list[list], forming_open_ms: int, frozen_slope_1w: float, depth: int = DEPTH_1W) -> list[list]:
    """Frozen 1w copy; the forming close is pinned by the stored 1w slope."""
    prior = [r for r in completed_1w if int(r[0]) < int(forming_open_ms)][-(int(depth) - 1) :]
    if len(prior) < 2 or abs(float(frozen_slope_1w)) >= 0.20:
        return []
    lookback = min(20, len(prior)) or 1
    base = float(prior[-(lookback - 1)][4]) if lookback > 1 else float(prior[-1][4])
    close = base * (1.0 + float(frozen_slope_1w))
    return [*prior, [int(forming_open_ms), close, close, close, close, 0.0]]


def frozen_copy_matches(vec: list[float], rows_1d: list[list], rows_1w: list[list], *, tol: float = 1e-9) -> bool:
    vals = htf_dim_values(rows_1d, rows_1w)
    if vals is None:
        return False
    return all(abs(float(vec[d]) - float(vals[d])) <= tol for d in (*FROZEN_1D_SIGNATURE_DIMS, DIM_SLOPE_1W))


def is_legacy_4h_vector(vec: list[float]) -> bool:
    """Built before 4h lost authority: nonzero 4h slope and 4h inside the EMA mean."""
    return len(vec) > DIM_SLOPE_4H and float(vec[DIM_SLOPE_4H]) != 0.0


def legacy_4h_term(vec: list[float], rows_4h: list[list]) -> float | None:
    """4h EMA term of a legacy vector, only when ``rows_4h`` reproduce its stored 4h slope."""
    if not rows_4h:
        return None
    if abs(_slope_norm(_summarize_tf(rows_4h).get("slope", 0.0)) - float(vec[DIM_SLOPE_4H])) > 1e-9:
        return None
    return ema_term(rows_4h)


def rebuild_vector(
    old: list[float],
    *,
    new_1d: list[list],
    new_1w: list[list],
    frozen_ema_1d: float,
    frozen_ema_1w: float,
    legacy_ema_4h: float | None = None,
    keep_legacy_4h: bool = False,
) -> list[float] | None:
    """Swap the 1d/1w-derived dims of ``old`` for values from ``new_1d``/``new_1w``.

    A legacy vector (4h inside the mean) is moved to the current contract only
    when ``legacy_ema_4h`` is known: 4h slope zeroed and 4h removed from the mean.
    With ``keep_legacy_4h`` its 4h slope and 4h share of the mean stay as built.
    Returns None when the bars cannot produce the dims or dim 136 does not
    decompose into half-step EMA terms (the vector was not built from the frozen copies).
    """
    vals = htf_dim_values(new_1d, new_1w)
    if vals is None or len(old) <= DIM_MEAN_EMA:
        return None
    legacy = is_legacy_4h_vector(old)
    convert = legacy and legacy_ema_4h is not None
    if legacy and not convert and not keep_legacy_4h:
        return None
    count = EMA_TERM_COUNT + (1 if legacy else 0)
    others = float(old[DIM_MEAN_EMA]) * count - float(frozen_ema_1d) - float(frozen_ema_1w)
    if convert:
        others -= float(legacy_ema_4h)
    snapped = round(others * 2.0) / 2.0
    if abs(others - snapped) > 1e-6 or snapped < 0.0:
        return None
    out = [float(x) for x in old]
    for dim, value in vals.items():
        out[dim] = float(value)
    if convert:
        out[DIM_SLOPE_4H] = 0.0
        count = EMA_TERM_COUNT
    out[DIM_MEAN_EMA] = _clip((snapped + ema_term(new_1d) + ema_term(new_1w)) / float(count), 0.0, 1.0)
    return out


def week_open_ms(t_ms: int) -> int:
    return align_open_ms(int(t_ms), "1w")


def day_open_ms(t_ms: int) -> int:
    return align_open_ms(int(t_ms), "1d")


def rebuild_marker(t_ms: int, *, old: list[float] | None = None, new: list[float] | None = None) -> dict[str, Any]:
    legacy = old is not None and is_legacy_4h_vector(old)
    converted = legacy and new is not None and not is_legacy_4h_vector(new)
    dims = sorted({*HTF_DERIVED_DIMS, DIM_SLOPE_4H}) if converted else list(HTF_DERIVED_DIMS)
    return {"v": REBUILD_VERSION, "dims": dims, "asof_ms": int(t_ms), "legacy_4h_converted": converted, "legacy_4h_kept": legacy and not converted}


def _iso_to_ms(raw: Any) -> int | None:
    from datetime import datetime, timezone

    s = str(raw or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _decision_ms(decision_id: Any, ts_utc: Any) -> int | None:
    tail = str(decision_id or "").rsplit("_", 1)[-1]
    if tail.isdigit() and len(tail) >= 12:
        return int(tail)
    return _iso_to_ms(ts_utc)


def _with_marker(raw_ctx: Any, t_ms: int, *, old: list[float] | None = None, new: list[float] | None = None) -> str:
    import json

    try:
        ctx = json.loads(raw_ctx) if raw_ctx else {}
    except (TypeError, ValueError):
        ctx = {}
    if not isinstance(ctx, dict):
        ctx = {"_prior_context": ctx}
    ctx["_htf_rebuild"] = rebuild_marker(t_ms, old=old, new=new)
    return json.dumps(ctx)


def rebuild_contaminated_rows(
    conn: Any,
    *,
    bars_1d: dict[str, list[list]],
    bars_1w: dict[str, list[list]],
    boundary_ms: int,
    bars_4h: dict[str, list[list]] | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Find DAY vectors carrying the frozen 1d/1w copies since ``boundary_ms`` and rebuild them.

    ``bars_1d`` / ``bars_1w`` are completed exchange bars per symbol (ascending).
    Only ``ai_inference_log.features_json/ctx_json`` and
    ``ai_outcome_training_rows.features_json/context_json`` are written, and only
    when ``apply`` is true. Rows already carrying the rebuild marker are skipped.
    """
    import json

    report: dict[str, Any] = {"boundary_ms": int(boundary_ms), "symbols": {}, "inference": {}, "training": {}, "dims": list(HTF_DERIVED_DIMS)}
    first = conn.execute(
        "SELECT MIN(id) FROM ai_inference_log WHERE LOWER(COALESCE(strategy_id,''))='day' AND feature_dim=145 AND ts_utc >= ?",
        (_ms_to_iso(boundary_ms),),
    ).fetchone()
    first_id = int(first[0]) if first and first[0] is not None else None
    if first_id is None:
        return report

    import hashlib

    def _digest(fj: Any) -> str:
        return hashlib.sha1(str(fj).encode()).hexdigest()

    # Pass 1 streams the window and keeps only ids/timestamps (production holds ~90k vectors).
    cur = conn.execute(
        """
        SELECT id, decision_id, symbol, ts_utc, feature_version, model_artifact, features_json, ctx_json
        FROM ai_inference_log
        WHERE id >= ? AND LOWER(COALESCE(strategy_id,''))='day' AND feature_dim=145
        ORDER BY id ASC
        """,
        (first_id,),
    )
    from collections import Counter

    # The frozen copy is the one signature that repeats across the window; live
    # vectors before a symbol froze each carry their own.
    seen: list[tuple[int, str, int, tuple[float, ...], float, str, str, str]] = []
    counts: dict[str, Counter] = {}
    for rid, did, sym, ts_utc, fver, artifact, fj, cj in cur:
        if cj and "_htf_rebuild" in str(cj):
            continue
        try:
            vec = [float(x) for x in json.loads(fj)]
        except (TypeError, ValueError):
            continue
        if len(vec) != 145:
            continue
        t_ms = _decision_ms(did, ts_utc)
        if t_ms is None:
            continue
        key = tuple(vec[d] for d in FROZEN_1D_SIGNATURE_DIMS)
        counts.setdefault(sym, Counter())[key] += 1
        seen.append((int(rid), sym, t_ms, key, vec[DIM_SLOPE_1W], str(fver), str(artifact), _digest(fj)))
    signature: dict[str, tuple[float, ...]] = {s: c.most_common(1)[0][0] for s, c in counts.items() if c.most_common(1)[0][1] > 1}
    frozen_1w_slopes: dict[str, set[float]] = {}
    contaminated_inf: list[tuple[int, str, int]] = []
    versions: set[str] = set()
    artifacts: set[str] = set()
    json_to_ms: dict[str, list[int]] = {}
    for rid, sym, t_ms, key, slope_1w, fver, artifact, digest in seen:
        if signature.get(sym) != key:
            continue
        frozen_1w_slopes.setdefault(sym, set()).add(slope_1w)
        contaminated_inf.append((rid, sym, t_ms))
        versions.add(fver)
        artifacts.add(artifact)
        json_to_ms.setdefault(digest, []).append(t_ms)
    del seen

    # Frozen copies: forming bars at the boundary, pinned by the frozen dims.
    frozen: dict[str, dict[str, Any]] = {}
    for sym, key in signature.items():
        d1 = bars_1d.get(sym) or []
        w1 = bars_1w.get(sym) or []
        chosen = None
        for forming in (day_open_ms(boundary_ms), day_open_ms(boundary_ms) - interval_ms("1d")):
            cand = reconstruct_frozen_daily(d1, forming, key[0])
            vals = htf_dim_values(cand, cand) if cand else None
            if vals and all(abs(vals[d] - key[i]) <= 1e-9 for i, d in enumerate(FROZEN_1D_SIGNATURE_DIMS)):
                chosen = (forming, cand)
                break
        weekly: dict[float, float] = {}
        for slope in sorted(frozen_1w_slopes.get(sym, set())):
            wcopy = reconstruct_frozen_weekly(w1, week_open_ms(boundary_ms), slope)
            if wcopy and abs(_slope_norm(_summarize_tf(wcopy).get("slope", 0.0)) - slope) <= 1e-9:
                weekly[slope] = ema_term(wcopy)
        frozen[sym] = {
            "daily_ok": chosen is not None,
            "daily_forming_open_ms": chosen[0] if chosen else None,
            "ema_1d": ema_term(chosen[1]) if chosen else None,
            "ema_1w_by_slope": weekly,
        }

    cache: dict[tuple[str, int, int], tuple[list[list], list[list]]] = {}

    def _bars(sym: str, t_ms: int) -> tuple[list[list], list[list]]:
        k = (sym, day_open_ms(t_ms), week_open_ms(t_ms))
        if k not in cache:
            cache[k] = (completed_asof(bars_1d.get(sym) or [], "1d", t_ms, DEPTH_1D), completed_asof(bars_1w.get(sym) or [], "1w", t_ms, DEPTH_1W))
        return cache[k]

    failures: Counter = Counter()
    legacy_converted = [0]
    legacy_kept = [0]

    def _rebuild(sym: str, vec: list[float], t_ms: int) -> list[float] | None:
        fz = frozen.get(sym) or {}
        if not fz.get("daily_ok"):
            failures["daily_copy_unvalidated"] += 1
            return None
        e1w = (fz.get("ema_1w_by_slope") or {}).get(vec[DIM_SLOPE_1W])
        if e1w is None:
            failures["weekly_copy_unvalidated"] += 1
            return None
        new_1d, new_1w = _bars(sym, t_ms)
        if len(new_1d) < DEPTH_1D or len(new_1w) < DEPTH_1W:
            failures["insufficient_clean_history"] += 1
            return None
        e4h = None
        if is_legacy_4h_vector(vec):
            src_4h = (bars_4h or {}).get(sym) or []
            # The live bundle may have served a 4h copy up to two bars old (bounded reuse).
            for lag in range(3):
                e4h = legacy_4h_term(vec, completed_asof(src_4h, "4h", t_ms - lag * interval_ms("4h"), DEPTH_4H))
                if e4h is not None:
                    break
            if e4h is None:
                legacy_kept[0] += 1
            else:
                legacy_converted[0] += 1
        out = rebuild_vector(
            vec,
            new_1d=new_1d,
            new_1w=new_1w,
            frozen_ema_1d=float(fz["ema_1d"]),
            frozen_ema_1w=float(e1w),
            legacy_ema_4h=e4h,
            keep_legacy_4h=True,
        )
        if out is None:
            failures["ema_mean_not_decomposable"] += 1
        return out

    inf_rebuilt = 0
    inf_failed = 0
    inf_applied = 0
    chunk = 1000
    for start in range(0, len(contaminated_inf), chunk):
        part = contaminated_inf[start : start + chunk]
        meta = {rid: (sym, t_ms) for rid, sym, t_ms in part}
        marks = ",".join("?" * len(part))
        updates: list[tuple[str, str, int]] = []
        for rid, fj, cj in conn.execute(f"SELECT id, features_json, ctx_json FROM ai_inference_log WHERE id IN ({marks})", list(meta)).fetchall():
            sym, t_ms = meta[int(rid)]
            vec = [float(x) for x in json.loads(fj)]
            new = _rebuild(sym, vec, t_ms)
            if new is None:
                inf_failed += 1
                continue
            inf_rebuilt += 1
            updates.append((json.dumps(new), _with_marker(cj, t_ms, old=vec, new=new), int(rid)))
        if apply and updates:
            conn.executemany("UPDATE ai_inference_log SET features_json=?, ctx_json=? WHERE id=?", updates)
            conn.commit()
            inf_applied += len(updates)

    tr_rows = conn.execute(
        """
        SELECT id, symbol, opened_at_utc, features_json, context_json, strategy_id
        FROM ai_outcome_training_rows
        WHERE features_json IS NOT NULL AND length(features_json) > 2
        ORDER BY id ASC
        """
    ).fetchall()
    tr_updates: list[tuple[str, str, int]] = []
    tr_failed = 0
    tr_contaminated: list[tuple[int, str, str]] = []
    for rid, sym, opened, fj, cj, sid in tr_rows:
        if cj and "_htf_rebuild" in str(cj):
            continue
        bus = str(sym or "").replace("/", "").replace("-", "").upper()
        if bus not in signature:
            continue
        try:
            vec = [float(x) for x in json.loads(fj)]
        except (TypeError, ValueError):
            continue
        if len(vec) != 145 or tuple(vec[d] for d in FROZEN_1D_SIGNATURE_DIMS) != signature[bus]:
            continue
        opened_ms = _iso_to_ms(opened)
        matches = json_to_ms.get(_digest(fj)) or []
        if matches and opened_ms is not None:
            t_ms: int | None = min(matches, key=lambda m: abs(m - opened_ms))
        else:
            t_ms = matches[0] if matches else opened_ms
        if t_ms is None:
            tr_failed += 1
            continue
        tr_contaminated.append((int(rid), bus, str(sid or "")))
        new = _rebuild(bus, vec, t_ms)
        if new is None:
            tr_failed += 1
            continue
        tr_updates.append((json.dumps(new), _with_marker(cj, t_ms, old=vec, new=new), int(rid)))

    if contaminated_inf:
        report["inference"] = {
            "contaminated": len(contaminated_inf),
            "rebuildable": inf_rebuilt,
            "failed": inf_failed,
            "first_id": contaminated_inf[0][0],
            "last_id": contaminated_inf[-1][0],
            "first_ms": min(r[2] for r in contaminated_inf),
            "last_ms": max(r[2] for r in contaminated_inf),
            "by_symbol": {s: sum(1 for r in contaminated_inf if r[1] == s) for s in signature},
            "feature_versions": sorted(versions),
            "model_artifacts": sorted(artifacts),
        }
    report["training"] = {
        "contaminated": len(tr_contaminated),
        "rebuildable": len(tr_updates),
        "failed": tr_failed,
        "ids": [r[0] for r in tr_contaminated],
        "strategy_ids": sorted({r[2] for r in tr_contaminated}),
    }
    report["failure_reasons"] = dict(failures)
    report["legacy_4h_converted"] = legacy_converted[0]
    report["legacy_4h_kept"] = legacy_kept[0]
    report["symbols"] = {
        s: {"signature": list(signature[s]), **{k: v for k, v in frozen[s].items() if k != "ema_1w_by_slope"}, "frozen_1w_copies": len(frozen[s]["ema_1w_by_slope"])} for s in signature
    }

    if apply:
        conn.executemany("UPDATE ai_outcome_training_rows SET features_json=?, context_json=? WHERE id=?", tr_updates)
        conn.commit()
        report["applied"] = {"inference": inf_applied, "training": len(tr_updates)}
    return report


def _ms_to_iso(ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).isoformat()


__all__ = [
    "DEPTH_1D",
    "DEPTH_1W",
    "FROZEN_1D_SIGNATURE_DIMS",
    "HTF_DERIVED_DIMS",
    "REBUILD_VERSION",
    "completed_asof",
    "day_open_ms",
    "ema_term",
    "frozen_copy_matches",
    "htf_dim_values",
    "is_legacy_4h_vector",
    "legacy_4h_term",
    "rebuild_marker",
    "rebuild_vector",
    "reconstruct_frozen_daily",
    "reconstruct_frozen_weekly",
    "week_open_ms",
]
