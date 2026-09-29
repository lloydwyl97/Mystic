"""Leak-free chronological validation of per-coin DAY direction models.

Writes evaluation artifacts outside ``models/active``; never promotes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backend.services.ai_model_promotion_holdout import _outcome_label
from backend.services.ai_model_promotion_pac import FEATURE_DIM_V2, FEATURE_VERSION_DAY_HTF, _row_net_pnl, _row_passes_filters
from backend.services.day_feature_health import zero_learning_blocked_feature_dims

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
EXCLUDED_CLOSE_MARKERS = ("MANUAL", "UNMATCHED", "RECONCIL", "CORRECTION", "DUST", "EXTERNAL", "ADMIN")
HOLDOUT_FRACTION = 0.2
N_FOLDS = 4
MIN_TRAIN = 30
BOUNDED_SIZE = 0.5


def _ts(raw: str) -> float:
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()


def _engine_of(conn: sqlite3.Connection, trade_id: str) -> str:
    row = conn.execute("SELECT engine_id FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' LIMIT 1", (trade_id,)).fetchone()
    return str(row[0] or "") if row else ""


def exclusion_reason(conn: sqlite3.Connection, row: sqlite3.Row) -> str | None:
    ctx = json.loads(row["context_json"] or "{}") if row["context_json"] else {}
    comps = json.loads(row["score_components_json"] or "{}") if row["score_components_json"] else {}
    if str(ctx.get("is_dust") or "").lower() in ("1", "true"):
        return "dust"
    trade_id = str(ctx.get("trade_id") or "")
    if trade_id.lower().startswith("scalp"):
        return "scalp_trade"
    engine = str(ctx.get("engine_id") or "") or (_engine_of(conn, trade_id) if trade_id else "")
    if engine and engine.upper() not in ("DAY_V2", "DAY"):
        return "non_day_engine"
    close = " ".join(str(comps.get(k) or "") for k in ("close_reason", "exit_reason_raw", "exit_reason_canonical")).upper()
    if any(m in close for m in EXCLUDED_CLOSE_MARKERS):
        return "manual_or_correction"
    return None


def load_rows(conn: sqlite3.Connection, symbol: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    conn.row_factory = sqlite3.Row
    base, ccxt = symbol, symbol.replace("USDT", "/USDT")
    raw = conn.execute(
        "SELECT * FROM ai_outcome_training_rows WHERE strategy_id='day' AND symbol IN (?,?) AND features_json IS NOT NULL ORDER BY closed_at_utc, id",
        (base, ccxt),
    ).fetchall()
    dropped: dict[str, int] = {}
    out = []
    for r in raw:
        if not _row_passes_filters(r, symbol_bus=base, min_fv=FEATURE_VERSION_DAY_HTF, min_dim=FEATURE_DIM_V2):
            dropped["pac_filter"] = dropped.get("pac_filter", 0) + 1
            continue
        why = exclusion_reason(conn, r)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        out.append(
            {
                "id": int(r["id"]),
                "opened": _ts(r["opened_at_utc"] or r["closed_at_utc"]),
                "closed": _ts(r["closed_at_utc"]),
                "x": zero_learning_blocked_feature_dims([float(v) for v in json.loads(r["features_json"])]),
                "y": _outcome_label(r),
                "net": float(_row_net_pnl(r)),
            }
        )
    return out, dropped


def purged_split(rows: list[dict[str, Any]], start: float, end: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validation = entries opened in [start, end); training = rows closed before the earliest validation entry."""
    val = [r for r in rows if start <= r["opened"] < end]
    if not val:
        return [], []
    cutoff = min(r["opened"] for r in val)
    train = [r for r in rows if r["closed"] < cutoff]
    return train, val


def fit(train: list[dict[str, Any]]) -> tuple[StandardScaler, RandomForestClassifier] | None:
    y = np.array([r["y"] for r in train])
    if len(train) < MIN_TRAIN or len(set(y.tolist())) < 2:
        return None
    X = np.array([r["x"] for r in train], dtype=float)
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42, n_jobs=1, class_weight="balanced")
    model.fit(scaler.transform(X), y)
    return scaler, model


def _pf(v: list[float]) -> float:
    g = sum(x for x in v if x > 0)
    b = -sum(x for x in v if x < 0)
    return round(g / b, 3) if b > 0 else (float("inf") if g > 0 else 0.0)


def _dd(v: list[float]) -> float:
    peak = cum = worst = 0.0
    for x in v:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return round(worst, 6)


def score(fitted, val: list[dict[str, Any]]) -> dict[str, Any]:
    nets = [r["net"] for r in val]
    base = {"n": len(val), "baseline_net": round(sum(nets), 6), "baseline_pf": _pf(nets), "baseline_dd": _dd(nets)}
    if fitted is None:
        return {**base, "status": "INSUFFICIENT_TRAIN"}
    scaler, model = fitted
    preds = model.predict(scaler.transform(np.array([r["x"] for r in val], dtype=float)))
    followed = [r["net"] for r, p in zip(val, preds, strict=True) if int(p) == 1]
    sized = [r["net"] * (1.0 if int(p) == 1 else BOUNDED_SIZE) for r, p in zip(val, preds, strict=True)]
    acc = float(np.mean(preds == np.array([r["y"] for r in val])))
    return {
        **base,
        "status": "OK",
        "accuracy": round(acc, 4),
        "buy_rate": round(float(np.mean(preds == 1)), 4),
        "filter_n": len(followed),
        "filter_net": round(sum(followed), 6),
        "filter_pf": _pf(followed),
        "filter_dd": _dd(followed),
        "sized_net": round(sum(sized), 6),
        "sized_pf": _pf(sized),
        "sized_dd": _dd(sized),
    }


def qualification(folds: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ok = [f for f in folds if f.get("status") == "OK"]
    net = sum(f[f"{prefix}_net"] for f in ok)
    base_net = sum(f["baseline_net"] for f in ok)
    worst_dd = min((f[f"{prefix}_dd"] for f in ok), default=0.0)
    base_dd = min((f["baseline_dd"] for f in ok), default=0.0)
    gains = sum(max(f[f"{prefix}_net"], 0) for f in ok)
    checks = {
        "positive_net": net > 0,
        "pf_gt_1": all(f[f"{prefix}_pf"] > 1 for f in ok) if ok else False,
        "majority_folds_positive": sum(f[f"{prefix}_net"] > 0 for f in ok) > len(folds) / 2,
        "beats_deterministic": net > base_net,
        "drawdown_ok": worst_dd >= base_dd * 1.25,
        "adequate_sample": all(f["n"] >= 20 for f in ok) and len(ok) == len(folds),
        "leakage_zero": all(f.get("overlap", 1) == 0 for f in folds),
    }
    return {
        "net": round(net, 6),
        "baseline_net": round(base_net, 6),
        "gross_positive": round(gains, 6),
        "positive_folds": sum(f[f"{prefix}_net"] > 0 for f in ok),
        "checks": checks,
        "qualified": all(checks.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--out-dir", default="models/clean_eval/day")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    per_symbol = {s: load_rows(conn, s) for s in SYMBOLS}
    opened = sorted(r["opened"] for rows, _ in per_symbol.values() for r in rows)
    edges = [opened[int(len(opened) * q)] for q in np.linspace(0.2, 1.0, N_FOLDS + 1)[:-1]] + [opened[-1] + 1]
    report: dict[str, Any] = {"generated_at": stamp, "fold_edges_utc": [datetime.fromtimestamp(e, tz=UTC).isoformat() for e in edges], "symbols": {}}
    pooled: list[dict[str, Any]] = [{"fold": k + 1, "n": 0, "baseline_net": 0.0, "filter_net": 0.0, "sized_net": 0.0, "overlap": 0} for k in range(N_FOLDS)]
    pooled_vals: list[dict[str, list[float]]] = [{"base": [], "filter": [], "sized": []} for _ in range(N_FOLDS)]

    for sym, (rows, dropped) in per_symbol.items():
        folds = []
        for k in range(N_FOLDS):
            train, val = purged_split(rows, edges[k], edges[k + 1])
            overlap = len({r["id"] for r in train} & {r["id"] for r in val})
            res = score(fit(train), val) if val else {"n": 0, "status": "NO_VALIDATION"}
            res.update({"fold": k + 1, "train_n": len(train), "overlap": overlap})
            if train:
                res["train_window"] = [datetime.fromtimestamp(min(r["opened"] for r in train), tz=UTC).isoformat(), datetime.fromtimestamp(max(r["closed"] for r in train), tz=UTC).isoformat()]
            if val:
                res["validation_window"] = [datetime.fromtimestamp(min(r["opened"] for r in val), tz=UTC).isoformat(), datetime.fromtimestamp(max(r["opened"] for r in val), tz=UTC).isoformat()]
            folds.append(res)
            if res.get("status") == "OK":
                p = pooled[k]
                p["n"] += res["n"]
                p["overlap"] += overlap
                for key in ("baseline_net", "filter_net", "sized_net"):
                    p[key] = round(p[key] + res[key], 6)
                fitted = fit(train)
                preds = fitted[1].predict(fitted[0].transform(np.array([r["x"] for r in val], dtype=float)))
                for r, pr in zip(val, preds, strict=True):
                    pooled_vals[k]["base"].append(r["net"])
                    pooled_vals[k]["sized"].append(r["net"] * (1.0 if int(pr) == 1 else BOUNDED_SIZE))
                    if int(pr) == 1:
                        pooled_vals[k]["filter"].append(r["net"])

        n_hold = max(1, int(len(rows) * HOLDOUT_FRACTION))
        holdout = rows[-n_hold:]
        final_train, final_val = purged_split(rows, holdout[0]["opened"], rows[-1]["opened"] + 1)
        fitted = fit(final_train)
        final = score(fitted, final_val)
        final["overlap"] = len({r["id"] for r in final_train} & {r["id"] for r in final_val})
        meta: dict[str, Any] = {
            "symbol": sym,
            "version": f"day_clean_eval_{stamp}",
            "feature_version": FEATURE_VERSION_DAY_HTF,
            "feature_dim": FEATURE_DIM_V2,
            "eligible_rows": len(rows),
            "dropped": dropped,
            "train_n": len(final_train),
            "validation_n": len(final_val),
            "train_window": [datetime.fromtimestamp(min(r["opened"] for r in final_train), tz=UTC).isoformat(), datetime.fromtimestamp(max(r["closed"] for r in final_train), tz=UTC).isoformat()]
            if final_train
            else None,
            "validation_window": [datetime.fromtimestamp(min(r["opened"] for r in final_val), tz=UTC).isoformat(), datetime.fromtimestamp(max(r["opened"] for r in final_val), tz=UTC).isoformat()]
            if final_val
            else None,
            "train_max_id": max((r["id"] for r in final_train), default=0),
            "validation_ids": [r["id"] for r in final_val],
            "metrics": final,
        }
        if fitted is not None:
            blob = pickle.dumps({"scaler": fitted[0], "model": fitted[1], "meta": meta})
            path = out_dir / f"{sym}_direction_clean_{stamp}.pkl"
            path.write_bytes(blob)
            meta["artifact_path"] = str(path)
            meta["artifact_sha256"] = hashlib.sha256(blob).hexdigest()
        (out_dir / f"{sym}_direction_clean_{stamp}.json").write_text(json.dumps(meta, indent=1, default=str))
        report["symbols"][sym] = {"dropped": dropped, "eligible": len(rows), "folds": folds, "final": {k: v for k, v in meta.items() if k != "validation_ids"}}

    for k, p in enumerate(pooled):
        v = pooled_vals[k]
        p.update(
            {
                "baseline_pf": _pf(v["base"]),
                "filter_pf": _pf(v["filter"]),
                "sized_pf": _pf(v["sized"]),
                "baseline_dd": _dd(v["base"]),
                "filter_dd": _dd(v["filter"]),
                "sized_dd": _dd(v["sized"]),
                "status": "OK" if p["n"] else "EMPTY",
            }
        )
    report["pooled_folds"] = pooled
    report["qualification"] = {"filter_report_only": qualification(pooled, "filter"), "bounded_sizing": qualification(pooled, "sized")}
    (out_dir / f"day_clean_validation_{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
