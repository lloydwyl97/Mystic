#!/usr/bin/env python3
"""DAY model governance: freeze, status, promotion switch, rollback, forward re-evaluation.

freeze                       register serving models + metadata, write ACTIVE pointers, disable promotion
status                       per-coin registry / pointer / promotion status
disable|enable --reason R    automatic promotion switch
rollback --symbol S --reason R
evaluate                     score serving models on outcomes opened after their training data ended
mark-legacy-inference [--apply]   tag unrebuildable inference rows layout_reconstruction_incomplete
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services import ai_model_registry as registry
from backend.services.live_strategy_contracts import per_coin_artifact_file
from backend.utils.path_helpers import ensure_model_directories

SID = "day"


def _active(sym: str) -> Path:
    return per_coin_artifact_file(Path(ensure_model_directories()["active"]), SID, sym)


def _load(path: Path) -> dict[str, Any]:
    art = pickle.loads(path.read_bytes())
    return art if isinstance(art, dict) else {}


def _db_active_row(db: str, sym: str, sha: str) -> dict[str, Any]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, model_id, promoted_at, created_at, promotion_reason, validation_metrics_json FROM ai_model_versions "
            "WHERE strategy_id=? AND symbol=? AND artifact_hash=? ORDER BY id DESC LIMIT 1",
            (SID, sym, sha),
        ).fetchone()
    if not row:
        return {}
    vm = json.loads(row["validation_metrics_json"] or "{}")
    keep = ("promotion_path", "holdout_sample_count", "holdout_window", "candidate_holdout", "active_holdout", "train_outcome_max_id")
    return {
        "db_version_row_id": row["id"],
        "model_id": row["model_id"],
        "promoted_at": row["promoted_at"] or row["created_at"],
        "promotion_reason": row["promotion_reason"],
        "promotion_metrics": {k: vm.get(k) for k in keep if k in vm},
    }


def cmd_freeze(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sym in TRADING_SYMBOLS:
        path = _active(sym)
        if not path.exists():
            out[sym] = {"error": "active_missing"}
            continue
        sha = registry.sha256_file(path)
        art = _load(path)
        meta = {
            "role": "frozen_incumbent",
            **registry.artifact_meta(path),
            **_db_active_row(args.db, sym, sha),
        }
        if not meta.get("training_fingerprint"):
            basis = {k: art.get(k) for k in ("trained_at", "train_outcome_max_id", "holdout_window", "train_samples", "feature_version", "feature_dim")}
            meta["training_fingerprint"] = hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()
            meta["training_fingerprint_basis"] = "reconstructed_from_artifact_fields"
        reg = registry.register_artifact(SID, sym, path, meta)
        registry.ensure_incumbent_registered(SID, sym, path, reason="frozen_by_governance")
        out[sym] = {"version": reg["version"], "path": reg["path"], "trained_at": meta.get("trained_at"), "promoted_at": meta.get("promoted_at")}
    if registry.promotion_enabled(SID)[0] or not registry.read_json(registry.promotion_control_path(SID)):
        registry.set_promotion_enabled(SID, False, args.reason)
    out["promotion_enabled"] = registry.promotion_enabled(SID)[0]
    return out


def cmd_status(_args: argparse.Namespace) -> dict[str, Any]:
    return {sym: registry.symbol_status(SID, sym, _active(sym)) for sym in TRADING_SYMBOLS}


def cmd_switch(args: argparse.Namespace) -> dict[str, Any]:
    return registry.set_promotion_enabled(SID, args.cmd == "enable", args.reason)


def cmd_rollback(args: argparse.Namespace) -> dict[str, Any]:
    ok, why = registry.rollback_to_previous(SID, args.symbol.upper(), _active(args.symbol.upper()), reason=args.reason)
    return {"ok": ok, "reason": why}


def cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from backend.services.ai_model_promotion_decision import classify_standalone, row_utilities, summarize_model
    from backend.services.ai_model_promotion_holdout import artifact_predictions, load_forward_rows

    out: dict[str, Any] = {}
    for sym in TRADING_SYMBOLS:
        path = _active(sym)
        art = _load(path)
        end = str(art.get("training_data_end") or art.get("trained_at") or "")
        fwd = load_forward_rows(strategy_id=SID, symbol_bus=sym, opened_after_utc=end.replace("T", " ")[:19], db_path=args.db)
        n = len(fwd["nets"])
        rec: dict[str, Any] = {
            "version": registry.sha256_file(path)[:16],
            "training_data_end": end,
            "forward_rows": n,
            "first_closed_at": fwd["first_closed_at"],
            "last_closed_at": fwd["last_closed_at"],
        }
        if n:
            preds = artifact_predictions(path, fwd["X"])
            util, bad = row_utilities(preds, fwd["nets"], fwd["good_bad"])
            rec["model"] = summarize_model(util, bad, preds, fwd["y"])
            all_util, all_bad = row_utilities(np.ones(n, dtype=np.int64), fwd["nets"], fwd["good_bad"])
            rec["take_every_trade"] = summarize_model(all_util, all_bad, np.ones(n), fwd["y"])
            rec["verdict"] = classify_standalone(util, bad, all_util)
        else:
            rec["verdict"] = "AMBIGUOUS"
        out[sym] = rec
    return out


def cmd_mark_legacy(args: argparse.Namespace) -> dict[str, Any]:
    uri = args.db if args.apply else f"file:{args.db}?mode=ro"
    with sqlite3.connect(uri, uri=not args.apply) as conn:
        rows = conn.execute("SELECT id, ctx_json FROM ai_inference_log WHERE ctx_json LIKE '%legacy_4h_kept\": true%'").fetchall()
        changed = 0
        for rid, cj in rows:
            ctx = json.loads(cj)
            marker = ctx.get("_htf_rebuild") or {}
            if marker.get("layout_reconstruction_incomplete"):
                continue
            marker.update(layout_reconstruction_incomplete=True, authoritative=False)
            ctx["_htf_rebuild"] = marker
            changed += 1
            if args.apply:
                conn.execute("UPDATE ai_inference_log SET ctx_json=? WHERE id=?", (json.dumps(ctx, separators=(",", ":")), rid))
        if args.apply:
            conn.commit()
    return {"legacy_rows": len(rows), "to_mark": changed, "applied": bool(args.apply)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DATABASE_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--reason", default="frozen pending repaired promotion path")
    sub.add_parser("status")
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("--reason", required=True)
    rb = sub.add_parser("rollback")
    rb.add_argument("--symbol", required=True)
    rb.add_argument("--reason", required=True)
    sub.add_parser("evaluate")
    ml = sub.add_parser("mark-legacy-inference")
    ml.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    handlers = {"freeze": cmd_freeze, "status": cmd_status, "disable": cmd_switch, "enable": cmd_switch, "rollback": cmd_rollback, "evaluate": cmd_evaluate, "mark-legacy-inference": cmd_mark_legacy}
    print(json.dumps(handlers[args.cmd](args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
