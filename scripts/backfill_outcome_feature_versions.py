#!/usr/bin/env python3
"""Patch ai_outcome_training_rows context_json _feature_version from features_json dim."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ai_training_pipeline import _FEATURE_DIM_V2, FEATURE_VERSION_DAY_HTF, _infer_outcome_feature_version
from backend.database_schema import DATABASE_PATH


def backfill_outcome_feature_versions(db_path: str = DATABASE_PATH) -> dict[str, int]:
    updated = 0
    skipped = 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, context_json, features_json, strategy_id
            FROM ai_outcome_training_rows
            WHERE strategy_id='day' OR strategy_id IS NULL
            """
        ).fetchall()
        for row in rows:
            rid = int(row["id"])
            inferred = _infer_outcome_feature_version(dict(row))
            if inferred < FEATURE_VERSION_DAY_HTF:
                skipped += 1
                continue
            ctx_raw = row["context_json"] or "{}"
            try:
                ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else {}
            except json.JSONDecodeError:
                ctx = {}
            if not isinstance(ctx, dict):
                ctx = {}
            cur = int(ctx.get("_feature_version") or ctx.get("feature_version") or 0)
            if cur >= FEATURE_VERSION_DAY_HTF:
                skipped += 1
                continue
            ctx["_feature_version"] = int(FEATURE_VERSION_DAY_HTF)
            ctx["feature_version"] = int(FEATURE_VERSION_DAY_HTF)
            conn.execute(
                "UPDATE ai_outcome_training_rows SET context_json=? WHERE id=?",
                (json.dumps(ctx, separators=(",", ":")), rid),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    return {"updated": updated, "skipped": skipped, "target_fv": FEATURE_VERSION_DAY_HTF, "target_dim": _FEATURE_DIM_V2}


if __name__ == "__main__":
    result = backfill_outcome_feature_versions()
    print(json.dumps(result, indent=2))
