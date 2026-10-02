#!/usr/bin/env python3
"""Rebuild SCALP_V2 derived edge/risk state from current-version candidate markouts.

Run with the app stopped. Backs up adaptive_metric_state and adaptive_linear_model
to suffixed tables first, then rebuilds edge residuals, gross path MAE and the
micro model. Nothing else is modified.

Decision-time raw expected move per candidate:
  - stored on the markout row (rows recorded after this version), else
  - exact value from the filled trade's entry context (signaled rows), else
  - closed-bar ATR estimate (opinion-path rows; this is ranking's own formula), else
  - unknown: the row teaches risk only.

  scripts/rebuild_scalp_edge_state.py --db mystic_trading.db --backup-suffix 20261002 [--calibration-out f.json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.scalp_v2.edge_state_rebuild import OhlcvPath, rebuild_scalp_edge_state
from backend.services.scalp_v2.executable_edge import scalp_executable_edge

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
FILL_MATCH_SEC = 5.0
CALIBRATION_FIELDS = (
    "live_cost_pct",
    "base_executable_edge_pct",
    "adaptive_residual_pct",
    "micro_residual_model_pct",
    "micro_residual_pct",
    "final_executable_edge_pct",
    "confidence",
    "risk_estimate_pct",
    "size_mult",
)


def _fill_raws(db: str) -> list[tuple[str, float, float, str]]:
    out: list[tuple[str, float, float, str]] = []
    conn = sqlite3.connect(db, timeout=15)
    try:
        for sym, ej in conn.execute("SELECT symbol, explainability_json FROM paper_trades WHERE engine_id='SCALP_V2' AND UPPER(side)='BUY' AND explainability_json IS NOT NULL"):
            try:
                ctx = (json.loads(ej) or {}).get("scalp_entry_context") or {}
                raw, cyc = float(ctx["expected_move_pct"]), float(ctx["cycle_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append((str(sym).upper().replace("/", "").replace("-", ""), cyc, raw, str(ctx.get("edge_source") or "atr_estimate")))
    finally:
        conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--backup-suffix", default=None)
    ap.add_argument("--calibration-out", default=None)
    args = ap.parse_args()

    path = OhlcvPath(args.db, SYMBOLS)
    fills = _fill_raws(args.db)
    sources = {"fill_exact": 0, "atr_reconstructed": 0, "signaled_unknown": 0}

    def raw_move_for(row: sqlite3.Row) -> tuple[float | None, str]:
        sym, at = str(row["symbol"]), float(row["evaluated_at"])
        match = [f for f in fills if f[0] == sym and abs(f[1] - at) <= FILL_MATCH_SEC]
        if match:
            sources["fill_exact"] += 1
            return match[0][2], match[0][3]
        if int(row["signaled"] or 0):
            sources["signaled_unknown"] += 1
            return None, "strategy"
        sources["atr_reconstructed"] += 1
        return path.raw_expected_move(sym, at), "atr_estimate"

    preds: list[dict] = []

    def on_decision(row: sqlite3.Row, view: dict, raw: float | None, source: str) -> None:
        if args.calibration_out is None:
            return
        feats = json.loads(row["features_json"] or "{}")
        sp = feats.get("spread_pct")
        edge = scalp_executable_edge(view, raw_expected_move_pct=raw or 0.0, spread_pct=float(sp) if sp is not None else None, impact_pct=0.0, edge_source=source)
        from backend.services.adaptive_learning import _forward_from_stored

        preds.append(
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "setup": row["setup"],
                "regime": row["regime"],
                "t": row["evaluated_at"],
                "signaled": row["signaled"],
                "horizon": row["label_horizon"],
                "raw": raw,
                "source": source,
                **{k: edge.as_dict()[k] for k in CALIBRATION_FIELDS},
                "realized": _forward_from_stored(row),
            }
        )

    stats = rebuild_scalp_edge_state(args.db, raw_move_for=raw_move_for, path_low=path.low_between, backup_suffix=args.backup_suffix, on_decision=on_decision)
    stats["raw_sources"] = sources
    if args.calibration_out:
        Path(args.calibration_out).write_text(json.dumps({"stats": stats, "preds": preds}))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
