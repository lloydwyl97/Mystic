"""Read-only scalp PnL attribution — symbol, setup, regime, exit, hold, cost burden."""

from __future__ import annotations

import sqlite3
from typing import Any

from backend.services.binance_scalp.config import get_scalp_config


def _hold_bucket(hold_sec: float | None) -> str:
    if hold_sec is None:
        return "unknown"
    s = float(hold_sec)
    if s < 60:
        return "0-60s"
    if s < 180:
        return "60-180s"
    if s < 300:
        return "180-300s"
    if s < 600:
        return "300-600s"
    if s < 900:
        return "600-900s"
    return "900s+"


def _fee_burden_bucket(fee_pct: float | None) -> str:
    if fee_pct is None:
        return "unknown"
    p = abs(float(fee_pct))
    if p < 0.001:
        return "low (<0.1%)"
    if p < 0.002:
        return "moderate (0.1-0.2%)"
    if p < 0.004:
        return "high (0.2-0.4%)"
    return "very_high (>=0.4%)"


def _rollup(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        k = str(row.get(key) or "unknown")
        b = buckets.setdefault(k, {"key": k, "trades": 0, "wins": 0, "losses": 0, "net_pnl_usd": 0.0})
        pnl = float(row.get("pnl_usd") or 0.0)
        b["trades"] += 1
        b["net_pnl_usd"] += pnl
        if pnl >= 0:
            b["wins"] += 1
        else:
            b["losses"] += 1
    out = list(buckets.values())
    for b in out:
        b["net_pnl_usd"] = round(b["net_pnl_usd"], 4)
        b["avg_pnl_usd"] = round(b["net_pnl_usd"] / b["trades"], 4) if b["trades"] else 0.0
    out.sort(key=lambda x: (-x["trades"], x["key"]))
    return out


def build_scalp_attribution_report(*, days: int | None = None, db_path: str | None = None) -> dict[str, Any]:
    path = db_path or get_scalp_config().database_path
    try:
        from backend.services.scalp_outcome_attribution import ensure_scalp_outcome_attribution_table

        ensure_scalp_outcome_attribution_table(path)
    except Exception:
        pass
    sell_where = "WHERE t.side='SELL'"
    params: tuple = ()
    if days is not None:
        sell_where += " AND datetime(t.created_at) >= datetime('now', ?)"
        params = (f"-{int(days)} days",)

    rows: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            sell_rows = conn.execute(
                f"""
                SELECT t.trade_id, t.symbol, t.pnl_usd, t.pnl_pct, t.exit_reason, t.created_at,
                       t.fee_usd, t.slippage_usd, t.notional, t.diagnostics_json,
                       b.diagnostics_json AS buy_diagnostics_json
                FROM scalp_paper_trades t
                LEFT JOIN scalp_paper_trades b
                  ON b.trade_id = REPLACE(t.trade_id, '_SELL', '') AND b.side = 'BUY'
                {sell_where}
                ORDER BY t.created_at DESC
                """,
                params,
            ).fetchall()
            attr_rows = {
                str(r["trade_id"]): dict(r)
                for r in conn.execute(
                    """
                    SELECT trade_id, scalp_setup, micro_regime, hold_seconds,
                           spread_at_entry, spread_at_exit, fees, net_pnl_after_fees,
                           exit_trigger_detail, exit_stale_review_count
                    FROM scalp_outcome_attribution
                    """
                ).fetchall()
            }
            # Also index by base trade id (SELL rows use {id}_SELL suffix).
            for tid, row in list(attr_rows.items()):
                if tid.endswith("_SELL"):
                    base = tid[:-5]
                    attr_rows.setdefault(base, row)
    except sqlite3.Error as exc:
        return {"engine": "scalp", "error": str(exc)[:200], "rows": []}

    for sr in sell_rows:
        tid = str(sr["trade_id"])
        base_tid = tid[:-5] if tid.endswith("_SELL") else tid
        attr = attr_rows.get(tid) or attr_rows.get(base_tid) or {}
        setup = str(attr.get("scalp_setup") or "")
        regime = str(attr.get("micro_regime") or "")
        hold_sec = attr.get("hold_seconds")
        if hold_sec is None:
            hold_sec = None
        spread_in = float(attr.get("spread_at_entry") or 0.0)
        spread_out = float(attr.get("spread_at_exit") or 0.0)
        fee_usd = float(sr["fee_usd"] or attr.get("fees") or 0.0)
        slip_usd = float(sr["slippage_usd"] or 0.0)
        notional = float(sr["notional"] or 0.0)
        cost_burden_pct = ((fee_usd + slip_usd) / notional) if notional > 0 else None
        if not setup:
            try:
                import json

                buy_raw = sr["buy_diagnostics_json"]
                if buy_raw:
                    buy_diag = json.loads(buy_raw)
                    setup_sig = buy_diag.get("setup_signal") or buy_diag.get("setup") or {}
                    if isinstance(setup_sig, dict):
                        setup = str(setup_sig.get("setup_name") or setup_sig.get("scalp_setup") or "")
                    setup = setup or str(buy_diag.get("setup_name") or buy_diag.get("scalp_setup") or "")
                    if not regime:
                        regime = str(buy_diag.get("micro_regime") or "")
                if not setup:
                    diag = json.loads(sr["diagnostics_json"] or "{}")
                    setup = str(diag.get("setup_name") or diag.get("scalp_setup") or "unknown")
                    if not regime:
                        regime = str(diag.get("micro_regime") or "unknown")
            except Exception:
                setup = setup or "unknown"
                regime = regime or "unknown"
        exit_reason = sr["exit_reason"] or "unknown"
        scratch_trigger = str(attr.get("exit_trigger_detail") or "") if exit_reason == "EARLY_SCRATCH_EXIT" else ""
        rows.append(
            {
                "trade_id": tid,
                "symbol": sr["symbol"],
                "pnl_usd": float(sr["pnl_usd"] or 0.0),
                "exit_reason": exit_reason,
                "setup": setup or "unknown",
                "regime": regime or "unknown",
                "hold_seconds": hold_sec,
                "hold_bucket": _hold_bucket(hold_sec),
                "fee_burden_bucket": _fee_burden_bucket(cost_burden_pct),
                "spread_at_entry": spread_in,
                "spread_at_exit": spread_out,
                "fee_usd": fee_usd,
                "slippage_usd": slip_usd,
                "cost_burden_pct": cost_burden_pct,
                "scratch_trigger_detail": scratch_trigger or None,
                "exit_stale_review_count": attr.get("exit_stale_review_count"),
            }
        )

    total_pnl = round(sum(r["pnl_usd"] for r in rows), 4)
    scratch_rows = [r for r in rows if r["exit_reason"] == "EARLY_SCRATCH_EXIT"]
    return {
        "engine": "scalp",
        "days": days,
        "closed_sells": len(rows),
        "total_net_pnl_usd": total_pnl,
        "by_symbol": _rollup(rows, "symbol"),
        "by_setup": _rollup(rows, "setup"),
        "by_regime": _rollup(rows, "regime"),
        "by_exit_reason": _rollup(rows, "exit_reason"),
        "by_hold_bucket": _rollup(rows, "hold_bucket"),
        "by_fee_burden": _rollup(rows, "fee_burden_bucket"),
        "early_scratch_exit": {
            "count": len(scratch_rows),
            "rate_of_all_exits": round(len(scratch_rows) / len(rows), 4) if rows else 0.0,
            "net_pnl_usd": round(sum(r["pnl_usd"] for r in scratch_rows), 4),
            "by_trigger_detail": _rollup(
                [{**r, "scratch_trigger_detail": r["scratch_trigger_detail"] or "unknown"} for r in scratch_rows],
                "scratch_trigger_detail",
            ),
        },
    }


__all__ = ["build_scalp_attribution_report"]
