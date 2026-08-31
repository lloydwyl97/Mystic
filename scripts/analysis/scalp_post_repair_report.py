#!/usr/bin/env python3
"""Read-only clean SCALP performance after breaker-recovery repair.

Not imported by runtime services. Does not write to SQLite, Redis, or env.
Does not combine Local and Ocean into one strategy statistic.

Canonical clean-start boundaries (do not move; do not rewrite history):
  Local  eval_after = 2026-08-23 02:02:38 UTC
  Ocean  eval_after = 2026-08-22 04:06:38 UTC

Usage on the host being measured:
  venv/bin/python3 scripts/analysis/scalp_post_repair_report.py
  venv/bin/python3 scripts/analysis/scalp_post_repair_report.py --host local
  venv/bin/python3 scripts/analysis/scalp_post_repair_report.py --host ocean --db /home/mystic/mystic/mystic_scalp.db
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    ROOT = Path(__file__).resolve().parents[2]
except (IndexError, OSError):
    ROOT = Path("/home/mystic/mystic")

# Frozen at runtime verification 2026-08-23. Do not edit to chase later PnL.
CANONICAL_EVAL_AFTER = {
    "local": "2026-08-23 02:02:38",
    "ocean": "2026-08-22 04:06:38",
}
HISTORICAL_BASELINE = {
    "local": {"sells": 269, "realized": 4.448221},
    "ocean": {"sells": 502, "realized": -10.79125},
}
REQUIRED_EXITS = (
    "PATH_EXECUTABLE_PROFIT",
    "NET_PROFIT_TARGET",
    "PATH_MAX_ADVERSE_STOP",
    "MAX_HOLD_HARD_LIMIT",
    "EARLY_SCRATCH_EXIT",
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CLASSIFICATION = "RUNTIME VALIDATED — STRATEGY SAMPLE STILL SMALL"


def detect_host() -> str:
    name = socket.gethostname().strip().lower()
    if name in {"mystic-prod", "mystic-ocean"} or name.endswith("-prod"):
        return "ocean"
    return "local"


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text.replace("+00:00", ""), fmt) if "%z" not in fmt else datetime.strptime(text, fmt)  # noqa: DTZ007
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _load_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 8) if values else None


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 8) if values else None


def _max_streak(flags: list[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _current_consec_losses(pnls: list[float]) -> int:
    n = 0
    for pnl in reversed(pnls):
        if pnl <= 0:
            n += 1
        else:
            break
    return n


def _rollup(pnls: list[float]) -> dict[str, Any]:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    gp = sum(wins)
    gl = abs(sum(losses))
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n, 6) if n else None,
        "net": round(sum(pnls), 6) if n else 0.0,
        "expectancy": round(sum(pnls) / n, 6) if n else None,
        "profit_factor": round(gp / gl, 6) if gl else ("inf" if gp > 0 else None),
        "avg_winner": round(sum(wins) / len(wins), 6) if wins else None,
        "avg_loser": round(sum(losses) / len(losses), 6) if losses else None,
        "largest_winner": round(max(wins), 6) if wins else None,
        "largest_loser": round(min(losses), 6) if losses else None,
        "max_consecutive_wins": _max_streak([p > 0 for p in pnls]),
        "max_consecutive_losses": _max_streak([p <= 0 for p in pnls]),
    }


def _entry_spread(buy_diag: dict, pos_diag: dict, sell_diag: dict) -> float | None:
    for src in (pos_diag, buy_diag):
        for key in ("spread_at_entry", "spread_pct", "redis_spread_pct"):
            val = _f(src.get(key))
            if val is not None:
                return val
    ranking = (pos_diag.get("symbol_ranking") or {}) if isinstance(pos_diag.get("symbol_ranking"), dict) else {}
    val = _f(ranking.get("spread_pct"))
    if val is not None:
        return val
    pre = sell_diag.get("preflight") if isinstance(sell_diag.get("preflight"), dict) else {}
    return _f(pre.get("spread_pct"))


def _exit_spread(sell_diag: dict) -> float | None:
    pre = sell_diag.get("preflight") if isinstance(sell_diag.get("preflight"), dict) else {}
    return _f(pre.get("spread_pct"))


def _rank_score(buy_diag: dict, pos_diag: dict, sell_diag: dict) -> float | None:
    for src in (buy_diag, pos_diag, sell_diag):
        val = _f(src.get("rank_score") if src.get("rank_score") is not None else src.get("legacy_rank_score"))
        if val is not None:
            return val
    ranking = pos_diag.get("symbol_ranking") if isinstance(pos_diag.get("symbol_ranking"), dict) else {}
    return _f(ranking.get("rank_score"))


def _entry_net_edge(buy_diag: dict, sell_diag: dict) -> float | None:
    for key in ("buy_ev", "predicted_net_return", "predicted_net_ev", "selected_expected_net_ev"):
        val = _f(buy_diag.get(key))
        if val is not None:
            return val
    pre = sell_diag.get("preflight") if isinstance(sell_diag.get("preflight"), dict) else {}
    return _f(pre.get("expected_net_edge_pct"))


def load_report(*, db_path: str, host: str, eval_after: str, threshold: int) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        meta = {}
        try:
            meta = dict(
                conn.execute(
                    """
                    SELECT consec_breaker_tripped_at, consec_breaker_recovery_until,
                           consec_breaker_eval_after, consec_breaker_reason
                    FROM scalp_meta WHERE id = 1
                    """
                ).fetchone()
                or {}
            )
        except sqlite3.OperationalError:
            meta = {}
        ledger = dict(conn.execute("SELECT realized_pnl, cash_balance, positions_value, principal FROM scalp_paper_ledger WHERE id=1").fetchone())
        hist = HISTORICAL_BASELINE[host]
        hist_sells = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl_usd),0) FROM scalp_paper_trades WHERE upper(side)='SELL' AND created_at<=?",
            (eval_after,),
        ).fetchone()
        sells = list(
            conn.execute(
                """
                SELECT id, trade_id, symbol, quantity, price, notional, fee_usd, slippage_usd,
                       pnl_usd, entry_price, exit_reason, created_at, diagnostics_json
                FROM scalp_paper_trades
                WHERE upper(side)='SELL' AND created_at > ?
                ORDER BY id
                """,
                (eval_after,),
            )
        )
        rows: list[dict[str, Any]] = []
        for sell in sells:
            tid = str(sell["trade_id"] or "").replace("_SELL", "")
            buy = conn.execute(
                "SELECT trade_id, price, notional, fee_usd, slippage_usd, created_at, diagnostics_json FROM scalp_paper_trades WHERE trade_id=?",
                (tid,),
            ).fetchone()
            pos = conn.execute(
                "SELECT max_favorable_pct, max_adverse_pct, diagnostics_json, entry_time FROM scalp_paper_positions WHERE trade_id=?",
                (tid,),
            ).fetchone()
            buy_diag = _load_json(buy["diagnostics_json"] if buy else None)
            sell_diag = _load_json(sell["diagnostics_json"])
            pos_diag = _load_json(pos["diagnostics_json"] if pos else None)
            entry_ts = buy["created_at"] if buy else None
            exit_ts = sell["created_at"]
            t0 = _parse_ts(entry_ts)
            t1 = _parse_ts(exit_ts)
            hold = (t1 - t0).total_seconds() if t0 and t1 else None
            fee = float(sell["fee_usd"] or 0) + float((buy["fee_usd"] if buy else 0) or 0)
            slip = float(sell["slippage_usd"] or 0) + float((buy["slippage_usd"] if buy else 0) or 0)
            net = float(sell["pnl_usd"] or 0)
            cost = fee + slip
            gross = net + cost
            mfe = _f(pos["max_favorable_pct"]) if pos else None
            mae = _f(pos["max_adverse_pct"]) if pos else None
            rows.append(
                {
                    "trade_id": tid,
                    "symbol": str(sell["symbol"] or "").upper(),
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "hold_sec": hold,
                    "entry_price": _f(buy["price"] if buy else sell["entry_price"]),
                    "exit_price": _f(sell["price"]),
                    "quantity": _f(sell["quantity"]),
                    "notional": _f(buy["notional"] if buy else sell["notional"]),
                    "fee_usd": round(fee, 8),
                    "slippage_usd": round(slip, 8),
                    "cost_usd": round(cost, 8),
                    "gross_pnl": round(gross, 8),
                    "net_pnl": round(net, 8),
                    "exit_reason": str(sell["exit_reason"] or "UNKNOWN"),
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                    "mfe_capture": round(net / (mfe * float(buy["notional"] if buy else sell["notional"] or 0)), 6) if mfe and mfe > 0 and buy else None,
                    "entry_spread": _entry_spread(buy_diag, pos_diag, sell_diag),
                    "exit_spread": _exit_spread(sell_diag),
                    "rank_score": _rank_score(buy_diag, pos_diag, sell_diag),
                    "entry_net_edge": _entry_net_edge(buy_diag, sell_diag),
                }
            )
        open_n = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
    finally:
        conn.close()

    pnls = [r["net_pnl"] for r in rows]
    holds = [float(r["hold_sec"]) for r in rows if r["hold_sec"] is not None]
    summary = _rollup(pnls)
    summary["avg_hold_sec"] = _mean(holds)
    summary["median_hold_sec"] = _median(holds)
    summary["classification"] = CLASSIFICATION

    by_symbol = {}
    for sym in SYMBOLS:
        by_symbol[sym] = _rollup([r["net_pnl"] for r in rows if r["symbol"] == sym])

    exit_counts = Counter(r["exit_reason"] for r in rows)
    by_exit = {}
    for reason in list(REQUIRED_EXITS) + sorted(k for k in exit_counts if k not in REQUIRED_EXITS):
        subset = [r for r in rows if r["exit_reason"] == reason]
        nets = [r["net_pnl"] for r in subset]
        by_exit[reason] = {
            "count": len(subset),
            "total_net": round(sum(nets), 6) if nets else 0.0,
            "average_net": round(sum(nets) / len(nets), 6) if nets else None,
        }

    costs = [r["cost_usd"] for r in rows]
    grosses = [r["gross_pnl"] for r in rows]
    fees = [r["fee_usd"] for r in rows]
    slips = [r["slippage_usd"] for r in rows]
    mfes = [float(r["mfe_pct"]) for r in rows if r["mfe_pct"] is not None]
    maes = [float(r["mae_pct"]) for r in rows if r["mae_pct"] is not None]
    captures = [float(r["mfe_capture"]) for r in rows if r["mfe_capture"] is not None]
    total_gross = sum(grosses)
    total_cost = sum(costs)
    execution = {
        "avg_cost_per_round_trip": _mean(costs),
        "avg_fee_usd": _mean(fees),
        "avg_slippage_usd": _mean(slips),
        "total_gross_pnl": round(total_gross, 6),
        "total_cost_usd": round(total_cost, 6),
        "gross_to_net_cost_drag": round(total_cost / total_gross, 6) if total_gross else None,
        "pct_gross_edge_lost_to_costs": round(100.0 * total_cost / total_gross, 4) if total_gross else None,
        "avg_mfe_pct": _mean(mfes),
        "avg_mae_pct": _mean(maes),
        "avg_mfe_capture": _mean(captures),
        "avg_entry_spread": _mean([float(r["entry_spread"]) for r in rows if r["entry_spread"] is not None]),
        "avg_exit_spread": _mean([float(r["exit_spread"]) for r in rows if r["exit_spread"] is not None]),
        "avg_rank_score": _mean([float(r["rank_score"]) for r in rows if r["rank_score"] is not None]),
        "avg_entry_net_edge": _mean([float(r["entry_net_edge"]) for r in rows if r["entry_net_edge"] is not None]),
    }

    live_eval = str(meta.get("consec_breaker_eval_after") or "").strip()
    breaker = {
        "state": "OPEN" if meta.get("consec_breaker_recovery_until") or meta.get("consec_breaker_tripped_at") else "CLOSED",
        "current_consecutive_losses": _current_consec_losses(pnls),
        "threshold": threshold,
        "last_trip_timestamp": meta.get("consec_breaker_tripped_at"),
        "recovery_timestamp": meta.get("consec_breaker_recovery_until"),
        "eval_after_live": live_eval or None,
        "eval_after_canonical": eval_after,
        "eval_after_matches_canonical": live_eval == eval_after,
        "trades_since_eval_window": len(rows),
        "reason": meta.get("consec_breaker_reason"),
    }

    return {
        "host": host,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "db_path": db_path,
        "canonical_eval_after": eval_after,
        "historical_baseline": {
            "sells": hist["sells"],
            "realized": hist["realized"],
            "observed_sells_at_or_before_eval_after": int(hist_sells[0]),
            "observed_realized_at_or_before_eval_after": round(float(hist_sells[1]), 6),
        },
        "ledger_realized_total": round(float(ledger.get("realized_pnl") or 0), 6),
        "open_positions": int(open_n),
        "classification": CLASSIFICATION,
        "clean_post_repair": summary,
        "by_symbol": by_symbol,
        "by_exit_reason": by_exit,
        "execution_quality": execution,
        "breaker": breaker,
        "trades": rows,
    }


def compare(local: dict[str, Any], ocean: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trades",
        "wins",
        "losses",
        "win_rate",
        "net",
        "expectancy",
        "profit_factor",
        "avg_winner",
        "avg_loser",
        "avg_hold_sec",
    )
    out = {}
    for key in keys:
        out[key] = {
            "local": (local.get("clean_post_repair") or {}).get(key),
            "ocean": (ocean.get("clean_post_repair") or {}).get(key),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only post-repair SCALP report")
    parser.add_argument("--host", choices=("local", "ocean", "auto"), default="auto")
    parser.add_argument("--db", default=str(ROOT / "mystic_scalp.db"))
    parser.add_argument("--eval-after", default="")
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()
    host = detect_host() if args.host == "auto" else args.host
    eval_after = args.eval_after or CANONICAL_EVAL_AFTER[host]
    threshold = args.threshold or (5 if host == "local" else 10)
    report = load_report(db_path=args.db, host=host, eval_after=eval_after, threshold=threshold)
    print(json.dumps(report, indent=2, default=str))
    if not args.json_only:
        clean = report["clean_post_repair"]
        print(
            f"\n{host.upper()} CLEAN POST-REPAIR  trades={clean['trades']} W/L={clean['wins']}/{clean['losses']} WR={clean['win_rate']} net={clean['net']} class={CLASSIFICATION}",
            file=os.sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
