"""Classify historical SCALP/DAY round trips using 1m OHLCV reconstruction.

Does not delete or rewrite trades. Used for post-repair analysis only.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

SCALP_TARGET_PCT = 0.0025
DAY_TARGET_PCT = 0.008
MEANINGFUL_MFE_FRAC = 0.50


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OSError, ValueError):
            return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(text, fmt)  # noqa: DTZ007
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _norm_symbol(symbol: str) -> str:
    s = str(symbol or "").upper().replace("/", "").replace("-", "")
    if s.endswith("USDT"):
        return s
    return f"{s}USDT" if s else ""


def _ohlcv_symbol(symbol: str) -> str:
    s = _norm_symbol(symbol)
    if s.endswith("USDT"):
        return f"{s[:-4]}-USDT"
    return s


def load_ohlcv(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = conn.execute("SELECT symbol, open, high, low, close, volume, ts FROM feature_ohlcv WHERE interval='1m' ORDER BY ts")
    for symbol, o, h, low, c, v, ts in rows:
        dt = _parse_ts(ts)
        if dt is None:
            continue
        out[_ohlcv_symbol(symbol)].append(
            {
                "ts": dt,
                "open": float(o or 0),
                "high": float(h or 0),
                "low": float(low or 0),
                "close": float(c or 0),
                "volume": float(v or 0),
            }
        )
    return out


def _bars_after(bars: list[dict[str, Any]], when: datetime, horizon_sec: int) -> list[dict[str, Any]]:
    end = when + timedelta(seconds=horizon_sec)
    return [b for b in bars if when < b["ts"] <= end]


def _bars_between(bars: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [b for b in bars if start <= b["ts"] <= end]


def _price_at_offset(bars: list[dict[str, Any]], when: datetime, offset_sec: int) -> float | None:
    target = when + timedelta(seconds=offset_sec)
    after = [b for b in bars if b["ts"] >= target]
    if after:
        return float(after[0]["close"])
    before = [b for b in bars if b["ts"] <= target]
    if before:
        return float(before[-1]["close"])
    return None


def reconstruct_excursions(
    bars: list[dict[str, Any]],
    *,
    entry_ts: datetime,
    exit_ts: datetime,
    entry_price: float,
) -> dict[str, Any]:
    hold_bars = _bars_between(bars, entry_ts, exit_ts)
    mfe = 0.0
    mae = 0.0
    t_mfe = None
    t_mae = None
    if entry_price > 0:
        for b in hold_bars:
            fav = (float(b["high"]) - entry_price) / entry_price
            adv = (float(b["low"]) - entry_price) / entry_price
            if fav > mfe:
                mfe = fav
                t_mfe = (b["ts"] - entry_ts).total_seconds()
            if adv < mae:
                mae = adv
                t_mae = (b["ts"] - entry_ts).total_seconds()
    return {
        "mfe_pct": round(mfe, 6),
        "mae_pct": round(mae, 6),
        "time_to_mfe_sec": t_mfe,
        "time_to_mae_sec": t_mae,
        "hold_bars": len(hold_bars),
    }


def reconstruct_post_exit(
    bars: list[dict[str, Any]],
    *,
    exit_ts: datetime,
    entry_price: float,
    target_pct: float,
    horizon_sec: int = 1200,
) -> dict[str, Any]:
    offsets = (30, 60, 180, 300, horizon_sec)
    path: dict[str, float | None] = {}
    hit_target_after = False
    hit_after_sec = None
    if entry_price > 0:
        after = _bars_after(bars, exit_ts, horizon_sec)
        for b in after:
            if float(b["high"]) >= entry_price * (1.0 + target_pct):
                hit_target_after = True
                hit_after_sec = (b["ts"] - exit_ts).total_seconds()
                break
    for sec in offsets:
        px = _price_at_offset(bars, exit_ts, sec)
        path[f"plus_{sec}s"] = round(px, 8) if px is not None else None
    return {
        "path": path,
        "hit_original_target_after_exit": hit_target_after,
        "hit_target_after_sec": hit_after_sec,
    }


def classify_loss(
    *,
    net_pnl: float,
    gross_pnl: float,
    costs: float,
    mfe_pct: float,
    target_pct: float,
    hit_target_after: bool,
    hit_after_sec: float | None,
) -> str:
    if net_pnl > 0:
        return "WIN"
    meaningful = target_pct * MEANINGFUL_MFE_FRAC
    cost_ate = gross_pnl > 0 and net_pnl <= 0 and costs >= max(gross_pnl * 0.80, 1e-9)
    exitish = mfe_pct >= target_pct or (hit_target_after and (hit_after_sec or 0) <= 300)
    entryish = mfe_pct < meaningful and not hit_target_after
    flags = []
    if entryish:
        flags.append("ENTRY")
    if exitish:
        flags.append("EXIT")
    if cost_ate:
        flags.append("COST")
    if len(flags) == 1:
        return f"{flags[0]}_FAILURE"
    if len(flags) > 1:
        return "COMBINATION"
    if mfe_pct >= meaningful and not exitish:
        return "COMBINATION"
    return "ENTRY_FAILURE"


def _pair_scalp_roundtrips(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT * FROM scalp_paper_trades ORDER BY id"))
    buys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    out: list[dict[str, Any]] = []
    for r in rows:
        rec = dict(r)
        side = str(rec.get("side") or "").upper()
        sym = _norm_symbol(str(rec.get("symbol") or ""))
        if side == "BUY":
            buys[sym].append(rec)
            continue
        if side != "SELL":
            continue
        entry = None
        if buys[sym]:
            entry = buys[sym].pop(0)
        diag = {}
        with_suppress = True
        try:
            diag = json.loads(rec.get("diagnostics_json") or "{}")
        except Exception:
            diag = {}
        entry_diag = {}
        if entry:
            try:
                entry_diag = json.loads(entry.get("diagnostics_json") or "{}")
            except Exception:
                entry_diag = {}
        entry_ts = _parse_ts((entry or {}).get("created_at")) or _parse_ts(diag.get("entry_time"))
        exit_ts = _parse_ts(rec.get("created_at"))
        entry_px = float(rec.get("entry_price") or (entry or {}).get("price") or 0)
        exit_px = float(rec.get("price") or 0)
        qty = float(rec.get("quantity") or 0)
        fees = float(rec.get("fee_usd") or 0)
        slip = float(rec.get("slippage_usd") or 0)
        net = float(rec.get("pnl_usd") or 0)
        gross = (exit_px - entry_px) * qty if entry_px and exit_px else net + fees + slip
        hold = None
        if entry_ts and exit_ts:
            hold = (exit_ts - entry_ts).total_seconds()
        out.append(
            {
                "sell_id": rec.get("id"),
                "symbol": sym,
                "setup": str(rec.get("strategy_id") or entry_diag.get("setup_name") or diag.get("setup_name") or ""),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "quantity": qty,
                "fees": fees,
                "slippage": slip,
                "gross_pnl": gross,
                "net_pnl": net,
                "exit_reason": str(rec.get("exit_reason") or ""),
                "hold_sec": hold,
                "entry_diag": entry_diag,
                "exit_diag": diag,
                "spread_est": float(entry_diag.get("spread_pct") or entry_diag.get("spread_at_entry") or 0),
                "rank_score": float((entry_diag.get("rank_score_at_entry") or entry_diag.get("rank_score") or 0) or 0),
                "soft_rank": bool(entry_diag.get("soft_rank_entry") or diag.get("soft_rank_entry")),
                "passed": entry_diag.get("passed"),
            }
        )
        _ = with_suppress
    return out


def classify_scalp_book(db_path: str, *, target_pct: float = SCALP_TARGET_PCT) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        bars_by_sym = load_ohlcv(conn)
        trades = _pair_scalp_roundtrips(conn)
    finally:
        conn.close()

    classified: list[dict[str, Any]] = []
    for t in trades:
        bars = bars_by_sym.get(_ohlcv_symbol(t["symbol"]), [])
        exc = {"mfe_pct": None, "mae_pct": None, "time_to_mfe_sec": None, "time_to_mae_sec": None}
        post = {"path": {}, "hit_original_target_after_exit": False, "hit_target_after_sec": None}
        if t["entry_ts"] and t["exit_ts"] and t["entry_price"] and bars:
            exc = reconstruct_excursions(bars, entry_ts=t["entry_ts"], exit_ts=t["exit_ts"], entry_price=t["entry_price"])
            post = reconstruct_post_exit(bars, exit_ts=t["exit_ts"], entry_price=t["entry_price"], target_pct=target_pct)
        cls = classify_loss(
            net_pnl=float(t["net_pnl"] or 0),
            gross_pnl=float(t["gross_pnl"] or 0),
            costs=float(t["fees"] or 0) + float(t["slippage"] or 0),
            mfe_pct=float(exc.get("mfe_pct") or 0),
            target_pct=target_pct,
            hit_target_after=bool(post.get("hit_original_target_after_exit")),
            hit_after_sec=post.get("hit_target_after_sec"),
        )
        scratch_q = {}
        if "SCRATCH" in str(t["exit_reason"] or "").upper():
            mfe = float(exc.get("mfe_pct") or 0)
            scratch_q = {
                "had_meaningful_mfe": mfe >= target_pct * MEANINGFUL_MFE_FRAC,
                "reached_target_after": bool(post.get("hit_original_target_after_exit")),
                "reached_target_after_sec": post.get("hit_target_after_sec"),
                "entry_immediately_wrong": mfe < target_pct * 0.15 and not post.get("hit_original_target_after_exit"),
                "costs_consumed_gross": float(t["gross_pnl"] or 0) > 0 and float(t["net_pnl"] or 0) <= 0,
            }
        row = {
            **t,
            **exc,
            **post,
            "class": cls,
            "scratch_questions": scratch_q,
            "entry_ts": t["entry_ts"].isoformat() if t["entry_ts"] else None,
            "exit_ts": t["exit_ts"].isoformat() if t["exit_ts"] else None,
        }
        classified.append(row)

    losses = [r for r in classified if r["class"] != "WIN"]
    n = len(classified)
    n_loss = len(losses) or 1
    counts = Counter(r["class"] for r in classified)
    loss_counts = Counter(r["class"] for r in losses)
    scratches = [r for r in classified if "SCRATCH" in str(r["exit_reason"] or "").upper()]
    return {
        "n_closes": n,
        "wins": sum(1 for r in classified if float(r["net_pnl"] or 0) > 0),
        "losses": n - sum(1 for r in classified if float(r["net_pnl"] or 0) > 0),
        "win_rate": round(sum(1 for r in classified if float(r["net_pnl"] or 0) > 0) / n, 4) if n else None,
        "net_pnl": round(sum(float(r["net_pnl"] or 0) for r in classified), 4),
        "expectancy": round(sum(float(r["net_pnl"] or 0) for r in classified) / n, 6) if n else None,
        "class_all": dict(counts),
        "class_losers": dict(loss_counts),
        "entry_failure_pct_losers": round(loss_counts.get("ENTRY_FAILURE", 0) / n_loss, 4),
        "exit_failure_pct_losers": round(loss_counts.get("EXIT_FAILURE", 0) / n_loss, 4),
        "cost_failure_pct_losers": round(loss_counts.get("COST_FAILURE", 0) / n_loss, 4),
        "combination_pct_losers": round(loss_counts.get("COMBINATION", 0) / n_loss, 4),
        "scratch_n": len(scratches),
        "scratch_had_mfe": sum(1 for r in scratches if (r.get("scratch_questions") or {}).get("had_meaningful_mfe")),
        "scratch_hit_target_after": sum(1 for r in scratches if (r.get("scratch_questions") or {}).get("reached_target_after")),
        "scratch_entry_wrong": sum(1 for r in scratches if (r.get("scratch_questions") or {}).get("entry_immediately_wrong")),
        "scratch_cost_consumed": sum(1 for r in scratches if (r.get("scratch_questions") or {}).get("costs_consumed_gross")),
        "by_symbol": _by_group(classified, "symbol"),
        "by_setup": _by_group(classified, "setup"),
        "by_exit": _by_group(classified, "exit_reason"),
        "rows": classified,
    }


def _by_group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "")].append(r)
    out = {}
    for k, items in groups.items():
        n = len(items)
        wins = sum(1 for r in items if float(r["net_pnl"] or 0) > 0)
        net = sum(float(r["net_pnl"] or 0) for r in items)
        out[k] = {
            "n": n,
            "wins": wins,
            "win_rate": round(wins / n, 4) if n else None,
            "net_pnl": round(net, 4),
            "expectancy": round(net / n, 6) if n else None,
            "classes": dict(Counter(r["class"] for r in items)),
        }
    return out


def classify_day_book(db_path: str, *, target_pct: float = DAY_TARGET_PCT) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        bars_by_sym = load_ohlcv(conn)
        sells = list(
            conn.execute(
                """
                SELECT id, symbol, price, entry_price, quantity, pnl, fees_paid, slippage_cost,
                       exit_reason, timestamp, created_at, entry_timestamp, hold_time_seconds,
                       explainability_json, diagnostics_json, regime
                FROM paper_trades WHERE UPPER(side)='SELL' ORDER BY id
                """
            )
        )
    finally:
        conn.close()

    classified = []
    for r in sells:
        rec = dict(r)
        sym = _norm_symbol(str(rec.get("symbol") or ""))
        entry_ts = _parse_ts(rec.get("entry_timestamp")) or _parse_ts(rec.get("created_at"))
        exit_ts = _parse_ts(rec.get("timestamp")) or _parse_ts(rec.get("created_at"))
        entry_px = float(rec.get("entry_price") or 0)
        exit_px = float(rec.get("price") or 0)
        qty = float(rec.get("quantity") or 0)
        net = float(rec.get("pnl") or 0)
        fees = float(rec.get("fees_paid") or 0)
        slip = float(rec.get("slippage_cost") or 0)
        gross = (exit_px - entry_px) * qty if entry_px and exit_px else net + fees + slip
        bars = bars_by_sym.get(_ohlcv_symbol(sym), [])
        exc = {"mfe_pct": None, "mae_pct": None}
        post = {"hit_original_target_after_exit": False, "hit_target_after_sec": None, "path": {}}
        if entry_ts and exit_ts and entry_px and bars:
            exc = reconstruct_excursions(bars, entry_ts=entry_ts, exit_ts=exit_ts, entry_price=entry_px)
            post = reconstruct_post_exit(bars, exit_ts=exit_ts, entry_price=entry_px, target_pct=target_pct, horizon_sec=3600)
        setup = ""
        try:
            ex = json.loads(rec.get("explainability_json") or "{}")
            setup = str(ex.get("setup_type_canonical") or ex.get("setup_type") or ex.get("entry_thesis") or "")
        except Exception:
            setup = ""
        cls = classify_loss(
            net_pnl=net,
            gross_pnl=gross,
            costs=fees + slip,
            mfe_pct=float(exc.get("mfe_pct") or 0),
            target_pct=target_pct,
            hit_target_after=bool(post.get("hit_original_target_after_exit")),
            hit_after_sec=post.get("hit_target_after_sec"),
        )
        classified.append(
            {
                "id": rec.get("id"),
                "symbol": sym,
                "setup": setup,
                "exit_reason": str(rec.get("exit_reason") or ""),
                "regime": rec.get("regime"),
                "net_pnl": net,
                "gross_pnl": gross,
                "hold_sec": rec.get("hold_time_seconds"),
                "class": cls,
                **exc,
                **post,
            }
        )
    n = len(classified)
    losses = [r for r in classified if r["class"] != "WIN"]
    n_loss = len(losses) or 1
    loss_counts = Counter(r["class"] for r in losses)
    wins = sum(1 for r in classified if float(r["net_pnl"] or 0) > 0)
    net = sum(float(r["net_pnl"] or 0) for r in classified)
    return {
        "n_closes": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": round(wins / n, 4) if n else None,
        "net_pnl": round(net, 4),
        "expectancy": round(net / n, 6) if n else None,
        "class_losers": dict(loss_counts),
        "entry_failure_pct_losers": round(loss_counts.get("ENTRY_FAILURE", 0) / n_loss, 4),
        "exit_failure_pct_losers": round(loss_counts.get("EXIT_FAILURE", 0) / n_loss, 4),
        "cost_failure_pct_losers": round(loss_counts.get("COST_FAILURE", 0) / n_loss, 4),
        "combination_pct_losers": round(loss_counts.get("COMBINATION", 0) / n_loss, 4),
        "by_symbol": _by_group(classified, "symbol"),
        "by_setup": _by_group(classified, "setup"),
        "by_exit": _by_group(classified, "exit_reason"),
        "progress_decay": _by_group([r for r in classified if "PROGRESS_DECAY" in str(r["exit_reason"] or "").upper()], "symbol"),
        "rows": classified,
    }
