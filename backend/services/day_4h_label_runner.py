"""Offline runner that matures DAY outcome labels for prior ranking decisions.

This module closes the forward learning loop: `day_4h_outcome_labeler` can compute a
label, but nothing walked the decision ledger and persisted one. It is read-only against
every production table and writes only to `day_decision_outcome_labels`.

It must never participate in live ranking, sizing, exits, or order routing. Every public
entry point is fail-open: any error yields an empty/partial summary and leaves trading
state untouched.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL, drop_bars_after
from backend.services.day_4h_outcome_labeler import (
    HORIZONS_SEC,
    LABEL_VERSION,
    MAX_LIFECYCLE_SEC,
    label_candidate,
)
from backend.services.day_asof_4h import FourHAsOfTracker
from backend.services.day_decision_label_contract import TABLE_LABELS, ensure_label_schema, normalize_label
from backend.services.day_decision_observability import TABLE_CANDIDATES, TABLE_GROUPS
from backend.services.day_trade_thesis import htf_4h_rise_broken

logger = logging.getLogger(__name__)

RUNNER_VERSION = "day_4h_label_runner_v1"
TABLE_CLOSES = "position_close_ledger"
DUST_WRITEOFF_REASON = "DUST_WRITEOFF"
RESIDUAL_LOSS_FRACTION = 0.99

# Earliest horizon that yields a usable label. Anything younger is left alone so we never
# write a label whose every markout is null.
MIN_MATURITY_SEC = HORIZONS_SEC["15m"]
# A label is only final once the longest horizon AND the lifecycle ceiling have elapsed.
FINAL_MATURITY_SEC = max(MAX_LIFECYCLE_SEC, HORIZONS_SEC["4h"])
DEFAULT_BATCH_LIMIT = 60

_BUY_TRADE_ID_RE = re.compile(r"buy_trade_id=([^;]+)")
_EPOCH_MS_SUFFIX_RE = re.compile(r"_(\d{10,13})$")


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_epoch_utc(ts: Any) -> float | None:
    """Epoch seconds for a bar timestamp, resolving naive values as UTC.

    `feature_ohlcv.ts` is stored without an offset (`2026-09-03 22:36:50.159198`). The
    shared `parse_epoch` resolves such strings in server-local time, which is a no-op on
    Ocean (Etc/UTC) but shifts every bar by the local offset anywhere else, silently
    pairing decisions with the wrong bars. Labels must not depend on the host timezone.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        value = float(ts)
        return value / 1000.0 if value > 1e12 else value
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def load_1m_bars_utc(conn: sqlite3.Connection, symbol: str) -> list[tuple[int, float, float, float, float, float]]:
    """`day_4h_outcome_labeler.load_1m_bars` with timezone-independent timestamps."""
    try:
        present = {str(r[1]) for r in conn.execute("PRAGMA table_info(feature_ohlcv)")}
    except sqlite3.OperationalError:
        return []
    if "ts" not in present:
        return []
    for name in (symbol, f"{symbol[:-4]}-USDT", f"{symbol[:-4]}/USDT"):
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM feature_ohlcv "
            "WHERE interval='1m' AND symbol=? ORDER BY ts ASC",
            (name,),
        ).fetchall()
        if not rows:
            continue
        out: list[tuple[int, float, float, float, float, float]] = []
        for ts, o, h, low, c, v in rows:
            ep = parse_epoch_utc(ts)
            if ep is None:
                continue
            out.append((int(ep), float(o or 0), float(h or 0), float(low or 0), float(c or 0), float(v or 0)))
        out.sort(key=lambda b: b[0])
        return out
    return []


def api_symbol(symbol: str) -> str:
    """`XRP/USDT` -> `XRPUSDT`. Leaves already-compact symbols untouched."""
    return str(symbol or "").replace("/", "").replace("-", "").upper()


def trade_id_epoch(trade_id: str | None) -> float | None:
    """DAY trade ids carry an epoch-ms suffix, e.g. `mystic_XRP/USDT_1788469212809`."""
    if not trade_id:
        return None
    match = _EPOCH_MS_SUFFIX_RE.search(str(trade_id))
    if not match:
        return None
    raw = float(match.group(1))
    return raw / 1000.0 if raw > 1e12 else raw


def buy_trade_id_from_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    match = _BUY_TRADE_ID_RE.search(str(detail))
    return match.group(1).strip() if match else None


def is_residual_writeoff(reason: Any, realized_profit: float | None, entry_px: float | None, qty: float | None) -> bool:
    """True when a close row is a dust/residual write-off rather than an economic exit.

    Residual tranches are booked at exactly minus their own notional
    (`realized_profit == -entry_price * quantity`), so summing them in would report a
    profitable round trip as a large loss. They appear both as `DUST_WRITEOFF` and, for
    leftover slivers, under the parent position's own exit reason. DAY accounting already
    excludes dust from realized P&L; labels follow the same rule.
    """
    if str(reason or "").upper() == DUST_WRITEOFF_REASON:
        return True
    if realized_profit is None or entry_px is None or qty is None:
        return False
    notional = float(entry_px) * float(qty)
    if notional <= 0:
        return False
    # A long spot exit cannot lose its whole notional unless the asset went to zero, so a
    # tranche booked at <= -99% of its own notional is a write-off, not a market outcome.
    return float(realized_profit) <= -RESIDUAL_LOSS_FRACTION * notional


def load_close_ledger(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Map originating buy trade id -> aggregated authoritative close. Read-only.

    One production entry can close in several tranches, so the round trip is the sum of
    its economic closes rather than any single row. Dust/residual write-offs are dropped
    (see `is_residual_writeoff`); including them turns real outcomes into large phantom
    losses.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute(
            f"SELECT symbol, closed_at_epoch, close_reason, realized_profit, "
            f"realized_profit_unknown, quantity, entry_price, exit_price, sell_trade_id, detail "
            f"FROM {TABLE_CLOSES} ORDER BY closed_at_epoch ASC"
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    grouped: dict[str, list[dict[str, Any]]] = {}
    dust_counts: dict[str, int] = {}
    for sym, closed_ep, reason, profit, profit_unknown, qty, entry_px, exit_px, sell_id, detail in rows:
        buy_id = buy_trade_id_from_detail(detail)
        if not buy_id:
            continue
        if is_residual_writeoff(reason, _num(profit), _num(entry_px), _num(qty)):
            dust_counts[str(buy_id)] = dust_counts.get(str(buy_id), 0) + 1
            continue
        grouped.setdefault(str(buy_id), []).append(
            {
                "symbol": api_symbol(sym),
                "exit_epoch": _num(closed_ep),
                "exit_reason": reason,
                "realized_profit": _num(profit),
                "realized_profit_unknown": bool(profit_unknown),
                "quantity": _num(qty) or 0.0,
                "entry_price": _num(entry_px),
                "exit_price": _num(exit_px),
                "sell_trade_id": sell_id,
            }
        )
    for buy_id, tranches in grouped.items():
        qty_total = sum(float(t["quantity"]) for t in tranches)
        dominant = max(tranches, key=lambda t: float(t["quantity"]))
        profits = [t["realized_profit"] for t in tranches]
        exits = [t["exit_epoch"] for t in tranches if t["exit_epoch"] is not None]
        weighted_exit = None
        if qty_total > 0:
            priced = [t for t in tranches if t["exit_price"] is not None]
            if priced:
                weighted_exit = sum(
                    float(t["exit_price"]) * float(t["quantity"]) for t in priced
                ) / max(sum(float(t["quantity"]) for t in priced), 1e-12)
        out[buy_id] = {
            "symbol": dominant["symbol"],
            "exit_epoch": max(exits) if exits else None,
            "exit_reason": dominant["exit_reason"],
            "realized_profit": None if any(p is None for p in profits) else sum(float(p) for p in profits),
            "realized_profit_unknown": any(t["realized_profit_unknown"] for t in tranches)
            or any(p is None for p in profits),
            "quantity": qty_total,
            "entry_price": dominant["entry_price"],
            "exit_price": weighted_exit if weighted_exit is not None else dominant["exit_price"],
            "sell_trade_id": dominant["sell_trade_id"],
            "tranches": len(tranches),
            "dust_tranches_excluded": dust_counts.get(buy_id, 0),
        }
    return out


def authoritative_fill(contract: dict[str, Any], closes: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Build the authoritative fill/exit record for a group, or None.

    Requires a real production round trip: a stamped buy fill joined to its close row.
    Never invents a fill.
    """
    trade_id = contract.get("trade_id") or contract.get("fill_trade_id")
    if not trade_id:
        return None
    close = closes.get(str(trade_id))
    if not close or close.get("realized_profit_unknown"):
        return None
    entry_px = _num(contract.get("fill_price")) or close.get("entry_price")
    exit_px = close.get("exit_price")
    exit_epoch = close.get("exit_epoch")
    if not entry_px or not exit_px or not exit_epoch:
        return None
    entry_epoch = trade_id_epoch(str(trade_id))
    notional = float(entry_px) * float(close.get("quantity") or 0.0)
    gross_bps = (float(exit_px) - float(entry_px)) / float(entry_px) * 1e4
    net_dollars = close.get("realized_profit")
    net_bps = None
    if net_dollars is not None and notional > 0:
        net_bps = float(net_dollars) / notional * 1e4
    commission_bps = None
    commission = _num(contract.get("commission"))
    if commission is not None and notional > 0:
        commission_bps = float(commission) / notional * 1e4
    holding = None
    if entry_epoch is not None:
        holding = max(0.0, float(exit_epoch) - float(entry_epoch))
    return {
        "entry_price": float(entry_px),
        "entry_epoch": entry_epoch,
        "exit_epoch": float(exit_epoch),
        "exit_price": float(exit_px),
        "exit_reason": close.get("exit_reason"),
        "gross_bps": gross_bps,
        "net_bps": net_bps,
        "net_dollars": net_dollars,
        "commission_bps": commission_bps,
        "spread_bps": None,
        "slippage_bps": None,
        "holding_seconds": holding,
        "maker_taker": contract.get("maker_taker"),
    }


def scan_first_4h_break(
    bars: list[tuple[int, float, ...]],
    *,
    decision_epoch: float,
    end_epoch: float,
) -> float | None:
    """Seconds from the decision to the first 4H structure break, or None.

    Semantically identical to `day_4h_outcome_labeler.first_4h_break_seconds`, but advances
    one `FourHAsOfTracker` monotonically instead of rebuilding the as-of bundle per 1m bar.
    The reference implementation is O(bars^2) and takes minutes per symbol on a real tape;
    this is linear. `tests/test_day_4h_label_runner.py` pins the two to equal results.
    """
    clipped = [b for b in bars if int(b[0]) <= float(end_epoch) + 1e-9]
    if not clipped:
        return None
    tracker = FourHAsOfTracker(bars_1m=clipped)
    last_state = False
    for bar in clipped:
        ep = int(bar[0])
        if ep < decision_epoch - 1e-9:
            continue
        bundle = tracker.advance(float(ep))
        # Copy before clipping: `advance` caches the bundle it returns.
        as_of = {**bundle, "4h": drop_bars_after(bundle.get("4h"), float(ep))}
        broken = bool(htf_4h_rise_broken(as_of, current_price=float(bar[4]), now_epoch=float(ep)))
        if broken and not last_state:
            return max(0.0, float(ep) - float(decision_epoch))
        last_state = broken
    return None


def mark_at_or_before(bars: list[tuple[int, float, ...]], epoch: float) -> float | None:
    """Point-in-time close at/just before `epoch`. Used as the counterfactual entry mark."""
    chosen = None
    for bar in bars:
        if int(bar[0]) > float(epoch) + 1e-9:
            break
        chosen = bar
    return None if chosen is None else float(chosen[4])


def label_row_and_json(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Column values plus `label_json` for one label, matching `persist_label` exactly."""
    row = normalize_label(payload)
    merged = {**row, **payload, "label_version": LABEL_VERSION}
    return row, json.dumps(merged, default=str)[:32000]


def persist_labels(db_path: str | Path, payloads: list[dict[str, Any]]) -> int:
    """Write a batch of labels in one transaction. Fail-open; returns rows written.

    `persist_label` opens three connections per label, which would mean thousands of
    short-lived writers against the live trading database on a backlog run. The stored
    result is identical (pinned by `test_batch_persist_matches_persist_label`).
    """
    if not payloads:
        return 0
    written = 0
    try:
        ensure_label_schema(db_path)
        conn = sqlite3.connect(str(db_path), timeout=30)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("DAY_4H_LABEL_RUNNER batch persist open failed: %s", exc)
        return 0
    try:
        for payload in payloads:
            row, label_json = label_row_and_json(payload)
            if not row["decision_group_id"] or not row["symbol"]:
                continue
            markouts = row["markouts"]
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_LABELS}(
                    decision_group_id, symbol, created_at, provenance,
                    markout_15m_net_bps, markout_30m_net_bps, markout_1h_net_bps,
                    markout_2h_net_bps, markout_4h_net_bps,
                    mfe_bps, mae_bps, time_to_mfe_sec, time_to_mae_sec, cost_cover,
                    production_exit_gross_bps, commission_bps, spread_bps, slippage_bps,
                    production_exit_net_bps, holding_time_sec, capture_ratio, exit_reason,
                    regret_vs_best_eligible_bps, regret_vs_hold_bps, label_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["decision_group_id"],
                    row["symbol"],
                    _now_iso(),
                    row["provenance"],
                    markouts.get("15m"),
                    markouts.get("30m"),
                    markouts.get("1h"),
                    markouts.get("2h"),
                    markouts.get("4h"),
                    row["mfe_bps"],
                    row["mae_bps"],
                    row["time_to_mfe_sec"],
                    row["time_to_mae_sec"],
                    1 if row["cost_cover"] else 0,
                    row["production_exit_gross_bps"],
                    row["commission_bps"],
                    row["spread_bps"],
                    row["slippage_bps"],
                    row["production_exit_net_bps"],
                    row["holding_time_sec"],
                    row["capture_ratio"],
                    row["exit_reason"],
                    row["regret_vs_best_eligible_bps"],
                    row["regret_vs_hold_bps"],
                    label_json,
                ),
            )
            written += 1
        conn.commit()
    except Exception as exc:
        logger.debug("DAY_4H_LABEL_RUNNER batch persist failed: %s", exc)
        written = 0
    finally:
        conn.close()
    return written


def completed_labels(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Groups/symbols already labeled to completion. Avoids rewriting settled history."""
    try:
        rows = conn.execute(
            f"SELECT decision_group_id, symbol, label_json FROM {TABLE_LABELS}"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    done: set[tuple[str, str]] = set()
    for gid, symbol, raw in rows:
        payload = _loads(raw)
        if payload.get("label_completed_at") and payload.get("label_version") == LABEL_VERSION:
            done.add((str(gid), str(symbol)))
    return done


def pending_groups(
    conn: sqlite3.Connection,
    *,
    now_epoch: float,
    limit: int = DEFAULT_BATCH_LIMIT,
    min_maturity_sec: float = MIN_MATURITY_SEC,
) -> list[dict[str, Any]]:
    """Decision groups old enough to label and not already finalized. Oldest first."""
    cutoff = datetime.fromtimestamp(float(now_epoch) - float(min_maturity_sec), tz=timezone.utc)
    try:
        rows = conn.execute(
            f"SELECT decision_group_id, created_at, selected_action, selected_symbol, contract_json "
            f"FROM {TABLE_GROUPS} WHERE created_at <= ? ORDER BY created_at ASC",
            (cutoff.isoformat(),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    done = completed_labels(conn)
    out: list[dict[str, Any]] = []
    for gid, created, action, selected, contract in rows:
        decision_epoch = parse_epoch_utc(created)
        if decision_epoch is None:
            continue
        symbols = [*COINS, HOLD_SYMBOL]
        if all((str(gid), sym) in done for sym in symbols):
            continue
        out.append(
            {
                "decision_group_id": str(gid),
                "created_at": created,
                "decision_epoch": float(decision_epoch),
                "selected_action": action,
                "selected_symbol": str(selected or HOLD_SYMBOL),
                "contract": _loads(contract),
                "done_symbols": {sym for sym in symbols if (str(gid), sym) in done},
            }
        )
        if len(out) >= int(limit):
            break
    return out


def candidate_symbols(conn: sqlite3.Connection, decision_group_id: str) -> list[str]:
    try:
        rows = conn.execute(
            f"SELECT symbol FROM {TABLE_CANDIDATES} WHERE decision_group_id=?",
            (decision_group_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [str(r[0]) for r in rows]


def run_label_batch(
    db_path: str | Path,
    *,
    now_epoch: float | None = None,
    limit: int = DEFAULT_BATCH_LIMIT,
    min_maturity_sec: float = MIN_MATURITY_SEC,
) -> dict[str, Any]:
    """Label matured decision groups. Fail-open; returns a summary and never raises."""
    now = _now_epoch() if now_epoch is None else float(now_epoch)
    summary: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "label_version": LABEL_VERSION,
        "groups_scanned": 0,
        "labels_written": 0,
        "authoritative": 0,
        "reconstructed": 0,
        "unknown": 0,
        "hold": 0,
        "errors": 0,
    }
    try:
        ensure_label_schema(db_path)
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, timeout=30)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("DAY_4H_LABEL_RUNNER open failed: %s", exc)
        summary["errors"] += 1
        return summary
    try:
        groups = pending_groups(conn, now_epoch=now, limit=limit, min_maturity_sec=min_maturity_sec)
        if not groups:
            return summary
        closes = load_close_ledger(conn)
        bars_cache: dict[str, list[tuple[int, float, ...]]] = {}
        payloads: list[dict[str, Any]] = []
        for group in groups:
            summary["groups_scanned"] += 1
            contract = group["contract"]
            decision_epoch = group["decision_epoch"]
            selected = group["selected_symbol"]
            fill = authoritative_fill(contract, closes)
            present = set(candidate_symbols(conn, group["decision_group_id"])) or {*COINS, HOLD_SYMBOL}
            for symbol in (*COINS, HOLD_SYMBOL):
                if symbol not in present or symbol in group["done_symbols"]:
                    continue
                try:
                    if symbol == HOLD_SYMBOL:
                        payload = label_candidate(
                            decision_group_id=group["decision_group_id"],
                            symbol=symbol,
                            decision_epoch=decision_epoch,
                            entry_px=None,
                            bars=[],
                            now_epoch=now,
                        )
                        summary["hold"] += 1
                    else:
                        if symbol not in bars_cache:
                            bars_cache[symbol] = load_1m_bars_utc(conn, symbol)
                        bars = bars_cache[symbol]
                        is_selected = symbol == selected
                        symbol_fill = fill if (is_selected and fill) else None
                        entry_px = (
                            symbol_fill["entry_price"]
                            if symbol_fill
                            else mark_at_or_before(bars, decision_epoch)
                        )
                        end = min(now, decision_epoch + MAX_LIFECYCLE_SEC)
                        if symbol_fill and symbol_fill.get("exit_epoch"):
                            end = min(end, float(symbol_fill["exit_epoch"]))
                        break_sec = scan_first_4h_break(
                            bars, decision_epoch=decision_epoch, end_epoch=end
                        )
                        payload = label_candidate(
                            decision_group_id=group["decision_group_id"],
                            symbol=symbol,
                            decision_epoch=decision_epoch,
                            entry_px=entry_px,
                            bars=bars,
                            now_epoch=now,
                            fill=symbol_fill,
                            break_seconds=break_sec,
                        )
                        payload["selected"] = is_selected
                        prov = str(payload.get("provenance") or "unknown")
                        if prov == "authoritative":
                            summary["authoritative"] += 1
                        elif prov == "reconstructed":
                            summary["reconstructed"] += 1
                        else:
                            summary["unknown"] += 1
                    payload["runner_version"] = RUNNER_VERSION
                    payloads.append(payload)
                except Exception as exc:
                    logger.debug(
                        "DAY_4H_LABEL_RUNNER label failed group=%s symbol=%s: %s",
                        group["decision_group_id"],
                        symbol,
                        exc,
                    )
                    summary["errors"] += 1
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("DAY_4H_LABEL_RUNNER batch failed: %s", exc)
        summary["errors"] += 1
        payloads = []
    finally:
        conn.close()
    summary["labels_written"] = persist_labels(db_path, payloads)
    if payloads and not summary["labels_written"]:
        summary["errors"] += 1
    return summary


__all__ = [
    "DEFAULT_BATCH_LIMIT",
    "DUST_WRITEOFF_REASON",
    "FINAL_MATURITY_SEC",
    "MIN_MATURITY_SEC",
    "RUNNER_VERSION",
    "api_symbol",
    "authoritative_fill",
    "buy_trade_id_from_detail",
    "completed_labels",
    "is_residual_writeoff",
    "label_row_and_json",
    "load_1m_bars_utc",
    "load_close_ledger",
    "mark_at_or_before",
    "parse_epoch_utc",
    "pending_groups",
    "persist_labels",
    "run_label_batch",
    "scan_first_4h_break",
    "trade_id_epoch",
]
