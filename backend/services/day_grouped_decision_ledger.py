"""Point-in-time grouped DAY decision ledger.

Builds one decision_group_id per production ranking timestamp from
decision_book_tape (BTC/ETH/SOL/XRP/HOLD). Does not select trades, change
sizing, or admit rejects as fills.

Path-EV values are taken from stored tape extras only. This module never
calls score_four_coins or substitutes today's model output for missing
historical as-of scores.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.config.execution_cost_model import (
    expected_exchange_commission_rt_pct,
    expected_slippage_rt_pct,
    expected_spread_pct,
)
from backend.services.day_production_lifecycle_replay import (
    SYMBOLS,
    parse_epoch,
    parse_feature_vector,
)

HOLD_SYMBOL = "HOLD"
COINS = SYMBOLS
OUTCOME_CLASSES = (
    "selected_execute",
    "ranking_loser",
    "stale_invalid",
    "spread_impact_ineligible",
    "symbol_open",
    "no_slot_or_capital",
    "blocked_after_ranking",
    "no_order_match",
    "terminal_fill_failure",
    "partial_fill",
    "hold",
    "other",
)
# Only terminal_fill_failure may be called fill_failed.
FILL_FAILED_ALIAS = "terminal_fill_failure"
AUTHORITY_SYMBOLS = (*COINS, HOLD_SYMBOL)
FILL_JOIN_SEC = 90
BOOK_STALE_SEC = 30.0


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT") and s != "HOLD":
        s = s + "T"
    return s


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def group_id_for(ts_utc: str) -> str:
    digest = hashlib.sha1(str(ts_utc).encode("utf-8")).hexdigest()[:16]
    return f"dg_{digest}"


@dataclass
class CandidateRow:
    decision_group_id: str
    ts_utc: str
    epoch: int
    bar_epoch: int
    symbol: str
    selected_action: str
    live_selected: bool
    path_ev: float
    hold_ev: float
    p_buy: float | None
    p_hold: float | None
    p_sell: float | None
    feature_schema_version: str | None
    feature_dim: int | None
    features: list[float] | None
    model_version: str | None
    data_source: str
    freshness_sec: float | None
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread_bps: float | None
    predicted_impact_bps: float | None
    expected_slippage_bps: float | None
    expected_commission_bps: float | None
    slot_occupancy: int | None
    cash: float | None
    symbol_already_open: bool
    max_open: bool
    proposed_notional: float | None
    outcome_class: str
    eligibility_reason: str
    field_authority: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionGroup:
    decision_group_id: str
    ts_utc: str
    epoch: int
    bar_epoch: int
    selected_action: str
    selected_symbol: str
    path_ev_winner: str
    why_selected: str
    model_version: str | None
    candidates: list[CandidateRow]
    missing_fields: list[str] = field(default_factory=list)

    def coin_rows(self) -> list[CandidateRow]:
        return [c for c in self.candidates if c.symbol in COINS]

    def hold_row(self) -> CandidateRow | None:
        for c in self.candidates:
            if c.symbol == HOLD_SYMBOL:
                return c
        return None


def _ev_key(symbol: str) -> str:
    return f"{symbol[:-4].lower()}_path_ev" if symbol.endswith("USDT") else "hold_ev"


def _nearest_inference(
    by_symbol: dict[str, list[dict[str, Any]]],
    symbol: str,
    epoch: int,
) -> dict[str, Any] | None:
    rows = by_symbol.get(symbol) or []
    chosen = None
    for row in rows:
        if int(row["epoch"]) <= epoch:
            chosen = row
        else:
            break
    if chosen is None:
        return None
    if epoch - int(chosen["epoch"]) > 900:
        return None
    return chosen


def _load_inferences_by_symbol(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(ai_inference_log)")}
    if "ts_utc" not in present:
        return {s: [] for s in COINS}
    cols = ["symbol", "ts_utc", "prob_buy", "prob_hold", "prob_sell"]
    if "feature_version" in present:
        cols.append("feature_version")
    if "feature_dim" in present:
        cols.append("feature_dim")
    if "features_json" in present:
        cols.append("features_json")
    if "model_artifact" in present:
        cols.append("model_artifact")
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM ai_inference_log WHERE strategy_id='day' ORDER BY ts_utc ASC").fetchall()
    idx = {name: i for i, name in enumerate(cols)}
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in COINS}
    for row in rows:
        sym = _api(row[idx["symbol"]])
        if sym not in out:
            continue
        ep = parse_epoch(row[idx["ts_utc"]])
        if ep is None:
            continue
        feat = None
        dim = int(row[idx["feature_dim"]]) if "feature_dim" in idx and row[idx["feature_dim"]] else 145
        if "features_json" in idx:
            feat = parse_feature_vector(row[idx["features_json"]], expected_dim=dim)
        out[sym].append(
            {
                "symbol": sym,
                "epoch": ep,
                "p_buy": float(row[idx["prob_buy"]] or 0.0),
                "p_hold": float(row[idx["prob_hold"]] or 0.0),
                "p_sell": float(row[idx["prob_sell"]] or 0.0),
                "feature_version": str(row[idx["feature_version"]]) if "feature_version" in idx else None,
                "feature_dim": dim if feat is not None else None,
                "features": feat,
                "model_artifact": str(row[idx["model_artifact"]]) if "model_artifact" in idx else None,
            }
        )
    return out


def _load_fills(
    conn: sqlite3.Connection,
    *,
    mode: str | None = None,
    strategy_id: str = "day",
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "side" not in present:
        return []
    where = ["side='BUY'"]
    args: list[Any] = []
    if "strategy_id" in present and strategy_id:
        where.append("strategy_id=?")
        args.append(strategy_id)
    if mode and "mode" in present:
        where.append("mode=?")
        args.append(mode)
    if start and "timestamp" in present:
        where.append("timestamp>=?")
        args.append(start)
    if end and "timestamp" in present:
        where.append("timestamp<?")
        args.append(end)
    cols = ["symbol", "timestamp", "quantity", "price", "decision_id"]
    for extra in ("entry_bar_timestamp", "remaining_position", "order_id", "trade_id", "id", "status"):
        if extra in present:
            cols.append(extra)
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM paper_trades WHERE {' AND '.join(where)}", args).fetchall()
    idx = {name: i for i, name in enumerate(cols)}
    out = []
    for row in rows:
        ep = parse_epoch(row[idx["timestamp"]])
        if ep is None:
            continue
        qty = float(row[idx["quantity"]] or 0.0)
        rem = float(row[idx["remaining_position"]] or 0.0) if "remaining_position" in idx else qty
        # remaining==0 is a later close, not an incomplete fill.
        still_open_partial = rem > 1e-12 and rem + 1e-12 < qty
        status = str(row[idx["status"]] or "") if "status" in idx else ""
        out.append(
            {
                "symbol": _api(row[idx["symbol"]]),
                "epoch": ep,
                "quantity": qty,
                "price": float(row[idx["price"]] or 0.0),
                "decision_id": str(row[idx["decision_id"]] or ""),
                "order_id": str(row[idx["order_id"]] or "") if "order_id" in idx else "",
                "trade_id": str(row[idx["trade_id"]] or "") if "trade_id" in idx else "",
                "fill_id": str(row[idx["id"]] or "") if "id" in idx else "",
                "status": status,
                "remaining": rem,
                "partial": still_open_partial,
            }
        )
    sells = _load_sells(conn, mode=mode, strategy_id=strategy_id, start=start)
    for fill in out:
        fill["sell"] = _next_sell(sells, fill["symbol"], fill["epoch"])
    return out


def _load_sells(
    conn: sqlite3.Connection,
    *,
    mode: str | None = None,
    strategy_id: str = "day",
    start: str | None = None,
) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "side" not in present:
        return []
    where = ["side='SELL'"]
    args: list[Any] = []
    if "strategy_id" in present and strategy_id:
        where.append("strategy_id=?")
        args.append(strategy_id)
    if mode and "mode" in present:
        where.append("mode=?")
        args.append(mode)
    if start and "timestamp" in present:
        where.append("timestamp>=?")
        args.append(start)
    cols = ["symbol", "timestamp", "price", "pnl_pct_net", "pnl_pct", "exit_reason", "hold_time_seconds", "fees_paid", "pnl"]
    have = [c for c in cols if c in present]
    rows = conn.execute(
        f"SELECT {', '.join(have)} FROM paper_trades WHERE {' AND '.join(where)} ORDER BY timestamp ASC",
        args,
    ).fetchall()
    idx = {name: i for i, name in enumerate(have)}
    out = []
    for row in rows:
        ep = parse_epoch(row[idx["timestamp"]])
        if ep is None:
            continue
        raw_pnl = row[idx["pnl_pct_net"]] if "pnl_pct_net" in idx else None
        raw_pnl_legacy = row[idx["pnl_pct"]] if "pnl_pct" in idx else None
        raw_fees = row[idx["fees_paid"]] if "fees_paid" in idx else None
        pnl = float(raw_pnl) if raw_pnl not in (None, "") else (float(raw_pnl_legacy) if raw_pnl_legacy not in (None, "") else None)
        out.append(
            {
                "symbol": _api(row[idx["symbol"]]),
                "epoch": ep,
                "price": float(row[idx["price"]] or 0.0) if "price" in idx else 0.0,
                "pnl_pct_net": pnl,
                "exit_reason": str(row[idx["exit_reason"]] or "") if "exit_reason" in idx else "",
                "hold_sec": float(row[idx["hold_time_seconds"]] or 0.0) if "hold_time_seconds" in idx else 0.0,
                "fees_paid": float(raw_fees) if raw_fees not in (None, "") else None,
                "pnl": float(row[idx["pnl"]]) if "pnl" in idx and row[idx["pnl"]] not in (None, "") else None,
            }
        )
    return out


def _next_sell(sells: list[dict[str, Any]], symbol: str, buy_epoch: int) -> dict[str, Any] | None:
    for sell in sells:
        if sell["symbol"] == symbol and int(sell["epoch"]) >= buy_epoch:
            return sell
    return None


def _load_rejects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(day_decision_records)")}
    if "final_decision" not in present:
        return []
    has_id = "decision_id" in present
    select = "decision_id, created_at, symbol, final_decision, first_hard_block, detail_json" if has_id else "created_at, symbol, final_decision, first_hard_block, detail_json"
    rows = conn.execute(f"SELECT {select} FROM day_decision_records").fetchall()
    out = []
    for row in rows:
        if has_id:
            decision_id, created, symbol, final, block, detail = row
        else:
            decision_id, created, symbol, final, block, detail = "", *row
        ep = parse_epoch(created)
        if ep is None:
            continue
        payload = _loads(detail)
        bar = parse_epoch(payload.get("bar_timestamp")) if payload.get("bar_timestamp") is not None else None
        out.append(
            {
                "decision_id": str(decision_id or ""),
                "epoch": ep,
                "bar_epoch": bar,
                "symbol": _api(symbol),
                "final_decision": str(final or ""),
                "first_hard_block": str(block or ""),
            }
        )
    return out


def _load_engine_rejects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(portfolio_engine_rejects)")}
    if "reason" not in present:
        return []
    cols = ["ts", "symbol", "reason"]
    if "bar_timestamp" in present:
        cols.append("bar_timestamp")
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM portfolio_engine_rejects").fetchall()
    idx = {name: i for i, name in enumerate(cols)}
    out = []
    for row in rows:
        ep = parse_epoch(row[idx["ts"]])
        if ep is None:
            continue
        out.append(
            {
                "epoch": ep,
                "symbol": _api(row[idx["symbol"]]),
                "reason": str(row[idx["reason"]] or ""),
                "bar_epoch": parse_epoch(row[idx["bar_timestamp"]]) if "bar_timestamp" in idx else None,
            }
        )
    return out


def _open_set_timeline(conn: sqlite3.Connection) -> list[tuple[int, set[str]]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "side" not in present:
        return []
    where = "strategy_id='day'" if "strategy_id" in present else "1=1"
    rows = conn.execute(f"SELECT symbol, side, timestamp FROM paper_trades WHERE {where} ORDER BY timestamp ASC").fetchall()
    open_set: set[str] = set()
    timeline: list[tuple[int, set[str]]] = []
    for symbol, side, ts in rows:
        ep = parse_epoch(ts)
        if ep is None:
            continue
        sym = _api(symbol)
        if str(side).upper() == "BUY":
            open_set.add(sym)
        elif str(side).upper() == "SELL":
            open_set.discard(sym)
        timeline.append((ep, set(open_set)))
    return timeline


def _open_at(timeline: list[tuple[int, set[str]]], epoch: int) -> set[str]:
    current: set[str] = set()
    for ep, symbols in timeline:
        if ep > epoch:
            break
        current = symbols
    return current


def _match_fill(
    fills: list[dict[str, Any]],
    symbol: str,
    epoch: int,
    *,
    decision_id: str = "",
) -> dict[str, Any] | None:
    if decision_id:
        for fill in fills:
            if fill.get("decision_id") == decision_id:
                return fill
    best = None
    best_dt = FILL_JOIN_SEC + 1
    for fill in fills:
        if fill["symbol"] != symbol:
            continue
        dt = abs(int(fill["epoch"]) - epoch)
        if dt <= FILL_JOIN_SEC and dt < best_dt:
            best = fill
            best_dt = dt
    return best


def _match_record(rows: list[dict[str, Any]], symbol: str, epoch: int, bar_epoch: int) -> dict[str, Any] | None:
    best = None
    best_dt = FILL_JOIN_SEC + 1
    for row in rows:
        if row["symbol"] != symbol:
            continue
        if row.get("bar_epoch") == bar_epoch:
            return row
        dt = abs(int(row["epoch"]) - epoch)
        if dt <= FILL_JOIN_SEC and dt < best_dt:
            best = row
            best_dt = dt
    return best


def _classify_reject_reason(reason: str) -> str | None:
    text = str(reason or "").upper()
    if not text:
        return None
    if any(tok in text for tok in ("STALE", "INVALID", "MISSING_BAR", "NO_DATA")):
        return "stale_invalid"
    if any(tok in text for tok in ("SPREAD", "IMPACT", "SLIP")):
        return "spread_impact_ineligible"
    if any(tok in text for tok in ("ALREADY_IN_TRADE", "DUPLICATE", "POSITION_ALREADY", "SAME_SYMBOL")):
        return "symbol_open"
    if any(tok in text for tok in ("MAX_OPEN", "NO_SLOT", "CASH", "CAPITAL", "MIN_NOTIONAL", "DEPLOYED")):
        return "no_slot_or_capital"
    return None


def _tape_groups(
    conn: sqlite3.Connection,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(decision_book_tape)")}
    if "engine" not in present:
        return []
    rows = conn.execute(
        """
        SELECT ts_utc, symbol, selected_action, selection_reason, buy_ev, hold_ev,
               model_version, best_bid, best_ask, mid, spread_pct, book_source,
               book_age_sec, extras_json
        FROM decision_book_tape
        WHERE engine='day'
        ORDER BY ts_utc ASC, id ASC
        """
    ).fetchall()
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        grouped[str(row[0])].append(row)
    events = []
    for ts_utc, items in grouped.items():
        by_symbol: dict[str, tuple] = {}
        extras = {}
        for item in items:
            sym = _api(item[1]) if str(item[1]).upper() != HOLD_SYMBOL else HOLD_SYMBOL
            if sym in AUTHORITY_SYMBOLS and sym not in by_symbol:
                by_symbol[sym] = item
            extras = _loads(item[13]) or extras
        events.append({"ts_utc": ts_utc, "rows": by_symbol, "extras": extras})
    events.sort(key=lambda e: e["ts_utc"])
    if start:
        events = [e for e in events if str(e["ts_utc"]) >= start]
    if end:
        events = [e for e in events if str(e["ts_utc"]) < end]
    return events


def _selected_without_fill_class(*, reject_reason: str, mapped: str | None, order_id: str, status: str) -> tuple[str, str]:
    """Classify a ranking-selected coin that has no authoritative fill.

    A missing tape-to-fill join is not a fill failure.
    """
    st = str(status or "").upper()
    if order_id and st in {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED"}:
        return "terminal_fill_failure", f"submitted_order_{st.lower()}_zero_fill"
    if mapped in {"stale_invalid", "spread_impact_ineligible", "symbol_open", "no_slot_or_capital"}:
        return "blocked_after_ranking", reject_reason or mapped
    if reject_reason:
        return "blocked_after_ranking", reject_reason
    return "no_order_match", "selected_no_authoritative_fill_or_order"


def build_groups(
    conn: sqlite3.Connection,
    *,
    mode: str | None = None,
    strategy_id: str = "day",
    start: str | None = None,
    end: str | None = None,
) -> list[DecisionGroup]:
    """Recover grouped PIT decisions. Rejects are classified, never counted as fills."""
    tape = _tape_groups(conn, start=start, end=end)
    inferences = _load_inferences_by_symbol(conn)
    fills = _load_fills(conn, mode=mode, strategy_id=strategy_id, start=start, end=end)
    decisions = _load_rejects(conn)
    engine_rejects = _load_engine_rejects(conn)
    open_tl = _open_set_timeline(conn)
    groups: list[DecisionGroup] = []
    for event in tape:
        extras = event["extras"]
        ts_utc = str(event["ts_utc"])
        epoch = parse_epoch(ts_utc) or 0
        bar_epoch = (epoch // 900) * 900
        gid = group_id_for(ts_utc)
        winner = str(extras.get("path_ev_winner") or extras.get("selected_symbol") or HOLD_SYMBOL)
        selected_action = str(extras.get("selected_action") or "HOLD")
        selected_symbol = _api(str(extras.get("selected_symbol") or "")) if extras.get("selected_symbol") else ""
        if selected_action.upper().startswith("BUY_") and not selected_symbol:
            selected_symbol = _api(selected_action[4:])
        if selected_action.upper() == "HOLD":
            selected_symbol = ""
        why = str(extras.get("why_selected") or extras.get("selection_reason") or "")
        model_version = extras.get("path_net_model_id")
        open_now = _open_at(open_tl, epoch)
        missing: list[str] = []
        if len(event["rows"]) < 5:
            missing.append("incomplete_tape_group")
        candidates: list[CandidateRow] = []
        for symbol in AUTHORITY_SYMBOLS:
            raw = event["rows"].get(symbol)
            path_ev = 0.0
            if symbol == HOLD_SYMBOL:
                path_ev = 0.0
            elif extras.get(_ev_key(symbol)) not in (None, ""):
                path_ev = float(extras[_ev_key(symbol)])
            elif raw is not None and raw[4] not in (None, ""):
                path_ev = float(raw[4])
                missing.append(f"{symbol}_path_ev_row_fallback")
            else:
                missing.append(f"{symbol}_path_ev_missing")
            inf = _nearest_inference(inferences, symbol, epoch) if symbol in COINS else None
            bid = float(raw[7]) if raw is not None and raw[7] not in (None, "") else None
            ask = float(raw[8]) if raw is not None and raw[8] not in (None, "") else None
            mid = float(raw[9]) if raw is not None and raw[9] not in (None, "") else None
            spread_pct = float(raw[10]) if raw is not None and raw[10] not in (None, "") else None
            book_source = str(raw[11] or "missing") if raw is not None else "missing"
            book_age = float(raw[12]) if raw is not None and raw[12] not in (None, "") else None
            live_selected = bool(symbol == selected_symbol and selected_action.upper().startswith("BUY"))
            if symbol == HOLD_SYMBOL:
                live_selected = selected_action.upper() == "HOLD" or not selected_symbol
            rec = _match_record(decisions, symbol, epoch, bar_epoch) if symbol in COINS else None
            fill = _match_fill(fills, symbol, epoch, decision_id=str((rec or {}).get("decision_id") or "")) if symbol in COINS else None
            erej = _match_record(engine_rejects, symbol, epoch, bar_epoch) if symbol in COINS else None
            reject_reason = ""
            if rec and rec["final_decision"] == "reject":
                reject_reason = rec["first_hard_block"]
            if erej:
                reject_reason = reject_reason or str(erej.get("reason") or "")
            mapped = _classify_reject_reason(reject_reason)
            symbol_open = symbol in open_now
            max_open = len(open_now) >= 4
            outcome = "other"
            eligibility = "unknown"
            if symbol == HOLD_SYMBOL:
                outcome = "hold"
                eligibility = "zero_benchmark"
            elif book_source == "missing" and mid is None and (book_age is None or book_age > BOOK_STALE_SEC):
                outcome = "stale_invalid"
                eligibility = "missing_book_and_mid"
            elif mapped == "stale_invalid":
                outcome = "stale_invalid"
                eligibility = reject_reason or "stale_invalid"
            elif mapped == "spread_impact_ineligible":
                outcome = "spread_impact_ineligible"
                eligibility = reject_reason or "spread_impact"
            elif live_selected and fill and fill.get("partial"):
                outcome = "partial_fill"
                eligibility = "selected_partial_fill"
            elif live_selected and fill:
                outcome = "selected_execute"
                eligibility = "selected_and_filled"
            elif live_selected and not fill:
                outcome, eligibility = _selected_without_fill_class(
                    reject_reason=reject_reason,
                    mapped=mapped,
                    order_id=str((fill or {}).get("order_id") or ""),
                    status=str((fill or {}).get("status") or ""),
                )
            elif mapped == "symbol_open" or symbol_open:
                outcome = "symbol_open"
                eligibility = reject_reason or "symbol_already_open"
            elif mapped == "no_slot_or_capital" or (max_open and not live_selected):
                outcome = "no_slot_or_capital"
                eligibility = reject_reason or "max_open_or_capital"
            else:
                outcome = "ranking_loser"
                eligibility = "valid_unselected_alternative"
            if symbol in COINS and inf is None:
                missing.append(f"{symbol}_inference_asof")
            if symbol in COINS and (bid is None or ask is None):
                missing.append(f"{symbol}_bid_ask")
            field_authority = {
                "path_ev": "exact_tape_extras" if extras.get(_ev_key(symbol)) not in (None, "") or symbol == HOLD_SYMBOL else "missing",
                "features_145": "exact_inference_log" if inf and inf.get("features") else "missing",
                "p_buy": "exact_inference_log" if inf is not None else "missing",
                "book": "exact_tape" if bid is not None and ask is not None else "missing",
                "slot_cash": "reconstructed_from_paper_trades",
                "old_rank_deltas": "missing_for_unchosen",
            }
            candidates.append(
                CandidateRow(
                    decision_group_id=gid,
                    ts_utc=ts_utc,
                    epoch=epoch,
                    bar_epoch=bar_epoch,
                    symbol=symbol,
                    selected_action=selected_action if live_selected else "HOLD",
                    live_selected=live_selected,
                    path_ev=float(path_ev),
                    hold_ev=0.0,
                    p_buy=None if inf is None else float(inf["p_buy"]),
                    p_hold=None if inf is None else float(inf["p_hold"]),
                    p_sell=None if inf is None else float(inf["p_sell"]),
                    feature_schema_version=None if inf is None else inf.get("feature_version"),
                    feature_dim=None if inf is None else inf.get("feature_dim"),
                    features=None if inf is None else inf.get("features"),
                    model_version=str(model_version or (raw[6] if raw is not None else "") or "") or None,
                    data_source=book_source,
                    freshness_sec=book_age,
                    best_bid=bid,
                    best_ask=ask,
                    mid=mid,
                    spread_bps=None if spread_pct is None else float(spread_pct) * 1e4,
                    predicted_impact_bps=expected_slippage_rt_pct() * 1e4 / 2.0,
                    expected_slippage_bps=expected_slippage_rt_pct() * 1e4,
                    expected_commission_bps=expected_exchange_commission_rt_pct() * 1e4,
                    slot_occupancy=len(open_now),
                    cash=None,
                    symbol_already_open=symbol_open,
                    max_open=max_open,
                    proposed_notional=None,
                    outcome_class=outcome,
                    eligibility_reason=eligibility,
                    field_authority=field_authority,
                    extras={
                        "why_selected": why,
                        "path_ev_winner": winner,
                        "reject_reason": reject_reason,
                        "fill_decision_id": (fill or {}).get("decision_id"),
                        "fill_qty": (fill or {}).get("quantity"),
                        "fill_price": (fill or {}).get("price"),
                        "fill_id": (fill or {}).get("fill_id"),
                        "order_id": (fill or {}).get("order_id"),
                        "sell": (fill or {}).get("sell"),
                        "ranking_selected": live_selected,
                        "execute_decision": bool(rec and rec.get("final_decision") == "execute"),
                        "order_submitted": bool((fill or {}).get("order_id")),
                        "expected_spread_bps": expected_spread_pct(symbol) * 1e4 if symbol in COINS else 0.0,
                    },
                )
            )
        groups.append(
            DecisionGroup(
                decision_group_id=gid,
                ts_utc=ts_utc,
                epoch=epoch,
                bar_epoch=bar_epoch,
                selected_action=selected_action,
                selected_symbol=selected_symbol,
                path_ev_winner=winner,
                why_selected=why,
                model_version=str(model_version) if model_version else None,
                candidates=candidates,
                missing_fields=sorted(set(missing)),
            )
        )
    return groups


def ledger_counts(groups: list[DecisionGroup]) -> dict[str, Any]:
    classes: dict[str, int] = dict.fromkeys(OUTCOME_CLASSES, 0)
    missing_field_rows = 0
    executable = 0
    for group in groups:
        if group.missing_fields:
            missing_field_rows += 1
        for cand in group.candidates:
            classes[cand.outcome_class] = classes.get(cand.outcome_class, 0) + 1
            if cand.outcome_class in ("ranking_loser", "selected_execute", "partial_fill"):
                executable += 1
    return {
        "groups": len(groups),
        "candidates": sum(len(g.candidates) for g in groups),
        "executable_candidates": executable,
        "outcome_classes": classes,
        "groups_with_missing_fields": missing_field_rows,
    }
