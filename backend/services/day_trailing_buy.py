"""DAY trailing-buy executor. Timing only. Reuses production BUY submission."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.config.day_entry_execution import (
    BOOK_STALE_SEC,
    ENTRY_AUTHORITY_TRAILING_BUY,
    trailing_buy_max_wait_seconds,
    trailing_buy_mode_status,
)
from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_trailing_buy_store import (
    CANCELED,
    EXPIRED,
    FAILED,
    FILLED,
    SUBMITTING,
    TRAIL_LOW,
    WAIT_DIP,
    claim_submitting,
    create_intent,
    load_active_intents,
    load_intent,
    load_intent_by_symbol,
    mark_order_accepted,
    mark_terminal,
    release_submitting_for_retry,
    update_watch,
)
from backend.services.spread_book_telemetry import read_market_book

logger = logging.getLogger(__name__)

ENTRY_AUTHORITY = ENTRY_AUTHORITY_TRAILING_BUY
DAY_TRADE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _api(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def honest_round_trip_cost_bps(symbol: str) -> float:
    return float(honest_all_in_rt_pct(symbol)) * 10000.0


def required_improvement_bps(symbol: str) -> float:
    return max(10.0, honest_round_trip_cost_bps(symbol) + 3.0)


def rebound_bps_from_spread(arm_spread_bps: float) -> float:
    return max(4.0, 2.0 * float(arm_spread_bps or 0.0))


def min_dip_bps(symbol: str, arm_spread_bps: float) -> float:
    return required_improvement_bps(symbol) + rebound_bps_from_spread(arm_spread_bps)


def dip_trigger_ask(arm_ask: float, min_dip: float) -> float:
    return float(arm_ask) * (1.0 - float(min_dip) / 10000.0)


def rebound_trigger_ask(lowest_ask: float, rebound: float) -> float:
    return float(lowest_ask) * (1.0 + float(rebound) / 10000.0)


def retained_improvement_ceiling_ask(arm_ask: float, improvement: float) -> float:
    return float(arm_ask) * (1.0 - float(improvement) / 10000.0)


def dip_achieved_bps(arm_ask: float, current_ask: float) -> float:
    if arm_ask <= 0 or current_ask <= 0:
        return 0.0
    return max(0.0, (float(arm_ask) - float(current_ask)) / float(arm_ask) * 10000.0)


def rebound_achieved_bps(lowest_ask: float, current_ask: float) -> float:
    if lowest_ask <= 0 or current_ask <= 0:
        return 0.0
    return max(0.0, (float(current_ask) - float(lowest_ask)) / float(lowest_ask) * 10000.0)


def fill_improvement_vs_arm_bps(arm_ask: float, fill_ask: float) -> float:
    if arm_ask <= 0 or fill_ask <= 0:
        return 0.0
    return (float(arm_ask) - float(fill_ask)) / float(arm_ask) * 10000.0


@dataclass(frozen=True)
class ObserveDecision:
    action: str
    status: str
    reason: str = ""
    lowest_ask: float = 0.0
    lowest_ask_ts: float = 0.0
    current_ask: float = 0.0


def observe_book(
    intent: dict[str, Any],
    *,
    ask: float,
    now: float,
    book_fresh: bool,
    thesis_invalid: bool = False,
    validity_reason: str = "",
) -> ObserveDecision:
    """Pure state machine. Never submits. Never chases a lost improvement."""
    status = str(intent.get("status") or "")
    arm_ask = float(intent.get("arm_ask") or 0.0)
    min_dip = float(intent.get("min_dip_bps") or 0.0)
    rebound = float(intent.get("rebound_bps") or 0.0)
    improvement = float(intent.get("required_improvement_bps") or 0.0)
    lowest = float(intent.get("lowest_ask") or 0.0)
    lowest_ts = float(intent.get("lowest_ask_ts") or 0.0)
    expires_at = float(intent.get("expires_at") or 0.0)
    px = float(ask or 0.0)
    if status not in {WAIT_DIP, TRAIL_LOW}:
        return ObserveDecision("hold", status, current_ask=px)
    if now >= expires_at > 0:
        return ObserveDecision("expire", EXPIRED, "TIMEOUT", current_ask=px)
    if validity_reason:
        return ObserveDecision("cancel", CANCELED, str(validity_reason), current_ask=px)
    if thesis_invalid:
        return ObserveDecision("cancel", CANCELED, "THESIS_4H_INVALID", current_ask=px)
    if not book_fresh or px <= 0:
        return ObserveDecision("cancel", CANCELED, "STALE_MARKET_BOOK", current_ask=px)
    if status == WAIT_DIP:
        if px <= dip_trigger_ask(arm_ask, min_dip):
            return ObserveDecision("trail", TRAIL_LOW, "MIN_DIP_REACHED", lowest_ask=px, lowest_ask_ts=now, current_ask=px)
        return ObserveDecision("watch", WAIT_DIP, current_ask=px)
    if px < lowest - 1e-12 or lowest <= 0:
        return ObserveDecision("new_low", TRAIL_LOW, "NEW_LOW", lowest_ask=px, lowest_ask_ts=now, current_ask=px)
    ceiling = retained_improvement_ceiling_ask(arm_ask, improvement)
    if px >= rebound_trigger_ask(lowest, rebound):
        if px <= ceiling + 1e-12:
            return ObserveDecision("submit", TRAIL_LOW, "REBOUND_CONFIRMED", lowest_ask=lowest, lowest_ask_ts=lowest_ts, current_ask=px)
        if px >= arm_ask - 1e-12:
            return ObserveDecision("expire", EXPIRED, "IMPROVEMENT_LOST", lowest_ask=lowest, lowest_ask_ts=lowest_ts, current_ask=px)
        return ObserveDecision("watch", TRAIL_LOW, "REBOUND_ABOVE_IMPROVEMENT", lowest_ask=lowest, lowest_ask_ts=lowest_ts, current_ask=px)
    return ObserveDecision("watch", TRAIL_LOW, lowest_ask=lowest, lowest_ask_ts=lowest_ts, current_ask=px)


def formulas_for_symbol(symbol: str, *, arm_spread_bps: float) -> dict[str, float]:
    cost = honest_round_trip_cost_bps(symbol)
    improvement = required_improvement_bps(symbol)
    rebound = rebound_bps_from_spread(arm_spread_bps)
    dip = improvement + rebound
    return {
        "round_trip_cost_bps": cost,
        "required_improvement_bps": improvement,
        "rebound_bps": rebound,
        "min_dip_bps": dip,
        "arm_spread_bps": float(arm_spread_bps),
    }


def operator_row(intent: dict[str, Any], *, current_ask: float | None = None) -> dict[str, Any]:
    ask = float(current_ask if current_ask is not None else (intent.get("current_ask") or 0.0))
    arm_ask = float(intent.get("arm_ask") or 0.0)
    lowest = float(intent.get("lowest_ask") or 0.0)
    return {
        "intent_id": intent.get("intent_id"),
        "decision_id": intent.get("decision_id"),
        "symbol": intent.get("symbol"),
        "state": intent.get("status"),
        "arm_ask": arm_ask,
        "current_ask": ask,
        "lowest_ask": lowest,
        "dip_achieved_bps": round(dip_achieved_bps(arm_ask, ask), 4),
        "min_dip_bps": float(intent.get("min_dip_bps") or 0.0),
        "rebound_achieved_bps": round(rebound_achieved_bps(lowest, ask), 4) if lowest > 0 else 0.0,
        "rebound_bps": float(intent.get("rebound_bps") or 0.0),
        "required_improvement_bps": float(intent.get("required_improvement_bps") or 0.0),
        "retained_improvement_bps": round(dip_achieved_bps(arm_ask, ask), 4),
        "arm_ts": intent.get("arm_ts"),
        "expires_at": intent.get("expires_at"),
        "cancel_reason": intent.get("cancel_reason") or "",
        "order_id": intent.get("order_id") or "",
        "client_order_id": intent.get("client_order_id") or "",
        "fill_id": intent.get("fill_id") or "",
        "trade_id": intent.get("trade_id") or "",
    }


def select_ranked_arm_stream(
    ranked_candidates: list[Any] | None,
    stream_candidates: list[Any] | None,
) -> list[Any]:
    """Keep existing rank order, then fill any missing top-4 stream symbols."""
    seen: set[str] = set()
    out: list[Any] = []
    for cand in list(ranked_candidates or []):
        api = _api(getattr(cand, "symbol", ""))
        if api not in DAY_TRADE_SYMBOLS or api in seen:
            continue
        seen.add(api)
        out.append(cand)
    extras: list[Any] = []
    for cand in list(stream_candidates or []):
        api = _api(getattr(cand, "symbol", ""))
        if api not in DAY_TRADE_SYMBOLS or api in seen:
            continue
        seen.add(api)
        extras.append(cand)

    def _extra_key(cand: Any) -> tuple[float, float, str]:
        dd = getattr(cand, "decision_data", None) or {}
        try:
            score = float(dd.get("final_selection_score") or dd.get("selection_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            conf = float(getattr(cand, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        return (-score, -conf, str(getattr(cand, "symbol", "") or ""))

    extras.sort(key=_extra_key)
    return out + extras


def available_economic_slots(*, held: int, pending_orders: int, max_positions: int) -> int:
    return max(0, int(max_positions) - int(held) - int(pending_orders))


def remaining_watch_notional_cap(*, free_cash: float, remaining_new_slots: int) -> float:
    """Split leftover cash across remaining new watches so all four can arm."""
    if remaining_new_slots <= 0:
        return 0.0
    return max(0.0, float(free_cash) / float(remaining_new_slots))


def sync_book_redis(redis_client: Any = None) -> Any:
    """Always use the sync Redis client. Integration's client is async."""
    from backend.config.redis_config import get_redis_client

    _ = redis_client
    return get_redis_client()


def fresh_executable_book(redis_client: Any, symbol: str) -> dict[str, Any] | None:
    book = read_market_book(redis_client, symbol)
    if not book:
        return None
    if not bool(book.get("fresh")):
        return None
    if float(book.get("freshness_sec") or 0.0) > BOOK_STALE_SEC:
        return None
    if float(book.get("ask") or 0.0) <= 0 or float(book.get("bid") or 0.0) <= 0:
        return None
    return book


def _thesis_invalid(intent: dict[str, Any], ask: float) -> bool:
    level = float(intent.get("thesis_invalid_level") or 0.0)
    return bool(level > 0 and ask > 0 and ask <= level)


async def arm_selected_candidate(
    engine: Any,
    *,
    symbol: str,
    quantity: float,
    stop_price: float,
    atr: float,
    confidence: float,
    bar_timestamp: int,
    explainability: Any,
    decision_id: str,
    sleeve: str,
    decision_data: dict[str, Any],
    redis_client: Any,
) -> dict[str, Any] | None:
    ok, err, mode = trailing_buy_mode_status()
    engine.day_entry_execution_error = "" if ok else err
    engine.day_entry_execution_mode = mode
    if not ok:
        logger.error("DAY_ENTRY_EXECUTION_FAIL_CLOSED %s — HOLD/NO_NEW_ENTRY", err)
        return None
    existing = load_intent_by_symbol(engine.db_path, symbol)
    if existing and str(existing.get("status") or "") in {WAIT_DIP, TRAIL_LOW, SUBMITTING}:
        logger.info(
            "TRAILING_BUY_PRESERVED symbol=%s intent=%s status=%s arm_ask=%.8f lowest_ask=%.8f",
            symbol,
            existing.get("intent_id"),
            existing.get("status"),
            float(existing.get("arm_ask") or 0.0),
            float(existing.get("lowest_ask") or 0.0),
        )
        return {"trailing_buy_armed": True, "intent": existing, "idempotent": True, "preserved": True}
    book = fresh_executable_book(redis_client, symbol)
    if not book:
        logger.info("TRAILING_BUY_ARM_BLOCKED %s STALE_OR_MISSING_BOOK", symbol)
        return None
    arm_ask = float(book["ask"])
    arm_bid = float(book["bid"])
    arm_mid = float(book["midpoint"])
    live_spread = float(book.get("spread_bps") or 0.0)
    formulas = formulas_for_symbol(symbol, arm_spread_bps=live_spread)
    discovery: dict[str, Any] = {}
    try:
        from backend.services.day_path_net import load_recent_bars
        from backend.services.day_setup_discovery import classify_setup, may_arm_setup, structured_min_dip_bps

        raw_bars = load_recent_bars(str(getattr(engine, "db_path", "") or ""), symbol)
        bars = []
        for row in raw_bars:
            ts = row.get("ts")
            epoch = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(float(ts or 0) or 0)
            bars.append((epoch, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume") or 0.0)))
        discovery = classify_setup(bars, symbol=symbol, ts=int(time.time()), atr=float(atr or 0.0), ask=arm_ask)
        if not may_arm_setup(discovery):
            logger.info(
                "TRAILING_BUY_ARM_BLOCKED %s setup_class=%s reason=%s",
                symbol,
                discovery.get("setup_class"),
                discovery.get("reason"),
            )
            return None
        if str(discovery.get("setup_class") or "") == "STRUCTURED_PULLBACK_RECLAIM":
            formulas["min_dip_bps"] = structured_min_dip_bps(
                symbol,
                float(atr or 0.0),
                arm_ask,
                float(formulas["rebound_bps"]),
            )
    except Exception:
        logger.exception("TRAILING_BUY_SETUP_DISCOVERY_FAILED symbol=%s", symbol)
    notional = float(quantity) * arm_ask
    reserved, reserve_reason = engine._try_reserve_entry(
        symbol,
        notional,
        decision_id=str(decision_id or ""),
        sleeve=str(sleeve or ""),
        ttl_sec=float(trailing_buy_max_wait_seconds()) + 60.0,
    )
    if not reserved:
        logger.info("TRAILING_BUY_ARM_BLOCKED %s reservation=%s", symbol, reserve_reason)
        return None
    now = time.time()
    exp = explainability.to_dict() if hasattr(explainability, "to_dict") else {}
    created_ok, create_reason, intent = create_intent(
        engine.db_path,
        fields={
            "decision_id": decision_id,
            "inference_id": str(decision_data.get("ai_inference_log_id") or decision_data.get("inference_log_id") or ""),
            "symbol": symbol,
            "setup": str(decision_data.get("setup_type") or decision_data.get("entry_thesis") or getattr(explainability, "setup_type", "") or ""),
            "arm_ts": now,
            "arm_bid": arm_bid,
            "arm_ask": arm_ask,
            "arm_midpoint": arm_mid,
            "decision_score": float(decision_data.get("final_selection_score") or decision_data.get("rank_score") or 0.0),
            "predicted_ev": float(decision_data.get("selected_net_expected_value") or decision_data.get("candidate_net_ev") or 0.0),
            **formulas,
            "expires_at": now + float(trailing_buy_max_wait_seconds()),
            "quantity": float(quantity),
            "stop_price": float(stop_price or 0.0),
            "atr": float(atr or 0.0),
            "confidence": float(confidence or 0.0),
            "bar_timestamp": int(bar_timestamp or 0),
            "sleeve": str(sleeve or ""),
            "notional_usd": notional,
            "thesis_invalid_level": float(decision_data.get("thesis_invalid_level") or getattr(explainability, "thesis_invalid_level", 0.0) or 0.0),
            "reservation_id": str((engine._entry_reservations.get(symbol) or {}).get("reservation_id") or ""),
            "payload": {
                "explainability": exp,
                "decision_data": dict(decision_data or {}),
                "entry_authority": ENTRY_AUTHORITY,
                "setup_discovery": discovery,
            },
        },
    )
    if not created_ok or not intent:
        engine._release_entry_reservation(symbol, decision_id=str(decision_id or ""), reason=create_reason)
        logger.info("TRAILING_BUY_ARM_BLOCKED %s create=%s", symbol, create_reason)
        return None
    if create_reason == "IDEMPOTENT_EXISTING":
        return {"trailing_buy_armed": True, "intent": intent, "idempotent": True}
    logger.info(
        "TRAILING_BUY_ARMED symbol=%s intent=%s decision=%s arm_ask=%.8f min_dip=%.4f rebound=%.4f improvement=%.4f expires=%.0f",
        symbol,
        intent.get("intent_id"),
        decision_id,
        arm_ask,
        formulas["min_dip_bps"],
        formulas["rebound_bps"],
        formulas["required_improvement_bps"],
        float(intent.get("expires_at") or 0.0),
    )
    return {"trailing_buy_armed": True, "intent": intent, "idempotent": False}


def _rebuild_explainability(payload: dict[str, Any], symbol: str) -> Any:
    from backend.services.portfolio_engine import TradeExplainability

    raw = dict((payload or {}).get("explainability") or {})
    exp = TradeExplainability(
        trade_id=str(raw.get("trade_id") or ""),
        symbol=str(raw.get("symbol") or symbol),
        side="BUY",
        timestamp=str(raw.get("timestamp") or ""),
    )
    for key, value in raw.items():
        if hasattr(exp, key):
            with_context = True
            try:
                setattr(exp, key, value)
            except Exception:
                with_context = False
            if not with_context:
                continue
    provenance = dict(getattr(exp, "entry_provenance", None) or raw.get("entry_provenance") or {})
    provenance["entry_authority"] = ENTRY_AUTHORITY
    exp.entry_provenance = provenance
    return exp


async def _pre_submit_safety(engine: Any, intent: dict[str, Any], ask: float) -> tuple[bool, str]:
    symbol = str(intent.get("symbol") or "")
    if _api(symbol) not in DAY_TRADE_SYMBOLS:
        return False, "SYMBOL_NOT_EXECUTABLE"
    allowed, why = engine._check_kill_switch_buy()
    if not allowed:
        return False, str(why or "KILL_OR_PAUSE")
    if getattr(engine, "_trading_paused", False):
        return False, f"TRADING_PAUSED:{getattr(engine, '_pause_reason', '')}"
    can_open, open_why = await engine._can_open_position(
        symbol,
        float(intent.get("notional_usd") or 0.0),
        decision_id=str(intent.get("decision_id") or ""),
    )
    if not can_open:
        return False, str(open_why or "CANNOT_OPEN")
    ns = symbol
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        ns = CanonicalSymbolFormatter.to_canonical(symbol)
    except Exception:
        ns = symbol
    if ns in getattr(engine, "open_positions", {}) or symbol in getattr(engine, "open_positions", {}):
        return False, "POSITION_ALREADY_OPEN"
    pending = set(engine._pending_buy_order_symbols())
    if ns in pending or symbol in pending:
        return False, "PENDING_BUY_EXISTS"
    if _thesis_invalid(intent, ask):
        return False, "THESIS_4H_INVALID"
    return True, ""


async def _submit_claimed(engine: Any, intent: dict[str, Any], ask: float) -> dict[str, Any] | None:
    symbol = str(intent.get("symbol") or "")
    payload = dict(intent.get("payload") or {})
    explainability = _rebuild_explainability(payload, symbol)
    safe, reason = await _pre_submit_safety(engine, intent, ask)
    if not safe:
        mark_terminal(engine.db_path, str(intent["intent_id"]), CANCELED, reason=reason, current_ask=ask)
        engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason=reason)
        logger.info("TRAILING_BUY_SUBMIT_BLOCKED %s %s", symbol, reason)
        return None
    result = await engine.execute_buy_fifo(
        symbol=symbol,
        quantity=float(intent.get("quantity") or 0.0),
        price=float(ask),
        stop_price=float(intent.get("stop_price") or 0.0),
        atr=float(intent.get("atr") or 0.0),
        confidence=float(intent.get("confidence") or 0.0),
        bar_timestamp=int(intent.get("bar_timestamp") or 0),
        explainability=explainability,
        decision_id=str(intent.get("decision_id") or ""),
        sleeve=str(intent.get("sleeve") or ""),
        entry_authority=ENTRY_AUTHORITY,
        client_order_id=str(intent.get("client_order_id") or ""),
        trailing_buy_intent_id=str(intent.get("intent_id") or ""),
    )
    if result:
        mark_order_accepted(
            engine.db_path,
            str(intent["intent_id"]),
            order_id=str(result.get("order_id") or result.get("exchange_order_id") or ""),
            fill_id=str(result.get("fill_id") or ""),
            trade_id=str(result.get("trade_id") or ""),
        )
        improvement = fill_improvement_vs_arm_bps(float(intent.get("arm_ask") or 0.0), float(result.get("price") or ask))
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FILLED,
            reason=ENTRY_AUTHORITY,
            order_id=str(result.get("order_id") or result.get("exchange_order_id") or ""),
            fill_id=str(result.get("fill_id") or ""),
            trade_id=str(result.get("trade_id") or ""),
            order_accepted=True,
            current_ask=float(result.get("price") or ask),
        )
        result = dict(result)
        result["entry_authority"] = ENTRY_AUTHORITY
        result["trailing_buy_intent_id"] = intent.get("intent_id")
        result["entry_improvement_bps"] = improvement
        logger.info(
            "TRAILING_BUY_FILLED symbol=%s intent=%s fill=%.8f arm_ask=%.8f improvement_bps=%.4f",
            symbol,
            intent.get("intent_id"),
            float(result.get("price") or ask),
            float(intent.get("arm_ask") or 0.0),
            improvement,
        )
        return result
    # execute_buy_fifo returned None: no fill. Retry only if no order was accepted.
    if release_submitting_for_retry(engine.db_path, str(intent["intent_id"])):
        logger.info("TRAILING_BUY_SUBMIT_RETRYABLE %s no order accepted", symbol)
    else:
        mark_terminal(engine.db_path, str(intent["intent_id"]), FAILED, reason="SUBMIT_UNCONFIRMED")
        engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason="SUBMIT_UNCONFIRMED")
    return None


async def cycle_trailing_buy_intents(engine: Any, redis_client: Any) -> dict[str, Any]:
    ok, err, mode = trailing_buy_mode_status()
    engine.day_entry_execution_error = "" if ok else err
    engine.day_entry_execution_mode = mode
    summary = {"mode_ok": ok, "error": err, "cycled": 0, "submitted": 0, "filled": 0, "closed": 0}
    if not ok:
        return summary
    books = sync_book_redis(redis_client)
    for intent in load_active_intents(engine.db_path):
        summary["cycled"] += 1
        symbol = str(intent.get("symbol") or "")
        if str(intent.get("status") or "") == SUBMITTING:
            await recover_submitting_intent(engine, intent)
            continue
        book = read_market_book(books, symbol)
        ask = float((book or {}).get("ask") or 0.0)
        fresh = bool(book and book.get("fresh") and float(book.get("freshness_sec") or 0.0) <= BOOK_STALE_SEC)
        validity = ""
        try:
            from backend.services.day_path_net import load_recent_bars
            from backend.services.day_setup_discovery import live_intent_validity

            raw_bars = load_recent_bars(str(getattr(engine, "db_path", "") or ""), symbol)
            bars = []
            for row in raw_bars:
                ts = row.get("ts")
                epoch = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(float(ts or 0) or 0)
                bars.append(
                    (
                        epoch,
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row.get("volume") or 0.0),
                    )
                )
            validity = live_intent_validity(intent, ask=ask, now=time.time(), bars=bars)
        except Exception:
            logger.exception("TRAILING_BUY_VALIDITY_FAILED symbol=%s", symbol)
        decision = observe_book(
            intent,
            ask=ask,
            now=time.time(),
            book_fresh=fresh,
            thesis_invalid=_thesis_invalid(intent, ask),
            validity_reason=validity,
        )
        if decision.action in {"expire", "cancel"}:
            mark_terminal(
                engine.db_path,
                str(intent["intent_id"]),
                decision.status,
                reason=decision.reason,
                current_ask=decision.current_ask,
            )
            engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason=decision.reason)
            summary["closed"] += 1
            continue
        update_watch(
            engine.db_path,
            str(intent["intent_id"]),
            status=decision.status,
            lowest_ask=decision.lowest_ask or None,
            lowest_ask_ts=decision.lowest_ask_ts or None,
            current_ask=decision.current_ask,
        )
        if decision.action != "submit":
            continue
        claimed, claimed_intent = claim_submitting(engine.db_path, str(intent["intent_id"]))
        if not claimed or not claimed_intent:
            continue
        summary["submitted"] += 1
        filled = await _submit_claimed(engine, claimed_intent, decision.current_ask)
        if filled:
            summary["filled"] += 1
    return summary


async def recover_submitting_intent(engine: Any, intent: dict[str, Any]) -> None:
    """Adopt an existing order/fill. Never guess a resubmit."""
    symbol = str(intent.get("symbol") or "")
    decision_id = str(intent.get("decision_id") or "")
    cid = str(intent.get("client_order_id") or "")
    local = _local_fill(engine, symbol=symbol, decision_id=decision_id, client_order_id=cid)
    if local:
        mark_order_accepted(
            engine.db_path,
            str(intent["intent_id"]),
            order_id=str(local.get("order_id") or ""),
            fill_id=str(local.get("fill_id") or ""),
            trade_id=str(local.get("trade_id") or ""),
        )
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FILLED,
            reason="RECOVERED_LOCAL_FILL",
            order_id=str(local.get("order_id") or ""),
            fill_id=str(local.get("fill_id") or ""),
            trade_id=str(local.get("trade_id") or ""),
            order_accepted=True,
        )
        logger.info("TRAILING_BUY_RECOVERED_FILL intent=%s trade=%s", intent.get("intent_id"), local.get("trade_id"))
        return
    exchange = await _exchange_order(engine, client_order_id=cid, symbol=symbol)
    if exchange and str(exchange.get("state") or "") == "open":
        mark_order_accepted(engine.db_path, str(intent["intent_id"]), order_id=str(exchange.get("order_id") or ""))
        logger.info("TRAILING_BUY_RECOVERED_OPEN_ORDER intent=%s order=%s", intent.get("intent_id"), exchange.get("order_id"))
        return
    if exchange and str(exchange.get("state") or "") == "filled":
        mark_order_accepted(
            engine.db_path,
            str(intent["intent_id"]),
            order_id=str(exchange.get("order_id") or ""),
            fill_id=str(exchange.get("fill_id") or ""),
            trade_id=str(exchange.get("trade_id") or ""),
        )
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FILLED,
            reason="RECOVERED_EXCHANGE_FILL",
            order_id=str(exchange.get("order_id") or ""),
            fill_id=str(exchange.get("fill_id") or ""),
            trade_id=str(exchange.get("trade_id") or ""),
            order_accepted=True,
        )
        return
    if bool(intent.get("order_accepted")):
        logger.warning("TRAILING_BUY_RECOVER_HOLD intent=%s accepted but fill unseen", intent.get("intent_id"))
        return
    if exchange is None and not bool(intent.get("order_accepted")):
        if release_submitting_for_retry(engine.db_path, str(intent["intent_id"])):
            logger.info("TRAILING_BUY_RECOVER_RETRY intent=%s proven_no_order", intent.get("intent_id"))
            return
    mark_terminal(engine.db_path, str(intent["intent_id"]), FAILED, reason="RECOVER_UNCONFIRMED")
    engine._release_entry_reservation(symbol, decision_id=decision_id, reason="RECOVER_UNCONFIRMED")


def _local_fill(engine: Any, *, symbol: str, decision_id: str, client_order_id: str) -> dict[str, Any] | None:
    import sqlite3

    conn = sqlite3.connect(str(engine.db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = None
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "paper_trades" not in tables:
            return None
        if decision_id:
            row = conn.execute(
                """
                SELECT trade_id, symbol, price FROM paper_trades
                WHERE decision_id=? AND UPPER(side)='BUY'
                ORDER BY rowid DESC LIMIT 1
                """,
                (decision_id,),
            ).fetchone()
        if row is None and client_order_id:
            try:
                row = conn.execute(
                    """
                    SELECT trade_id, symbol, price FROM paper_trades
                    WHERE explainability_json LIKE ? AND UPPER(side)='BUY'
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (f"%{client_order_id}%",),
                ).fetchone()
            except sqlite3.Error:
                row = None
        if row is None:
            return None
        return {"trade_id": row["trade_id"], "symbol": row["symbol"], "price": row["price"]}
    finally:
        conn.close()


async def _exchange_order(engine: Any, *, client_order_id: str, symbol: str) -> dict[str, Any] | None:
    if not client_order_id:
        return None
    live = getattr(engine, "_live_service", None)
    if live is None:
        return None
    fetch = getattr(live, "fetch_order", None) or getattr(live, "get_order", None)
    if fetch is None:
        return {"state": "unknown"}
    try:
        raw = await fetch(client_order_id=client_order_id, symbol=symbol)
    except TypeError:
        try:
            raw = await fetch(client_order_id)
        except Exception:
            return {"state": "unknown"}
    except Exception:
        return None
    if not raw:
        return None
    status = str(raw.get("status") or raw.get("state") or "").lower()
    if status in {"filled", "closed"}:
        return {"state": "filled", "order_id": str(raw.get("id") or raw.get("order_id") or ""), "fill_id": str(raw.get("fill_id") or ""), "trade_id": str(raw.get("trade_id") or "")}
    if status in {"open", "new", "partially_filled", "partial"}:
        return {"state": "open", "order_id": str(raw.get("id") or raw.get("order_id") or "")}
    if status in {"canceled", "cancelled", "expired", "rejected"}:
        return None
    return {"state": "unknown", "order_id": str(raw.get("id") or "")}


async def recover_trailing_buy_intents(engine: Any) -> int:
    rows = load_active_intents(engine.db_path)
    n = 0
    for intent in rows:
        if str(intent.get("status") or "") == SUBMITTING:
            await recover_submitting_intent(engine, intent)
            n += 1
    if rows:
        logger.info("TRAILING_BUY_RECOVERED active=%s submitting_checked=%s", len(rows), n)
    return len(load_active_intents(engine.db_path))


def cancel_legacy_open_buy_orders(engine: Any, *, reason: str = "LEGACY_IMMEDIATE_BUY_RETIRED") -> list[dict[str, Any]]:
    """Cancel unfilled automated DAY BUY orders only. Never cancel SELL."""
    canceled: list[dict[str, Any]] = []
    pending = getattr(engine, "_pending_orders", None) or {}
    for order in list(pending.values()):
        if str(getattr(order, "side", "") or "").upper() != "BUY":
            continue
        canceled.append(
            {
                "order_id": str(getattr(order, "order_id", "") or ""),
                "symbol": str(getattr(order, "symbol", "") or ""),
                "reason": reason,
            }
        )
    return canceled


def load_operator_intents(db_path: str, redis_client: Any = None) -> list[dict[str, Any]]:
    rows = []
    books = sync_book_redis(redis_client)
    for intent in load_active_intents(db_path):
        ask = None
        book = read_market_book(books, str(intent.get("symbol") or ""))
        if book:
            ask = float(book.get("ask") or 0.0)
        rows.append(operator_row(intent, current_ask=ask))
    return rows


def get_intent(db_path: str, intent_id: str) -> dict[str, Any] | None:
    return load_intent(db_path, intent_id)


__all__ = [
    "DAY_TRADE_SYMBOLS",
    "ENTRY_AUTHORITY",
    "ObserveDecision",
    "arm_selected_candidate",
    "available_economic_slots",
    "cancel_legacy_open_buy_orders",
    "cycle_trailing_buy_intents",
    "dip_achieved_bps",
    "dip_trigger_ask",
    "fill_improvement_vs_arm_bps",
    "formulas_for_symbol",
    "fresh_executable_book",
    "get_intent",
    "honest_round_trip_cost_bps",
    "load_operator_intents",
    "min_dip_bps",
    "observe_book",
    "operator_row",
    "rebound_achieved_bps",
    "rebound_bps_from_spread",
    "rebound_trigger_ask",
    "recover_submitting_intent",
    "recover_trailing_buy_intents",
    "remaining_watch_notional_cap",
    "required_improvement_bps",
    "retained_improvement_ceiling_ask",
    "select_ranked_arm_stream",
    "sync_book_redis",
]
