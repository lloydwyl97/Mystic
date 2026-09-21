"""DAY trailing-buy executor. Timing only. Reuses production BUY submission."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.config.day_entry_execution import (
    BOOK_STALE_SEC,
    ENTRY_AUTHORITY_TRAILING_BUY,
    trailing_buy_max_wait_seconds,
    trailing_buy_mode_active,
    trailing_buy_mode_status,
)
from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_entry_spendable import is_terminal_buy_cash_reason, money
from backend.services.day_path_input_validity import parse_bar_ts
from backend.services.day_trailing_buy_store import (
    CANCELED,
    EXPIRED,
    FAILED,
    FILLED,
    IN_FLIGHT_STATES,
    ORDER_OPEN,
    PARTIALLY_FILLED,
    SUBMITTING,
    TRAIL_LOW,
    WAIT_DIP,
    claim_submitting,
    create_intent,
    load_active_intents,
    load_intent,
    load_intent_by_symbol,
    mark_in_flight,
    mark_order_accepted,
    mark_terminal,
    release_submitting_for_retry,
    update_watch,
)
from backend.services.spread_book_telemetry import read_market_book

logger = logging.getLogger(__name__)


def _bar_epoch(ts: Any) -> int:
    parsed = parse_bar_ts(ts)
    return int(parsed.timestamp()) if parsed is not None else 0


ENTRY_AUTHORITY = ENTRY_AUTHORITY_TRAILING_BUY
DAY_TRADE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
ORDER_ACCEPTED = "ORDER_ACCEPTED"
FILL_ADOPTED = "FILL_ADOPTED"
HARD_SAFETY_REJECTED = "HARD_SAFETY_REJECTED"
EXCHANGE_REJECTED = "EXCHANGE_REJECTED"
TRANSIENT_RETRY = "TRANSIENT_RETRY"
AMBIGUOUS_RECONCILE = "AMBIGUOUS_RECONCILE"
BELOW_MIN_NOTIONAL = "BELOW_MIN_NOTIONAL"
INTENT_EXPIRED = "INTENT_EXPIRED"
SUCCESSOR_EXPIRE_REASONS = frozenset({"TIMEOUT", "IMPROVEMENT_LOST"})

_TRANSIENT_MARKERS = (
    "LIVE_BUY_ERROR",
    "TIMEOUT",
    "RATE_LIMIT",
    "TRANSIENT",
    "PROTECTED_LIMIT_BUY_NOT_FILLED",
    "NETWORK",
    "CONNECTION",
    "EXIT_MARK_STALE",
    "PROTECTED_PREFLIGHT",
)
_MIN_NOTIONAL_MARKERS = ("BELOW_MIN_NOTIONAL", "below_min_notional")
_EXCHANGE_MARKERS = ("EXCHANGE_REJECTED", "BINANCE", "-2010", "-1013", "-2011", "-1015")
_DETERMINISTIC_MARKERS = (
    "ARTIFACT_CONTRACT",
    "ENTRY_CONTEXT",
    "ENTRY_EXIT",
    "INSUFFICIENT_CASH",
    "INSUFFICIENT_EXECUTABLE",
    "KILL",
    "TRADING_PAUSED",
    "HARD_FUSE",
    "LIVE_TEST_GATE",
    "LIVE_EXECUTION_UNAVAILABLE",
    "POSITION_ALREADY_OPEN",
    "PENDING_BUY",
    "ENTRY_RESERVED",
    "THESIS_4H",
    "COMPLETED_4H",
    "MAX_POSITIONS",
    "ACCOUNT_OVERALLOCATED",
    "DELEVERAGING",
    "EXCHANGE_CONSTRAINT",
    "SYMBOL_NOT_EXECUTABLE",
    "BUY_BLOCKED_LEGACY",
    "CASH_INVARIANT",
    "HARD_SAFETY",
    "STALE_MARKET",
    "LAST_LOOK",
)


def classify_trailing_submit_outcome(reject: str) -> tuple[str, bool]:
    """Map execute_buy_fifo reject text to an explicit outcome and retry flag."""
    text = str(reject or "").strip()
    upper = text.upper()
    if not text:
        return f"{AMBIGUOUS_RECONCILE}:UNSPECIFIED", False
    if upper == "TIMEOUT" or upper.startswith("INTENT_EXPIRED"):
        return INTENT_EXPIRED, False
    if any(marker in text or marker.upper() in upper for marker in _MIN_NOTIONAL_MARKERS):
        return BELOW_MIN_NOTIONAL if upper == "BELOW_MIN_NOTIONAL" else f"{BELOW_MIN_NOTIONAL}:{text}", False
    if any(marker in upper for marker in _EXCHANGE_MARKERS) and "LIVE_BUY_ERROR" not in upper:
        return f"{EXCHANGE_REJECTED}:{text}", False
    if any(marker in upper for marker in _TRANSIENT_MARKERS):
        return f"{TRANSIENT_RETRY}:{text}", True
    if is_terminal_buy_cash_reason(text) or any(marker in upper for marker in _DETERMINISTIC_MARKERS):
        return f"{HARD_SAFETY_REJECTED}:{text}", False
    if "AMBIGUOUS" in upper or "UNCONFIRMED" in upper:
        return f"{AMBIGUOUS_RECONCILE}:{text}", False
    return f"{HARD_SAFETY_REJECTED}:{text}", False


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
    _ = thesis_invalid  # 4H / stamped thesis cannot cancel a live intent
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


_OBSERVE_LOGGED: dict[str, tuple[str, str]] = {}


def log_observe_decision(intent: dict[str, Any], decision: ObserveDecision, *, ask: float) -> None:
    """Emit one line per state/reason transition of the trailing-buy state machine.

    The submit bracket is single-digit bps wide, so which branch fired -- and how
    wide the bracket was at that moment -- is not recoverable after the fact from
    the intent row alone. Repeats of an unchanged (status, reason) are suppressed.
    """
    intent_id = str(intent.get("intent_id") or "")
    key = (str(decision.status or ""), str(decision.reason or ""))
    if _OBSERVE_LOGGED.get(intent_id) == key:
        return
    _OBSERVE_LOGGED[intent_id] = key
    arm_ask = float(intent.get("arm_ask") or 0.0)
    lowest = float(decision.lowest_ask or intent.get("lowest_ask") or 0.0)
    trigger = rebound_trigger_ask(lowest, float(intent.get("rebound_bps") or 0.0)) if lowest > 0 else 0.0
    ceiling = retained_improvement_ceiling_ask(arm_ask, float(intent.get("required_improvement_bps") or 0.0)) if arm_ask > 0 else 0.0
    width = ((ceiling - trigger) / arm_ask * 10000.0) if arm_ask > 0 and trigger > 0 else 0.0
    logger.info(
        "TRAILING_BUY_OBSERVE symbol=%s intent=%s action=%s status=%s reason=%s ask=%.8f arm=%.8f low=%.8f dip_got=%.2fbps dip_req=%.2fbps submit_window=[%.8f,%.8f] width=%.2fbps",
        intent.get("symbol"),
        intent_id,
        decision.action,
        decision.status,
        decision.reason or "-",
        float(ask or 0.0),
        arm_ask,
        lowest,
        dip_achieved_bps(arm_ask, float(ask or 0.0)),
        float(intent.get("min_dip_bps") or 0.0),
        trigger,
        ceiling,
        width,
    )
    if decision.action in {"expire", "cancel", "submit"}:
        _OBSERVE_LOGGED.pop(intent_id, None)


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
        # Key must match the persisted column name; the store reads "spread_bps".
        "spread_bps": float(arm_spread_bps),
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


def remaining_watch_notional_cap(*, free_cash: object, remaining_new_slots: int):
    """Split leftover cash across remaining new watches so all four can arm."""
    from backend.services.day_entry_spendable import remaining_slot_cap

    return remaining_slot_cap(free_cash=free_cash, remaining_new_slots=remaining_new_slots)


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


def _persist_hold(
    engine: Any,
    *,
    symbol: str,
    reason: str,
    authority: str,
    decision_id: str = "",
    intent_id: str = "",
    trailing_status: str = "",
    observed: dict[str, Any] | None = None,
    required: dict[str, Any] | None = None,
) -> None:
    """Record a HOLD category. Telemetry only — never changes the decision."""
    try:
        from backend.services.day_decision_state import build_hold_record, classify_hold_category, persist_hold_record

        persist_hold_record(
            str(getattr(engine, "db_path", "") or ""),
            build_hold_record(
                symbol=str(symbol or ""),
                category=classify_hold_category(reject_reason=reason, trailing_status=trailing_status),
                reason=reason,
                authority=authority,
                decision_id=str(decision_id or ""),
                intent_id=str(intent_id or ""),
                observed=observed,
                required=required,
            ),
        )
    except Exception:
        logger.debug("trailing-buy hold persist skipped", exc_info=True)


def _thesis_invalid(intent: dict[str, Any], ask: float) -> bool:
    level = float(intent.get("thesis_invalid_level") or 0.0)
    return bool(level > 0 and ask > 0 and ask <= level)


def _open_position_blocks_buy(engine: Any, symbol: str, ns: str) -> bool:
    """True only for a held lot. DUST_PENDING leftover inventory never blocks a BUY."""
    open_positions = getattr(engine, "open_positions", None) or {}
    pos = open_positions.get(ns)
    if pos is None:
        pos = open_positions.get(symbol)
    blocks_fn = getattr(engine, "_day_position_blocks_new_entry", None)
    if callable(blocks_fn):
        blocked = bool(blocks_fn(pos))
    elif pos is None or str(getattr(pos, "status", "ACTIVE") or "ACTIVE") == "DUST_PENDING":
        blocked = False
    else:
        blocked = float(getattr(pos, "quantity", 0) or 0) > 0
    if pos is not None and not blocked and str(getattr(pos, "status", "") or "") == "DUST_PENDING":
        logger.info(
            "TRAILING_BUY_DUST_NOT_HELD symbol=%s qty=%s (does not block BUY)",
            symbol,
            getattr(pos, "quantity", 0),
        )
    return blocked


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
    reserved_notional: object | None = None,
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
        _persist_hold(
            engine,
            symbol=symbol,
            reason="STALE_OR_MISSING_BOOK",
            authority="arm_selected_candidate",
            decision_id=str(decision_id or ""),
        )
        return None
    arm_ask = float(book["ask"])
    arm_bid = float(book["bid"])
    arm_mid = float(book["midpoint"])
    live_spread = float(book.get("spread_bps") or 0.0)
    formulas = formulas_for_symbol(symbol, arm_spread_bps=live_spread)
    discovery: dict[str, Any] = {}
    from backend.services.day_setup_discovery import classify_setup, may_arm_setup

    try:
        from backend.config.day_setup_discovery import SETUP_DISCOVERY_LOOKBACK_BARS
        from backend.services.day_path_net import load_recent_bars

        raw_bars = load_recent_bars(
            str(getattr(engine, "db_path", "") or ""),
            symbol,
            n=SETUP_DISCOVERY_LOOKBACK_BARS,
        )
        bars = []
        for row in raw_bars:
            epoch = _bar_epoch(row.get("ts"))
            bars.append((epoch, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume") or 0.0)))
        discovery = classify_setup(bars, symbol=symbol, ts=int(time.time()), atr=float(atr or 0.0), ask=arm_ask)
        # 240m structure / 4H discovery is telemetry only. It must not change
        # the authorized dip distance or delay a BUY.
    except Exception:
        logger.exception("TRAILING_BUY_SETUP_DISCOVERY_FAILED symbol=%s", symbol)
        discovery = {}
    if discovery:
        logger.info(
            "TRAILING_BUY_SETUP_TELEMETRY %s setup_class=%s reason=%s asof=%s early_need=%s structure_need_min=%s veto=false",
            symbol,
            discovery.get("setup_class"),
            discovery.get("reason"),
            discovery.get("asof_bars"),
            discovery.get("early_trend_need_bars"),
            discovery.get("structure_need_minutes"),
        )
    _ = may_arm_setup(discovery)  # 4H/240m discovery cannot veto a live arm
    from backend.services.day_entry_spendable import money

    notional = money(reserved_notional) if reserved_notional is not None else money(quantity) * money(arm_ask)
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
            "notional_usd": float(notional),
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


async def rearm_successor_after_expire(
    engine: Any,
    expired: dict[str, Any],
    redis_client: Any,
    *,
    reason: str,
) -> dict[str, Any] | None:
    """Arm a fresh WAIT_DIP immediately after a natural expire.

    New arm_ask from the live book. No inherited tracked low. No second
    reservation while a watcher is already active. Ranking clock is not
    required. Does not change dip, rebound, or improvement thresholds.
    """
    if str(reason or "") not in SUCCESSOR_EXPIRE_REASONS:
        return None
    symbol = str(expired.get("symbol") or "")
    if not symbol:
        return None
    existing = load_intent_by_symbol(engine.db_path, symbol)
    if existing:
        return None
    ns = symbol
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        ns = CanonicalSymbolFormatter.to_canonical(symbol)
    except Exception:
        ns = symbol
    if _open_position_blocks_buy(engine, symbol, ns):
        logger.info("TRAILING_BUY_SUCCESSOR_SKIPPED symbol=%s reason=POSITION_OPEN", symbol)
        return None
    payload = dict(expired.get("payload") or {})
    expl = _rebuild_explainability(payload, symbol)
    decision_data = dict(payload.get("decision_data") or {})
    out = await arm_selected_candidate(
        engine,
        symbol=symbol,
        quantity=float(expired.get("quantity") or 0.0),
        stop_price=float(expired.get("stop_price") or 0.0),
        atr=float(expired.get("atr") or 0.0),
        confidence=float(expired.get("confidence") or 0.0),
        bar_timestamp=int(time.time()),
        explainability=expl,
        decision_id=str(expired.get("decision_id") or ""),
        sleeve=str(expired.get("sleeve") or ""),
        decision_data=decision_data,
        redis_client=redis_client,
        reserved_notional=expired.get("notional_usd"),
    )
    if out and out.get("intent"):
        nxt = out["intent"]
        logger.info(
            "TRAILING_BUY_SUCCESSOR_ARMED symbol=%s prior=%s next=%s reason=%s arm_ask=%.8f lowest_ask=%.8f",
            symbol,
            expired.get("intent_id"),
            nxt.get("intent_id"),
            reason,
            float(nxt.get("arm_ask") or 0.0),
            float(nxt.get("lowest_ask") or 0.0),
        )
    return out


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
        why = str(open_why or "CANNOT_OPEN")
        if why == "HOLD_CONSEC_LOSSES":
            logger.info("HOLD_CONSEC_LOSSES_TELEMETRY intent=%s symbol=%s submit_not_vetoed", intent.get("intent_id"), symbol)
            return True, ""
        if why == "INSUFFICIENT_CASH" or why.startswith("INSUFFICIENT_CASH:"):
            from backend.config.protected_execution import MAKER_FEE, USE_PROTECTED_LIMIT_EXECUTION
            from backend.config.trading_economics import TAKER_FEE
            from backend.services.day_entry_spendable import plan_executable_buy

            own_res, _own_ns = engine._own_entry_reservation(symbol, str(intent.get("decision_id") or ""))
            pending_other = engine._pending_buy_notional(
                exclude_symbol=_own_ns if own_res else "",
                exclude_decision_id=str(intent.get("decision_id") or ""),
            )
            constraints = (getattr(engine, "_symbol_constraints", None) or {}).get(symbol) or {}
            plan = plan_executable_buy(
                requested_qty=intent.get("quantity") or 0,
                price=ask,
                commission_rate=MAKER_FEE if USE_PROTECTED_LIMIT_EXECUTION else TAKER_FEE,
                spendable=money(getattr(engine, "_available_balance", 0) or 0) - money(pending_other),
                qty_step=constraints.get("qty_step") or 0,
                min_qty=constraints.get("min_qty") or 0,
                min_notional=constraints.get("min_notional") or 0,
                allocation=own_res.get("notional") if own_res else intent.get("notional_usd"),
            )
            if plan.ok:
                return True, ""
            return False, plan.reason
        return False, why
    ns = symbol
    try:
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        ns = CanonicalSymbolFormatter.to_canonical(symbol)
    except Exception:
        ns = symbol
    if _open_position_blocks_buy(engine, symbol, ns):
        return False, "POSITION_ALREADY_OPEN"
    pending = set(engine._pending_buy_order_symbols())
    if ns in pending or symbol in pending:
        return False, "PENDING_BUY_EXISTS"
    if _thesis_invalid(intent, ask):
        # Price-vs-stamped-stop only. 4H structure cannot cancel the intent.
        logger.info(
            "TRAILING_BUY_THESIS_LEVEL_TELEMETRY symbol=%s intent=%s ask=%s level=%s authority=TELEMETRY_ONLY_NO_TRADE_AUTHORITY",
            symbol,
            intent.get("intent_id"),
            ask,
            intent.get("thesis_invalid_level"),
        )
    return True, ""


async def _submit_claimed(engine: Any, intent: dict[str, Any], ask: float) -> dict[str, Any] | None:
    symbol = str(intent.get("symbol") or "")
    payload = dict(intent.get("payload") or {})
    explainability = _rebuild_explainability(payload, symbol)
    safe, reason = await _pre_submit_safety(engine, intent, ask)
    if not safe:
        outcome, _retryable = classify_trailing_submit_outcome(reason)
        engine.last_buy_outcome = outcome
        mark_terminal(engine.db_path, str(intent["intent_id"]), CANCELED, reason=reason, current_ask=ask)
        engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason=reason)
        logger.info("TRAILING_BUY_SUBMIT_BLOCKED %s %s", symbol, reason)
        logger.info("TRAILING_BUY_SUBMIT_OUTCOME symbol=%s intent=%s outcome=%s", symbol, intent.get("intent_id"), outcome)
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
        # The order filled, so the reservation became a position. CONSUMED, not
        # released: the cash was spent, not handed back.
        _consume_intent_reservation(engine, intent)
        order_id = str(result.get("order_id") or result.get("exchange_order_id") or "")
        identity = _identity_for_order(engine, order_id)
        mark_order_accepted(
            engine.db_path,
            str(intent["intent_id"]),
            order_id=order_id,
            fill_id=str(result.get("fill_id") or "") or identity.get("fill_id", ""),
            trade_id=str(result.get("trade_id") or "") or identity.get("trade_id", ""),
        )
        improvement = fill_improvement_vs_arm_bps(float(intent.get("arm_ask") or 0.0), float(result.get("price") or ask))
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FILLED,
            reason=ENTRY_AUTHORITY,
            order_id=order_id,
            fill_id=str(result.get("fill_id") or "") or identity.get("fill_id", ""),
            trade_id=str(result.get("trade_id") or "") or identity.get("trade_id", ""),
            order_accepted=True,
            current_ask=float(result.get("price") or ask),
        )
        result = dict(result)
        result["entry_authority"] = ENTRY_AUTHORITY
        result["trailing_buy_intent_id"] = intent.get("intent_id")
        result["entry_improvement_bps"] = improvement
        filled = bool(result.get("fill_id") or result.get("filled") or result.get("average"))
        outcome = FILL_ADOPTED if filled else ORDER_ACCEPTED
        result["outcome"] = outcome
        engine.last_buy_outcome = outcome
        logger.info("TRAILING_BUY_SUBMIT_OUTCOME symbol=%s intent=%s outcome=%s", symbol, intent.get("intent_id"), outcome)
        logger.info(
            "TRAILING_BUY_FILLED symbol=%s intent=%s fill=%.8f arm_ask=%.8f improvement_bps=%.4f",
            symbol,
            intent.get("intent_id"),
            float(result.get("price") or ask),
            float(intent.get("arm_ask") or 0.0),
            improvement,
        )
        return result
    reject = str(getattr(engine, "last_buy_reject_reason", "") or "")
    outcome, retryable = classify_trailing_submit_outcome(reject)
    engine.last_buy_outcome = outcome
    logger.info("TRAILING_BUY_SUBMIT_OUTCOME symbol=%s intent=%s outcome=%s reject=%s", symbol, intent.get("intent_id"), outcome, reject or "UNSPECIFIED")
    if is_terminal_buy_cash_reason(reject) or (not retryable and outcome.startswith(HARD_SAFETY_REJECTED)):
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            CANCELED,
            reason=reject or outcome,
            current_ask=ask,
        )
        engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason=reject or outcome)
        logger.info("TRAILING_BUY_SUBMIT_TERMINAL %s %s", symbol, reject or outcome)
        return None
    if outcome.startswith(EXCHANGE_REJECTED) or outcome == BELOW_MIN_NOTIONAL or outcome.startswith(f"{BELOW_MIN_NOTIONAL}:"):
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            CANCELED,
            reason=reject or outcome,
            current_ask=ask,
        )
        engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason=reject or outcome)
        logger.info("TRAILING_BUY_SUBMIT_TERMINAL %s %s", symbol, reject or outcome)
        return None
    if outcome.startswith(AMBIGUOUS_RECONCILE):
        logger.info("TRAILING_BUY_SUBMIT_HOLD %s %s", symbol, outcome)
        return None
    if retryable and release_submitting_for_retry(engine.db_path, str(intent["intent_id"])):
        logger.info("TRAILING_BUY_SUBMIT_RETRYABLE %s %s", symbol, outcome)
        return None
    mark_terminal(engine.db_path, str(intent["intent_id"]), FAILED, reason=outcome)
    engine._release_entry_reservation(symbol, decision_id=str(intent.get("decision_id") or ""), reason=outcome)
    return None


async def cycle_trailing_buy_intents(engine: Any, redis_client: Any) -> dict[str, Any]:
    ok, err, mode = trailing_buy_mode_status()
    engine.day_entry_execution_error = "" if ok else err
    engine.day_entry_execution_mode = mode
    summary = {"mode_ok": ok, "error": err, "cycled": 0, "submitted": 0, "filled": 0, "closed": 0, "successor": 0}
    if not ok:
        return summary
    books = sync_book_redis(redis_client)
    for intent in load_active_intents(engine.db_path):
        summary["cycled"] += 1
        symbol = str(intent.get("symbol") or "")
        if str(intent.get("status") or "") in IN_FLIGHT_STATES:
            _persist_hold(
                engine,
                symbol=symbol,
                reason=str(intent.get("status") or SUBMITTING),
                authority="cycle_trailing_buy_intents",
                decision_id=str(intent.get("decision_id") or ""),
                intent_id=str(intent.get("intent_id") or ""),
                trailing_status=str(intent.get("status") or SUBMITTING),
            )
            await recover_submitting_intent(engine, intent)
            continue
        book = read_market_book(books, symbol)
        ask = float((book or {}).get("ask") or 0.0)
        fresh = bool(book and book.get("fresh") and float(book.get("freshness_sec") or 0.0) <= BOOK_STALE_SEC)
        # Setup-discovery / 4H structure cannot cancel a live intent.
        # live_intent_validity is permanently unenforced; do not load bars for it.
        decision = observe_book(
            intent,
            ask=ask,
            now=time.time(),
            book_fresh=fresh,
            thesis_invalid=_thesis_invalid(intent, ask),
            validity_reason="",
        )
        log_observe_decision(intent, decision, ask=ask)
        try:
            from backend.services.day_decision_state import record_trailing_observe

            record_trailing_observe(str(engine.db_path), intent, decision, ask=ask)
        except Exception:
            logger.debug("hold-state persist skipped", exc_info=True)
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
            if decision.action == "expire" and decision.reason in SUCCESSOR_EXPIRE_REASONS:
                successor = await rearm_successor_after_expire(
                    engine,
                    intent,
                    redis_client,
                    reason=decision.reason,
                )
                if successor:
                    summary["successor"] += 1
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


def _identity_for_order(engine: Any, exchange_order_id: str) -> dict[str, str]:
    """Recover Mystic's trade id and the venue fill ids for one exchange order.

    ``live_exchange_fills`` is the canonical live venue record: one append-only
    row per confirmed fill, carrying the exchange order id, Mystic's trade id,
    the intent id, the decision id and the per-fill venue trade ids. It is the
    only place the venue order id and Mystic's own identifiers are already
    joined, so it is what restart adoption reads rather than re-deriving links.
    """
    if not exchange_order_id:
        return {}
    try:
        from backend.services.live_order_identity import fills_for_order

        rows = fills_for_order(engine.db_path, str(exchange_order_id))
    except Exception:
        logger.exception("TRAILING_BUY_IDENTITY_LOOKUP_FAILED order=%s", exchange_order_id)
        return {}
    buys = [r for r in rows if str(r.get("side") or "").upper() == "BUY"] or rows
    if not buys:
        return {}
    row = buys[-1]
    fill_ids: list[str] = []
    try:
        fill_ids = [str(x) for x in json.loads(str(row.get("fill_ids_json") or "[]")) if str(x).strip()]
    except (TypeError, ValueError):
        fill_ids = []
    return {
        "trade_id": str(row.get("mystic_trade_id") or ""),
        "fill_id": ",".join(fill_ids),
        "decision_id": str(row.get("decision_id") or ""),
        "client_order_id": str(row.get("client_order_id") or ""),
    }


def _consume_intent_reservation(engine: Any, intent: dict[str, Any]) -> None:
    """Retire a filled intent's reservation as CONSUMED, exactly once.

    Uses the reservation_id stored on the intent row rather than the engine's
    in-memory reservation map, because adoption runs after a restart when that
    map is empty. A filled reservation left ACTIVE was being relabelled EXPIRED
    by the staleness sweep, so deployed capital looked abandoned.
    """
    reservation_id = str(intent.get("reservation_id") or "")
    decision_id = str(intent.get("decision_id") or "")
    symbol = str(intent.get("symbol") or "")
    try:
        from backend.services.day_entry_reservations import consume_reservation

        consumed = consume_reservation(
            engine.db_path,
            reservation_id=reservation_id,
            decision_id=decision_id,
            symbol=symbol,
        )
    except Exception:
        logger.exception(
            "RESERVATION_CONSUME_FAILED intent=%s reservation=%s — reservation may remain ACTIVE",
            intent.get("intent_id"),
            reservation_id,
        )
        return
    # Drop the in-memory hold too, or the engine keeps counting spent capital as
    # reserved until the next reload.
    with contextlib.suppress(Exception):
        from backend.services.portfolio_engine import normalize_symbol

        engine._entry_reservations.pop(normalize_symbol(symbol), None)
    if not consumed:
        logger.info(
            "RESERVATION_ALREADY_TERMINAL intent=%s reservation=%s — no second transition",
            intent.get("intent_id"),
            reservation_id,
        )


async def recover_submitting_intent(engine: Any, intent: dict[str, Any]) -> None:
    """Adopt an existing order/fill. Never guess a resubmit."""
    symbol = str(intent.get("symbol") or "")
    decision_id = str(intent.get("decision_id") or "")
    cid = str(intent.get("client_order_id") or "")
    local = _local_fill(engine, symbol=symbol, decision_id=decision_id, client_order_id=cid)
    if local:
        order_id = str(local.get("order_id") or "")
        identity = _identity_for_order(engine, order_id)
        fill_id = str(local.get("fill_id") or "") or identity.get("fill_id", "")
        trade_id = str(local.get("trade_id") or "") or identity.get("trade_id", "")
        mark_order_accepted(
            engine.db_path,
            str(intent["intent_id"]),
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
        )
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FILLED,
            reason="RECOVERED_LOCAL_FILL",
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
            order_accepted=True,
        )
        _consume_intent_reservation(engine, intent)
        logger.info("TRAILING_BUY_RECOVERED_FILL intent=%s trade=%s", intent.get("intent_id"), trade_id)
        return
    exchange = await _exchange_order(
        engine,
        client_order_id=cid,
        symbol=symbol,
        order_id=str(intent.get("order_id") or ""),
    )
    state = str(exchange.get("state") or "") if exchange else ""
    if state in {"open", "accepted"}:
        # Live at the venue. Never resubmit; adopt the open order.
        filled_qty = float(exchange.get("filled") or 0.0)
        in_flight = PARTIALLY_FILLED if filled_qty > 0 else ORDER_OPEN
        mark_in_flight(
            engine.db_path,
            str(intent["intent_id"]),
            in_flight,
            order_id=str(exchange.get("order_id") or ""),
            reason=f"RECOVERED_VENUE_{state.upper()}",
        )
        logger.info(
            "TRAILING_BUY_RECOVERED_%s_ORDER intent=%s order=%s status=%s",
            state.upper(),
            intent.get("intent_id"),
            exchange.get("order_id"),
            in_flight,
        )
        return
    if state == "canceled":
        # Proven dead at the venue with nothing filled: release the reservation
        # and let the intent retry rather than failing it as unconfirmed.
        if float(exchange.get("filled") or 0.0) <= 0.0 and release_submitting_for_retry(engine.db_path, str(intent["intent_id"])):
            logger.info(
                "TRAILING_BUY_RECOVER_RETRY intent=%s venue_canceled order=%s",
                intent.get("intent_id"),
                exchange.get("order_id"),
            )
            return
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            CANCELED,
            reason="RECOVERED_VENUE_CANCELED",
            order_id=str(exchange.get("order_id") or ""),
        )
        engine._release_entry_reservation(symbol, decision_id=decision_id, reason="RECOVERED_VENUE_CANCELED")
        return
    if state == "filled":
        # The venue knows nothing about Mystic's trade id, so exchange["trade_id"]
        # and exchange["fill_id"] are always blank here. Recover both from the
        # canonical live fill record, which is keyed by exchange order id and was
        # already written when the fill settled.
        order_id = str(exchange.get("order_id") or "")
        identity = _identity_for_order(engine, order_id)
        fill_id = str(exchange.get("fill_id") or "") or identity.get("fill_id", "")
        trade_id = str(exchange.get("trade_id") or "") or identity.get("trade_id", "")
        mark_order_accepted(
            engine.db_path,
            str(intent["intent_id"]),
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
        )
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FILLED,
            reason="RECOVERED_EXCHANGE_FILL",
            order_id=order_id,
            fill_id=fill_id,
            trade_id=trade_id,
            order_accepted=True,
        )
        _consume_intent_reservation(engine, intent)
        logger.info(
            "TRAILING_BUY_RECOVERED_EXCHANGE_FILL intent=%s order=%s trade=%s fill_ids=%s",
            intent.get("intent_id"),
            order_id,
            trade_id or "UNRESOLVED",
            fill_id or "UNRESOLVED",
        )
        return
    if bool(intent.get("order_accepted")):
        logger.warning("TRAILING_BUY_RECOVER_HOLD intent=%s accepted but fill unseen", intent.get("intent_id"))
        return
    age = _intent_age_sec(intent)
    if state == "unknown":
        reason = str((exchange or {}).get("reason") or "VENUE_LOOKUP_UNRESOLVED")
        if age < VENUE_LOOKUP_HOLD_SEC:
            logger.warning(
                "TRAILING_BUY_RECOVER_HOLD intent=%s venue_state=unknown age=%.1f reason=%s",
                intent.get("intent_id"),
                age,
                reason,
            )
            return
        mark_terminal(engine.db_path, str(intent["intent_id"]), FAILED, reason=f"VENUE_LOOKUP_TIMEOUT:{reason}"[:120])
        engine._release_entry_reservation(symbol, decision_id=decision_id, reason="VENUE_LOOKUP_TIMEOUT")
        return
    if exchange is None and not bool(intent.get("order_accepted")):
        if age < SUBMITTING_STALE_SEC:
            logger.info(
                "TRAILING_BUY_RECOVER_HOLD intent=%s proven_no_order age=%.1f waiting_stale=%.1f",
                intent.get("intent_id"),
                age,
                SUBMITTING_STALE_SEC,
            )
            return
        mark_terminal(
            engine.db_path,
            str(intent["intent_id"]),
            FAILED,
            reason="VENUE_ORDER_NOT_FOUND",
        )
        engine._release_entry_reservation(symbol, decision_id=decision_id, reason="VENUE_ORDER_NOT_FOUND")
        logger.info("TRAILING_BUY_RECOVER_FAILED intent=%s VENUE_ORDER_NOT_FOUND age=%.1f", intent.get("intent_id"), age)
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


SUBMITTING_STALE_SEC = 30.0
VENUE_LOOKUP_HOLD_SEC = 90.0


def _intent_age_sec(intent: dict[str, Any]) -> float:
    try:
        return max(0.0, time.time() - float(intent.get("updated_at") or intent.get("arm_ts") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _classify_fetch_error(message: str) -> str:
    text = str(message or "")
    if "-2013" in text or "Order does not exist" in text:
        return "not_found"
    if "-1100" in text or "Illegal characters" in text:
        return "bad_id_shape"
    return "transport"


async def _exchange_order(engine: Any, *, client_order_id: str, symbol: str, order_id: str = "") -> dict[str, Any] | None:
    """Resolve venue state. None = proven no order. unknown = lookup failed."""
    oid = str(order_id or "").strip()
    cid = str(client_order_id or "").strip()
    live = getattr(engine, "_live_service", None)
    if live is None:
        return {"state": "unknown", "reason": "LIVE_SERVICE_UNAVAILABLE"}
    fetch = getattr(live, "fetch_order", None) or getattr(live, "get_order", None)
    if fetch is None:
        return {"state": "unknown", "reason": "FETCH_ORDER_UNAVAILABLE"}
    raw: dict[str, Any] | None = None
    last_err = ""

    async def _call(ref: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        nonlocal last_err
        try:
            res = await fetch("binanceus", ref, symbol, params) if params else await fetch("binanceus", ref, symbol)
        except TypeError:
            try:
                res = await fetch("binanceus", ref, symbol)
            except Exception as exc:
                last_err = str(exc)
                return {"_error": last_err}
        except Exception as exc:
            last_err = str(exc)
            return {"_error": last_err}
        if not isinstance(res, dict):
            return None
        if str(res.get("status") or "") == "error":
            last_err = str(res.get("message") or res.get("code") or "error")
            return {"_error": last_err, "code": str(res.get("code") or "")}
        body = res.get("order") if str(res.get("status") or "") in {"success", "ok"} else (res.get("order") or None)
        if body is None and "id" in res:
            body = res
        return body if isinstance(body, dict) else None

    if oid.isdigit():
        got = await _call(oid)
        if got and not got.get("_error"):
            raw = got
        elif got and _classify_fetch_error(str(got.get("_error") or "")) == "transport":
            logger.info("TRAILING_BUY_RECOVER_FETCH_TRANSPORT symbol=%s order=%s err=%s", symbol, oid, last_err[:160])
            return {"state": "unknown", "reason": last_err[:160]}
    if raw is None and cid:
        got = await _call(cid, {"origClientOrderId": cid})
        if got and not got.get("_error"):
            raw = got
        else:
            kind = _classify_fetch_error(str((got or {}).get("_error") or last_err))
            if kind == "not_found":
                return None
            if kind == "bad_id_shape":
                return None
            if kind == "transport":
                logger.info("TRAILING_BUY_RECOVER_FETCH_TRANSPORT symbol=%s cid=%s err=%s", symbol, cid, last_err[:160])
                return {"state": "unknown", "reason": last_err[:160]}
    if raw is None and not oid and not cid:
        return None
    if raw is None:
        return None
    status = str(raw.get("status") or raw.get("state") or "").lower()
    oid = str(raw.get("id") or raw.get("order_id") or oid or cid)
    filled = float(raw.get("filled") or 0.0)
    if status in {"filled", "closed"} and filled > 0:
        return {
            "state": "filled",
            "order_id": oid,
            "fill_id": str(raw.get("fill_id") or ""),
            "trade_id": str(raw.get("trade_id") or ""),
        }
    if status in {"open", "new", "partially_filled", "partial"}:
        return {"state": "open", "order_id": oid, "filled": filled}
    if status in {"canceled", "cancelled", "expired", "rejected"}:
        # Terminal at the venue with nothing filled: safe to release for retry.
        # Returning None here made "cancelled" indistinguishable from "lookup
        # failed", so a known-dead order was treated as unresolved.
        return {"state": "canceled", "order_id": oid, "filled": filled}
    if status in {"accepted", "pending_new", "pending"}:
        return {"state": "accepted", "order_id": oid, "filled": filled}
    return {"state": "unknown", "order_id": oid, "filled": filled}


async def recover_trailing_buy_intents(engine: Any) -> int:
    rows = load_active_intents(engine.db_path)
    n = 0
    for intent in rows:
        if str(intent.get("status") or "") in IN_FLIGHT_STATES:
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
    "SUCCESSOR_EXPIRE_REASONS",
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
    "rearm_successor_after_expire",
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
