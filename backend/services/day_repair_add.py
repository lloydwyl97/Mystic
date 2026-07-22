"""
DAY-only controlled one-time position repair add evaluation.

Called from PortfolioEngine.process_repair_adds_once — not a separate bridge.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.config.repair_add_economics import (
    MAX_REPAIR_ADDS_PER_POSITION,
    REPAIR_ADD_COOLDOWN_SEC,
    REPAIR_ADD_ENABLED,
    REPAIR_ADD_MAX_TOTAL_SYMBOL_ALLOCATION_PCT,
    REPAIR_ADD_MIN_CONFIDENCE,
    REPAIR_ADD_MIN_RECOVERY_IMPROVEMENT_PCT,
    REPAIR_ADD_REQUIRED_FEATURE_DIM,
    REPAIR_ADD_REQUIRED_FEATURE_VERSION,
    REPAIR_ADD_SIZE_PCT_OF_ORIGINAL,
    REPAIR_ADD_TRIGGER_NET_PNL,
)
from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.day_active_market_bundle import validate_day_active_bundle
from backend.utils.symbols import normalize_symbol, to_exchange_symbol

logger = logging.getLogger(__name__)


@dataclass
class RepairAddEvaluation:
    eligible: bool
    blockers: list[str]
    add_qty: float = 0.0
    add_notional: float = 0.0
    mark_price: float = 0.0
    net_pnl_pct: float = 0.0
    old_recovery_price: float = 0.0
    new_recovery_price: float = 0.0
    recovery_improvement_pct: float = 0.0
    signal_confidence: float = 0.0
    original_cost: float = 0.0


def sell_recovery_price(entry_price: float) -> float:
    """Mark price required to clear MIN_NET_PROFIT_TO_SELL after roundtrip cost."""
    gross_floor = float(MIN_NET_PROFIT_TO_SELL) + float(ESTIMATED_ROUNDTRIP_COST)
    return float(entry_price) * (1.0 + gross_floor)


def net_pnl_pct(mark: float, entry: float) -> float:
    if entry <= 0 or mark <= 0:
        return 0.0
    return (mark - entry) / entry - float(ESTIMATED_ROUNDTRIP_COST)


def recovery_improvement_pct(
    old_entry: float,
    old_qty: float,
    add_qty: float,
    add_price: float,
) -> tuple[float, float, float]:
    """Return (old_recovery, new_recovery, improvement_fraction)."""
    if old_qty <= 0 or add_qty <= 0 or add_price <= 0 or old_entry <= 0:
        return 0.0, 0.0, 0.0
    old_rec = sell_recovery_price(old_entry)
    new_qty = old_qty + add_qty
    new_entry = (old_entry * old_qty + add_price * add_qty) / new_qty
    new_rec = sell_recovery_price(new_entry)
    if old_rec <= 0:
        return old_rec, new_rec, 0.0
    improvement = (old_rec - new_rec) / old_rec
    return old_rec, new_rec, improvement


def _validate_day_signal(signal: dict[str, str]) -> tuple[bool, list[str], float]:
    blockers: list[str] = []
    if not signal:
        return False, ["missing_ai_signal"], 0.0

    side = str(signal.get("side") or signal.get("action") or signal.get("prediction") or "").strip().lower()
    if side != "buy":
        blockers.append(f"side={side or 'missing'}")

    cf = str(signal.get("content_fresh") or signal.get("signal_content_fresh") or "").strip()
    if cf == "0":
        blockers.append("content_not_fresh")
    stale = str(signal.get("signal_content_stale") or "").strip()
    if stale == "1":
        blockers.append("signal_content_stale")

    try:
        fv = int(float(signal.get("feature_version") or 0))
    except (TypeError, ValueError):
        fv = 0
    if fv != REPAIR_ADD_REQUIRED_FEATURE_VERSION:
        blockers.append(f"feature_version={fv}")

    try:
        fd = int(float(signal.get("feature_dim") or 0))
    except (TypeError, ValueError):
        fd = 0
    if fd != REPAIR_ADD_REQUIRED_FEATURE_DIM:
        blockers.append(f"feature_dim={fd}")

    if not str(signal.get("context_audit_emit") or "").strip():
        blockers.append("missing_context_audit_emit")

    try:
        conf = float(signal.get("confidence") or signal.get("winner_probability") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf + 1e-12 < REPAIR_ADD_MIN_CONFIDENCE:
        blockers.append(f"confidence={conf:.3f}<{REPAIR_ADD_MIN_CONFIDENCE}")

    sig_ts = str(signal.get("signal_content_timestamp") or signal.get("timestamp") or "").strip()
    if not sig_ts:
        blockers.append("missing_signal_timestamp")
    else:
        try:
            float(sig_ts)
        except (TypeError, ValueError):
            blockers.append("signal_timestamp_parse")

    return len(blockers) == 0, blockers, conf


def evaluate_repair_add(
    *,
    symbol: str,
    quantity: float,
    entry_price: float,
    repair_add_count: int,
    last_repair_add_ts: float,
    original_position_cost: float,
    mark_price: float,
    total_equity: float,
    cash_balance: float,
    signal: dict[str, str],
    day_bundle: dict[str, Any] | None,
    day_missing: list[str] | None,
) -> RepairAddEvaluation:
    blockers: list[str] = []
    api_sym = to_exchange_symbol(symbol)
    if api_sym not in DAY_TRADE_SYMBOLS:
        blockers.append("not_top4")

    if not REPAIR_ADD_ENABLED:
        blockers.append("repair_add_disabled")

    if quantity <= 0 or entry_price <= 0:
        blockers.append("invalid_position")

    if mark_price <= 0:
        blockers.append("missing_mark_price")

    net = net_pnl_pct(mark_price, entry_price)
    if net + 1e-12 >= REPAIR_ADD_TRIGGER_NET_PNL:
        blockers.append(f"net_pnl={net * 100:.2f}%>={REPAIR_ADD_TRIGGER_NET_PNL * 100:.2f}%")

    if repair_add_count >= MAX_REPAIR_ADDS_PER_POSITION:
        blockers.append(f"repair_add_count={repair_add_count}")

    now = time.time()
    if last_repair_add_ts > 0 and (now - last_repair_add_ts) < REPAIR_ADD_COOLDOWN_SEC:
        blockers.append(f"cooldown_remaining={REPAIR_ADD_COOLDOWN_SEC - (now - last_repair_add_ts):.0f}s")

    ok_sig, sig_blockers, conf = _validate_day_signal(signal)
    if not ok_sig:
        blockers.extend(sig_blockers)

    if day_bundle is None or day_missing is None:
        blockers.append("missing_day_bundle_context")
    else:
        ok_bd, ms = validate_day_active_bundle(day_bundle if isinstance(day_bundle, dict) else {})
        if not ok_bd or ms:
            blockers.append(f"day_bundle_invalid:{';'.join(ms or day_missing or [])[:120]}")
        if day_missing:
            blockers.append(f"day_missing:{';'.join(day_missing[:8])}")

    orig_cost = float(original_position_cost or 0.0)
    if orig_cost <= 0:
        orig_cost = quantity * entry_price

    current_cost = quantity * entry_price
    max_symbol_cost = float(total_equity) * REPAIR_ADD_MAX_TOTAL_SYMBOL_ALLOCATION_PCT
    headroom = max(0.0, max_symbol_cost - current_cost)
    target_add_notional = orig_cost * REPAIR_ADD_SIZE_PCT_OF_ORIGINAL
    underwater = net + 1e-12 < REPAIR_ADD_TRIGGER_NET_PNL
    if headroom <= 0:
        if underwater:
            # Underwater repair adds may exceed entry allocation cap; still cash-limited.
            add_notional = min(target_add_notional, float(cash_balance))
        else:
            _alloc_pct = (current_cost / total_equity * 100) if total_equity > 0 else 0.0
            blockers.append(f"allocation_at_cap={_alloc_pct:.1f}%")
            add_notional = 0.0
    else:
        add_notional = min(target_add_notional, headroom, float(cash_balance))
    if add_notional <= 0:
        blockers.append("add_notional_zero")

    add_qty = add_notional / mark_price if mark_price > 0 else 0.0
    if add_qty <= 0:
        blockers.append("add_qty_zero")

    old_rec, new_rec, imp = recovery_improvement_pct(entry_price, quantity, add_qty, mark_price)
    if imp + 1e-12 < REPAIR_ADD_MIN_RECOVERY_IMPROVEMENT_PCT:
        blockers.append(f"recovery_improve={imp * 100:.3f}%<{REPAIR_ADD_MIN_RECOVERY_IMPROVEMENT_PCT * 100:.3f}%")

    eligible = len(blockers) == 0
    return RepairAddEvaluation(
        eligible=eligible,
        blockers=blockers,
        add_qty=add_qty,
        add_notional=add_notional,
        mark_price=mark_price,
        net_pnl_pct=net,
        old_recovery_price=old_rec,
        new_recovery_price=new_rec,
        recovery_improvement_pct=imp,
        signal_confidence=conf,
        original_cost=orig_cost,
    )


def repair_add_trade_ids_json(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x) for x in val if x]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return []
