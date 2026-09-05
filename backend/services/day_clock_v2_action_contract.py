"""CLOCK-V2 action semantics. Research only — never selects, sizes, or authorizes.

Five distinct concepts that the capture-v1 schema conflated into one boolean:

1. ACTION AVAILABILITY      - could production have selected this action at this bar
2. LEGACY RANK MEMBERSHIP   - was the symbol in the old 15m scored-candidate list
3. PRODUCTION SELECTION     - did the four-coin path-EV argmax pick it
4. EXECUTION AUTHORIZATION  - did post-selection gates permit an order
5. ACTUAL FILL              - did an order fill

Traced contract (backend/services/day_direct_path_ev_authority.select_action):
the argmax runs over DAY_TRADE_SYMBOLS and drops a coin only when its
``valid`` entry is false, i.e. path_input_valid and path_net_status=="predicted".
The legacy ``candidates`` list is consumed by ``old_rank_telemetry`` for stamps
only (OLD_RANK_EXECUTION_AUTHORITY is False). Absence from that list therefore
does NOT make an action unavailable, which is why production selected and filled
symbols that capture-v1 recorded as NO_SCORED_CANDIDATE.
"""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = "day_clock_v2_action_contract_v1"
HOLD_SYMBOL = "HOLD"
DAY_UNIVERSE: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

# --- hard availability blockers (real production gates) ---
NOT_IN_UNIVERSE = "NOT_IN_DAY_UNIVERSE"
PATH_INPUT_INVALID = "PATH_INPUT_INVALID"
DUPLICATE_SAME_SYMBOL = "DUPLICATE_SAME_SYMBOL"
MAX_OPEN_LIMIT = "MAX_OPEN_LIMIT"
AVAILABILITY_UNKNOWN = "AVAILABILITY_UNKNOWN_NO_PATH_TELEMETRY"

HARD_UNAVAILABLE_REASONS: tuple[str, ...] = (
    NOT_IN_UNIVERSE,
    PATH_INPUT_INVALID,
    DUPLICATE_SAME_SYMBOL,
    MAX_OPEN_LIMIT,
)

# --- legacy rank-candidate membership (telemetry only) ---
NO_SCORED_CANDIDATE = "NO_SCORED_CANDIDATE"
LEGACY_MEMBERSHIP_REASONS: tuple[str, ...] = (NO_SCORED_CANDIDATE,)

# --- legacy final-rank provenance ---
LEGACY_RANK_GENUINE = "LEGACY_RANK_SCORE_GENUINE"
LEGACY_RANK_ABSENT = "NO_LEGACY_RANK_SCORE_CANDIDATE_ABSENT"
LEGACY_RANK_NOT_SCORED = "NO_LEGACY_RANK_SCORE_CANDIDATE_UNSCORED"
LEGACY_RANK_PATH_EV_SUBSTITUTED = "PATH_EV_SUBSTITUTED_NOT_LEGACY_RANK"
LEGACY_RANK_HOLD = "HOLD_REFERENCE_ZERO"

# Telemetry-only conditions that production logs but explicitly does not enforce.
# Kept as a named allow-list so a future reader cannot mistake them for gates.
NON_BLOCKING_TELEMETRY: tuple[str, ...] = (
    "NO_SCORED_CANDIDATE",
    "COIN_PAUSED",
    "LOW_CONFIDENCE",
    "PNL_ADAPT_PENALTY",
    "REGIME_ROUTE_ADVISORY",
    "BUCKET_QUALITY_ADVISORY",
    "ENTRY_QUALITY_CHECK",
    "BUY_MARGIN_BELOW_THRESHOLD",
)


def _api(symbol: Any) -> str:
    text = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if text.endswith("USD") and not text.endswith("USDT"):
        text += "T"
    return text


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_action_availability(
    *,
    symbol: str,
    path_input_valid: Any = None,
    path_invalid_reason: Any = None,
    open_symbols: Any = None,
    slots_used: Any = None,
    slot_count: Any = None,
) -> dict[str, Any]:
    """Could production have selected this action at this decision bar?

    HOLD is always available. A coin is available when it is in the DAY universe,
    has valid path-EV input, is not already held, and a slot is free. Nothing here
    depends on the legacy scored-candidate list.
    """
    sym = _api(symbol) if symbol != HOLD_SYMBOL else HOLD_SYMBOL
    if sym == HOLD_SYMBOL:
        return {
            "action_available": True,
            "action_unavailable_reason": None,
            "availability_contract_version": CONTRACT_VERSION,
        }
    if sym not in DAY_UNIVERSE:
        return {
            "action_available": False,
            "action_unavailable_reason": NOT_IN_UNIVERSE,
            "availability_contract_version": CONTRACT_VERSION,
        }
    if path_input_valid is None:
        # Never assert availability we cannot prove from stored point-in-time data.
        return {
            "action_available": None,
            "action_unavailable_reason": AVAILABILITY_UNKNOWN,
            "availability_contract_version": CONTRACT_VERSION,
        }
    if not bool(path_input_valid):
        reason = str(path_invalid_reason or "") or PATH_INPUT_INVALID
        return {
            "action_available": False,
            "action_unavailable_reason": reason,
            "availability_contract_version": CONTRACT_VERSION,
        }
    held = {_api(s) for s in (open_symbols or [])}
    if sym in held:
        return {
            "action_available": False,
            "action_unavailable_reason": DUPLICATE_SAME_SYMBOL,
            "availability_contract_version": CONTRACT_VERSION,
        }
    used = _num(slots_used)
    total = _num(slot_count)
    if used is not None and total is not None and total > 0 and used >= total:
        return {
            "action_available": False,
            "action_unavailable_reason": MAX_OPEN_LIMIT,
            "availability_contract_version": CONTRACT_VERSION,
        }
    return {
        "action_available": True,
        "action_unavailable_reason": None,
        "availability_contract_version": CONTRACT_VERSION,
    }


def evaluate_legacy_rank_candidate(
    *,
    symbol: str,
    candidate_present: bool,
    exclusion_reason: Any = None,
) -> dict[str, Any]:
    """Legacy 15m scored-candidate list membership. Telemetry, not a gate."""
    sym = _api(symbol) if symbol != HOLD_SYMBOL else HOLD_SYMBOL
    if sym == HOLD_SYMBOL:
        return {
            "legacy_rank_candidate_present": True,
            "legacy_rank_candidate_reason": None,
        }
    present = bool(candidate_present)
    if present:
        return {
            "legacy_rank_candidate_present": True,
            "legacy_rank_candidate_reason": str(exclusion_reason) if exclusion_reason else None,
        }
    return {
        "legacy_rank_candidate_present": False,
        "legacy_rank_candidate_reason": str(exclusion_reason or NO_SCORED_CANDIDATE),
    }


def evaluate_legacy_final_rank(
    *,
    symbol: str,
    candidate_present: bool,
    final_selection_score: Any = None,
    recorded_final_rank_score: Any = None,
    path_ev: Any = None,
) -> dict[str, Any]:
    """Legacy final-rank provenance. Never substitutes path-EV for a rank score.

    The legacy score is produced by the 15m ranking chain (bandit, adaptive
    weights, symbol trust, thesis, haircuts) over a constructed BuyCandidate.
    A coin with no candidate has no such score, so the honest value is NULL.
    """
    sym = _api(symbol) if symbol != HOLD_SYMBOL else HOLD_SYMBOL
    if sym == HOLD_SYMBOL:
        return {
            "legacy_final_rank_score": 0.0,
            "legacy_final_rank_score_valid": True,
            "legacy_final_rank_reason": LEGACY_RANK_HOLD,
        }
    genuine = _num(final_selection_score)
    if bool(candidate_present) and genuine is not None:
        return {
            "legacy_final_rank_score": genuine,
            "legacy_final_rank_score_valid": True,
            "legacy_final_rank_reason": LEGACY_RANK_GENUINE,
        }
    if bool(candidate_present):
        return {
            "legacy_final_rank_score": None,
            "legacy_final_rank_score_valid": False,
            "legacy_final_rank_reason": LEGACY_RANK_NOT_SCORED,
        }
    recorded = _num(recorded_final_rank_score)
    ev = _num(path_ev)
    reason = LEGACY_RANK_ABSENT
    if recorded is not None and ev is not None and recorded == ev:
        reason = LEGACY_RANK_PATH_EV_SUBSTITUTED
    return {
        "legacy_final_rank_score": None,
        "legacy_final_rank_score_valid": False,
        "legacy_final_rank_reason": reason,
    }


def evaluate_action_row(
    *,
    symbol: str,
    candidate_present: bool,
    exclusion_reason: Any = None,
    path_input_valid: Any = None,
    path_invalid_reason: Any = None,
    open_symbols: Any = None,
    slots_used: Any = None,
    slot_count: Any = None,
    final_selection_score: Any = None,
    recorded_final_rank_score: Any = None,
    path_ev: Any = None,
    production_selected: bool = False,
    execute_authorized: Any = None,
    filled: Any = None,
) -> dict[str, Any]:
    """All five concepts for one action. Pure."""
    out: dict[str, Any] = {}
    out.update(
        evaluate_action_availability(
            symbol=symbol,
            path_input_valid=path_input_valid,
            path_invalid_reason=path_invalid_reason,
            open_symbols=open_symbols,
            slots_used=slots_used,
            slot_count=slot_count,
        )
    )
    out.update(
        evaluate_legacy_rank_candidate(
            symbol=symbol,
            candidate_present=candidate_present,
            exclusion_reason=exclusion_reason,
        )
    )
    out.update(
        evaluate_legacy_final_rank(
            symbol=symbol,
            candidate_present=candidate_present,
            final_selection_score=final_selection_score,
            recorded_final_rank_score=recorded_final_rank_score,
            path_ev=path_ev,
        )
    )
    out["production_selected"] = bool(production_selected)
    out["execute_authorized"] = None if execute_authorized is None else bool(execute_authorized)
    out["filled"] = None if filled is None else bool(filled)
    return out


# --- Part 4: selected-action invariant ---

INVARIANT_ID = "clock_v2_selected_action_availability_v1"
VIOLATION_SELECTED_UNAVAILABLE = "SELECTED_ACTION_RECORDED_UNAVAILABLE"
VIOLATION_FILLED_UNAVAILABLE = "FILLED_ACTION_RECORDED_UNAVAILABLE"
VIOLATION_LEGACY_MEMBERSHIP_USED_AS_AVAILABILITY = "LEGACY_MEMBERSHIP_USED_AS_AVAILABILITY"


def selected_action_invariant(
    *,
    rows: list[dict[str, Any]],
    selected_symbol: Any,
    filled: bool = False,
) -> dict[str, Any]:
    """A production-selected action must be recorded available.

    Returns ``pass=False`` with a named violation when it is not. A violation is
    only acceptable if a hard production gate explains it; a legacy
    candidate-list omission never does.
    """
    selected = _api(selected_symbol) if selected_symbol and str(selected_symbol) != HOLD_SYMBOL else HOLD_SYMBOL
    by_sym = {(_api(r.get("symbol")) if str(r.get("symbol")) != HOLD_SYMBOL else HOLD_SYMBOL): r for r in rows or []}
    row = by_sym.get(selected)
    if selected == HOLD_SYMBOL:
        return {
            "invariant": INVARIANT_ID,
            "pass": True,
            "selected_symbol": HOLD_SYMBOL,
            "violations": [],
            "proven_production_defect": None,
        }
    if row is None:
        return {
            "invariant": INVARIANT_ID,
            "pass": False,
            "selected_symbol": selected,
            "violations": [VIOLATION_SELECTED_UNAVAILABLE],
            "proven_production_defect": "SELECTED_ACTION_ROW_MISSING",
        }
    available = row.get("action_available")
    reason = str(row.get("action_unavailable_reason") or "")
    violations: list[str] = []
    defect: str | None = None
    if available is not True:
        if reason in HARD_UNAVAILABLE_REASONS:
            # Production selected an action a hard gate had already closed.
            defect = f"PRODUCTION_SELECTED_HARD_BLOCKED_ACTION:{reason}"
            violations.append(VIOLATION_SELECTED_UNAVAILABLE)
        else:
            violations.append(VIOLATION_SELECTED_UNAVAILABLE)
        if reason in LEGACY_MEMBERSHIP_REASONS:
            violations.append(VIOLATION_LEGACY_MEMBERSHIP_USED_AS_AVAILABILITY)
        if filled:
            violations.append(VIOLATION_FILLED_UNAVAILABLE)
    return {
        "invariant": INVARIANT_ID,
        "pass": not violations,
        "selected_symbol": selected,
        "action_available": available,
        "action_unavailable_reason": row.get("action_unavailable_reason"),
        "legacy_rank_candidate_present": row.get("legacy_rank_candidate_present"),
        "violations": violations,
        "proven_production_defect": defect,
    }


# --- Part 7: deterministic point-in-time reconstruction of capture-v1 rows ---

RECONSTRUCTED_PIT = "RECONSTRUCTED_PIT"
NOT_RECONSTRUCTABLE = "NOT_RECONSTRUCTABLE_INSUFFICIENT_PIT_DATA"


def reconstruct_group_action_state(contract: dict[str, Any]) -> dict[str, Any]:
    """Recover corrected action semantics from a stored capture-v1 group contract.

    Uses only fields recorded at decision time: per-symbol path_input_valid,
    open_symbols, slots_used, slot_count, the legacy candidate flag, and the
    legacy final_selection_score. Returns ``NOT_RECONSTRUCTABLE`` when any coin
    lacks path telemetry, rather than guessing availability.
    """
    payload = dict(contract or {})
    rows = list(payload.get("candidates") or [])
    by_sym = {str(r.get("symbol")): dict(r) for r in rows}
    open_symbols = payload.get("open_symbols") or []
    slots_used = payload.get("slots_used")
    slot_count = payload.get("slot_count")
    out_rows: list[dict[str, Any]] = []
    reconstructable = True
    for sym in (*DAY_UNIVERSE, HOLD_SYMBOL):
        row = by_sym.get(sym)
        if row is None:
            reconstructable = False
            continue
        if sym != HOLD_SYMBOL and row.get("path_input_valid") is None:
            reconstructable = False
        resolved = evaluate_action_row(
            symbol=sym,
            candidate_present=bool(row.get("eligible")),
            exclusion_reason=row.get("exclusion_reason"),
            path_input_valid=row.get("path_input_valid"),
            path_invalid_reason=row.get("path_invalid_reason"),
            open_symbols=open_symbols,
            slots_used=slots_used,
            slot_count=slot_count,
            # capture-v1 never stored final_selection_score separately; a genuine
            # legacy score is present only when the candidate existed and the
            # recorded score differs from raw path_ev.
            final_selection_score=(row.get("final_rank_score") if row.get("eligible") else None),
            recorded_final_rank_score=row.get("final_rank_score"),
            path_ev=row.get("path_ev"),
            production_selected=(_api(payload.get("selected_symbol")) == sym),
        )
        resolved["symbol"] = sym
        out_rows.append(resolved)
    status = RECONSTRUCTED_PIT if reconstructable else NOT_RECONSTRUCTABLE
    invariant = selected_action_invariant(
        rows=out_rows,
        selected_symbol=payload.get("selected_symbol"),
        filled=str(payload.get("lifecycle_state") or payload.get("final_lifecycle_state") or "") == "filled",
    )
    return {
        "decision_group_id": payload.get("decision_group_id"),
        "reconstruction_status": status,
        "trainable_support_eligible": reconstructable,
        "contract_version": CONTRACT_VERSION,
        "rows": out_rows,
        "selected_action_invariant": invariant,
    }


__all__ = [
    "AVAILABILITY_UNKNOWN",
    "CONTRACT_VERSION",
    "DAY_UNIVERSE",
    "DUPLICATE_SAME_SYMBOL",
    "HARD_UNAVAILABLE_REASONS",
    "HOLD_SYMBOL",
    "INVARIANT_ID",
    "LEGACY_MEMBERSHIP_REASONS",
    "LEGACY_RANK_ABSENT",
    "LEGACY_RANK_GENUINE",
    "LEGACY_RANK_PATH_EV_SUBSTITUTED",
    "MAX_OPEN_LIMIT",
    "NON_BLOCKING_TELEMETRY",
    "NOT_RECONSTRUCTABLE",
    "NO_SCORED_CANDIDATE",
    "PATH_INPUT_INVALID",
    "RECONSTRUCTED_PIT",
    "VIOLATION_FILLED_UNAVAILABLE",
    "VIOLATION_LEGACY_MEMBERSHIP_USED_AS_AVAILABILITY",
    "VIOLATION_SELECTED_UNAVAILABLE",
    "evaluate_action_availability",
    "evaluate_action_row",
    "evaluate_legacy_final_rank",
    "evaluate_legacy_rank_candidate",
    "reconstruct_group_action_state",
    "selected_action_invariant",
]
