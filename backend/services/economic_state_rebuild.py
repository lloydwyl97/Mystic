"""Rebuild the current economic version's adaptive state from current-version evidence.

Run with the portfolio engine stopped. Nothing before an engine's economic
anchor is read. Every label reaches the learner at the time it was knowable,
in order, through the same functions the live loop uses:

- DAY realized closes (entered at or after the anchor) at close time.
- DAY qualified and frequency-blocked candidates reconstructed from the stored
  decisions and bars, under the actual held symbols, fills and 24h caps. The
  first record of each opportunity that was never filled learns its lifecycle
  label at its exit time. Lifecycles still running are inserted as candidate
  rows for the live resolver.
- SCALP recorded candidates re-scored at their decision time and resolved at
  their stamped horizon (residual, claim calibration, micro model and weight,
  risk), plus SCALP realized closes at close time. Claims recorded under an
  earlier economic version are restated as the claim the current strategies
  make. The production rows are stamped with the economic version and whether
  their label is already in the rebuilt state, so the live resolver never
  counts one twice.

Idempotent: the current version's state, linear model and rebuilt candidate
rows are replaced. Refuses once live candidates of the current version exist.
``rebuild_scalp`` does the SCALP half alone, for a SCALP learner-format change.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.economic_replay import (
    DAY_ENGINE,
    DAY_SYMBOLS,
    SCALP_ENGINE,
    BarStore,
    DayCandidate,
    RepairedDayPolicy,
    ScalpPolicy,
    day_lifecycle,
    load_scalp_rows,
    reconstruct_day_candidates,
    simulate_scalp,
    tag_regimes,
)

REBUILT_MARK = '"rebuilt":true'


def _iso_epoch(value: Any) -> float | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def actual_closes(db_path: str, engine_id: str, since: float, store: BarStore) -> list[dict[str, Any]]:
    """Realized closes of ``engine_id`` entered at or after ``since``, as the live
    close learner sees them: net = gross - estimated round-trip cost, keys from the
    entry's adaptive decision, MFE/MAE from 1m bars over the hold."""
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
    from backend.services.adaptive_learning import _norm_symbol
    from backend.services.strategy_version import is_current_version

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: list[dict[str, Any]] = []
    try:
        sells = conn.execute(
            "SELECT * FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? ORDER BY timestamp",
            (engine_id,),
        ).fetchall()
        for sell in sells:
            entered, closed = _iso_epoch(sell["entry_timestamp"]), _iso_epoch(sell["timestamp"])
            if entered is None or closed is None or entered < since:
                continue
            if not is_current_version(engine_id, sell) or "DUST" in str(sell["exit_reason"] or "").upper():
                continue
            entry, exit_px = float(sell["entry_price"] or 0.0), float(sell["price"] or 0.0)
            buy = conn.execute(
                "SELECT adaptive_decision_json FROM paper_trades WHERE UPPER(side)='BUY' AND UPPER(COALESCE(engine_id,''))=? "
                "AND ((COALESCE(decision_id,'')!='' AND decision_id=?) OR trade_id=?) ORDER BY rowid DESC LIMIT 1",
                (engine_id, sell["decision_id"], sell["trade_id"]),
            ).fetchone()
            try:
                decision = json.loads((buy["adaptive_decision_json"] if buy else "") or "{}")
            except (TypeError, ValueError):
                decision = {}
            if entry <= 0 or exit_px <= 0 or not isinstance(decision, dict) or not decision.get("setup"):
                continue
            symbol = _norm_symbol(sell["symbol"])
            bars = [b for b in store.minute_path(symbol, entered, closed + 60.0) if b[0] + 60.0 > entered]
            high = max([b[2] for b in bars] + [entry, exit_px])
            low = min([b[3] for b in bars] + [entry, exit_px])
            out.append(
                {
                    "symbol": symbol,
                    "setup": str(decision["setup"]),
                    "regime": str(decision.get("regime") or ""),
                    "decision": decision,
                    "strategy_version": str(sell["strategy_version"]),
                    "entered_at": entered,
                    "closed_at": closed,
                    "entry_price": entry,
                    "net": (exit_px - entry) / entry - ESTIMATED_ROUNDTRIP_COST,
                    "mfe": (high - entry) / entry,
                    "mae": (entry - low) / entry,
                    "minutes": (closed - entered) / 60.0,
                    "high": high,
                }
            )
    finally:
        conn.close()
    return out


def _match_fill(cands_by_symbol: dict[str, list[DayCandidate]], fill: dict[str, Any]) -> DayCandidate | None:
    """The bar decision the fill executed: same symbol, closed within the 16 minutes before entry."""
    best = None
    for c in cands_by_symbol.get(fill["symbol"], []):
        if fill["entered_at"] - 960.0 <= c.decided_at <= fill["entered_at"] and (best is None or c.decided_at > best.decided_at):
            best = c
    return best


def rebuild_day(
    cands: list[DayCandidate], fills: list[dict[str, Any]], store: BarStore, policy: RepairedDayPolicy, *, roundtrip_cost: float, now: float, freq_symbol: int = 2, freq_total: int = 8
) -> dict[str, Any]:
    """Feed ``policy`` the DAY evidence in time order under the actual fills."""
    import heapq
    import itertools

    from backend.services.adaptive_learning import CANDIDATE_QUALIFIED, CANDIDATE_QUALIFIED_BLOCKED, continuation_ratio
    from backend.services.day_v2.lifecycle_sim import LifecycleParams
    from backend.services.day_v2.winner_contract import objective_level

    by_symbol: dict[str, list[DayCandidate]] = defaultdict(list)
    for c in cands:
        by_symbol[c.symbol].append(c)
    matched = {id(f): _match_fill(by_symbol, f) for f in fills}
    fill_of = {id(m): f for f in fills if (m := matched[id(f)]) is not None}
    heap: list[tuple[float, int, Any]] = []
    seq = itertools.count()
    consumed: dict[str, float] = {}
    first_record: set[str] = set()
    pending: list[dict[str, Any]] = []
    counts = {"closes": 0, "lifecycles": 0, "records": 0, "blocked": 0, "pending": 0, "fills_matched": sum(1 for m in matched.values() if m is not None), "fills": len(fills)}

    def learn_close(fill: dict[str, Any], cand: DayCandidate | None) -> None:
        continuation = None
        if cand is not None:
            params = LifecycleParams.from_signal(cand.signal, entry_price=fill["entry_price"], entry_time=fill["entered_at"], adaptive=fill["decision"])
            goal = objective_level(fill["setup"], fill["entry_price"], params.atr_1h, params.objective_structural, atr_mult=params.objective_atr_mult, structural_emphasis=params.structural_emphasis)
            continuation = continuation_ratio(entry_price=fill["entry_price"], highest_price=fill["high"], objective=goal)
        policy.al.learn_from_close(
            policy.db,
            engine=DAY_ENGINE,
            symbol=fill["symbol"],
            setup=fill["setup"],
            regime=fill["regime"],
            strategy_version=fill["strategy_version"],
            net_pct=fill["net"],
            mfe_pct=fill["mfe"],
            mae_pct=fill["mae"],
            hold_min=fill["minutes"],
            continuation=continuation,
            version_current=True,
            is_dust=False,
            entered_at=fill["entered_at"],
            now=fill["closed_at"],
        )
        counts["closes"] += 1

    def learn_lifecycle(c: DayCandidate, label: dict[str, Any]) -> None:
        filled_at = consumed.get(c.opportunity_id)
        if filled_at is None or filled_at > float(label["exit_time"]):
            policy.learn_lifecycle(c, label, float(label["exit_time"]))
            counts["lifecycles"] += 1

    for f in fills:
        if f["closed_at"] <= now:
            heapq.heappush(heap, (f["closed_at"], next(seq), lambda f=f: learn_close(f, matched[id(f)])))
    by_t: dict[float, list[DayCandidate]] = defaultdict(list)
    for c in cands:
        by_t[c.decided_at].append(c)
    for t in sorted(by_t):
        while heap and heap[0][0] <= t:
            heapq.heappop(heap)[2]()
        held = {f["symbol"] for f in fills if f["entered_at"] < t < f["closed_at"]}
        live = [c for c in by_t[t] if c.symbol not in held and c.opportunity_id not in consumed]
        if not live:
            continue
        recent = [f for f in fills if t - 86400.0 < f["entered_at"] <= t]
        ok, blocked = [], []
        for c in live:
            capped = sum(1 for f in recent if f["symbol"] == c.symbol) >= freq_symbol or len(recent) >= freq_total
            (blocked if capped else ok).append(c)
        decisions = {id(c): policy.decide(c, t) for c in live}
        ranked = policy.rank([{"symbol": c.symbol, "signal": c.signal, "ask_price": c.ask, "adaptive": decisions[id(c)], "_c": c} for c in ok], roundtrip_cost) if ok else []
        records = [(row["_c"], CANDIDATE_QUALIFIED) for row in ranked] + [(c, CANDIDATE_QUALIFIED_BLOCKED) for c in blocked]
        counts["blocked"] += len(blocked)
        for c, state in records:
            counts["records"] += 1
            adaptive = decisions[id(c)]
            fill = fill_of.get(id(c))
            if fill is not None:
                consumed[c.opportunity_id] = t
            elif c.opportunity_id not in first_record:
                label = day_lifecycle(c, adaptive, store, roundtrip_cost=roundtrip_cost, now=now)
                if label.get("final") and label.get("net") is not None:
                    heapq.heappush(heap, (float(label["exit_time"]), next(seq), lambda c=c, label=label: learn_lifecycle(c, label)))
                elif not label.get("final"):
                    pending.append({"candidate": c, "adaptive": adaptive, "state": state})
            first_record.add(c.opportunity_id)
    while heap and heap[0][0] <= now:
        heapq.heappop(heap)[2]()
    pending = [p for p in pending if p["candidate"].opportunity_id not in consumed]
    counts["pending"] = len(pending)
    return {"counts": counts, "pending": pending}


def _live_version_rows(db_path: str, versions: tuple[str, ...]) -> int:
    conn = sqlite3.connect(db_path, timeout=15)
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(adaptive_candidate_markouts)")}
        if "economic_version" not in cols:
            return 0
        marks = ",".join("?" for _ in versions)
        row = conn.execute(
            f"SELECT COUNT(*) FROM adaptive_candidate_markouts WHERE economic_version IN ({marks}) AND INSTR(COALESCE(economic_json,''), ?)=0",
            (*versions, REBUILT_MARK),
        ).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def _live_current_rows(db_path: str) -> int:
    from backend.services.adaptive_learning import current_economic_version

    return _live_version_rows(db_path, (current_economic_version(DAY_ENGINE), current_economic_version(SCALP_ENGINE)))


def _learn_scalp_close(db: str, close: dict[str, Any]) -> None:
    from backend.services.adaptive_learning import learn_from_close

    learn_from_close(
        db,
        engine=SCALP_ENGINE,
        symbol=close["symbol"],
        setup=close["setup"],
        regime=close["regime"],
        strategy_version=close["strategy_version"],
        net_pct=close["net"],
        mfe_pct=close["mfe"],
        mae_pct=close["mae"],
        hold_min=close["minutes"],
        continuation=None,
        version_current=True,
        is_dust=False,
        entered_at=close["entered_at"],
        now=close["closed_at"],
    )


def _scalp_scratch(db_path: str, store: BarStore, scratch_db: str, *, anchor: float, now: float) -> dict[str, Any]:
    """SCALP's current version rebuilt into ``scratch_db``: every recorded candidate
    re-scored at its decision time and resolved at its horizon, realized closes at
    close time. Claim rows recorded under another economic version carry the claim
    the current strategies make (``unfloored_claim``). Returns the scratch state,
    micro model and one production-row stamp per candidate:
    (economic_version, economic_json, learned, raw_expected_move or None, id)."""
    from backend.services import adaptive_learning as al
    from backend.services.scalp_v2 import executable_edge as edge_mod

    version = al.current_economic_version(SCALP_ENGINE)
    closes = actual_closes(db_path, SCALP_ENGINE, anchor, store)
    rows = load_scalp_rows(db_path, anchor, now, al.current_strategy_version(SCALP_ENGINE), store=store, unfloor_unless_version=version)
    policy = ScalpPolicy("repaired", scratch_db, al, edge_mod, store)
    result = simulate_scalp(rows, policy, now=now, events=[(c["closed_at"], lambda c=c: _learn_scalp_close(scratch_db, c)) for c in closes])
    conn = sqlite3.connect(scratch_db)
    conn.row_factory = sqlite3.Row
    try:
        scratch_rows = {int(r["id"]): r for r in conn.execute("SELECT id, learned, economic_json FROM adaptive_candidate_markouts")}
        state = conn.execute("SELECT * FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (SCALP_ENGINE, version)).fetchall()
        model = conn.execute("SELECT * FROM adaptive_linear_model WHERE engine_id=? AND model=?", (SCALP_ENGINE, al._model_key(SCALP_ENGINE, "micro_edge"))).fetchone()
    finally:
        conn.close()
    stamps: list[tuple[str, str, int, float | None, int]] = []
    for c in rows:
        sid = result["scratch_ids"].get(c["id"])
        srow = scratch_rows.get(int(sid)) if sid is not None else None
        if srow is None:
            continue
        econ = json.loads(srow["economic_json"] or "{}") or {}
        econ["rebuilt"] = True
        if c["directional"]:
            econ["raw_claim_recorded"] = c["raw_recorded"]
        stamps.append((version, json.dumps(econ, separators=(",", ":"), default=str), int(srow["learned"] or 0), c["raw"] if c["directional"] else None, c["id"]))
    claims = [c for c in rows if c["directional"]]
    return {
        "version": version,
        "rows": rows,
        "closes": closes,
        "result": result,
        "state": state,
        "model": model,
        "stamps": stamps,
        "summary": {
            "scalp_economic_version": version,
            "scalp_rows": len(rows),
            "scalp_claims": len(claims),
            "scalp_claims_restated": sum(1 for c in claims if abs(c["raw"] - c["raw_recorded"]) > 1e-12),
            "scalp_claims_without_bars": sum(1 for c in claims if c["unfloor_missing"]),
            "scalp_rows_learned": sum(1 for s in stamps if s[2]),
            "scalp_closes": len(closes),
            "scalp_state_rows": len(state),
            "scalp_admitted_in_rebuild": sum(1 for p in result["predictions"] if p["admitted"]),
            "micro_model_n": float(json.loads(model["payload"]).get("n", 0)) if model is not None else 0,
            "micro_weight": al.micro_weight(scratch_db, now=now),
        },
    }


def _write_scalp(conn: sqlite3.Connection, scratch: dict[str, Any]) -> None:
    from backend.services import adaptive_learning as al

    conn.execute("DELETE FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (SCALP_ENGINE, scratch["version"]))
    for r in scratch["state"]:
        conn.execute(
            "INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["engine_id"], r["economic_version"], r["symbol"], r["setup"], r["regime"], r["metric"], r["n"], r["ewma"], r["m2"], r["updated_at"]),
        )
    conn.execute("DELETE FROM adaptive_linear_model WHERE engine_id=? AND model=?", (SCALP_ENGINE, al._model_key(SCALP_ENGINE, "micro_edge")))
    model = scratch["model"]
    if model is not None:
        conn.execute("INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at) VALUES (?, ?, ?, ?)", (model["engine_id"], model["model"], model["payload"], model["updated_at"]))
    conn.executemany(
        "UPDATE adaptive_candidate_markouts SET economic_version=?, economic_json=?, learned=?, raw_expected_move=COALESCE(?, raw_expected_move) WHERE id=?",
        scratch["stamps"],
    )


def _scalp_setup_summary(db_path: str, now: float) -> dict[str, Any]:
    from backend.services import adaptive_learning as al

    conn = sqlite3.connect(db_path)
    try:
        setups = sorted(
            {str(r[0]) for r in conn.execute("SELECT DISTINCT setup FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (SCALP_ENGINE, al.current_economic_version(SCALP_ENGINE)))}
        )
    finally:
        conn.close()
    out: dict[str, Any] = {}
    for setup in setups:
        claim = al.scalp_claim_calibration(db_path, "", setup, "", now=now)
        out[setup] = {
            "claim_gross_setup_bps": round(claim["claim_gross_setup"] * 1e4, 2),
            "claim_capture": round(claim["claim_capture"], 4),
            "claim_weight": round(claim["n_claim"], 2),
        }
    return out


def rebuild_scalp(db_path: str, *, now: float | None = None, apply: bool = False, workdir: str | None = None) -> dict[str, Any]:
    """Rebuild (``apply``) or dry-run SCALP's current economic version only.

    For a SCALP learner-format change: DAY state and rows are untouched. The
    stored claim of a restated row is kept in its ``economic_json`` as
    ``raw_claim_recorded`` and ``raw_expected_move`` holds the restated claim,
    so rows the live resolver has not yet labelled learn the same claim.
    Refuses once live rows of the version exist.
    """
    from backend.services import adaptive_learning as al
    from backend.services.strategy_version import economic_anchor

    moment = float(now if now is not None else time.time())
    version = al.current_economic_version(SCALP_ENGINE)
    out: dict[str, Any] = {"now": moment, "applied": False, "refused": "", "scalp_economic_version": version}
    live = _live_version_rows(db_path, (version,))
    if live:
        out["refused"] = f"LIVE_SCALP_VERSION_ROWS={live}"
        return out
    anchor = float(economic_anchor(SCALP_ENGINE)["epoch"])
    store = BarStore(db_path, DAY_SYMBOLS, since=anchor - 86400)
    scratch_root = workdir or ("/dev/shm" if Path("/dev/shm").is_dir() else None)
    with tempfile.TemporaryDirectory(prefix="scalp_rebuild_", dir=scratch_root) as tmp:
        scratch = _scalp_scratch(db_path, store, f"{tmp}/scalp.db", anchor=anchor, now=moment)
        out.update(scratch["summary"])
        out["scalp_setups"] = _scalp_setup_summary(f"{tmp}/scalp.db", moment)
        if not apply:
            return out
        conn = al._connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _write_scalp(conn, scratch)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        out["applied"] = True
    return out


def _setup_summary(db_path: str, now: float) -> dict[str, Any]:
    from backend.services import adaptive_learning as al

    out: dict[str, Any] = {}
    conn = sqlite3.connect(db_path)
    try:
        setups = sorted(
            {str(r[0]) for r in conn.execute("SELECT DISTINCT setup FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (DAY_ENGINE, al.current_economic_version(DAY_ENGINE)))}
        )
    finally:
        conn.close()
    for setup in setups:
        d = al.day_decision(db_path, "", setup, "", now=now)
        out[setup] = {
            "setup_expected_net_bps": round(d["economic"]["levels"]["setup"] * 1e4, 2),
            "setup_weight": round(d["economic"]["level_weights"]["setup"], 2),
            "size_mult_at_setup_level": round(al.day_size_mult(d["economic"]["levels"]["setup"], max(0.0, d["mae"]) + d["economic"]["expected_cost"]), 4),
        }
    return out


def rebuild(db_path: str, *, now: float | None = None, apply: bool = False, workdir: str | None = None) -> dict[str, Any]:
    """Rebuild (``apply``) or dry-run the current economic version's state in ``db_path``."""
    from backend.config.trading_economics import canonical_roundtrip_cost_pct
    from backend.services import adaptive_learning as al
    from backend.services.day_v2.lifecycle_sim import LifecycleParams
    from backend.services.strategy_version import economic_anchor

    moment = float(now if now is not None else time.time())
    out: dict[str, Any] = {"now": moment, "applied": False, "refused": ""}
    live = _live_current_rows(db_path)
    if live:
        out["refused"] = f"LIVE_CURRENT_VERSION_ROWS={live}"
        return out
    day_anchor = float(economic_anchor(DAY_ENGINE)["epoch"])
    scalp_anchor = float(economic_anchor(SCALP_ENGINE)["epoch"])
    store = BarStore(db_path, DAY_SYMBOLS, since=day_anchor - 10 * 86400)
    cost = canonical_roundtrip_cost_pct()
    cands, _near = reconstruct_day_candidates(db_path, day_anchor, moment, store)
    tag_regimes(db_path, cands)
    day_fills = actual_closes(db_path, DAY_ENGINE, day_anchor, store)
    day_version = al.current_economic_version(DAY_ENGINE)
    scratch_root = workdir or ("/dev/shm" if Path("/dev/shm").is_dir() else None)
    with tempfile.TemporaryDirectory(prefix="econ_rebuild_", dir=scratch_root) as tmp:
        day_db = f"{tmp}/day.db"
        day_policy = RepairedDayPolicy(day_db)
        day = rebuild_day(cands, day_fills, store, day_policy, roundtrip_cost=cost, now=moment)
        scalp = _scalp_scratch(db_path, store, f"{tmp}/scalp.db", anchor=scalp_anchor, now=moment)
        dconn = sqlite3.connect(day_db)
        dconn.row_factory = sqlite3.Row
        try:
            day_state = dconn.execute("SELECT * FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (DAY_ENGINE, day_version)).fetchall()
        finally:
            dconn.close()
        out.update(
            {
                "day_economic_version": day_version,
                "day": day["counts"],
                "day_candidates": len(cands),
                "day_state_rows": len(day_state),
                **scalp["summary"],
                "day_setups": _setup_summary(day_db, moment),
            }
        )
        if not apply:
            return out
        conn = al._connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (DAY_ENGINE, day_version))
            for r in day_state:
                conn.execute(
                    "INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["engine_id"], r["economic_version"], r["symbol"], r["setup"], r["regime"], r["metric"], r["n"], r["ewma"], r["m2"], r["updated_at"]),
                )
            conn.execute("DELETE FROM adaptive_candidate_markouts WHERE engine_id=? AND economic_version=? AND INSTR(COALESCE(economic_json,''), ?)>0", (DAY_ENGINE, day_version, REBUILT_MARK))
            _write_scalp(conn, scalp)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        inserted = 0
        for p in day["pending"]:
            c, adaptive = p["candidate"], p["adaptive"]
            economic = {**(adaptive.get("economic") or {}), "candidate_state": p["state"], "rebuilt": True}
            if al.record_candidate(
                db_path,
                engine=DAY_ENGINE,
                symbol=c.symbol,
                setup=c.setup,
                regime=c.regime_tag,
                ref_price=c.ask,
                roundtrip_cost=cost,
                signaled=True,
                evaluated_at=c.decided_at,
                candidate_state=p["state"],
                lifecycle=LifecycleParams.from_signal(c.signal, entry_price=c.ask, entry_time=c.decided_at, adaptive=adaptive),
                economic=economic,
                opportunity_id=c.opportunity_id,
            ):
                inserted += 1
        out["day_pending_inserted"] = inserted
        out["applied"] = True
    return out


__all__ = ["REBUILT_MARK", "actual_closes", "rebuild", "rebuild_day", "rebuild_scalp"]
