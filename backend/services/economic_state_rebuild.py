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
from backend.services.strategy_version import current_lifecycle_label, exit_contract_of

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
                    "exit_reason": str(sell["exit_reason"] or ""),
                    "opportunity_id": str(sell["scalp_opportunity_id"] or "") if "scalp_opportunity_id" in set(sell.keys()) else "",
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
    counts = {"closes": 0, "lifecycles": 0, "policy_gaps": 0, "records": 0, "blocked": 0, "pending": 0, "fills_matched": sum(1 for m in matched.values() if m is not None), "fills": len(fills)}

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
            exit_reason=fill.get("exit_reason"),
        )
        counts["closes"] += 1
        if cand is not None and exit_contract_of(DAY_ENGINE, entered_at=fill["entered_at"], exit_reason=fill.get("exit_reason")) == "CURRENT":
            label = day_lifecycle(cand, fill["decision"], store, roundtrip_cost=roundtrip_cost, now=now)
            if label.get("final") and label.get("net") is not None and current_lifecycle_label(label.get("reason")):
                at = max(float(fill["closed_at"]), float(label["exit_time"]))
                if at <= now:
                    heapq.heappush(heap, (at, next(seq), lambda c=cand, gap=float(fill["net"]) - float(label["net"]), at=at: learn_gap(c, gap, at)))

    def learn_gap(c: DayCandidate, gap: float, at: float) -> None:
        policy.al.observe(policy.db, engine=DAY_ENGINE, symbol=c.symbol, setup=c.setup, regime=c.regime_tag, metric="policy_gap", value=gap, strategy_version=policy.version, now=at)
        counts["policy_gaps"] += 1

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
        exit_reason=close.get("exit_reason"),
    )


def _scalp_policy_gaps(scratch_db: str, rows: list[dict[str, Any]], closes: list[dict[str, Any]], scratch_ids: dict[int, int | None], *, now: float) -> int:
    """Realized minus the admitted claim's own-horizon markout, for current-policy
    closes, folded in time order at the later of the close and the label."""
    from backend.services import adaptive_learning as al

    gaps: list[tuple[float, dict[str, Any], float]] = []
    conn = sqlite3.connect(scratch_db)
    conn.row_factory = sqlite3.Row
    try:
        for close in closes:
            if exit_contract_of(SCALP_ENGINE, entered_at=close["entered_at"], exit_reason=close.get("exit_reason")) != "CURRENT":
                continue
            claims = [c for c in rows if c["directional"] and c["symbol"] == close["symbol"] and close["entered_at"] - 30.0 <= c["t"] <= close["entered_at"] + 1.0]
            if not claims:
                continue
            claim = max(claims, key=lambda c: c["t"])
            sid = scratch_ids.get(claim["id"])
            row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (sid,)).fetchone() if sid is not None else None
            if row is None:
                continue
            market = al._market_label(SCALP_ENGINE, row, json.loads(row["markouts_json"] or "{}"))
            if market is None:
                continue
            at = max(float(close["closed_at"]), float(row["evaluated_at"]) + (float(row["label_horizon"] or 0) or 600.0))
            if at <= now:
                gaps.append((at, dict(row), float(close["net"]) - market))
    finally:
        conn.close()
    for at, row, gap in sorted(gaps, key=lambda g: g[0]):
        al.observe(
            scratch_db, engine=SCALP_ENGINE, symbol=row["symbol"], setup=row["setup"], regime=row["regime"], metric="policy_gap", value=gap, strategy_version=str(row["strategy_version"]), now=at
        )
    return len(gaps)


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
    policy_gaps = _scalp_policy_gaps(scratch_db, rows, closes, result["scratch_ids"], now=now)
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
            "scalp_policy_gaps": policy_gaps,
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
                lifecycle=LifecycleParams.from_signal(c.signal, entry_price=c.ask, entry_time=c.entry_time, adaptive=adaptive),
                economic=economic,
                opportunity_id=c.opportunity_id,
            ):
                inserted += 1
        out["day_pending_inserted"] = inserted
        out["applied"] = True
    return out


# --------------------------------------------------------------------------- regeneration
#
# ``regenerate`` rebuilds an engine's derived economic state on a live
# database from its authoritative rows: the recorded candidate rows (their
# stored markouts, inputs and decision-time economics), the realized closes in
# ``trade_learning_outcomes`` and the 1m bars. Every label reaches a scratch
# learner at the moment it became knowable, through ``resolve_markouts``,
# ``learn_from_close`` and ``record_policy_outcome`` themselves, so the rebuilt
# state follows the live version and exit-contract rules. DAY evidence from
# before the first recorded candidate is the bar reconstruction ``rebuild``
# uses. Continuation state (``hold_*``) and ``trade_continuation`` are kept.

PRESERVED_METRICS: tuple[str, ...] = ("trade_continuation",)
PRESERVED_METRIC_PREFIXES: tuple[str, ...] = ("hold_adv_", "hold_remaining_")
_LINK_WINDOW_SEC = 30.0


def preserved_metric(metric: str) -> bool:
    """State ``regenerate`` keeps as it is."""
    name = str(metric or "")
    return name in PRESERVED_METRICS or name.startswith(PRESERVED_METRIC_PREFIXES)


def _json(raw: Any) -> dict[str, Any]:
    try:
        out = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def realized_closes(db_path: str, engine_id: str) -> list[dict[str, Any]]:
    """Realized closes as the live close learner received them, in close order.

    One per ``trade_learning_outcomes`` row of the engine entered at or after its
    economic anchor (manual and dust write-offs excluded): the learned realized
    net, MFE/MAE while holding, hold time, close reason and versions, keyed by
    the entry decision's setup and regime, with the SELL's opportunity id.
    """
    from backend.services import adaptive_learning as al

    engine = str(engine_id or "").upper()
    anchor = al.anchor_epoch(engine)
    excluded = al.POLICY_SEED_EXCLUDED_EXITS
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: list[dict[str, Any]] = []
    try:
        marks = ",".join("?" * len(excluded))
        outcomes = conn.execute(
            f"SELECT * FROM trade_learning_outcomes WHERE UPPER(COALESCE(engine_id,''))=? AND entry_timestamp>=? AND net_profit_pct IS NOT NULL "
            f"AND COALESCE(close_reason,'') NOT IN ({marks}) ORDER BY exit_timestamp",
            (engine, anchor, *excluded),
        ).fetchall()
        for o in outcomes:
            extra = _json(o["extra_json"])
            holding = _json(o["indicators_while_holding_json"])
            entered, closed = float(o["entry_timestamp"]), float(o["exit_timestamp"])
            buy, sell = _close_rows(conn, engine, o["symbol"], str(extra.get("original_trade_id") or ""), closed)
            decision = _json(buy["adaptive_decision_json"]) if buy is not None else {}
            sell_decision = _json(sell["adaptive_decision_json"]) if sell is not None else {}
            setup = str(decision.get("setup") or o["setup"] or "")
            if not setup:
                continue
            hold = o["hold_seconds"]
            out.append(
                {
                    "symbol": al._norm_symbol(o["symbol"]),
                    "setup": setup,
                    "regime": str(decision.get("regime") or ""),
                    "strategy_version": str(extra.get("strategy_version") or o["strategy_version"] or ""),
                    "version_current": bool(extra.get("version_current")),
                    "is_dust": bool(extra.get("is_dust")),
                    "net": float(o["net_profit_pct"]),
                    "mfe": holding.get("mfe_pct"),
                    "mae": holding.get("mae_pct"),
                    "hold_min": float(hold) / 60.0 if hold else None,
                    "entered_at": entered,
                    "closed_at": closed,
                    "exit_reason": str(o["close_reason"] or ""),
                    "contract": exit_contract_of(engine, entered_at=entered, exit_reason=str(o["close_reason"] or "")),
                    "opportunity_id": str((sell["scalp_opportunity_id"] if sell is not None else "") or (buy["scalp_opportunity_id"] if buy is not None else "") or ""),
                    "position_trade_id": str(extra.get("original_trade_id") or (buy["trade_id"] if buy is not None else "") or ""),
                    "candidate_id": _lineage_candidate(sell_decision, decision),
                }
            )
    finally:
        conn.close()
    return out


_TRADE_COLUMNS = "trade_id, decision_id, scalp_opportunity_id, adaptive_decision_json"


def _close_rows(conn: sqlite3.Connection, engine: str, symbol: str, position_trade_id: str, closed: float) -> tuple[Any, Any]:
    """(BUY, SELL) of one realized close, by identity first.

    The position's own BUY (``original_trade_id``) and the SELL sharing its
    decision id; a live exit's realized timestamp can trail its SELL row by more
    than any fixed window. Without that identity, the nearest SELL on the
    symbol inside ``_LINK_WINDOW_SEC`` and its BUY.
    """
    buy = None
    if position_trade_id:
        buy = conn.execute(
            f"SELECT {_TRADE_COLUMNS} FROM paper_trades WHERE UPPER(side)='BUY' AND UPPER(COALESCE(engine_id,''))=? AND trade_id=? ORDER BY rowid DESC LIMIT 1",
            (engine, position_trade_id),
        ).fetchone()
    if buy is not None and str(buy["decision_id"] or ""):
        sell = conn.execute(
            f"SELECT {_TRADE_COLUMNS} FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? AND decision_id=? "
            "ORDER BY ABS(strftime('%s', substr(timestamp, 1, 19)) - ?) LIMIT 1",
            (engine, buy["decision_id"], closed),
        ).fetchone()
        if sell is not None:
            return buy, sell
    sell = conn.execute(
        f"SELECT {_TRADE_COLUMNS} FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? AND symbol=? "
        "AND ABS(strftime('%s', substr(timestamp, 1, 19)) - ?) <= ? ORDER BY ABS(strftime('%s', substr(timestamp, 1, 19)) - ?) LIMIT 1",
        (engine, symbol, closed, _LINK_WINDOW_SEC, closed),
    ).fetchone()
    if sell is not None and buy is None:
        buy = conn.execute(
            f"SELECT {_TRADE_COLUMNS} FROM paper_trades WHERE UPPER(side)='BUY' AND UPPER(COALESCE(engine_id,''))=? "
            "AND ((COALESCE(decision_id,'')!='' AND decision_id=?) OR trade_id=?) ORDER BY rowid DESC LIMIT 1",
            (engine, sell["decision_id"], sell["trade_id"]),
        ).fetchone()
    return buy, sell


def _lineage_candidate(*decisions: dict[str, Any]) -> int | None:
    """The entry's own candidate row id from SELL close lineage or BUY entry lineage."""
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        close = decision.get("close_lineage")
        entry = close.get("entry") if isinstance(close, dict) else None
        for source in (entry, decision.get("lineage")):
            if isinstance(source, dict) and source.get("candidate_id") not in (None, ""):
                try:
                    return int(source["candidate_id"])
                except (TypeError, ValueError):
                    continue
    return None


def _candidate_rows(db_path: str, engine_id: str) -> list[dict[str, Any]]:
    from backend.services import adaptive_learning as al

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND economic_version=? ORDER BY id",
            (engine_id, al.current_economic_version(engine_id)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def link_closes(engine_id: str, rows: list[dict[str, Any]], closes: list[dict[str, Any]]) -> dict[int, int]:
    """Close index -> its own candidate row id.

    The close's lineage ``candidate_id`` when it names a current row. Otherwise
    DAY: the opportunity's fills in order, one per close (an opportunity can be
    filled again after an earlier position closed); SCALP: the admitted claim on
    the symbol recorded in the 30 s before entry."""
    links: dict[int, int] = {}
    ids = {int(r["id"]) for r in rows}
    used: set[int] = set()
    for i, close in enumerate(closes):
        cand = close.get("candidate_id")
        if cand is not None and int(cand) in ids and int(cand) not in used:
            links[i] = int(cand)
            used.add(int(cand))
    if engine_id == DAY_ENGINE:
        fills: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            opp = str(r.get("opportunity_id") or "")
            if opp and int(r.get("filled") or 0):
                fills[opp].append(int(r["id"]))
        for i, close in enumerate(closes):
            if i in links:
                continue
            free = [row_id for row_id in fills.get(str(close.get("opportunity_id") or ""), []) if row_id not in used]
            if free:
                links[i] = free[0]
                used.add(free[0])
        return links
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if int(r.get("signaled") or 0):
            by_symbol[str(r["symbol"])].append(r)
    for i, close in enumerate(closes):
        if i in links:
            continue
        found = [r for r in by_symbol.get(close["symbol"], []) if close["entered_at"] - _LINK_WINDOW_SEC <= float(r["evaluated_at"]) <= close["entered_at"] + 1.0 and int(r["id"]) not in used]
        if found:
            row = max(found, key=lambda r: (float(r["evaluated_at"]), int(r["id"])))
            links[i] = int(row["id"])
            used.add(int(row["id"]))
    return links


def _due_marks(stored: dict[str, Any], evaluated_at: float, unit: float, until: float) -> dict[str, Any]:
    """The stored forward marks already due at ``until`` (lifecycle excluded)."""
    out: dict[str, Any] = {}
    for key, value in stored.items():
        if key == "lifecycle":
            continue
        try:
            due = evaluated_at + (float(key[:-1]) if key.endswith("s") else float(key) * unit)
        except ValueError:
            continue
        if due <= until + 1e-9:
            out[key] = value
    return out


def _lifecycle_final_at(label: dict[str, Any], params: Any) -> float | None:
    from backend.services.day_v2.lifecycle_sim import DAY_LIFECYCLE_GRACE_SEC, DAY_LIFECYCLE_MAX_MIN

    if not label.get("final"):
        return None
    horizon_end = float(params.entry_time) + DAY_LIFECYCLE_MAX_MIN * 60.0 if params is not None else None
    reason = str(label.get("reason") or "")
    if reason == "HORIZON_MARK_PARTIAL" or label.get("net") is None:
        return (horizon_end + DAY_LIFECYCLE_GRACE_SEC) if horizon_end is not None else None
    return float(label.get("exit_time") or 0.0) or None


def _replay(
    source_db: str,
    scratch_db: str,
    engine_id: str,
    rows: list[dict[str, Any]],
    closes: list[dict[str, Any]],
    *,
    now: float,
    closes_after: float,
    resimulate: bool,
) -> dict[str, Any]:
    """Feed the scratch learner every row label and close at its knowable time."""
    from backend.services import adaptive_learning as al
    from backend.services.day_v2.lifecycle_sim import LifecycleParams, ohlcv_bars_1m

    def bars(sym: str, start: float, end: float) -> list:
        return ohlcv_bars_1m(source_db, sym, start, end)

    def low(sym: str, start: float, end: float) -> float | None:
        return al.ohlcv_low_between(source_db, sym, start, end)

    unit = 60.0 if engine_id == DAY_ENGINE else 1.0
    default_h = 60.0 if engine_id == DAY_ENGINE else 600.0
    conn = al._connect(scratch_db)
    cols = [str(r[1]) for r in conn.execute("PRAGMA table_info(adaptive_candidate_markouts)")]
    blank = {"markouts_json": "{}", "learned": 0, "resolved": 1, "lifecycle_learned": 0, "policy_learned": 0, "realized_net": None, "filled": 0}
    for r in rows:
        values = {c: (blank[c] if c in blank else r.get(c)) for c in cols if c in r or c in blank}
        conn.execute(f"INSERT INTO adaptive_candidate_markouts ({','.join(values)}) VALUES ({','.join('?' * len(values))})", tuple(values.values()))
    conn.commit()
    links = link_closes(engine_id, rows, closes)
    linked_rows = {row_id: i for i, row_id in links.items()}
    events: list[tuple[float, int, str, Any]] = []
    counts = {"rows": len(rows), "closes": 0, "closes_current": 0, "closes_retired": 0, "links": len(links), "labels": 0, "lifecycles_pending": 0, "lifecycles_resimulated": 0}
    for r in rows:
        row_id = int(r["id"])
        stored = _json(r.get("markouts_json"))
        evaluated = float(r["evaluated_at"])
        if int(r.get("filled") or 0) or row_id in linked_rows:
            close = closes[linked_rows[row_id]] if row_id in linked_rows else None
            fill_at = float(close["entered_at"]) if close is not None else evaluated
            opp = str((close or {}).get("opportunity_id") or r.get("opportunity_id") or "") or f"row:{row_id}"
            if close is not None and not close.get("opportunity_id"):
                close["opportunity_id"] = opp
            events.append((max(evaluated, fill_at), 0, "fill", (row_id, opp)))
        label_h = float(r.get("label_horizon") or 0) or default_h
        t_fix = evaluated + label_h * unit
        if t_fix <= now:
            events.append((t_fix, 1, "label", (row_id, t_fix)))
        if engine_id == DAY_ENGINE and str(r.get("lifecycle_json") or ""):
            params = LifecycleParams.from_json(r["lifecycle_json"])
            if resimulate:
                label = al._lifecycle_label(r, moment=now, bars_1m=bars)
                old = stored.get("lifecycle") if isinstance(stored.get("lifecycle"), dict) else {}
                if label.get("final") and old.get("net") != label.get("net"):
                    counts["lifecycles_resimulated"] += 1
            else:
                life = stored.get("lifecycle")
                label = {"final": True, **life} if isinstance(life, dict) else {"final": False}
                if label.get("final") and params is not None:
                    label["exit_time"] = float(params.entry_time) + float(life.get("minutes") or 0.0) * 60.0
            final_at = _lifecycle_final_at(label, params)
            if final_at is None or final_at > now:
                counts["lifecycles_pending"] += 1
            else:
                events.append((max(final_at, evaluated), 1, "life", (row_id, max(final_at, evaluated), None if resimulate else stored.get("lifecycle"))))
    for i, close in enumerate(closes):
        if close["closed_at"] > closes_after and close["closed_at"] <= now:
            events.append((close["closed_at"], 2, "close", i))
    events.sort(key=lambda e: (e[0], e[1]))
    by_id = {int(r["id"]): r for r in rows}

    def stage(row_id: int, until: float, lifecycle: dict | None) -> None:
        r = by_id[row_id]
        current = _json(conn.execute("SELECT markouts_json FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()[0])
        marks = {**_due_marks(_json(r.get("markouts_json")), float(r["evaluated_at"]), unit, until), **{k: v for k, v in current.items() if k == "lifecycle"}}
        life_json = str(r.get("lifecycle_json") or "")
        if lifecycle is not None:
            marks["lifecycle"] = lifecycle
        elif not resimulate and "lifecycle" not in marks:
            life_json = ""
        conn.execute("UPDATE adaptive_candidate_markouts SET markouts_json=?, lifecycle_json=?, resolved=0 WHERE id=?", (json.dumps(marks), life_json, row_id))

    staged: list[int] = []

    def flush(moment: float) -> None:
        if not staged:
            return
        conn.commit()
        al.resolve_markouts(scratch_db, lambda _sym, _ts: None, now=moment, path_low=low, bars_1m=bars)
        marks = ",".join("?" * len(staged))
        conn.execute(f"UPDATE adaptive_candidate_markouts SET resolved=1 WHERE id IN ({marks})", staged)
        for row_id in staged:
            conn.execute("UPDATE adaptive_candidate_markouts SET lifecycle_json=? WHERE id=?", (str(by_id[row_id].get("lifecycle_json") or ""), row_id))
        conn.commit()
        counts["labels"] += len(staged)
        staged.clear()

    current_t = None
    for t, _order, kind, payload in events:
        if current_t is not None and (t != current_t or kind not in {"life", "label"} or len(staged) >= 150):
            flush(current_t)
        current_t = t
        if kind == "fill":
            row_id, opp = payload
            conn.execute("UPDATE adaptive_candidate_markouts SET filled=1, opportunity_id=? WHERE id=?", (opp, row_id))
            conn.commit()
        elif kind in ("label", "life"):
            row_id = payload[0]
            stage(row_id, payload[1], payload[2] if kind == "life" else None)
            staged.append(row_id)
        else:
            close = closes[payload]
            counts["closes"] += 1
            counts["closes_current" if close["contract"] == "CURRENT" else "closes_retired"] += 1
            al.learn_from_close(
                scratch_db,
                engine=engine_id,
                symbol=close["symbol"],
                setup=close["setup"],
                regime=close["regime"],
                strategy_version=close["strategy_version"],
                net_pct=close["net"],
                mfe_pct=close["mfe"],
                mae_pct=close["mae"],
                hold_min=close["hold_min"],
                continuation=None,
                version_current=close["version_current"],
                is_dust=close["is_dust"],
                entered_at=close["entered_at"],
                now=close["closed_at"],
                opportunity_id=str(close.get("opportunity_id") or ""),
                exit_reason=close["exit_reason"],
                candidate_id=links.get(payload),
            )
    if current_t is not None:
        flush(current_t)
    conn.close()
    return counts


def _engine_state(db_path: str, engine_id: str) -> list[dict[str, Any]]:
    from backend.services import adaptive_learning as al

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?",
                (engine_id, al.current_economic_version(engine_id)),
            )
        ]
    finally:
        conn.close()


def decision_snapshot(db_path: str, engine_id: str, rows: list[dict[str, Any]], *, now: float) -> dict[tuple[str, str, str], dict[str, float]]:
    """What the next candidate of each recorded key would read: DAY market alpha,
    policy gap, policy value and size; SCALP market edge, policy gap, final
    executable policy value and size, from the row's own claim and features."""
    from backend.services import adaptive_learning as al
    from backend.services.scalp_v2 import executable_edge as edge_mod
    from backend.services.scalp_v2.raw_move_source import is_directional

    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        key = (str(r["symbol"]), str(r["setup"]), str(r["regime"]))
        if key in out:
            continue
        if engine_id == DAY_ENGINE:
            d = al.day_decision(db_path, *key, now=now)
            econ = d["economic"]
            out[key] = {"market_alpha": econ["market_alpha"], "policy_gap": econ["policy_gap"], "policy_value": econ["policy_value"], "size_mult": d["size_mult"], "risk": d["mae"]}
            continue
        if not is_directional(r.get("raw_move_source")):
            continue
        view = al.scalp_decision(db_path, *key, _json(r.get("features_json")), now=now)
        edge = edge_mod.scalp_executable_edge(view, raw_expected_move_pct=float(r.get("raw_expected_move") or 0.0), spread_pct=None, impact_pct=0.0, edge_source=str(r.get("raw_move_source") or ""))
        out[key] = {"market_edge": edge.market_edge_pct, "policy_gap": edge.policy_gap_pct, "policy_value": edge.final_executable_edge_pct, "size_mult": edge.size_mult, "risk": edge.risk_estimate_pct}
    return out


def compare_snapshots(before: dict, after: dict) -> dict[str, Any]:
    """Largest absolute difference per field over the keys both snapshots price."""
    keys = sorted(set(before) & set(after))
    fields = sorted({f for k in keys for f in before[k]})
    worst = {f: max((abs(float(after[k][f]) - float(before[k][f])) for k in keys), default=0.0) for f in fields}
    where = {f: "|".join(max(keys, key=lambda k, f=f: abs(float(after[k][f]) - float(before[k][f])))) for f in fields if keys}
    return {"keys": len(keys), "max_abs_diff": worst, "max_abs_diff_key": where}


def regenerate(
    db_path: str,
    engine_id: str,
    *,
    now: float | None = None,
    apply: bool = False,
    workdir: str | None = None,
    resimulate: bool = True,
) -> dict[str, Any]:
    """Rebuild ``engine_id``'s derived economic state from its authoritative rows.

    Dry run unless ``apply`` (run it with the portfolio engine stopped). Applied,
    the engine's current-version state other than ``preserved_metric`` rows and
    the SCALP micro model are replaced, and each candidate row is stamped with
    whether its labels are in the rebuilt state, so the live resolver learns only
    labels that are still pending. ``resimulate`` replays DAY lifecycle labels
    under the current exit policy instead of reading the stored label.
    """
    from backend.config.trading_economics import canonical_roundtrip_cost_pct
    from backend.services import adaptive_learning as al
    from backend.services.strategy_version import economic_anchor

    engine = str(engine_id or "").upper()
    moment = float(now if now is not None else time.time())
    version = al.current_economic_version(engine)
    out: dict[str, Any] = {"engine": engine, "economic_version": version, "now": moment, "applied": False}
    rows = _candidate_rows(db_path, engine)
    closes = realized_closes(db_path, engine)
    live_rows = [r for r in rows if REBUILT_MARK not in str(r.get("economic_json") or "")]
    cutoff = min((float(r["evaluated_at"]) for r in live_rows), default=moment)
    scratch_root = workdir or ("/dev/shm" if Path("/dev/shm").is_dir() else None)
    with tempfile.TemporaryDirectory(prefix="econ_regen_", dir=scratch_root) as tmp:
        scratch = f"{tmp}/{engine.lower()}.db"
        al._connect(scratch).close()
        closes_after = float("-inf")
        if engine == DAY_ENGINE:
            anchor = float(economic_anchor(DAY_ENGINE)["epoch"])
            store = BarStore(db_path, DAY_SYMBOLS, since=anchor - 10 * 86400)
            cands, _near = reconstruct_day_candidates(db_path, anchor, cutoff, store)
            tag_regimes(db_path, cands)
            boot = rebuild_day(cands, actual_closes(db_path, DAY_ENGINE, anchor, store), store, RepairedDayPolicy(scratch), roundtrip_cost=canonical_roundtrip_cost_pct(), now=cutoff)
            out["bootstrap"] = {"until": cutoff, **boot["counts"]}
            closes_after = cutoff
        out["replay"] = _replay(db_path, scratch, engine, rows, closes, now=moment, closes_after=closes_after, resimulate=resimulate)
        before = _engine_state(db_path, engine)
        after = _engine_state(scratch, engine)
        recent = rows[-400:]
        out["decisions"] = compare_snapshots(decision_snapshot(db_path, engine, recent, now=moment), decision_snapshot(scratch, engine, recent, now=moment))
        out["metrics"] = {
            m: {"before_n": round(sum(float(r["n"]) for r in before if r["metric"] == m), 3), "after_n": round(sum(float(r["n"]) for r in after if r["metric"] == m), 3)}
            for m in sorted({r["metric"] for r in before} | {r["metric"] for r in after})
            if not preserved_metric(m)
        }
        if not apply:
            return out
        sconn = sqlite3.connect(scratch)
        sconn.row_factory = sqlite3.Row
        try:
            flags = {int(r["id"]): r for r in sconn.execute("SELECT id, learned, lifecycle_learned, policy_learned, realized_net, markouts_json FROM adaptive_candidate_markouts")}
            model = sconn.execute("SELECT * FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine, al._model_key(engine, "micro_edge"))).fetchone()
        finally:
            sconn.close()
        conn = al._connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            for r in conn.execute("SELECT DISTINCT metric FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?", (engine, version)).fetchall():
                if not preserved_metric(r[0]):
                    conn.execute("DELETE FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND metric=?", (engine, version, r[0]))
            for r in after:
                if preserved_metric(r["metric"]):
                    continue
                conn.execute(
                    "INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (r["engine_id"], r["economic_version"], r["symbol"], r["setup"], r["regime"], r["metric"], r["n"], r["ewma"], r["m2"], r["updated_at"]),
                )
            if engine == SCALP_ENGINE:
                conn.execute("DELETE FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine, al._model_key(engine, "micro_edge")))
                if model is not None:
                    conn.execute(
                        "INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at) VALUES (?,?,?,?)", (model["engine_id"], model["model"], model["payload"], model["updated_at"])
                    )
            for r in rows:
                f = flags.get(int(r["id"]))
                if f is None:
                    continue
                marks = _json(r.get("markouts_json"))
                life = _json(f["markouts_json"]).get("lifecycle")
                if engine == DAY_ENGINE and resimulate and isinstance(life, dict):
                    marks["lifecycle"] = life
                conn.execute(
                    "UPDATE adaptive_candidate_markouts SET learned=?, lifecycle_learned=?, policy_learned=?, realized_net=COALESCE(?, realized_net), markouts_json=? WHERE id=?",
                    (int(f["learned"] or 0), int(f["lifecycle_learned"] or 0), int(f["policy_learned"] or 0), f["realized_net"], json.dumps(marks), int(r["id"])),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        out["applied"] = True
    return out


__all__ = [
    "PRESERVED_METRICS",
    "REBUILT_MARK",
    "actual_closes",
    "compare_snapshots",
    "decision_snapshot",
    "link_closes",
    "preserved_metric",
    "realized_closes",
    "rebuild",
    "rebuild_day",
    "rebuild_scalp",
    "regenerate",
]
