"""Deterministic, causal economic replay for DAY_V2 and SCALP_V2.

Reconstructs current-version decisions from stored raw market data and stored
decision-time inputs, then evaluates two learning policies on the same market
sequence. Every decision reads only state learned from labels that were final
before the decision time; a label becomes visible to a policy only at the time
its outcome was knowable (lifecycle exit, markout horizon or trade close).

DAY: every closed 15m bar since the economic anchor runs through the live
``evaluate_entry_signal``; each qualified signal is labeled by the live exit
contract (``day_v2.lifecycle_sim``) from the decision ask. Production entry
constraints are applied (one DAY position per symbol, consumed opportunities,
24h frequency caps, DAY slots). Policies differ only in rank and bounded size.

SCALP: the stored strategy claims (``adaptive_candidate_markouts`` joined to
``scalp_v2_decisions`` detail) are re-scored by each policy's executable edge;
admitted claims are labeled by the SCALP exit contract over 1m bars.
"""

from __future__ import annotations

import heapq
import importlib.util
import itertools
import json
import math
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

DAY_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
INTERVAL_SEC = {"1m": 60, "15m": 900, "1h": 3600, "4h": 14400}
BAR_LIMITS = {"15m": 60, "1h": 20, "4h": 15}
DAY_ENGINE = "DAY_V2"
SCALP_ENGINE = "SCALP_V2"


def _db_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("-", "").replace("USDT", "-USDT")


def _epoch(ts: Any) -> float | None:
    if ts is None:
        return None
    text = str(ts).replace("Z", "+00:00").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text.split("+")[0], fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def iso_epoch(value: str) -> float:
    d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


class BarStore:
    """In-memory closed bars per (symbol, interval), oldest first."""

    def __init__(self, db_path: str, symbols: Iterable[str] = DAY_SYMBOLS, since: float | None = None) -> None:
        self.bars: dict[tuple[str, str], list[dict[str, float]]] = {}
        self.epochs: dict[tuple[str, str], list[float]] = {}
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            lo = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if since else "0000"
            for sym in symbols:
                for interval in INTERVAL_SEC:
                    rows = conn.execute(
                        "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE symbol=? AND interval=? AND ts>=? ORDER BY ts",
                        (_db_symbol(sym), interval, lo),
                    ).fetchall()
                    out = []
                    for ts, o, h, low, c, v in rows:
                        e = _epoch(ts)
                        if e is None or None in (o, h, low, c, v):
                            continue
                        out.append({"ts": ts, "ts_epoch": e, "open": float(o), "high": float(h), "low": float(low), "close": float(c), "volume": float(v)})
                    self.bars[(sym, interval)] = out
                    self.epochs[(sym, interval)] = [b["ts_epoch"] for b in out]
        finally:
            conn.close()

    def closed(self, symbol: str, interval: str, limit: int, as_of: float) -> list[dict[str, float]]:
        """Same contract as candle_contract.load_closed_bars: closed bars only, newest ``limit``."""
        import bisect

        sec = INTERVAL_SEC[interval]
        epochs = self.epochs.get((symbol, interval), [])
        hi = bisect.bisect_right(epochs, as_of + 2 - sec)
        return self.bars[(symbol, interval)][max(0, hi - limit) : hi]

    def minute_path(self, symbol: str, start: float, end: float) -> list[tuple[float, float, float, float, float]]:
        import bisect

        epochs = self.epochs.get((symbol, "1m"), [])
        lo = bisect.bisect_left(epochs, math.floor(start / 60.0) * 60.0)
        hi = bisect.bisect_left(epochs, end)
        return [(b["ts_epoch"], b["open"], b["high"], b["low"], b["close"]) for b in self.bars[(symbol, "1m")][lo:hi]]

    def close_at(self, symbol: str, epoch: float) -> float | None:
        """Close of the 1m bar containing ``epoch`` (same rule as adaptive_learning.ohlcv_quote)."""
        import bisect

        epochs = self.epochs.get((symbol, "1m"), [])
        i = bisect.bisect_right(epochs, epoch) - 1
        if i < 0:
            return None
        bar = self.bars[(symbol, "1m")][i]
        return bar["close"] if bar["ts_epoch"] <= epoch < bar["ts_epoch"] + 62 else None


@dataclass
class DayCandidate:
    symbol: str
    decided_at: float
    setup: str
    regime_signal: str
    ask: float
    signal: Any
    recorded_ask: bool
    lifecycle: dict[str, Any] = field(default_factory=dict)
    regime_tag: str = ""

    @property
    def opportunity_id(self) -> str:
        return str(getattr(self.signal, "opportunity_id", "") or f"{self.symbol}:{self.setup}:{self.decided_at}")


def _recorded_day_asks(db_path: str, since: float) -> dict[tuple[str, int], tuple[float, str]]:
    """(symbol, 15m bar) -> (decision ask, regime tag) for DAY candidates the live system recorded."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT symbol, evaluated_at, ref_price, regime FROM adaptive_candidate_markouts WHERE engine_id='DAY_V2' AND signaled=1 AND evaluated_at>=?",
            (since,),
        ).fetchall()
    finally:
        conn.close()
    return {(str(s), int(float(t) // 900)): (float(p), str(r or "")) for s, t, p, r in rows}


def reconstruct_day_candidates(db_path: str, since: float, until: float, store: BarStore, *, half_spread: float = 0.00006) -> tuple[list[DayCandidate], list[dict[str, Any]]]:
    """Qualified DAY signals on every closed 15m bar in [since, until), plus near-qualified states.

    The ask is the live recorded decision ask when the live system recorded the
    candidate, else the signal-bar close plus the canonical half-spread estimate.
    """
    from backend.services.day_v2.live_signal import ENABLED_SETUPS, evaluate_entry_signal, explain_no_signal

    recorded = _recorded_day_asks(db_path, since)
    out: list[DayCandidate] = []
    near: list[dict[str, Any]] = []
    t = math.ceil(since / 900.0) * 900.0
    while t < until:
        for sym in DAY_SYMBOLS:
            b15 = store.closed(sym, "15m", BAR_LIMITS["15m"], t)
            if len(b15) < 32 or b15[-1]["ts_epoch"] + 900 < t - 1:
                continue
            b1h = store.closed(sym, "1h", BAR_LIMITS["1h"], t)
            b4h = store.closed(sym, "4h", BAR_LIMITS["4h"], t)
            sig = evaluate_entry_signal(sym, b15, b1h, b4h)
            key = (sym, int(t // 900))
            if sig is not None:
                rec = recorded.get(key)
                ask = rec[0] if rec else float(b15[-1]["close"]) * (1.0 + half_spread)
                out.append(DayCandidate(sym, t, sig.setup, sig.regime, ask, sig, rec is not None, regime_tag=rec[1] if rec else ""))
            else:
                exp = explain_no_signal(sym, b15, b1h, b4h)
                unmet = list(exp.get("unmet") or [])
                closest = str(exp.get("closest") or "")
                if len(unmet) == 1 and closest in ENABLED_SETUPS:
                    near.append({"symbol": sym, "decided_at": t, "setup": closest, "unmet": unmet[0], "ref": float(b15[-1]["close"]) * (1.0 + half_spread)})
        t += 900.0
    return out, near


def label_day_lifecycles(cands: list[DayCandidate], store: BarStore, *, roundtrip_cost: float, now: float, adaptive_for: Callable[[DayCandidate], dict] | None = None) -> None:
    from backend.services.day_v2.lifecycle_sim import DAY_LIFECYCLE_MAX_MIN, LifecycleParams, simulate_lifecycle

    for c in cands:
        params = LifecycleParams.from_signal(c.signal, entry_price=c.ask, entry_time=c.decided_at, adaptive=adaptive_for(c) if adaptive_for else None)
        path = store.minute_path(c.symbol, c.decided_at, c.decided_at + DAY_LIFECYCLE_MAX_MIN * 60 + 60)
        c.lifecycle = simulate_lifecycle(params, path, roundtrip_cost=roundtrip_cost, now=now)


def tag_regimes(db_path: str, cands: list[DayCandidate]) -> None:
    """Regime tag at each decision, from bars opened before it, unless the live system recorded one."""
    from backend.services.adaptive_learning import market_regime_tag

    cache: dict[tuple[str, float], str] = {}
    for c in cands:
        if c.regime_tag:
            continue
        key = (c.symbol, c.decided_at)
        if key not in cache:
            cache[key] = market_regime_tag(db_path, c.symbol, as_of=c.decided_at) or ""
        c.regime_tag = cache[key]


def day_lifecycle(c: DayCandidate, adaptive: dict | None, store: BarStore, *, roundtrip_cost: float, now: float) -> dict[str, Any]:
    """Live DAY exit contract from the decision ask, with the deciding policy's multipliers."""
    from backend.services.adaptive_learning import continuation_ratio
    from backend.services.day_v2.lifecycle_sim import DAY_LIFECYCLE_MAX_MIN, LifecycleParams, simulate_lifecycle
    from backend.services.day_v2.winner_contract import objective_level

    params = LifecycleParams.from_signal(c.signal, entry_price=c.ask, entry_time=c.decided_at, adaptive=adaptive)
    path = store.minute_path(c.symbol, c.decided_at, c.decided_at + DAY_LIFECYCLE_MAX_MIN * 60 + 60)
    label = simulate_lifecycle(params, path, roundtrip_cost=roundtrip_cost, now=now)
    if label.get("final") and label.get("net") is not None:
        objective = objective_level(
            c.setup,
            c.ask,
            params.atr_1h,
            params.objective_structural,
            atr_mult=params.objective_atr_mult,
            structural_emphasis=params.structural_emphasis,
        )
        label["continuation"] = continuation_ratio(entry_price=c.ask, highest_price=c.ask * (1.0 + float(label["mfe"])), objective=objective)
    return label


# --------------------------------------------------------------------------- policies


def load_baseline_modules(sha: str, repo: str | Path) -> dict[str, ModuleType]:
    """The learner, DAY ranking and SCALP executable edge exactly as committed at ``sha``.

    Loaded as separate modules so the baseline and the repaired code run side by
    side on the same market sequence. Each keeps its own state in its own DB.
    """
    out: dict[str, ModuleType] = {}
    with tempfile.TemporaryDirectory(prefix=f"baseline_{sha}_") as tmp:
        for name, rel in (
            ("adaptive_learning", "backend/services/adaptive_learning.py"),
            ("ranking", "backend/services/day_v2/ranking.py"),
            ("executable_edge", "backend/services/scalp_v2/executable_edge.py"),
        ):
            src = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{rel}"], capture_output=True, text=True, check=True).stdout
            path = Path(tmp) / f"baseline_{name}.py"
            path.write_text(src)
            spec = importlib.util.spec_from_file_location(f"baseline_{sha}_{name}", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load baseline {rel}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            out[name] = module
    return out


class RepairedDayPolicy:
    """The repaired learner: hierarchical expected net over realized closes and
    lifecycle replays of unfilled qualified candidates; rank and size from it."""

    name = "repaired"
    records_blocked = True
    learns_lifecycle = True

    def __init__(self, db_path: str) -> None:
        from backend.services import adaptive_learning as al

        self.al = al
        self.db = db_path
        self.version = al.current_strategy_version(DAY_ENGINE)

    def decide(self, c: DayCandidate, now: float) -> dict[str, Any]:
        return self.al.day_decision(self.db, c.symbol, c.setup, c.regime_tag, now=now)

    def rank(self, rows: list[dict[str, Any]], cost: float) -> list[dict[str, Any]]:
        from backend.services.day_v2.ranking import rank_day_candidates

        return rank_day_candidates(rows, list(DAY_SYMBOLS), cost)

    def forward_horizon_min(self, adaptive: dict[str, Any]) -> float | None:
        return None

    def learn_close(self, c: DayCandidate, label: dict[str, Any], at: float) -> None:
        self.al.learn_from_close(
            self.db,
            engine=DAY_ENGINE,
            symbol=c.symbol,
            setup=c.setup,
            regime=c.regime_tag,
            strategy_version=self.version,
            net_pct=label["net"],
            mfe_pct=label.get("mfe"),
            mae_pct=label.get("mae"),
            hold_min=label.get("minutes"),
            continuation=label.get("continuation"),
            version_current=True,
            is_dust=False,
            now=at,
        )

    def learn_lifecycle(self, c: DayCandidate, label: dict[str, Any], at: float) -> None:
        self.al.observe(self.db, engine=DAY_ENGINE, symbol=c.symbol, setup=c.setup, regime=c.regime_tag, metric="lifecycle_net", value=label["net"], strategy_version=self.version, now=at)

    def learn_forward(self, c: DayCandidate, value: float, at: float) -> None:
        return None


class BaselineDayPolicy:
    """The committed baseline learner: single-level shrinkage over realized trade
    net and the fixed-horizon forward markout of every qualified record, rank =
    objective edge + learned net, size from the learned net per unit of MAE."""

    name = "current"
    records_blocked = False
    learns_lifecycle = False

    def __init__(self, db_path: str, modules: dict[str, ModuleType]) -> None:
        self.al = modules["adaptive_learning"]
        self.ranking = modules["ranking"]
        self.db = db_path
        self.version = self.al.current_strategy_version(DAY_ENGINE)

    def decide(self, c: DayCandidate, now: float) -> dict[str, Any]:
        return self.al.day_decision(self.db, c.symbol, c.setup, c.regime_tag)

    def rank(self, rows: list[dict[str, Any]], cost: float) -> list[dict[str, Any]]:
        return self.ranking.rank_day_candidates(rows, list(DAY_SYMBOLS), cost)

    def forward_horizon_min(self, adaptive: dict[str, Any]) -> float | None:
        return self.al._snap_horizon(float(adaptive.get("time_to_mfe_min") or 60.0), self.al.DAY_HORIZONS_MIN)

    def learn_close(self, c: DayCandidate, label: dict[str, Any], at: float) -> None:
        mae = label.get("mae")
        for metric, value in (
            ("trade_net", label["net"]),
            ("trade_mfe", label.get("mfe")),
            ("trade_mae", abs(mae) if mae is not None else None),
            ("trade_time_to_mfe_min", label.get("minutes")),
            ("trade_continuation", label.get("continuation")),
        ):
            if value is not None:
                self.al.observe(self.db, engine=DAY_ENGINE, symbol=c.symbol, setup=c.setup, regime=c.regime_tag, metric=metric, value=float(value), strategy_version=self.version, now=at)

    def learn_lifecycle(self, c: DayCandidate, label: dict[str, Any], at: float) -> None:
        return None

    def learn_forward(self, c: DayCandidate, value: float, at: float) -> None:
        self.al.observe(self.db, engine=DAY_ENGINE, symbol=c.symbol, setup=c.setup, regime=c.regime_tag, metric="markout_forward", value=value, strategy_version=self.version, now=at)


# --------------------------------------------------------------------------- DAY simulation


def simulate_day(
    cands: list[DayCandidate],
    store: BarStore,
    policy: Any,
    *,
    roundtrip_cost: float,
    now: float,
    max_slots: int = 4,
    freq_symbol: int = 2,
    freq_total: int = 8,
) -> dict[str, Any]:
    """Run one policy over the reconstructed DAY candidate sequence.

    Production entry constraints: a held symbol is not evaluated, a filled
    opportunity is consumed, rolling-24h caps (``freq_symbol`` per symbol,
    ``freq_total`` overall) block, and at most ``max_slots`` DAY positions are
    open. Candidates are funded in the policy's rank order; each fill is held
    until its lifecycle exit. Every label reaches the policy at the time its
    outcome was knowable, never earlier.
    """
    by_t: dict[float, list[DayCandidate]] = defaultdict(list)
    for c in cands:
        by_t[c.decided_at].append(c)
    heap: list[tuple[float, int, Callable[[], None]]] = []
    seq = itertools.count()
    open_until: dict[str, float] = {}
    consumed: dict[str, float] = {}
    fills: list[tuple[float, str]] = []
    first_record: set[str] = set()
    trades: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    def push(at: float, fn: Callable[[], None]) -> None:
        heapq.heappush(heap, (at, next(seq), fn))

    def lifecycle_learn(c: DayCandidate, label: dict[str, Any]) -> Callable[[], None]:
        def run() -> None:
            filled_at = consumed.get(c.opportunity_id)
            if filled_at is None or filled_at > float(label["exit_time"]):
                policy.learn_lifecycle(c, label, float(label["exit_time"]))

        return run

    for t in sorted(by_t):
        while heap and heap[0][0] <= t:
            heapq.heappop(heap)[2]()
        live = [c for c in by_t[t] if open_until.get(c.symbol, 0.0) <= t and c.opportunity_id not in consumed]
        if not live:
            continue
        recent = [f for f in fills if t - 86400.0 < f[0] <= t]
        ok, blocked = [], []
        for c in live:
            capped = sum(1 for f in recent if f[1] == c.symbol) >= freq_symbol or len(recent) >= freq_total
            (blocked if capped else ok).append(c)
        decisions = {id(c): policy.decide(c, t) for c in live}
        ranked = policy.rank([{"symbol": c.symbol, "signal": c.signal, "ask_price": c.ask, "adaptive": decisions[id(c)], "_c": c} for c in ok], roundtrip_cost) if ok else []
        open_count = sum(1 for v in open_until.values() if v > t)
        recorded = [(row["_c"], row.get("rank") or {}, "QUALIFIED") for row in ranked]
        if policy.records_blocked:
            recorded += [(c, {}, "QUALIFIED_BLOCKED") for c in blocked]
        for c, rank, state in recorded:
            adaptive = decisions[id(c)]
            label = day_lifecycle(c, adaptive, store, roundtrip_cost=roundtrip_cost, now=now)
            final = bool(label.get("final")) and label.get("net") is not None
            filled = state == "QUALIFIED" and open_count < max_slots and final
            size = float(adaptive.get("size_mult") or 1.0)
            records.append(
                {
                    "symbol": c.symbol,
                    "setup": c.setup,
                    "t": t,
                    "state": state,
                    "filled": filled,
                    "expected_net": float(adaptive.get("expected_net") or 0.0),
                    "size": size,
                    "net": label.get("net") if final else None,
                    "score": rank.get("score"),
                }
            )
            if filled:
                open_count += 1
                consumed[c.opportunity_id] = t
                fills.append((t, c.symbol))
                open_until[c.symbol] = float(label["exit_time"])
                trades.append(
                    {
                        "symbol": c.symbol,
                        "setup": c.setup,
                        "regime": c.regime_tag,
                        "entry": t,
                        "exit": float(label["exit_time"]),
                        "net": float(label["net"]),
                        "size": size,
                        "reason": label.get("reason"),
                        "mfe": label.get("mfe"),
                        "mae": label.get("mae"),
                        "expected_net": float(adaptive.get("expected_net") or 0.0),
                    }
                )
                push(float(label["exit_time"]), lambda c=c, label=label: policy.learn_close(c, label, float(label["exit_time"])))
            elif policy.learns_lifecycle and final and c.opportunity_id not in first_record:
                push(float(label["exit_time"]), lifecycle_learn(c, label))
            first_record.add(c.opportunity_id)
            horizon = policy.forward_horizon_min(adaptive)
            if horizon is not None:
                price = store.close_at(c.symbol, t + horizon * 60.0)
                if price is not None and c.ask > 0:
                    value = (price - c.ask) / c.ask - roundtrip_cost
                    push(t + horizon * 60.0, lambda c=c, value=value, at=t + horizon * 60.0: policy.learn_forward(c, value, at))
    while heap:
        heapq.heappop(heap)[2]()
    return {"trades": trades, "records": records}


def day_report(trades: list[dict[str, Any]], folds: list[tuple[float, float]]) -> dict[str, Any]:
    """Size-weighted P&L in base-slot units plus per-setup and per-fold breakdowns."""

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows = sorted(rows, key=lambda r: r["entry"])
        out = summarize_returns([r["net"] for r in rows], [r["size"] for r in rows])
        out["unweighted_mean_net_bps"] = statistics.fmean([r["net"] for r in rows]) * 1e4 if rows else 0.0
        out["avg_size"] = statistics.fmean([r["size"] for r in rows]) if rows else 0.0
        out["false_entry_rate"] = sum(1 for r in rows if r["net"] <= 0) / len(rows) if rows else 0.0
        out["never_developed_rate"] = sum(1 for r in rows if float(r.get("mfe") or 0.0) < 0.00066) / len(rows) if rows else 0.0
        out["avg_expected_net_bps"] = statistics.fmean([r["expected_net"] for r in rows]) * 1e4 if rows else 0.0
        return out

    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in trades:
        by_setup[r["setup"]].append(r)
    return {
        "all": block(trades),
        "after_first_fold": block([r for r in trades if folds and r["entry"] >= folds[0][1]]),
        "by_setup": {k: block(v) for k, v in sorted(by_setup.items())},
        "folds": [block([r for r in trades if lo <= r["entry"] < hi]) for lo, hi in folds],
    }


# --------------------------------------------------------------------------- SCALP


def unfloored_claim(c: dict[str, Any], store: BarStore) -> float | None:
    """The claim the current strategies make for a row recorded when VWAP reclaim
    floored its projection at 12 bps and range bounce claimed recovery + 8 bps.

    Exact where the recorded value was the projection itself (above the floor);
    otherwise the projection rebuilt from the 15 1m bars closed by the decision,
    priced at the decision ask, and bounded by the recorded value. None when the
    bars are missing.
    """
    import bisect

    from backend.services.binance_scalp.strategies.common import directional_claim_pct
    from backend.services.binance_scalp.strategies.range_bounce_scalp import bounce_projection
    from backend.services.binance_scalp.strategies.vwap_ema_reclaim import _REACH_MIN_PCT, reclaim_projection

    recorded, setup, cur = float(c["raw"]), str(c["setup"]).upper(), float(c["ref"])
    if setup == "VWAP_EMA_RECLAIM" and recorded > _REACH_MIN_PCT + 1e-9:
        return recorded
    if setup not in ("VWAP_EMA_RECLAIM", "RANGE_BOUNCE_SCALP"):
        return recorded
    key = (c["symbol"], "1m")
    hi = bisect.bisect_right(store.epochs.get(key, []), float(c["t"]) - 60.0)
    bars = store.bars.get(key, [])[max(0, hi - 15) : hi]
    if setup == "VWAP_EMA_RECLAIM":
        return min(directional_claim_pct(reclaim_projection(bars, cur)), recorded) if len(bars) >= 15 else None
    if len(bars) < 10:
        return None
    to_high = directional_claim_pct(bounce_projection(bars, cur))
    return recorded if abs(to_high - recorded) < 5e-5 else min(to_high, recorded)


def load_scalp_rows(db_path: str, since: float, until: float, strategy_version: str, *, store: BarStore | None = None, unfloor_unless_version: str | None = None) -> list[dict[str, Any]]:
    """Recorded SCALP candidates in [since, until), oldest first, with decision-time inputs.

    Every row with a strategy-claim source is a claim, a 0 projection included.
    With ``store``, claim rows recorded under any economic version other than
    ``unfloor_unless_version`` carry ``unfloored_claim`` as ``raw`` (``raw_recorded``
    keeps the stored value; ``unfloor_missing`` flags a row without bars, priced at 0).
    """
    from backend.services.scalp_v2.raw_move_source import is_directional, normalize_raw_move_source

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND strategy_version=? AND evaluated_at>=? AND evaluated_at<? ORDER BY evaluated_at, id",
            (SCALP_ENGINE, strategy_version, since, until),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        try:
            feats = json.loads(r["features_json"] or "{}")
        except (TypeError, ValueError):
            feats = {}
        raw = max(0.0, float(r["raw_expected_move"])) if r["raw_expected_move"] is not None else 0.0
        source = normalize_raw_move_source(r["raw_move_source"]) if r["raw_move_source"] else "NONE"
        spread = feats.get("spread_pct") if isinstance(feats, dict) else None
        c = {
            "id": int(r["id"]),
            "symbol": str(r["symbol"]),
            "setup": str(r["setup"]),
            "regime": str(r["regime"]),
            "t": float(r["evaluated_at"]),
            "ref": float(r["ref_price"]),
            "cost": float(r["roundtrip_cost"] or 0.0),
            "signaled": bool(r["signaled"]),
            "raw": raw,
            "raw_recorded": raw,
            "unfloor_missing": False,
            "source": source,
            "directional": is_directional(source),
            "features": feats if isinstance(feats, dict) else {},
            "spread": float(spread) if spread is not None else None,
        }
        if store is not None and c["directional"] and str(r["economic_version"] or "") != str(unfloor_unless_version or ""):
            claim = unfloored_claim(c, store)
            c["unfloor_missing"] = claim is None
            c["raw"] = 0.0 if claim is None else claim
        out.append(c)
    return out


def scalp_exit_sim(c: dict[str, Any], view: dict[str, Any], store: BarStore) -> dict[str, Any] | None:
    """SCALP exit contract on 1m bars from the decision ask: adverse stop (learned
    distance, never wider than the contract bound), net target, horizon.

    Bars opening after the decision only. Within one bar the stop is assumed to
    trade before the target (conservative). Net is in the markout units.
    """
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
    from backend.services.scalp_v2.exit_calibration import scalp_v2_max_adverse_net_pct
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN

    entry, t0, cost = c["ref"], c["t"], c["cost"]
    if entry <= 0:
        return None
    contract = float(scalp_v2_max_adverse_net_pct(c["symbol"]))
    risk = float(view.get("risk_estimate") or 0.0)
    adverse = min(contract, risk + float(ESTIMATED_ROUNDTRIP_COST)) if risk > 0 else contract
    target = float(view.get("target_pct") or 0.0)
    hold = min(float(SCALP_V2_TIME_STOP_MIN), max(1.0, float(view.get("hold_min") or SCALP_V2_TIME_STOP_MIN)))
    end = t0 + hold * 60.0
    last = None
    for t_open, _o, high, low, close in store.minute_path(c["symbol"], t0, end + 60.0):
        if t_open < t0:
            continue
        if (low - entry) / entry - cost <= -adverse:
            return {"net": -adverse, "exit": t_open + 60.0, "reason": "ADVERSE_STOP"}
        if target > 0 and (high - entry) / entry - cost >= target:
            return {"net": target, "exit": t_open + 60.0, "reason": "NET_PROFIT"}
        last = (t_open + 60.0, (close - entry) / entry - cost)
        if t_open + 60.0 >= end:
            return {"net": last[1], "exit": last[0], "reason": "TIME_STOP"}
    if last is None:
        return None
    return {"net": last[1], "exit": last[0], "reason": "TIME_STOP_PARTIAL"}


class ScalpPolicy:
    """One SCALP learner version: its own scalp_decision, executable edge,
    record_candidate and resolve_markouts, writing to its own state DB."""

    def __init__(self, name: str, db_path: str, learner: ModuleType, edge_module: ModuleType, store: BarStore, *, full_api: bool | None = None) -> None:
        self.name = name
        self.al = learner
        self.edge_mod = edge_module
        self.db = db_path
        self.store = store
        # full_api: the learner takes ``now=`` and persists the decision-time economics.
        self.repaired = (name == "repaired") if full_api is None else bool(full_api)

    def view(self, c: dict[str, Any], now: float) -> dict[str, Any]:
        if self.repaired:
            return self.al.scalp_decision(self.db, c["symbol"], c["setup"], c["regime"], c["features"], now=now)
        return self.al.scalp_decision(self.db, c["symbol"], c["setup"], c["regime"], c["features"])

    def edge(self, view: dict[str, Any], c: dict[str, Any]) -> Any:
        return self.edge_mod.scalp_executable_edge(view, raw_expected_move_pct=c["raw"], spread_pct=c["spread"], impact_pct=0.0, edge_source=c["source"])

    def record(self, c: dict[str, Any], economic: dict[str, Any] | None) -> int | None:
        kwargs: dict[str, Any] = {
            "engine": SCALP_ENGINE,
            "symbol": c["symbol"],
            "setup": c["setup"],
            "regime": c["regime"],
            "ref_price": c["ref"],
            "roundtrip_cost": c["cost"],
            "signaled": c["signaled"],
            "evaluated_at": c["t"],
            "features": c["features"],
            "raw_expected_move": c["raw"] if c["directional"] else (c["raw"] or None),
            "raw_move_source": c["source"],
        }
        if self.repaired:
            kwargs["economic"] = economic
        return self.al.record_candidate(self.db, **kwargs)

    def resolve(self, now: float) -> None:
        def low(sym: str, a: float, b: float) -> float | None:
            lows = [bar[3] for bar in self.store.minute_path(sym, a, b) if bar[0] >= a]
            return min(lows) if lows else None

        while self.al.resolve_markouts(self.db, self.store.close_at, now=now, path_low=low) > 0:
            continue


def simulate_scalp(rows: list[dict[str, Any]], policy: ScalpPolicy, *, now: float, events: list[tuple[float, Callable[[], None]]] | None = None) -> dict[str, Any]:
    """Re-score every recorded SCALP claim with ``policy`` at its decision time.

    Every candidate is recorded into the policy's own state and resolved at its
    horizon exactly as live, so admitted and rejected claims both teach. A claim
    with final executable edge > 0 is a trade (one per symbol at a time), exited
    by ``scalp_exit_sim``. ``events`` are (time, callback) learning updates
    applied once their time has passed (realized closes).
    """
    from backend.services.adaptive_learning import _mark_at

    open_until: dict[str, float] = {}
    trades: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    scratch_id: dict[int, int | None] = {}
    pending = sorted(events or [], key=lambda e: e[0])
    for c in rows:
        while pending and pending[0][0] <= c["t"]:
            pending.pop(0)[1]()
        policy.resolve(c["t"])
        economic = None
        if c["directional"]:
            view = policy.view(c, c["t"])
            edge = policy.edge(view, c)
            final = float(edge.final_executable_edge_pct)
            if policy.repaired:
                economic = edge.as_dict().get("economic")
            preds.append(
                {
                    "id": c["id"],
                    "t": c["t"],
                    "symbol": c["symbol"],
                    "setup": c["setup"],
                    "regime": c["regime"],
                    "raw": c["raw"],
                    "cost": c["cost"],
                    "final": final,
                    "base": float(edge.base_executable_edge_pct),
                    "micro": float(edge.micro_residual_pct),
                    "micro_model": float(edge.micro_residual_model_pct),
                    "admitted": final > 0,
                }
            )
            if final > 0 and open_until.get(c["symbol"], 0.0) <= c["t"]:
                out = scalp_exit_sim(c, view, policy.store)
                if out is not None:
                    open_until[c["symbol"]] = float(out["exit"])
                    trades.append(
                        {
                            "symbol": c["symbol"],
                            "setup": c["setup"],
                            "entry": c["t"],
                            "net": float(out["net"]),
                            "cost": c["cost"],
                            "size": float(edge.size_mult),
                            "reason": out["reason"],
                            "expected_net": final,
                            "mfe": None,
                        }
                    )
        scratch_id[c["id"]] = policy.record(c, economic)
    for at, fn in pending:
        if at <= now:
            fn()
    policy.resolve(now)
    labels: dict[int, tuple[float | None, float | None]] = {}
    conn = sqlite3.connect(policy.db)
    conn.row_factory = sqlite3.Row
    try:
        for r in conn.execute("SELECT id, markouts_json, label_horizon FROM adaptive_candidate_markouts"):
            marks = json.loads(r["markouts_json"] or "{}")
            labels[int(r["id"])] = (_mark_at(marks, float(r["label_horizon"] or 600.0)), _mark_at(marks, 600.0))
    finally:
        conn.close()
    for p in preds:
        sid = scratch_id.get(p["id"])
        own, common = labels.get(sid, (None, None)) if sid is not None else (None, None)
        p["label_own_horizon"] = own
        p["label"] = common
    return {"trades": trades, "predictions": preds, "scratch_ids": scratch_id}


def micro_value(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Incremental value of the micro term on resolved claims: squared error of
    the final edge with and without it, and admissions it changed."""
    rows = [p for p in preds if p.get("label") is not None]
    if not rows:
        return {"claims": 0}
    with_micro = statistics.fmean([(p["final"] - p["label"]) ** 2 for p in rows])
    without = statistics.fmean([(p["final"] - p["micro"] - p["label"]) ** 2 for p in rows])
    flipped = [p for p in rows if (p["final"] > 0) != (p["final"] - p["micro"] > 0)]
    return {
        "claims": len(rows),
        "rmse_bps_with_micro": math.sqrt(with_micro) * 1e4,
        "rmse_bps_without_micro": math.sqrt(without) * 1e4,
        "mean_abs_micro_bps": statistics.fmean([abs(p["micro"]) for p in rows]) * 1e4,
        "admissions_changed": len(flipped),
        "label_bps_of_changed": [round(p["label"] * 1e4, 2) for p in flipped],
    }


def _avg_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def discrimination(preds: list[dict[str, Any]], label_key: str = "label") -> dict[str, Any]:
    """How well ``final`` orders claims by realized ``label_key``, independent of
    how many are admitted: tie-corrected Spearman correlation, AUC for a positive
    label, and the mean realized label of the top fifth by ``final``."""
    rows = [p for p in preds if p.get(label_key) is not None]
    if len(rows) < 3:
        return {"claims": len(rows)}
    finals = [p["final"] for p in rows]
    labels = [p[label_key] for p in rows]
    rf, rl = _avg_ranks(finals), _avg_ranks(labels)
    mf, ml = statistics.fmean(rf), statistics.fmean(rl)
    cov = sum((a - mf) * (b - ml) for a, b in zip(rf, rl, strict=True))
    den = math.sqrt(sum((a - mf) ** 2 for a in rf) * sum((b - ml) ** 2 for b in rl))
    pos = [r for r, lab in zip(rf, labels, strict=True) if lab > 0]
    n_pos, n_neg = len(pos), len(rows) - len(pos)
    auc = (sum(pos) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg) if n_pos and n_neg else None
    top = sorted(rows, key=lambda p: p["final"], reverse=True)[: max(1, len(rows) // 5)]
    return {
        "claims": len(rows),
        "spearman": cov / den if den > 0 else 0.0,
        "auc_positive": auc,
        "top_fifth_label_bps": statistics.fmean([p[label_key] for p in top]) * 1e4,
        "all_label_bps": statistics.fmean(labels) * 1e4,
    }


def scalp_report(result: dict[str, Any], folds: list[tuple[float, float]]) -> dict[str, Any]:
    preds = [p for p in result["predictions"] if p.get("label") is not None]
    admitted = [p for p in preds if p["admitted"]]

    def bias(rows: list[dict[str, Any]]) -> float | None:
        return statistics.fmean([p["final"] - p["label"] for p in rows]) * 1e4 if rows else None

    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        return statistics.fmean([p[key] for p in rows]) * 1e4 if rows else None

    trades = sorted(result["trades"], key=lambda r: r["entry"])
    later = [t for t in trades if folds and t["entry"] >= folds[0][1]]
    later_preds = [p for p in preds if folds and p["t"] >= folds[0][1]]
    sized = [(t["net"], t.get("cost", 0.0), t["size"]) for t in trades]
    out = {
        "claims_scored": len(preds),
        "claims_admitted": len(admitted),
        "prediction_bias_bps_all": bias(preds),
        "prediction_bias_bps_admitted": bias(admitted),
        "predicted_net_bps_admitted": mean(admitted, "final"),
        "realized_label_bps_admitted": mean(admitted, "label"),
        "realized_label_bps_all": mean(preds, "label"),
        "predicted_gross_bps_all": statistics.fmean([p["final"] + p.get("cost", 0.0) for p in preds]) * 1e4 if preds else None,
        "realized_gross_bps_all": statistics.fmean([p["label"] + p.get("cost", 0.0) for p in preds]) * 1e4 if preds else None,
        "trade_gross": sum((n + c) * s for n, c, s in sized),
        "trade_cost": sum(c * s for _n, c, s in sized),
        "after_first_fold": {
            "claims_scored": len(later_preds),
            "claims_admitted": sum(1 for p in later_preds if p["admitted"]),
            "prediction_bias_bps_all": bias(later_preds),
            "prediction_bias_bps_admitted": bias([p for p in later_preds if p["admitted"]]),
            "realized_label_bps_admitted": mean([p for p in later_preds if p["admitted"]], "label"),
            "trades": summarize_returns([t["net"] for t in later], [t["size"] for t in later]),
        },
        "trades": summarize_returns([t["net"] for t in trades], [t["size"] for t in trades]),
        "folds": [summarize_returns([t["net"] for t in trades if lo <= t["entry"] < hi], [t["size"] for t in trades if lo <= t["entry"] < hi]) for lo, hi in folds],
        "fold_bias_bps": [bias([p for p in preds if lo <= p["t"] < hi]) for lo, hi in folds],
        "rmse_bps_all": math.sqrt(statistics.fmean([(p["final"] - p["label"]) ** 2 for p in preds])) * 1e4 if preds else None,
        "discrimination": discrimination(preds, "label"),
        "discrimination_own_horizon": discrimination(result["predictions"], "label_own_horizon"),
        "fold_discrimination": [discrimination([p for p in preds if lo <= p["t"] < hi], "label") for lo, hi in folds],
        "micro": micro_value(preds),
    }
    out["trades"]["unweighted_mean_net_bps"] = statistics.fmean([t["net"] for t in trades]) * 1e4 if trades else 0.0
    return out


def chronological_folds(start: float, end: float, n: int = 4) -> list[tuple[float, float]]:
    step = (end - start) / n
    return [(start + i * step, start + (i + 1) * step if i < n - 1 else end + 1.0) for i in range(n)]


def summarize_returns(values: list[float], weights: list[float] | None = None) -> dict[str, Any]:
    """Trade count, net, PF, max drawdown and mean/median net for a return sequence (fractions)."""
    w = weights if weights is not None else [1.0] * len(values)
    pnl = [v * x for v, x in zip(values, w, strict=True)]
    gains = sum(p for p in pnl if p > 0)
    losses = -sum(p for p in pnl if p < 0)
    peak = cum = dd = 0.0
    for p in pnl:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {
        "trades": len(values),
        "net": cum,
        "pf": (gains / losses) if losses > 0 else (math.inf if gains > 0 else 0.0),
        "max_drawdown": dd,
        "mean_net_bps": statistics.fmean(values) * 1e4 if values else 0.0,
        "median_net_bps": statistics.median(values) * 1e4 if values else 0.0,
        "win_rate": sum(1 for v in values if v > 0) / len(values) if values else 0.0,
    }


def run_replay(db_path: str, *, baseline_sha: str, repo: str | Path, now: float | None = None, folds: int = 4) -> dict[str, Any]:
    """CURRENT (``baseline_sha``) vs REPAIRED on the same post-anchor market sequence.

    Both start with empty state at each engine's economic anchor and learn
    causally. DAY outcomes are lifecycle labels of the funded candidates; SCALP
    outcomes are the SCALP exit contract over 1m bars for admitted claims.
    """
    import time as _time

    from backend.config.trading_economics import canonical_roundtrip_cost_pct
    from backend.services import adaptive_learning as al
    from backend.services.scalp_v2 import executable_edge as edge_mod
    from backend.services.strategy_version import economic_anchor

    moment = float(now if now is not None else _time.time())
    baseline = load_baseline_modules(baseline_sha, repo)
    day_anchor = float(economic_anchor(DAY_ENGINE)["epoch"])
    scalp_anchor = float(economic_anchor(SCALP_ENGINE)["epoch"])
    store = BarStore(db_path, DAY_SYMBOLS, since=day_anchor - 10 * 86400)
    cost = canonical_roundtrip_cost_pct()
    cands, _near = reconstruct_day_candidates(db_path, day_anchor, moment, store)
    tag_regimes(db_path, cands)
    day_folds = chronological_folds(day_anchor, moment, folds)
    scalp_folds = chronological_folds(scalp_anchor, moment, folds)
    out: dict[str, Any] = {
        "now": moment,
        "baseline_sha": baseline_sha,
        "day_anchor": day_anchor,
        "scalp_anchor": scalp_anchor,
        "day_candidates": len(cands),
        "day_folds": day_folds,
        "scalp_folds": scalp_folds,
        "policies": {},
    }
    rows = load_scalp_rows(db_path, scalp_anchor, moment, al.current_strategy_version(SCALP_ENGINE))
    out["scalp_rows"] = len(rows)
    out["scalp_claims"] = sum(1 for r in rows if r["directional"])
    scratch = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="econ_replay_", dir=scratch) as tmp:
        day_policies = (BaselineDayPolicy(f"{tmp}/day_current.db", baseline), RepairedDayPolicy(f"{tmp}/day_repaired.db"))
        scalp_policies = (
            ScalpPolicy("current", f"{tmp}/scalp_current.db", baseline["adaptive_learning"], baseline["executable_edge"], store),
            ScalpPolicy("repaired", f"{tmp}/scalp_repaired.db", al, edge_mod, store),
        )
        for day_policy, scalp_policy in zip(day_policies, scalp_policies, strict=True):
            day = simulate_day(cands, store, day_policy, roundtrip_cost=cost, now=moment)
            scalp = simulate_scalp(rows, scalp_policy, now=moment)
            out["policies"][day_policy.name] = {
                "day": day_report(day["trades"], day_folds),
                "day_trades": day["trades"],
                "scalp": scalp_report(scalp, scalp_folds),
            }
    return out


def run_scalp_replay(db_path: str, *, baseline_sha: str, repo: str | Path, now: float | None = None, folds: int = 4) -> dict[str, Any]:
    """SCALP at ``baseline_sha`` on the recorded claims, the same code on unfloored
    claims, and the working tree on unfloored claims, over the same candidates.
    Each starts empty at the SCALP anchor, learns causally from every claim
    (admitted or not) and trades by the exit contract."""
    import time as _time

    from backend.services import adaptive_learning as al
    from backend.services.scalp_v2 import executable_edge as edge_mod
    from backend.services.strategy_version import economic_anchor

    moment = float(now if now is not None else _time.time())
    baseline = load_baseline_modules(baseline_sha, repo)
    anchor = float(economic_anchor(SCALP_ENGINE)["epoch"])
    store = BarStore(db_path, DAY_SYMBOLS, since=anchor - 86400)
    version = al.current_strategy_version(SCALP_ENGINE)
    recorded = load_scalp_rows(db_path, anchor, moment, version)
    unfloored = load_scalp_rows(db_path, anchor, moment, version, store=store, unfloor_unless_version=al.current_economic_version(SCALP_ENGINE))
    bounds = chronological_folds(anchor, moment, folds)
    claims = [c for c in unfloored if c["directional"]]
    out: dict[str, Any] = {
        "now": moment,
        "baseline_sha": baseline_sha,
        "anchor": anchor,
        "folds": bounds,
        "rows": len(unfloored),
        "claims": len(claims),
        "claims_unfloor_missing": sum(1 for c in claims if c["unfloor_missing"]),
        "claims_changed": sum(1 for c in claims if abs(c["raw"] - c["raw_recorded"]) > 1e-12),
        "policies": {},
    }
    scratch = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="scalp_replay_", dir=scratch) as tmp:
        runs = (
            (f"A_{baseline_sha}", recorded, ScalpPolicy("baseline", f"{tmp}/a.db", baseline["adaptive_learning"], baseline["executable_edge"], store, full_api=True)),
            (f"A_{baseline_sha}_unfloored", unfloored, ScalpPolicy("baseline", f"{tmp}/a2.db", baseline["adaptive_learning"], baseline["executable_edge"], store, full_api=True)),
            ("B_calibrated", unfloored, ScalpPolicy("repaired", f"{tmp}/b.db", al, edge_mod, store)),
        )
        for name, rows, policy in runs:
            result = simulate_scalp(rows, policy, now=moment)
            out["policies"][name] = {**scalp_report(result, bounds), "trade_list": result["trades"], "predictions": result["predictions"]}
    return out


__all__ = [
    "BarStore",
    "BaselineDayPolicy",
    "DayCandidate",
    "RepairedDayPolicy",
    "ScalpPolicy",
    "chronological_folds",
    "day_lifecycle",
    "day_report",
    "discrimination",
    "iso_epoch",
    "label_day_lifecycles",
    "load_baseline_modules",
    "load_scalp_rows",
    "reconstruct_day_candidates",
    "run_replay",
    "run_scalp_replay",
    "scalp_exit_sim",
    "scalp_report",
    "simulate_day",
    "simulate_scalp",
    "summarize_returns",
    "tag_regimes",
    "unfloored_claim",
]
