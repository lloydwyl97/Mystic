"""Rebuild DAY and SCALP continuation state from current-version history.

A snapshot at time T is described only by information known at T. Its label
is the later executable net minus that snapshot's net, and it is not written
until that later time. Age is recorded on the snapshot and is not a sell rule.

The installed rows are the existing ``hold_remaining_up`` / ``hold_remaining_down``
metrics. Entry metrics are not read or written. Raw trades, fills, ownership
and accounting rows are not modified.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.services.adaptive_learning import (
    DAY_ENGINE,
    MEAN_FORM_METRICS,
    SCALP_ENGINE,
    _connect,
    _decay_factor,
    _lattice,
    _norm_symbol,
    current_economic_version,
    current_strategy_version,
    hold_remaining_metric,
    learned_hold_or_exit,
)
from backend.services.continuation_surface import (
    ADVANTAGE_VERSION,
    ContinuationMemory,
    horizons_for,
    record_advantage,
    state_features,
)
from backend.services.day_v2.lifecycle_sim import ohlcv_bars_1m
from backend.services.day_v2.live_exit_evaluator import catastrophic_threshold_price
from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT
from backend.services.strategy_version import economic_anchor, economic_version, is_current_version

LEARNING_VERSION = "CONTINUATION_REMAINING_V1"
SOURCE = "current_version_heartbeat_and_ohlcv"

DAY_HORIZONS_SEC = (15 * 60, 30 * 60, 60 * 60, 2 * 60 * 60, 4 * 60 * 60, 6 * 60 * 60)
SCALP_HORIZONS_SEC = (30, 60, 120, 300, 600, 1200, 1800, 3600)
# A 1m bar cannot stand in for a sub-minute scalp mark.
_BAR_HORIZON_FLOOR_SEC = 120.0

HOLD_METRICS = ("hold_remaining_up", "hold_remaining_down")


@dataclass
class Snapshot:
    time: float
    net: float
    mark: float
    mfe: float
    mae: float
    hold_sec: float
    high_water: float


@dataclass
class Position:
    engine: str
    trade_id: str
    symbol: str
    setup: str
    regime: str
    entry_price: float
    entry_time: float
    exit_time: float | None
    exit_price: float | None
    exit_reason: str
    atr: float
    anchor: float
    snapshots: list[Snapshot] = field(default_factory=list)
    closed: bool = False


@dataclass(frozen=True)
class Observation:
    engine: str
    symbol: str
    setup: str
    regime: str
    snapshot_time: float
    label_time: float
    unrealized_net: float
    remaining: float
    weight: float
    source: str
    horizon_sec: float
    trade_id: str


def _epoch(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None and math.isfinite(number) and number > 1_000_000_000:
        return number / 1000.0 if number > 10_000_000_000 else number
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _net(entry: float, price: float) -> float:
    return (float(price) - float(entry)) / float(entry) - float(ESTIMATED_ROUNDTRIP_COST)


def _horizons(engine: str) -> tuple[int, ...]:
    return SCALP_HORIZONS_SEC if engine == SCALP_ENGINE else DAY_HORIZONS_SEC


def _decision(raw: str | None) -> dict[str, Any]:
    try:
        loaded = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_positions(db_path: str, *, active_since: float | None = None) -> tuple[list[Position], Counter]:
    """Current-version positions entered at or after the engine's economic anchor."""
    skipped: Counter = Counter()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        buys = conn.execute("SELECT * FROM paper_trades WHERE UPPER(side)='BUY' AND UPPER(COALESCE(engine_id,'')) IN ('DAY_V2','SCALP_V2')").fetchall()
        positions: list[Position] = []
        for buy in buys:
            engine = str(buy["engine_id"] or "").upper()
            if not is_current_version(engine, buy):
                skipped[(engine, "not_current_version")] += 1
                continue
            entered = _epoch(buy["entry_timestamp"] or buy["timestamp"])
            anchor = economic_anchor(engine)
            if entered is None or anchor is None or entered < float(anchor["epoch"]):
                skipped[(engine, "before_anchor")] += 1
                continue
            entry = float(buy["entry_price"] or buy["price"] or 0.0)
            if entry <= 0:
                skipped[(engine, "no_entry_price")] += 1
                continue
            names = set(buy.keys())
            decision = _decision(buy["adaptive_decision_json"] if "adaptive_decision_json" in names else "")
            setup = str(decision.get("setup") or "")
            if not setup:
                skipped[(engine, "no_setup")] += 1
                continue
            sell = conn.execute(
                """
                SELECT timestamp, price, exit_reason FROM paper_trades
                WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? AND symbol=? AND entry_timestamp=?
                ORDER BY id DESC LIMIT 1
                """,
                (engine, buy["symbol"], buy["entry_timestamp"]),
            ).fetchone()
            exit_time = _epoch(sell["timestamp"]) if sell else None
            if active_since is not None and exit_time is not None and exit_time < active_since:
                skipped[(engine, "outside_recent_window")] += 1
                continue
            exit_price = float(sell["price"] or 0.0) if sell and sell["price"] else None
            if exit_price is not None and exit_price <= 0:
                exit_price = None
            snaps: list[Snapshot] = []
            for row in conn.execute(
                """
                SELECT epoch_ms, ts_utc, mark, net_unrealized_pct, mfe_pct, mae_pct, hold_seconds, highest_since_entry
                FROM ai_position_heartbeats WHERE trade_id=? ORDER BY epoch_ms
                """,
                (buy["trade_id"],),
            ):
                when = _epoch(row["epoch_ms"]) or _epoch(row["ts_utc"])
                mark = float(row["mark"] or 0.0)
                if when is None or mark <= 0 or when + 1.0 < entered:
                    skipped[(engine, "bad_heartbeat")] += 1
                    continue
                if exit_time is not None and when > exit_time + 1.0:
                    skipped[(engine, "heartbeat_after_exit")] += 1
                    continue
                stored = row["net_unrealized_pct"]
                try:
                    net = float(stored) if stored is not None else _net(entry, mark)
                except (TypeError, ValueError):
                    net = _net(entry, mark)
                if not math.isfinite(net):
                    skipped[(engine, "bad_heartbeat")] += 1
                    continue
                snaps.append(
                    Snapshot(
                        time=when,
                        net=net,
                        mark=mark,
                        mfe=float(row["mfe_pct"] or 0.0),
                        mae=float(row["mae_pct"] or 0.0),
                        hold_sec=float(row["hold_seconds"] or max(0.0, when - entered)),
                        high_water=float(row["highest_since_entry"] or mark),
                    )
                )
            try:
                atr = float(buy["atr_at_entry"] or 0.0)
            except (TypeError, ValueError, KeyError):
                atr = 0.0
            try:
                anchor_px = float(decision.get("structural_anchor") or 0.0)
            except (TypeError, ValueError):
                anchor_px = 0.0
            positions.append(
                Position(
                    engine=engine,
                    trade_id=str(buy["trade_id"]),
                    symbol=str(buy["symbol"]),
                    setup=setup,
                    regime=str(decision.get("regime") or ""),
                    entry_price=entry,
                    entry_time=entered,
                    exit_time=exit_time if exit_price else None,
                    exit_price=exit_price,
                    exit_reason=str(sell["exit_reason"] or "") if sell else "",
                    atr=atr,
                    anchor=anchor_px,
                    snapshots=snaps,
                    closed=exit_price is not None and exit_time is not None,
                )
            )
        return positions, skipped
    finally:
        conn.close()


def _bar_covering(bars: list[tuple[float, float, float, float, float]], moment: float) -> tuple[float, float, float, float, float] | None:
    """The 1m bar whose open contains ``moment``. None when that minute was not stored."""
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid][0] <= moment:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None
    bar = bars[lo - 1]
    if bar[0] <= moment < bar[0] + 60.0:
        return bar
    return None


def observations_for(position: Position, bars: list[tuple[float, float, float, float, float]], *, as_of: float) -> tuple[list[Observation], Counter]:
    """Labels for one position. A horizon with no stored mark is skipped, not filled in."""
    skipped: Counter = Counter()
    raw: list[Observation] = []
    exit_net = _net(position.entry_price, position.exit_price) if position.closed and position.exit_price else None
    for snap in position.snapshots:
        if snap.time > as_of:
            skipped[(position.engine, "snapshot_after_as_of")] += 1
            continue
        labels: list[tuple[float, float, str, float]] = []
        for later in position.snapshots:
            if snap.time < later.time <= as_of:
                labels.append((later.time, later.net - snap.net, "later_heartbeat", later.time - snap.time))
        if exit_net is not None and position.exit_time is not None and snap.time < position.exit_time <= as_of:
            labels.append((position.exit_time, exit_net - snap.net, "exit_fill", position.exit_time - snap.time))
        for horizon in _horizons(position.engine):
            if horizon < _BAR_HORIZON_FLOOR_SEC:
                skipped[(position.engine, "horizon_finer_than_stored_bars")] += 1
                continue
            target = snap.time + horizon
            if target > as_of:
                skipped[(position.engine, "horizon_not_yet_elapsed")] += 1
                continue
            bar = _bar_covering(bars, target)
            if bar is None:
                skipped[(position.engine, "horizon_bar_missing")] += 1
                continue
            label_time = bar[0] + 60.0
            if label_time <= snap.time or label_time > as_of:
                skipped[(position.engine, "horizon_bar_missing")] += 1
                continue
            labels.append((label_time, _net(position.entry_price, bar[4]) - snap.net, "ohlcv_horizon", float(horizon)))
        if not labels:
            skipped[(position.engine, "snapshot_without_future")] += 1
            continue
        for label_time, remaining, source, horizon in labels:
            if not math.isfinite(remaining):
                skipped[(position.engine, "non_finite_label")] += 1
                continue
            raw.append(
                Observation(
                    engine=position.engine,
                    symbol=position.symbol,
                    setup=position.setup,
                    regime=position.regime,
                    snapshot_time=snap.time,
                    label_time=label_time,
                    unrealized_net=snap.net,
                    remaining=remaining,
                    weight=0.0,
                    source=source,
                    horizon_sec=horizon,
                    trade_id=position.trade_id,
                )
            )
    if not raw:
        return [], skipped
    share = 1.0 / len(raw)
    return [Observation(**{**obs.__dict__, "weight": share}) for obs in raw], skipped


def build_observations(db_path: str, *, as_of: float | None = None) -> tuple[list[Observation], dict[str, Any]]:
    """Every usable current-version continuation label at or before ``as_of``."""
    clock = float(as_of if as_of is not None else time.time())
    positions, skipped = load_positions(db_path)
    span: dict[str, tuple[float, float]] = {}
    for position in positions:
        start, end = span.get(position.symbol, (position.entry_time, clock))
        span[position.symbol] = (min(start, position.entry_time), max(end, position.exit_time or clock, clock))
    bars_cache = {symbol: ohlcv_bars_1m(db_path, symbol, start, end + 60.0) for symbol, (start, end) in span.items()}
    observations: list[Observation] = []
    usable_positions: Counter = Counter()
    usable_snapshots: Counter = Counter()
    for position in positions:
        found, more = observations_for(position, bars_cache.get(position.symbol, []), as_of=clock)
        skipped.update(more)
        if found:
            usable_positions[position.engine] += 1
            usable_snapshots[position.engine] += len({obs.snapshot_time for obs in found if obs.trade_id == position.trade_id})
            observations.extend(found)
        else:
            skipped[(position.engine, "position_without_label")] += 1
    observations.sort(key=lambda obs: (obs.label_time, obs.trade_id, obs.snapshot_time, obs.source, obs.horizon_sec))
    inventory = _inventory(positions, skipped, usable_positions, usable_snapshots, observations)
    return observations, inventory


def _inventory(positions: list[Position], skipped: Counter, usable_positions: Counter, usable_snapshots: Counter, observations: list[Observation]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for engine in (DAY_ENGINE, SCALP_ENGINE):
        owned = [p for p in positions if p.engine == engine]
        heartbeats = sum(len(p.snapshots) for p in owned)
        anchor = economic_anchor(engine) or {}
        out[engine] = {
            "anchor_commit": anchor.get("commit", ""),
            "anchor_at": anchor.get("at", ""),
            "positions": len(owned),
            "closed": sum(1 for p in owned if p.closed),
            "open": sum(1 for p in owned if not p.closed),
            "heartbeats": heartbeats,
            "usable_positions": int(usable_positions[engine]),
            "usable_snapshots": int(usable_snapshots[engine]),
            "observations": sum(1 for obs in observations if obs.engine == engine),
            "unusable": {reason: count for (eng, reason), count in sorted(skipped.items()) if eng == engine},
        }
    return out


class _MemoryPosterior:
    """Same update and lattice as ``observe`` / ``continuation_terminal``, without a commit per label."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}

    def observe(self, obs: Observation) -> bool:
        engine_id = str(obs.engine or "").upper()
        version = current_strategy_version(engine_id)
        if not version or not math.isfinite(obs.remaining) or not (math.isfinite(obs.weight) and obs.weight > 0):
            return False
        metric = hold_remaining_metric(obs.unrealized_net)
        if metric not in MEAN_FORM_METRICS:
            return False
        key = (engine_id, current_economic_version(engine_id), _norm_symbol(obs.symbol), str(obs.setup or "").upper(), str(obs.regime or "").lower(), metric)
        moment = float(obs.label_time)
        row = self.rows.get(key)
        if row is None or float(row["n"]) <= 0:
            n, mean, m2 = obs.weight, obs.remaining, 0.0
        else:
            decay = _decay_factor(str(row["updated_at"] or ""), moment, engine_id)
            n = float(row["n"]) * decay + obs.weight
            alpha = obs.weight / n
            delta = obs.remaining - float(row["ewma"])
            mean = float(row["ewma"]) + alpha * delta
            m2 = max(0.0, float(row["m2"] or 0.0)) * decay + obs.weight * delta * (obs.remaining - mean)
        self.rows[key] = {
            "symbol": key[2],
            "setup": key[3],
            "regime": key[4],
            "metric": metric,
            "n": n,
            "ewma": mean,
            "m2": max(0.0, m2),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment)),
        }
        return True

    def terminal(self, engine: str, symbol: str, setup: str, regime: str, unrealized: float, now: float) -> float:
        metric = hold_remaining_metric(unrealized)
        engine_id = str(engine or "").upper()
        version = current_economic_version(engine_id)
        rows = [row for (eng, eco, _sym, _setup, _reg, met), row in self.rows.items() if eng == engine_id and eco == version and met == metric]
        lat = _lattice(rows, engine_id=engine_id, symbol=symbol, setup=setup, regime=regime, weights={metric: 1.0}, prior=0.0, now=now, include_engine=True)
        return float(unrealized) + float(lat["mean"])


def write_observations(state_db: str, observations: list[Observation]) -> dict[str, int]:
    """Fold labels into a learner database. One position's labels sum to weight 1.

    The update is the same one ``observe`` applies. One transaction keeps a
    historical rebuild from committing once per label.
    """
    wrote = {DAY_ENGINE: 0, SCALP_ENGINE: 0}
    conn = _connect(state_db)
    try:
        for obs in observations:
            engine_id = str(obs.engine or "").upper()
            version = current_strategy_version(engine_id)
            metric = hold_remaining_metric(obs.unrealized_net)
            if not version or metric not in MEAN_FORM_METRICS or not math.isfinite(obs.remaining) or not (math.isfinite(obs.weight) and obs.weight > 0):
                continue
            key = (engine_id, current_economic_version(engine_id), _norm_symbol(obs.symbol), str(obs.setup or "").upper(), str(obs.regime or "").lower(), metric)
            moment = float(obs.label_time)
            row = conn.execute(
                "SELECT n, ewma, m2, updated_at FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric=?",
                key,
            ).fetchone()
            if row is None or float(row["n"]) <= 0:
                n, mean, m2 = obs.weight, obs.remaining, 0.0
            else:
                decay = _decay_factor(str(row["updated_at"] or ""), moment, engine_id)
                n = float(row["n"]) * decay + obs.weight
                alpha = obs.weight / n
                delta = obs.remaining - float(row["ewma"])
                mean = float(row["ewma"]) + alpha * delta
                m2 = max(0.0, float(row["m2"] or 0.0)) * decay + obs.weight * delta * (obs.remaining - mean)
            conn.execute(
                """
                INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(engine_id, economic_version, symbol, setup, regime, metric) DO UPDATE SET
                    n=excluded.n, ewma=excluded.ewma, m2=excluded.m2, updated_at=excluded.updated_at
                """,
                (*key, n, mean, max(0.0, m2), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))),
            )
            wrote[engine_id] += 1
        conn.commit()
    finally:
        conn.close()
    return wrote


def install_continuation(target_db: str, state_db: str, inventory: dict[str, Any]) -> dict[str, int]:
    """Replace only current-version continuation rows. Trades and entry metrics stay."""
    _connect(target_db).close()
    src = sqlite3.connect(state_db)
    dst = sqlite3.connect(target_db, timeout=30)
    try:
        dst.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_learning_meta (
                engine_id TEXT NOT NULL,
                economic_version TEXT NOT NULL,
                learning_version TEXT NOT NULL,
                source TEXT NOT NULL,
                anchor_commit TEXT NOT NULL,
                anchor_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                observations INTEGER NOT NULL,
                PRIMARY KEY (engine_id, economic_version)
            )
            """
        )
        installed = {DAY_ENGINE: 0, SCALP_ENGINE: 0}
        dst.execute("BEGIN IMMEDIATE")
        for engine in (DAY_ENGINE, SCALP_ENGINE):
            version = economic_version(engine)
            dst.execute(
                "DELETE FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND metric IN ('hold_remaining_up','hold_remaining_down')",
                (engine, version),
            )
            rows = src.execute(
                """
                SELECT engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at
                FROM adaptive_metric_state
                WHERE engine_id=? AND economic_version=? AND metric IN ('hold_remaining_up','hold_remaining_down')
                """,
                (engine, version),
            ).fetchall()
            dst.executemany(
                """
                INSERT INTO adaptive_metric_state
                (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            installed[engine] = len(rows)
            info = inventory.get(engine) or {}
            dst.execute(
                """
                INSERT INTO continuation_learning_meta
                (engine_id, economic_version, learning_version, source, anchor_commit, anchor_at, updated_at, observations)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(engine_id, economic_version) DO UPDATE SET
                    learning_version=excluded.learning_version,
                    source=excluded.source,
                    anchor_commit=excluded.anchor_commit,
                    anchor_at=excluded.anchor_at,
                    updated_at=excluded.updated_at,
                    observations=excluded.observations
                """,
                (
                    engine,
                    version,
                    LEARNING_VERSION,
                    SOURCE,
                    str(info.get("anchor_commit") or ""),
                    str(info.get("anchor_at") or ""),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    int(info.get("observations") or 0),
                ),
            )
        dst.execute("COMMIT")
        return installed
    except Exception:
        dst.execute("ROLLBACK")
        raise
    finally:
        src.close()
        dst.close()


def _catastrophic(position: Position, mark: float, low: float) -> bool:
    if position.engine == SCALP_ENGINE:
        adverse = (position.entry_price - min(mark, low)) / position.entry_price
        return adverse >= SCALP_V2_CATASTROPHIC_PCT
    if position.atr <= 0:
        return False
    floor = catastrophic_threshold_price(position.entry_price, position.atr, position.anchor)
    return min(mark, low) <= floor


def _path_points(position: Position, bars: list[tuple[float, float, float, float, float]], as_of: float) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = [(position.entry_time, position.entry_price, position.entry_price)]
    for snap in position.snapshots:
        if position.entry_time <= snap.time <= as_of:
            points.append((snap.time, snap.mark, snap.mark))
    if position.closed and position.exit_time and position.exit_price and position.exit_time <= as_of:
        points.append((position.exit_time, position.exit_price, position.exit_price))
    for opened, _o, _h, low, close in bars:
        closed = opened + 60.0
        if position.entry_time < closed <= as_of:
            points.append((closed, close, low))
    points.sort(key=lambda item: (item[0], item[1]))
    return points


def _learned_row(position: Position, moment: float, mark: float, reason: str, bars: list[tuple[float, float, float, float, float]]) -> dict[str, Any]:
    learned_net = _net(position.entry_price, mark)
    mfe = max((snap.mfe for snap in position.snapshots if snap.time <= moment), default=0.0)
    for opened, _o, high, _low, _close in bars:
        if position.entry_time <= opened < moment:
            mfe = max(mfe, (high - position.entry_price) / position.entry_price)
    return {
        "net": learned_net,
        "hold_min": max(0.0, moment - position.entry_time) / 60.0,
        "mfe": mfe,
        "reason": reason,
        "engine": position.engine,
        "entry_time": position.entry_time,
    }


def _summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    def side(name: str) -> list[float]:
        return [float(row[name]) for row in trades]

    nets = side("net")
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    holds = sorted(side("hold_min"))
    peak = 0.0
    equity = 0.0
    drawdown = 0.0
    for net in nets:
        equity += net
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    favorable = sum(float(row.get("mfe") or 0.0) for row in trades)
    realized_gross = sum(float(row["net"]) + float(ESTIMATED_ROUNDTRIP_COST) for row in trades)
    return {
        "trades": len(trades),
        "net": sum(nets),
        "pf": (gross_win / gross_loss) if gross_loss else None,
        "avg_winner": (sum(wins) / len(wins)) if wins else None,
        "avg_loser": (sum(losses) / len(losses)) if losses else None,
        "dd": drawdown,
        "mfe_capture": (realized_gross / favorable) if favorable > 1e-8 else None,
        "avg_hold": (sum(holds) / len(holds)) if holds else None,
        "median_hold": holds[len(holds) // 2] if holds else None,
        "max_hold": holds[-1] if holds else None,
    }


def _actual_row(position: Position, bars: list[tuple[float, float, float, float, float]]) -> dict[str, Any]:
    actual_net = _net(position.entry_price, float(position.exit_price))
    hold = max(0.0, float(position.exit_time or position.entry_time) - position.entry_time) / 60.0
    mfe = max((snap.mfe for snap in position.snapshots), default=0.0)
    for opened, _o, high, _low, _close in bars:
        if position.entry_time <= opened <= float(position.exit_time or position.entry_time):
            mfe = max(mfe, (high - position.entry_price) / position.entry_price)
    return {
        "net": actual_net,
        "hold_min": hold,
        "mfe": mfe,
        "reason": position.exit_reason,
        "engine": position.engine,
        "entry_time": position.entry_time,
    }


def _classify(got: dict[str, Any], base: dict[str, Any]) -> str:
    if abs(got["net"] - base["net"]) < 1e-6 and abs(got["hold_min"] - base["hold_min"]) < 1.0:
        return "NO_MATERIAL_CHANGE"
    if got["net"] > base["net"] and got["hold_min"] > base["hold_min"]:
        return "HELD_LONGER_PROFITABLY"
    if got["net"] > base["net"] and got["hold_min"] < base["hold_min"]:
        return "EXIT_EARLIER_PROFITABLY"
    return "WORSE"


def walk_forward(db_path: str, *, as_of: float | None = None) -> dict[str, Any]:
    """Causal same-entry replay. Each decision sees only labels whose future has already arrived."""
    clock = float(as_of if as_of is not None else time.time())
    observations, inventory = build_observations(db_path, as_of=clock)
    positions, _skipped = load_positions(db_path)
    closed = [p for p in positions if p.closed]
    if not closed:
        return {"inventory": inventory, "engines": {}}
    earliest = min(p.entry_time for p in closed)
    bars_cache = {symbol: ohlcv_bars_1m(db_path, symbol, earliest, clock + 60.0) for symbol in {p.symbol for p in closed}}
    events: list[tuple[float, int, str, Any]] = [(obs.label_time, 0, "learn", obs) for obs in observations]
    paths: dict[str, list[tuple[float, float, float]]] = {}
    for position in closed:
        path = _path_points(position, bars_cache.get(position.symbol, []), clock)
        paths[position.trade_id] = path
        events.extend((point[0], 1, "decide", (position, point)) for point in path)
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    posterior = _MemoryPosterior()
    exited: dict[str, dict[str, Any]] = {}
    actuals = {position.trade_id: _actual_row(position, bars_cache.get(position.symbol, [])) for position in closed}
    reasons: Counter = Counter()
    for _when, _order, kind, payload in events:
        if kind == "learn":
            posterior.observe(payload)
            continue
        position, point = payload
        if position.trade_id in exited:
            continue
        moment, mark, low = point
        if moment + 1e-6 < position.entry_time:
            continue
        reason = _decision_reason(posterior, position, moment, mark, low, clock)
        if reason is None:
            continue
        reasons[(position.engine, reason)] += 1
        exited[position.trade_id] = _learned_row(position, moment, mark, reason, bars_cache.get(position.symbol, []))
    for position in closed:
        if position.trade_id in exited:
            continue
        last = paths[position.trade_id][-1]
        reasons[(position.engine, "DATA_END")] += 1
        exited[position.trade_id] = _learned_row(position, last[0], last[1], "DATA_END", bars_cache.get(position.symbol, []))
    return _replay_report(inventory, closed, actuals, exited, reasons)


def _decision_reason(posterior: _MemoryPosterior, position: Position, moment: float, mark: float, low: float, clock: float) -> str | None:
    """Exit reason at this mark, or None while the policy is still holding inside the path."""
    if _catastrophic(position, mark, low):
        return "CATASTROPHIC"
    unrealized = _net(position.entry_price, mark)
    terminal = posterior.terminal(position.engine, position.symbol, position.setup, position.regime, unrealized, moment)
    if learned_hold_or_exit(expected_terminal_net=terminal, unrealized_net=unrealized) == "exit":
        return "LEARNED"
    if position.exit_time is not None and moment + 1e-6 < position.exit_time:
        return None
    if moment + 60.0 < clock:
        return None
    return "DATA_END"


def _replay_report(
    inventory: dict[str, Any],
    closed: list[Position],
    actuals: dict[str, dict[str, Any]],
    exited: dict[str, dict[str, Any]],
    reasons: Counter,
) -> dict[str, Any]:
    report: dict[str, Any] = {"inventory": inventory, "engines": {}}
    for engine in (DAY_ENGINE, SCALP_ENGINE):
        rows = sorted((p for p in closed if p.engine == engine), key=lambda p: p.entry_time)
        classes = Counter(_classify(exited[pos.trade_id], actuals[pos.trade_id]) for pos in rows if pos.trade_id in exited)
        width = max(1, len(rows) // 4) if rows else 1
        folds = []
        for index in range(0, len(rows), width):
            chunk = rows[index : index + width]
            folds.append(
                {
                    "actual": _summarize([actuals[p.trade_id] for p in chunk]),
                    "learned": _summarize([exited[p.trade_id] for p in chunk if p.trade_id in exited]),
                }
            )
        later = rows[width:] if len(rows) > width else []
        by_reason: dict[str, dict[str, float]] = {}
        for pos in rows:
            label = actuals[pos.trade_id]["reason"] or "UNLABELED"
            bucket = by_reason.setdefault(label, {"n": 0.0, "actual_net": 0.0, "learned_net": 0.0, "actual_hold": 0.0, "learned_hold": 0.0})
            bucket["n"] += 1
            bucket["actual_net"] += actuals[pos.trade_id]["net"]
            bucket["actual_hold"] += actuals[pos.trade_id]["hold_min"]
            if pos.trade_id in exited:
                bucket["learned_net"] += exited[pos.trade_id]["net"]
                bucket["learned_hold"] += exited[pos.trade_id]["hold_min"]
        report["engines"][engine] = {
            "actual": _summarize([actuals[p.trade_id] for p in rows]),
            "learned": _summarize([exited[p.trade_id] for p in rows if p.trade_id in exited]),
            "classes": dict(classes),
            "learned_reasons": {reason: count for (eng, reason), count in sorted(reasons.items()) if eng == engine},
            "by_actual_reason": by_reason,
            "folds": folds,
            "oos_actual_net": sum(actuals[p.trade_id]["net"] for p in later),
            "oos_learned_net": sum(exited[p.trade_id]["net"] for p in later if p.trade_id in exited),
            "oos_actual": _summarize([actuals[p.trade_id] for p in later]),
            "oos_learned": _summarize([exited[p.trade_id] for p in later if p.trade_id in exited]),
            "oos_improved": (sum(exited[p.trade_id]["net"] for p in later if p.trade_id in exited) > sum(actuals[p.trade_id]["net"] for p in later)) if later else False,
        }
    return report


@dataclass(frozen=True)
class AdvantageLabel:
    engine: str
    symbol: str
    setup: str
    regime: str
    trade_id: str
    snapshot_time: float
    label_time: float
    horizon: int
    advantage: float
    weight: float
    features: dict[str, float]


def _safety_price(position: Position) -> float | None:
    """Price at which catastrophic protection would already have exited. None when it cannot fire."""
    if position.engine == SCALP_ENGINE:
        return position.entry_price * (1.0 - SCALP_V2_CATASTROPHIC_PCT)
    if position.atr <= 0:
        return None
    return catastrophic_threshold_price(position.entry_price, position.atr, position.anchor)


def _price_at_horizon(position: Position, snap: Snapshot, horizon: int, bars: list[tuple[float, float, float, float, float]], *, as_of: float) -> tuple[float, float] | None:
    """(mark, label_time) for one horizon, or None when that mark was not stored.

    A later bar after the historical exit is a real counterfactual. A path that
    would have hit catastrophic protection is labeled at that safety price.
    """
    target = snap.time + horizon
    if target > as_of + 1e-6:
        return None
    safety = _safety_price(position)
    if safety is not None:
        for opened, _o, _h, low, _close in bars:
            if snap.time <= opened < target and low <= safety:
                return safety, opened + 60.0
    if horizon < _BAR_HORIZON_FLOOR_SEC:
        band = 0.5 * horizon
        nearest: tuple[float, float] | None = None
        points: list[tuple[float, float]] = [(later.time, later.mark) for later in position.snapshots if snap.time < later.time <= as_of]
        if position.closed and position.exit_time and position.exit_price and snap.time < position.exit_time <= as_of:
            points.append((position.exit_time, position.exit_price))
        for moment, price in points:
            if abs(moment - target) > band:
                continue
            if nearest is None or abs(moment - target) < abs(nearest[0] - target):
                nearest = (moment, price)
        if nearest is None:
            return None
        if safety is not None and nearest[1] <= safety:
            return safety, nearest[0]
        return nearest[1], nearest[0]
    bar = _bar_covering(bars, target)
    if bar is None or bar[0] < snap.time:
        return None
    if safety is not None and bar[3] <= safety:
        return safety, bar[0] + 60.0
    return bar[4], bar[0] + 60.0


def _path_features(position: Position, snap: Snapshot, prev_net: float | None) -> dict[str, float]:
    return state_features(
        entry=position.entry_price,
        mark=snap.mark,
        net=snap.net,
        mfe=snap.mfe,
        mae=snap.mae,
        high_water=snap.high_water,
        prev_net=prev_net,
        age_sec=max(0.0, snap.time - position.entry_time),
    )


def advantage_labels_for(position: Position, bars: list[tuple[float, float, float, float, float]], *, as_of: float) -> tuple[list[AdvantageLabel], Counter]:
    """One label per stored horizon. Missing marks are omitted, not filled in."""
    skipped: Counter = Counter()
    drafts: list[tuple[Snapshot, int, float, float, dict[str, float]]] = []
    prev_net: float | None = None
    for snap in position.snapshots:
        if snap.time > as_of:
            skipped[(position.engine, "snapshot_after_as_of")] += 1
            continue
        features = _path_features(position, snap, prev_net)
        found = False
        for horizon in horizons_for(position.engine):
            marked = _price_at_horizon(position, snap, horizon, bars, as_of=as_of)
            if marked is None:
                skipped[(position.engine, "horizon_unavailable")] += 1
                continue
            price, label_time = marked
            if label_time <= snap.time or label_time > as_of + 1e-6:
                skipped[(position.engine, "horizon_unavailable")] += 1
                continue
            advantage = _net(position.entry_price, price) - snap.net
            if not math.isfinite(advantage):
                skipped[(position.engine, "non_finite_label")] += 1
                continue
            drafts.append((snap, horizon, label_time, advantage, features))
            found = True
        if not found:
            skipped[(position.engine, "snapshot_without_future")] += 1
        prev_net = snap.net
    if not drafts:
        return [], skipped
    # Weight by time since the previous state so a burst of heartbeats is one path, not many samples.
    ordered = [snap for snap in position.snapshots if snap.time <= as_of]
    gap_of = {}
    previous_time = position.entry_time
    for snap in ordered:
        gap_of[snap.time] = max(snap.time - previous_time, 1.0)
        previous_time = snap.time
    present = {item[0].time for item in drafts}
    total = sum(gap_of[moment] for moment in present)
    counts: Counter = Counter(item[0].time for item in drafts)
    labels: list[AdvantageLabel] = []
    for snap, horizon, label_time, advantage, features in drafts:
        share = (gap_of[snap.time] / total) / counts[snap.time] if total else 0.0
        labels.append(
            AdvantageLabel(
                engine=position.engine,
                symbol=position.symbol,
                setup=position.setup,
                regime=position.regime,
                trade_id=position.trade_id,
                snapshot_time=snap.time,
                label_time=label_time,
                horizon=horizon,
                advantage=advantage,
                weight=share,
                features=features,
            )
        )
    return labels, skipped


def build_advantage_labels(db_path: str, *, as_of: float | None = None) -> tuple[list[AdvantageLabel], dict[str, Any]]:
    clock = float(as_of if as_of is not None else time.time())
    positions, skipped = load_positions(db_path)
    span: dict[str, tuple[float, float]] = {}
    for position in positions:
        start, end = span.get(position.symbol, (position.entry_time, clock))
        span[position.symbol] = (min(start, position.entry_time), max(end, position.exit_time or clock, clock))
    bars_cache = {symbol: ohlcv_bars_1m(db_path, symbol, start, end + 60.0) for symbol, (start, end) in span.items()}
    labels: list[AdvantageLabel] = []
    usable_positions: Counter = Counter()
    usable_snapshots: Counter = Counter()
    for position in positions:
        found, more = advantage_labels_for(position, bars_cache.get(position.symbol, []), as_of=clock)
        skipped.update(more)
        if found:
            usable_positions[position.engine] += 1
            usable_snapshots[position.engine] += len({item.snapshot_time for item in found})
            labels.extend(found)
        else:
            skipped[(position.engine, "position_without_label")] += 1
    labels.sort(key=lambda item: (item.label_time, item.trade_id, item.snapshot_time, item.horizon))
    inventory = _inventory(positions, skipped, usable_positions, usable_snapshots, [])
    for engine in (DAY_ENGINE, SCALP_ENGINE):
        inventory[engine]["observations"] = sum(1 for item in labels if item.engine == engine)
        inventory[engine]["learning_version"] = ADVANTAGE_VERSION
    return labels, inventory


def continuation_predicts(prediction: dict[str, Any] | None) -> bool:
    """True when predicted hold advantage ranks with the realized advantage.

    Approximately zero rank means the surface does not yet know hold from exit.
    This is the deployment check, not a live sample gate.
    """
    pred = prediction or {}
    try:
        rank = float(pred.get("rank_correlation"))
        accuracy = float(pred.get("hold_exit_accuracy"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(rank) or not math.isfinite(accuracy):
        return False
    return rank >= 0.10 and accuracy > 0.5


def continuation_accepts(repaired: dict[str, Any], legacy: dict[str, Any]) -> bool:
    """Repaired OOS net beats the installed continuation without a material drawdown increase."""
    try:
        new_net = float(repaired["net"])
        old_net = float(legacy["net"])
        new_dd = float(repaired["dd"])
        old_dd = float(legacy["dd"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (new_net, old_net, new_dd, old_dd)):
        return False
    if int(repaired.get("trades") or 0) <= 0 or int(legacy.get("trades") or 0) <= 0:
        return False
    if new_net <= old_net:
        return False
    material = max(0.005, abs(old_dd) * 0.10)
    return new_dd <= old_dd + material


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        out = [0.0] * len(values)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out

    xs = ranks([item[0] for item in pairs])
    ys = ranks([item[1] for item in pairs])
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den <= 0:
        return None
    return num / den


def _feature_series(position: Position, bars: list[tuple[float, float, float, float, float]], path: list[tuple[float, float, float]]) -> dict[float, dict[str, float]]:
    """State at each path time, using only bars and heartbeats already reached."""
    entry = position.entry_price
    mfe = 0.0
    mae = 0.0
    high = entry
    prev_net: float | None = None
    index = 0
    series: dict[float, dict[str, float]] = {}
    snaps = [snap for snap in position.snapshots if snap.time >= position.entry_time]
    snap_i = 0
    for moment, mark, _low in path:
        while snap_i < len(snaps) and snaps[snap_i].time <= moment:
            mfe = max(mfe, snaps[snap_i].mfe)
            mae = max(mae, snaps[snap_i].mae)
            high = max(high, snaps[snap_i].high_water)
            snap_i += 1
        while index < len(bars) and bars[index][0] < moment:
            opened, _o, high_px, low_px, _close = bars[index]
            if opened >= position.entry_time:
                mfe = max(mfe, (high_px - entry) / entry)
                mae = max(mae, 0.0, (entry - low_px) / entry)
                high = max(high, high_px)
            index += 1
        net = _net(entry, mark)
        series[moment] = state_features(entry=entry, mark=mark, net=net, mfe=mfe, mae=mae, high_water=max(high, mark), prev_net=prev_net, age_sec=max(0.0, moment - position.entry_time))
        prev_net = net
    return series


def walk_advantage(db_path: str, *, as_of: float | None = None) -> dict[str, Any]:
    """Causal same-entry replay of the horizon advantage surface against the recorded exits."""
    clock = float(as_of if as_of is not None else time.time())
    labels, inventory = build_advantage_labels(db_path, as_of=clock)
    positions, _skipped = load_positions(db_path)
    closed = [p for p in positions if p.closed and p.exit_time is not None and p.exit_time <= clock]
    if not closed:
        return {"inventory": inventory, "engines": {}}
    earliest = min(p.entry_time for p in closed)
    bars_cache = {symbol: ohlcv_bars_1m(db_path, symbol, earliest, clock + 60.0) for symbol in {p.symbol for p in closed}}
    events: list[tuple[float, int, str, Any]] = [(item.label_time, 0, "learn", item) for item in labels]
    paths: dict[str, list[tuple[float, float, float]]] = {}
    feature_series: dict[str, dict[float, dict[str, float]]] = {}
    for position in closed:
        path = _path_points(position, bars_cache.get(position.symbol, []), clock)
        paths[position.trade_id] = path
        feature_series[position.trade_id] = _feature_series(position, bars_cache.get(position.symbol, []), path)
        events.extend((point[0], 1, "decide", (position, point)) for point in path)
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    surface = ContinuationMemory()
    policies = ("best", "blended")
    exited: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in policies}
    reasons: dict[str, Counter] = {name: Counter() for name in policies}
    actuals = {position.trade_id: _actual_row(position, bars_cache.get(position.symbol, [])) for position in closed}
    pending: dict[tuple[str, float], dict[str, Any]] = {}
    pairs: dict[str, list[tuple[float, float]]] = {engine: [] for engine in (DAY_ENGINE, SCALP_ENGINE)}
    label_index = {(item.trade_id, item.snapshot_time, item.horizon): item for item in labels}
    for _when, _order, kind, payload in events:
        if kind == "learn":
            surface.update(payload.engine, payload.symbol, payload.setup, payload.regime, payload.horizon, payload.features, payload.advantage, payload.weight, payload.label_time)
            continue
        position, point = payload
        moment, mark, low = point
        if moment + 1e-6 < position.entry_time:
            continue
        features = feature_series[position.trade_id].get(moment) or {}
        catastrophic = _catastrophic(position, mark, low)
        for name in policies:
            if position.trade_id in exited[name]:
                continue
            reason = "CATASTROPHIC" if catastrophic else None
            pred = 0.0
            chosen: int | None = None
            if reason is None:
                pred, chosen = surface.advantage(position.engine, position.symbol, position.setup, position.regime, features, moment, how=name)
                if pred < 0.0:
                    reason = "LEARNED"
            if reason is None and position.exit_time is not None and moment + 1e-6 < position.exit_time:
                if name == "best" and chosen is not None and any(abs(snap.time - moment) < 1.0 for snap in position.snapshots):
                    realized = label_index.get((position.trade_id, min(position.snapshots, key=lambda snap: abs(snap.time - moment)).time, chosen))
                    if realized is not None:
                        pending[(position.trade_id, moment)] = {"engine": position.engine, "pred": pred, "realized": realized.advantage}
                continue
            if reason is None and moment + 60.0 < clock:
                continue
            if reason is None:
                reason = "DATA_END"
            reasons[name][(position.engine, reason)] += 1
            exited[name][position.trade_id] = _learned_row(position, moment, mark, reason, bars_cache.get(position.symbol, []))
    for position in closed:
        for name in policies:
            if position.trade_id in exited[name]:
                continue
            last = paths[position.trade_id][-1]
            reasons[name][(position.engine, "DATA_END")] += 1
            exited[name][position.trade_id] = _learned_row(position, last[0], last[1], "DATA_END", bars_cache.get(position.symbol, []))
    for rec in pending.values():
        pairs[rec["engine"]].append((rec["pred"], rec["realized"]))
    report: dict[str, Any] = {"inventory": inventory, "engines": {}, "version": ADVANTAGE_VERSION}
    for engine in (DAY_ENGINE, SCALP_ENGINE):
        rows = sorted((p for p in closed if p.engine == engine), key=lambda p: p.entry_time)
        width = max(1, len(rows) // 4) if rows else 1
        later = rows[width:] if len(rows) > width else []
        block: dict[str, Any] = {"actual": _summarize([actuals[p.trade_id] for p in rows]), "observations": inventory[engine]["observations"]}
        scored = pairs[engine]
        correct = [1 for pred, realized in scored if (pred > 0) == (realized > 0) and pred != 0 and realized != 0]
        called = [1 for pred, realized in scored if pred != 0 and realized != 0]
        ordered = sorted(scored, key=lambda item: item[0])
        tercile = max(1, len(ordered) // 3) if ordered else 1
        bottom = ordered[:tercile]
        top = ordered[-tercile:] if ordered else []
        block["prediction"] = {
            "pairs": len(scored),
            "rank_correlation": _spearman(scored),
            "hold_exit_accuracy": (sum(correct) / len(called)) if called else None,
            "bottom_realized": (sum(item[1] for item in bottom) / len(bottom)) if bottom else None,
            "top_realized": (sum(item[1] for item in top) / len(top)) if top else None,
        }
        for name in policies:
            learned = [exited[name][p.trade_id] for p in rows]
            later_rows = [exited[name][p.trade_id] for p in later]
            later_actual = [actuals[p.trade_id] for p in later]
            folds = []
            for index in range(0, len(rows), width):
                chunk = rows[index : index + width]
                folds.append(
                    {
                        "actual": _summarize([actuals[p.trade_id] for p in chunk]),
                        "learned": _summarize([exited[name][p.trade_id] for p in chunk]),
                    }
                )
            block[name] = {
                "learned": _summarize(learned),
                "oos": _summarize(later_rows),
                "oos_actual": _summarize(later_actual),
                "classes": dict(Counter(_classify(exited[name][pos.trade_id], actuals[pos.trade_id]) for pos in rows)),
                "learned_reasons": {reason: count for (eng, reason), count in sorted(reasons[name].items()) if eng == engine},
                "oos_improved_vs_actual": (sum(item["net"] for item in later_rows) > sum(item["net"] for item in later_actual)) if later else False,
                "folds": folds,
                "held_under_one_minute": sum(1 for row in learned if float(row["hold_min"]) < 1.0),
            }
        report["engines"][engine] = block
    return report


def train_surface(labels: list[AdvantageLabel]) -> ContinuationMemory:
    """Replay horizon labels in label-time order into one surface."""
    surface = ContinuationMemory()
    ordered = sorted(labels, key=lambda item: (item.label_time, item.trade_id, item.snapshot_time, item.horizon))
    for item in ordered:
        surface.update(item.engine, item.symbol, item.setup, item.regime, item.horizon, item.features, item.advantage, item.weight, item.label_time)
    return surface


def write_surface(state_db: str, surface: ContinuationMemory) -> dict[str, int]:
    """Store a trained surface in one transaction. Entry metrics are not written."""
    from backend.services.continuation_surface import SURFACE_MODEL

    wrote = {DAY_ENGINE: 0, SCALP_ENGINE: 0}
    conn = _connect(state_db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for key, row in surface.rows.items():
            engine_id, economic, symbol, setup, regime, metric = key
            conn.execute(
                """
                INSERT INTO adaptive_metric_state
                (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(engine_id, economic_version, symbol, setup, regime, metric) DO UPDATE SET
                    n=excluded.n, ewma=excluded.ewma, m2=excluded.m2, updated_at=excluded.updated_at
                """,
                (engine_id, economic, symbol, setup, regime, metric, row["n"], row["ewma"], row["m2"], row["updated_at"]),
            )
            wrote[engine_id] = wrote.get(engine_id, 0) + 1
        horizons: dict[str, dict[str, Any]] = {}
        updated: dict[str, str] = {}
        for (engine_id, horizon), stats in surface.ridge.items():
            horizons.setdefault(engine_id, {})[str(int(horizon))] = stats
            updated[engine_id] = str(stats.get("updated_at") or "")
        for engine_id, payload in horizons.items():
            model = f"{SURFACE_MODEL}@{current_economic_version(engine_id)}"
            conn.execute(
                """
                INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(engine_id, model) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (engine_id, model, json.dumps({"horizons": payload}, separators=(",", ":")), updated.get(engine_id) or ""),
            )
        conn.execute("COMMIT")
        return wrote
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def remember_labels(db_path: str, labels: list[AdvantageLabel]) -> None:
    """Mark horizon labels already folded so a later pass does not train them again."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_advantage_log (
                engine_id TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                snapshot_time REAL NOT NULL,
                horizon INTEGER NOT NULL,
                PRIMARY KEY (engine_id, trade_id, snapshot_time, horizon)
            )
            """
        )
        conn.executemany(
            "INSERT OR IGNORE INTO continuation_advantage_log (engine_id, trade_id, snapshot_time, horizon) VALUES (?,?,?,?)",
            [(item.engine, item.trade_id, item.snapshot_time, item.horizon) for item in labels],
        )
        conn.commit()
    finally:
        conn.close()


_LAST_FOLD = 0.0


def fold_recent_advantages(db_path: str, *, now: float | None = None) -> int:
    """Fold horizon labels that have elapsed since the surface was installed.

    No-op until an advantage surface is the live continuation authority.
    One position's labels keep the same shared weight. Already-folded labels are skipped.
    """
    global _LAST_FOLD
    from backend.services.continuation_surface import advantage_authority

    active = {engine for engine in (DAY_ENGINE, SCALP_ENGINE) if advantage_authority(db_path, engine)}
    if not active:
        return 0
    moment = float(now if now is not None else time.time())
    if moment - _LAST_FOLD < 60.0:
        return 0
    _LAST_FOLD = moment
    positions, _skipped = load_positions(db_path, active_since=moment - 13 * 3600.0)
    positions = [position for position in positions if position.engine in active]
    if not positions:
        return 0
    span: dict[str, tuple[float, float]] = {}
    for position in positions:
        start, end = span.get(position.symbol, (position.entry_time, moment))
        span[position.symbol] = (min(start, position.entry_time), max(end, position.exit_time or moment, moment))
    bars_cache = {symbol: ohlcv_bars_1m(db_path, symbol, start, end + 60.0) for symbol, (start, end) in span.items()}
    fresh: list[AdvantageLabel] = []
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_advantage_log (
                engine_id TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                snapshot_time REAL NOT NULL,
                horizon INTEGER NOT NULL,
                PRIMARY KEY (engine_id, trade_id, snapshot_time, horizon)
            )
            """
        )
        known = {(row[0], row[1], float(row[2]), int(row[3])) for row in conn.execute("SELECT engine_id, trade_id, snapshot_time, horizon FROM continuation_advantage_log")}
    finally:
        conn.close()
    for position in positions:
        found, _more = advantage_labels_for(position, bars_cache.get(position.symbol, []), as_of=moment)
        for item in found:
            if item.engine not in active:
                continue
            if (item.engine, item.trade_id, item.snapshot_time, item.horizon) not in known:
                fresh.append(item)
    if not fresh:
        return 0
    persist_advantage(db_path, fresh)
    remember_labels(db_path, fresh)
    return len(fresh)


def persist_advantage(state_db: str, labels: list[AdvantageLabel]) -> dict[str, int]:
    wrote = {DAY_ENGINE: 0, SCALP_ENGINE: 0}
    for item in labels:
        if record_advantage(
            state_db,
            engine=item.engine,
            symbol=item.symbol,
            setup=item.setup,
            regime=item.regime,
            horizon=item.horizon,
            features=item.features,
            advantage=item.advantage,
            weight=item.weight,
            now=item.label_time,
        ):
            wrote[item.engine] += 1
    return wrote
