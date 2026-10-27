"""Hold-advantage surface for continuation.

The learned quantity is the incremental executable net of still being in the
position at a horizon, minus the executable net of liquidating now. It is not
a price target and it does not pay the position for remaining open.

A horizon with no stored mark is absent. Horizons that have been seen are
shrunk toward 0. The decision number is the best informed horizon: hold when
that advantage is positive, exit when it is negative. An uninformed horizon
does not vote. No sample-count gate and no extra threshold.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from backend.services.adaptive_learning import (
    HOLD_ADVANTAGE_HORIZONS,
    PRIOR_STRENGTH,
    _decay_factor,
    _lattice,
    _norm_symbol,
    current_economic_version,
    current_strategy_version,
    estimate,
    observe,
)

FEATURE_NAMES = ("net", "mfe", "mae", "giveback", "dist_high", "slope", "age_min")
ADVANTAGE_VERSION = "CONTINUATION_ADVANTAGE_V1"
SURFACE_MODEL = "continuation_surface"
DAY_ADVANTAGE_HORIZONS = (900, 1800, 3600, 7200, 14400, 21600, 43200)
SCALP_ADVANTAGE_HORIZONS = (30, 60, 120, 300, 600, 1200, 1800, 3600)


def horizons_for(engine: str) -> tuple[int, ...]:
    return SCALP_ADVANTAGE_HORIZONS if str(engine or "").upper() == "SCALP_V2" else DAY_ADVANTAGE_HORIZONS


def metric_name(horizon: int) -> str:
    return f"hold_adv_{int(horizon)}"


def feature_vector(features: dict[str, float] | None) -> list[float]:
    raw = features or {}
    out: list[float] = []
    for name in FEATURE_NAMES:
        try:
            value = float(raw.get(name) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        out.append(value if math.isfinite(value) else 0.0)
    return out


def state_features(
    *,
    entry: float,
    mark: float,
    net: float,
    mfe: float,
    mae: float,
    high_water: float,
    prev_net: float | None,
    age_sec: float,
) -> dict[str, float]:
    """Causal position state. Age is a coordinate, not a sell rule."""
    gross = ((mark - entry) / entry) if entry else 0.0
    favorable = max(float(mfe or 0.0), gross, 0.0)
    adverse = max(float(mae or 0.0), -min(gross, 0.0), 0.0)
    return {
        "net": float(net),
        "mfe": favorable,
        "mae": adverse,
        "giveback": max(0.0, favorable - max(gross, 0.0)),
        "dist_high": max(0.0, (high_water - mark) / entry) if entry and high_water else 0.0,
        "slope": 0.0 if prev_net is None else float(net) - float(prev_net),
        "age_min": max(0.0, float(age_sec)) / 60.0,
    }


def _blank_ridge() -> dict[str, Any]:
    width = len(FEATURE_NAMES)
    return {
        "n": 0.0,
        "updated_at": "",
        "sum_x": [0.0] * width,
        "sum_x2": [0.0] * width,
        "xtx": [[0.0] * width for _ in range(width)],
        "xty": [0.0] * width,
    }


def _decay_ridge(stats: dict[str, Any], now: float, engine: str) -> None:
    factor = _decay_factor(str(stats.get("updated_at") or ""), now, engine) if stats.get("updated_at") else 1.0
    if factor == 1.0:
        return
    stats["n"] = float(stats.get("n") or 0.0) * factor
    stats["sum_x"] = [float(v) * factor for v in stats["sum_x"]]
    stats["sum_x2"] = [float(v) * factor for v in stats["sum_x2"]]
    stats["xtx"] = [[float(v) * factor for v in row] for row in stats["xtx"]]
    stats["xty"] = [float(v) * factor for v in stats["xty"]]


def _standardize(stats: dict[str, Any], values: list[float]) -> list[float]:
    n = float(stats.get("n") or 0.0)
    if n <= 0:
        return list(values)
    z: list[float] = []
    for index, value in enumerate(values):
        mean = float(stats["sum_x"][index]) / n
        var = max(float(stats["sum_x2"][index]) / n - mean * mean, 1e-8)
        z.append((value - mean) / math.sqrt(var))
    return z


def _update_ridge(stats: dict[str, Any], features: dict[str, float] | None, target: float, weight: float, now: float, engine: str) -> None:
    if not math.isfinite(target) or not math.isfinite(weight) or weight <= 0:
        return
    values = feature_vector(features)
    if float(values[6]) <= 1e-12 or all(abs(v) <= 1e-12 for v in values[1:]):
        return
    _decay_ridge(stats, now, engine)
    z = _standardize(stats, values)
    for i, value in enumerate(values):
        stats["sum_x"][i] = float(stats["sum_x"][i]) + weight * value
        stats["sum_x2"][i] = float(stats["sum_x2"][i]) + weight * value * value
    for i, left in enumerate(z):
        stats["xty"][i] = float(stats["xty"][i]) + weight * left * target
        for j, right in enumerate(z):
            stats["xtx"][i][j] = float(stats["xtx"][i][j]) + weight * left * right
    stats["n"] = float(stats.get("n") or 0.0) + weight
    stats["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


def _ridge_dot(stats: dict[str, Any], features: dict[str, float] | None, now: float, engine: str) -> float:
    _decay_ridge(stats, now, engine)
    if float(stats.get("n") or 0.0) <= 0:
        return 0.0
    raw_ft = dict(features or {})
    if float(raw_ft.get("age_min") or 0.0) > 1e-12:
        mfe_v = max(0.0, float(raw_ft.get("mfe") or 0.0))
        mae_v = max(0.0, float(raw_ft.get("mae") or 0.0))
        gb_v = max(0.0, float(raw_ft.get("giveback") or 0.0))
        net_v = float(raw_ft.get("net") or 0.0)
        raw_ft["net"] = max(net_v, mfe_v - 0.00066)
        if mfe_v >= 0.0020:
            raw_ft["mae"] = max(mae_v, gb_v)
    vec = feature_vector(raw_ft)
    if float(vec[6]) <= 1e-12 or all(abs(v) <= 1e-12 for v in vec[1:]):
        return 0.0
    z = np.asarray(_standardize(stats, vec), dtype=float)
    ps = float(PRIOR_STRENGTH) / float(max(1, len(horizons_for(engine))))
    xtx = np.asarray(stats["xtx"], dtype=float) + ps * np.eye(len(FEATURE_NAMES))
    xty = np.asarray(stats["xty"], dtype=float)
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return 0.0
    for idx_f in (0, 1, 5):
        if float(beta[idx_f]) < 0.0 and float(z[idx_f]) < 0.0:
            z[idx_f] = 0.0
    for idx_a in (2, 3, 4):
        if float(beta[idx_a]) > 0.0 and float(z[idx_a]) > 0.0:
            z[idx_a] = 0.0
    dot = float(z @ beta)
    return dot if math.isfinite(dot) else 0.0


class ContinuationMemory:
    """In-memory copy of the horizon lattice and the state ridge."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
        self.ridge: dict[tuple[str, int], dict[str, Any]] = {}

    def update(self, engine: str, symbol: str, setup: str, regime: str, horizon: int, features: dict[str, float], advantage: float, weight: float, now: float) -> bool:
        engine_id = str(engine or "").upper()
        version = current_strategy_version(engine_id)
        metric = metric_name(horizon)
        if not version or metric not in HOLD_ADVANTAGE_METRICS or not math.isfinite(advantage) or not (math.isfinite(weight) and weight > 0):
            return False
        key = (engine_id, current_economic_version(engine_id), _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower(), metric)
        row = self.rows.get(key)
        if row is None or float(row["n"]) <= 0:
            n, mean, m2 = weight, advantage, 0.0
        else:
            decay = _decay_factor(str(row["updated_at"] or ""), now, engine_id)
            n = float(row["n"]) * decay + weight
            alpha = weight / n
            delta = advantage - float(row["ewma"])
            mean = float(row["ewma"]) + alpha * delta
            m2 = max(0.0, float(row["m2"] or 0.0)) * decay + weight * delta * (advantage - mean)
        self.rows[key] = {
            "symbol": key[2],
            "setup": key[3],
            "regime": key[4],
            "metric": metric,
            "n": n,
            "ewma": mean,
            "m2": max(0.0, m2),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        ridge_key = (engine_id, int(horizon))
        stats = self.ridge.setdefault(ridge_key, _blank_ridge())
        _update_ridge(stats, features, advantage, weight, now, engine_id)
        return True

    def horizon_view(self, engine: str, symbol: str, setup: str, regime: str, horizon: int, features: dict[str, float] | None, now: float) -> tuple[float, float]:
        """(posterior advantage, evidence weight). Evidence 0 means this horizon has not been seen."""
        engine_id = str(engine or "").upper()
        metric = metric_name(horizon)
        version = current_economic_version(engine_id)
        rows = [row for (eng, eco, _sym, _setup, _reg, met), row in self.rows.items() if eng == engine_id and eco == version and met == metric]
        lat = _lattice(rows, engine_id=engine_id, symbol=symbol, setup=setup, regime=regime, weights={metric: 1.0}, prior=0.0, now=now, include_engine=True)
        evidence = float(lat["level_weights"]["engine"])
        if evidence <= 0:
            return 0.0, 0.0
        stats = self.ridge.get((engine_id, int(horizon))) or _blank_ridge()
        return float(lat["mean"]) + _ridge_dot(stats, features, now, engine_id), evidence

    def advantage(self, engine: str, symbol: str, setup: str, regime: str, features: dict[str, float] | None, now: float, *, how: str = "best") -> tuple[float, int | None]:
        informed: list[tuple[int, float, float]] = []
        for horizon in horizons_for(engine):
            mean, evidence = self.horizon_view(engine, symbol, setup, regime, horizon, features, now)
            if evidence > 0:
                informed.append((horizon, mean, evidence))
        if not informed:
            return 0.0, None
        if how == "blended":
            weight = sum(item[2] for item in informed)
            return sum(item[1] * item[2] for item in informed) / weight, max(informed, key=lambda item: item[2])[0]
        chosen = max(informed, key=lambda item: item[1])
        return chosen[1], chosen[0]


HOLD_ADVANTAGE_METRICS = frozenset(metric_name(horizon) for horizon in HOLD_ADVANTAGE_HORIZONS)


_RIDGE_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_META_CACHE: dict[tuple[str, str], tuple[float, tuple[str, str] | None]] = {}


def _meta_row(db_path: str, engine: str) -> tuple[str, str] | None:
    import sqlite3

    from backend.services.adaptive_learning import _db_stamp

    engine_id = str(engine or "").upper()
    ckey = (*_db_stamp(db_path), engine_id)
    now_mono = time.monotonic()
    hit = _META_CACHE.get(ckey)
    if hit is not None and (now_mono - hit[0]) < 2.0:
        return hit[1]
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT learning_version, source FROM continuation_learning_meta WHERE engine_id=? AND economic_version=?",
            (engine_id, current_economic_version(engine_id)),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    res = (str(row[0] or ""), str(row[1] or "")) if row else None
    if len(_META_CACHE) > 32:
        _META_CACHE.clear()
    _META_CACHE[ckey] = (now_mono, res)
    return res


def advantage_authority(db_path: str, engine: str) -> bool:
    """True only after an advantage surface has been installed for this engine."""
    row = _meta_row(db_path, engine)
    return bool(row and row[0] == ADVANTAGE_VERSION)


def installed_aggregator(db_path: str, engine: str) -> str:
    """Which horizon combination the installed surface was accepted with."""
    row = _meta_row(db_path, engine)
    source = row[1] if row else ""
    prefix = "horizon_hold_advantage:"
    how = source[len(prefix) :] if source.startswith(prefix) else ""
    return how if how in {"best", "blended"} else "best"


def surface_trace(
    db_path: str,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    features: dict[str, float] | None,
    now: float,
    *,
    how: str = "best",
) -> dict[str, Any] | None:
    """Same surface read as ``surface_advantage``, with the horizon pieces kept."""
    informed: list[dict[str, Any]] = []
    engine_id = str(engine or "").upper()
    age_sec = max(0.0, float((features or {}).get("age_min") or 0.0)) * 60.0
    ridge = _load_ridge(db_path, engine_id) if age_sec > 0.0 else {}
    for horizon in horizons_for(engine_id):
        metric = metric_name(horizon)
        view = estimate(db_path, engine_id, symbol, setup, regime, metric, now=now, include_engine=True)
        evidence = float(view.get("level_weights", {}).get("engine") or 0.0)
        if evidence <= 0:
            continue
        posterior = float(view["mean"])
        fade_horizon = min(float(horizon), 3600.0)
        if posterior > 0.0 and age_sec > 0.0 and fade_horizon > 0.0:
            posterior = posterior * max(0.0, 1.0 - age_sec / fade_horizon)
        ridge_dot = _ridge_dot(ridge.get(str(horizon)) or _blank_ridge(), features, now, engine_id) if age_sec > 0.0 else 0.0
        informed.append(
            {
                "horizon": int(horizon),
                "posterior": posterior,
                "ridge": ridge_dot,
                "evidence": evidence,
                "value": posterior + ridge_dot,
            }
        )
    if not informed:
        return None
    if how == "blended":
        base_h = float(min(horizons_for(engine_id)))
        weights = [float(item["evidence"]) * ((base_h / max(base_h, float(item["horizon"]))) ** 0.35) for item in informed]
        weight = sum(weights)
        advantage = sum(float(item["value"]) * w for item, w in zip(informed, weights, strict=True)) / weight
    else:
        advantage = max(float(item["value"]) for item in informed)
    return {"advantage": float(advantage), "how": how, "horizons": informed, "coordinates": feature_vector(features)}


def surface_advantage(db_path: str, engine: str, symbol: str, setup: str, regime: str, features: dict[str, float] | None, now: float, *, how: str = "best") -> float | None:
    """Live read of the installed surface. None when this engine has no advantage rows."""
    trace = surface_trace(db_path, engine, symbol, setup, regime, features, now, how=how)
    return None if trace is None else float(trace["advantage"])


def _model_key(engine_id: str) -> str:
    return f"{SURFACE_MODEL}@{current_economic_version(engine_id)}"


def _load_ridge(db_path: str, engine_id: str) -> dict[str, Any]:
    import copy

    from backend.services.adaptive_learning import _connect, _db_stamp

    ckey = (*_db_stamp(db_path), engine_id)
    now_mono = time.monotonic()
    hit = _RIDGE_CACHE.get(ckey)
    if hit is not None and (now_mono - hit[0]) < 2.0:
        return copy.deepcopy(hit[1])
    try:
        conn = _connect(db_path)
    except Exception:
        return {}
    try:
        row = conn.execute("SELECT payload FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine_id, _model_key(engine_id))).fetchone()
    except Exception:
        return {}
    finally:
        conn.close()
    if len(_RIDGE_CACHE) > 32:
        _RIDGE_CACHE.clear()
    if not row or not row["payload"]:
        _RIDGE_CACHE[ckey] = (now_mono, {})
        return {}
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return {}
    res = (payload.get("horizons") or {}) if isinstance(payload, dict) else {}
    _RIDGE_CACHE[ckey] = (now_mono, res)
    return copy.deepcopy(res)


def record_advantage(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    horizon: int,
    features: dict[str, float],
    advantage: float,
    weight: float,
    now: float,
    conn: Any = None,
) -> bool:
    """Fold one horizon advantage into the live surface. Does not touch entry metrics.

    With ``conn`` both writes join the caller's transaction and are not committed here."""
    from backend.services.adaptive_learning import _connect

    engine_id = str(engine or "").upper()
    ok = observe(
        db_path,
        engine=engine_id,
        symbol=symbol,
        setup=setup,
        regime=regime,
        metric=metric_name(horizon),
        value=float(advantage),
        strategy_version=current_strategy_version(engine_id),
        now=now,
        weight=weight,
        conn=conn,
    )
    if not ok:
        return False
    own = conn is None
    if own:
        conn = _connect(db_path)
    try:
        key = _model_key(engine_id)
        row = conn.execute("SELECT payload FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine_id, key)).fetchone()
        try:
            payload = json.loads(row["payload"]) if row and row["payload"] else {}
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        horizons = payload.setdefault("horizons", {})
        stats = horizons.get(str(int(horizon))) or _blank_ridge()
        _update_ridge(stats, features, float(advantage), float(weight), float(now), engine_id)
        horizons[str(int(horizon))] = stats
        conn.execute(
            """
            INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(engine_id, model) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (engine_id, key, json.dumps(payload, separators=(",", ":")), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
    _RIDGE_CACHE.clear()
    return True


def install_surface(target_db: str, state_db: str, inventory: dict[str, Any], *, how: str, engines: tuple[str, ...] = ("DAY_V2", "SCALP_V2")) -> dict[str, int]:
    """Copy advantage rows and the ridge. Trades, fills and entry metrics stay."""
    import sqlite3

    from backend.services.adaptive_learning import _connect, _invalidate_adaptive_caches

    _connect(target_db).close()
    src = sqlite3.connect(state_db)
    dst = sqlite3.connect(target_db, timeout=30)
    metrics = tuple(metric_name(horizon) for horizon in HOLD_ADVANTAGE_HORIZONS)
    placeholders = ",".join("?" for _ in metrics)
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
        installed = dict.fromkeys(engines, 0)
        dst.execute("BEGIN IMMEDIATE")
        for engine in engines:
            version = current_economic_version(engine)
            dst.execute(
                f"DELETE FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND metric IN ({placeholders})",
                (engine, version, *metrics),
            )
            rows = src.execute(
                f"""
                SELECT engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at
                FROM adaptive_metric_state
                WHERE engine_id=? AND economic_version=? AND metric IN ({placeholders})
                """,
                (engine, version, *metrics),
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
            model = src.execute("SELECT payload, updated_at FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine, _model_key(engine))).fetchone()
            if model:
                dst.execute(
                    """
                    INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(engine_id, model) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                    """,
                    (engine, _model_key(engine), model[0], model[1]),
                )
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
                    ADVANTAGE_VERSION,
                    f"horizon_hold_advantage:{how}",
                    str(info.get("anchor_commit") or ""),
                    str(info.get("anchor_at") or ""),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    int(info.get("observations") or 0),
                ),
            )
        dst.execute("COMMIT")
        _RIDGE_CACHE.clear()
        _META_CACHE.clear()
        _invalidate_adaptive_caches()
        return installed
    except Exception:
        dst.execute("ROLLBACK")
        raise
    finally:
        src.close()
        dst.close()
