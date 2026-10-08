"""Research-only link from outside-venue state to the realized policy net.

The prediction is the model's value before this episode resolves. The update
runs after the policy net is stored, once. Live entry does not read it.
A missing or stale outside book is left missing. It is not stored as zero.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

FEATURES = (
    "coinbase_ret_5s",
    "coinbase_ret_15s",
    "coinbase_flow_5s",
    "coinbase_spread",
    "coinbase_dislocation",
    "coinbase_lead_5s",
    "kraken_ret_5s",
    "kraken_ret_15s",
    "kraken_flow_5s",
    "kraken_spread",
    "kraken_dislocation",
    "kraken_lead_5s",
)
_MODEL = """
CREATE TABLE IF NOT EXISTS external_policy_model (
    engine_id TEXT PRIMARY KEY,
    n REAL NOT NULL,
    bias REAL NOT NULL,
    w_json TEXT NOT NULL,
    mu_json TEXT NOT NULL,
    var_json TEXT NOT NULL,
    selected_sum REAL NOT NULL DEFAULT 0,
    taken INTEGER NOT NULL DEFAULT 0,
    abstain INTEGER NOT NULL DEFAULT 0,
    total_sum REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
)
"""
_APPLIED = """
CREATE TABLE IF NOT EXISTS external_policy_applied (
    episode_id INTEGER PRIMARY KEY,
    prediction REAL NOT NULL,
    net REAL NOT NULL,
    applied_at REAL NOT NULL
)
"""


def ensure_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute(_MODEL)
        conn.execute(_APPLIED)
        conn.commit()
    finally:
        conn.close()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def feature_vector(raw: dict[str, Any] | None) -> list[float] | None:
    """A complete outside-venue vector. One missing venue drops the observation."""
    if not isinstance(raw, dict) or int(raw.get("venues_fresh") or 0) < 1:
        return None
    out = []
    for name in FEATURES:
        value = _finite(raw.get(name))
        if value is None:
            return None
        out.append(value)
    return out


def _blank() -> dict[str, Any]:
    return {
        "n": 0.0,
        "bias": 0.0,
        "w": [0.0] * len(FEATURES),
        "mu": [0.0] * len(FEATURES),
        "var": [1.0] * len(FEATURES),
        "selected_sum": 0.0,
        "taken": 0,
        "abstain": 0,
        "total_sum": 0.0,
    }


def _load(conn: sqlite3.Connection, engine: str) -> dict[str, Any]:
    try:
        row = conn.execute("SELECT * FROM external_policy_model WHERE engine_id=?", (engine,)).fetchone()
    except sqlite3.OperationalError:
        return _blank()
    if row is None:
        return _blank()
    state = _blank()
    state["n"] = float(row[1])
    state["bias"] = float(row[2])
    state["w"] = [float(v) for v in json.loads(row[3])]
    state["mu"] = [float(v) for v in json.loads(row[4])]
    state["var"] = [float(v) for v in json.loads(row[5])]
    state["selected_sum"] = float(row[6])
    state["taken"] = int(row[7])
    state["abstain"] = int(row[8])
    state["total_sum"] = float(row[9])
    return state


def _predict_state(state: dict[str, Any], values: list[float]) -> float:
    if float(state["n"]) <= 0.0:
        return 0.0
    z = []
    for value, mu, var in zip(values, state["mu"], state["var"], strict=True):
        z.append(max(-5.0, min(5.0, (value - mu) / math.sqrt(var + 1e-9))))
    return float(state["bias"]) + sum(weight * zi for weight, zi in zip(state["w"], z, strict=True))


def predict_external(db_path: str, engine: str, values: list[float]) -> float:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return _predict_state(_load(conn, engine), values)
    finally:
        conn.close()


def capture_external(db_path: str, engine: str, symbol: str) -> dict[str, Any] | None:
    """Read the outside-venue snapshot and the prediction it implies. No model write."""
    from backend.services.external_venue_feed import read_discovery

    features = read_discovery(symbol)
    if features is None:
        return None
    try:
        ensure_schema(db_path)
    except sqlite3.Error:
        return None
    return freeze_external(db_path, engine, features)


def freeze_external(db_path: str, engine: str, features: dict[str, Any] | None) -> dict[str, Any] | None:
    """Prediction from the model as it stands now. Does not update the model."""
    values = feature_vector(features)
    if values is None:
        return None
    try:
        prediction = predict_external(db_path, engine, values)
    except sqlite3.Error:
        prediction = 0.0
    return {"features": features, "vector": values, "prediction": prediction, "prediction_at": time.time()}


def apply_external_outcome(conn: sqlite3.Connection, row: Any, net: float, moment: float) -> None:
    """One update after the policy net is stored. Already-applied episodes are skipped."""
    try:
        snap = json.loads(row["snapshot_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return
    block = snap.get("external_discovery") if isinstance(snap, dict) else None
    values = feature_vector((block or {}).get("features") if isinstance(block, dict) else None)
    if values is None:
        return
    try:
        seen = conn.execute("SELECT 1 FROM external_policy_applied WHERE episode_id=?", (int(row["id"]),)).fetchone()
    except sqlite3.OperationalError:
        return
    if seen is not None:
        return
    state = _load(conn, str(row["engine_id"]))
    prediction = _finite((block or {}).get("prediction"))
    if prediction is None:
        prediction = _predict_state(state, values)
    z = []
    for value, mu, var in zip(values, state["mu"], state["var"], strict=True):
        z.append(max(-5.0, min(5.0, (value - float(mu)) / math.sqrt(float(var) + 1e-9))))
    err = (sum(weight * zi for weight, zi in zip(state["w"], z, strict=True)) + float(state["bias"])) - float(net)
    step = 0.05 / (1.0 + sum(zi * zi for zi in z))
    state["w"] = [weight - step * err * zi - 0.05 * 1e-3 * weight for weight, zi in zip(state["w"], z, strict=True)]
    state["bias"] = float(state["bias"]) - 0.05 * err
    rate = 0.05
    state["mu"] = [(1.0 - rate) * float(mu) + rate * value for mu, value in zip(state["mu"], values, strict=True)]
    state["var"] = [
        max(1e-9, (1.0 - rate) * float(var) + rate * (value - mu) ** 2)
        for var, value, mu in zip(state["var"], values, state["mu"], strict=True)
    ]
    state["n"] = float(state["n"]) + 1.0
    state["total_sum"] = float(state["total_sum"]) + float(net)
    if prediction > 0.0:
        state["selected_sum"] = float(state["selected_sum"]) + float(net)
        state["taken"] = int(state["taken"]) + 1
    else:
        state["abstain"] = int(state["abstain"]) + 1
    conn.execute(
        """
        INSERT INTO external_policy_model (
            engine_id, n, bias, w_json, mu_json, var_json, selected_sum, taken, abstain, total_sum, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(engine_id) DO UPDATE SET
            n=excluded.n, bias=excluded.bias, w_json=excluded.w_json, mu_json=excluded.mu_json,
            var_json=excluded.var_json, selected_sum=excluded.selected_sum, taken=excluded.taken,
            abstain=excluded.abstain, total_sum=excluded.total_sum, updated_at=excluded.updated_at
        """,
        (
            str(row["engine_id"]),
            state["n"],
            state["bias"],
            json.dumps(state["w"]),
            json.dumps(state["mu"]),
            json.dumps(state["var"]),
            state["selected_sum"],
            state["taken"],
            state["abstain"],
            state["total_sum"],
            float(moment),
        ),
    )
    conn.execute(
        "INSERT INTO external_policy_applied (episode_id, prediction, net, applied_at) VALUES (?,?,?,?)",
        (int(row["id"]), float(prediction), float(net), float(moment)),
    )


def evidence(db_path: str, engine: str) -> dict[str, float | None]:
    """Selected policy net versus taking every resolved episode. Read only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        state = _load(conn, engine)
    finally:
        conn.close()
    n = int(state["taken"]) + int(state["abstain"])
    if n <= 0:
        return {"n": 0.0, "selected_mean": None, "always_mean": None}
    return {
        "n": float(n),
        "selected_mean": float(state["selected_sum"]) / n,
        "always_mean": float(state["total_sum"]) / n,
        "taken": float(state["taken"]),
    }
