"""Research-only state-conditioned policy learners. Not live entry authority.

A prediction is fit on episodes that have already resolved. The episode being
opened is not in that fit. The stored prediction is what gets scored. Closing
the episode makes it training data for the next one, and does not rewrite the
prediction already written on it.

DAY and SCALP keep separate training rows. The target is realized policy net.
A relative model predicts that net minus the same group's average, and it is
only a ranking aid.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import numpy as np

SCHEMA = "STATE_V1"
MAX_TRAIN = 2000
RIDGE_ALPHA = 10.0
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
VALUE_MODELS = ("ridge", "huber", "sgd", "tree", "ridge_no_alpha", "ridge_no_gap", "ridge_no_both")
CHALLENGER_DDL = """
CREATE TABLE IF NOT EXISTS policy_challenger_scores (
    decision_group_id TEXT NOT NULL,
    model TEXT NOT NULL,
    engine_id TEXT NOT NULL,
    decided_at REAL NOT NULL,
    spearman REAL,
    selected_net REAL,
    regret REAL,
    oracle_net REAL,
    oracle_positive REAL,
    capture REAL,
    fp_count INTEGER,
    fp_net REAL,
    missed_count INTEGER,
    missed_net REAL,
    abstained INTEGER,
    PRIMARY KEY (decision_group_id, model)
)
"""
HORIZON_DDL = """
CREATE TABLE IF NOT EXISTS policy_horizon_marks (
    episode_id INTEGER NOT NULL,
    horizon_sec INTEGER NOT NULL,
    marked_at REAL NOT NULL,
    bid REAL,
    net REAL,
    exit_net REAL,
    PRIMARY KEY (episode_id, horizon_sec)
)
"""
HORIZONS = (60, 300)

_DAY_STATE = (
    "atr15_pct",
    "atr1h_pct",
    "rsi",
    "bb_pct",
    "sma20_dist",
    "h1_ret",
    "bar_return",
    "range_location",
    "setup_htf",
    "setup_range",
    "setup_breakout",
    "setup_vwap",
    "setup_exhaustion",
)
_SCALP_STATE = (
    "microprice_pressure",
    "microprice_accel",
    "obi_l1",
    "obi_l5",
    "obi_l10",
    "obi_l20",
    "ofi_1s",
    "ofi_5s",
    "ofi_30s",
    "agg_flow_imbalance_5s",
    "adverse_selection_score",
    "spread_pct",
    "obi_l10_persistence_5s",
    "depth_fragility",
    "bid_absorption_score",
    "ask_absorption_score",
    "cancel_imbalance_5s",
    "near_touch_depth_loss",
)
_CONTEXT = ("market_alpha", "policy_gap", "policy_calibration", "live_rank", "expected_cost", "peer_mean", "own_minus_peer")
_SYMBOL = ("sym_btc", "sym_eth", "sym_sol", "sym_xrp")
DAY_FEATURES = _DAY_STATE + _CONTEXT + _SYMBOL
SCALP_FEATURES = _SCALP_STATE + _CONTEXT + _SYMBOL
FAMILIES = {
    "volatility": ("atr15_pct", "atr1h_pct"),
    "range": ("rsi", "bb_pct", "range_location"),
    "trend": ("sma20_dist", "h1_ret", "bar_return"),
    "setup": ("setup_htf", "setup_range", "setup_breakout", "setup_vwap", "setup_exhaustion"),
    "queue": ("obi_l1", "obi_l5", "obi_l10", "obi_l20", "depth_fragility", "near_touch_depth_loss", "cancel_imbalance_5s"),
    "microprice": ("microprice_pressure", "microprice_accel"),
    "flow": ("ofi_1s", "ofi_5s", "ofi_30s", "agg_flow_imbalance_5s", "bid_absorption_score", "ask_absorption_score", "adverse_selection_score"),
    "cost": ("spread_pct", "expected_cost"),
    "market_alpha": ("market_alpha",),
    "policy_gap": ("policy_gap",),
    "calibration": ("policy_calibration",),
    "cross_coin": ("live_rank", "peer_mean", "own_minus_peer", "sym_btc", "sym_eth", "sym_sol", "sym_xrp"),
}


def feature_names(engine: str) -> tuple[str, ...]:
    return DAY_FEATURES if str(engine).upper().startswith("DAY") else SCALP_FEATURES


def _num(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _canon(symbol: str) -> str:
    text = str(symbol or "").upper().replace("-", "").replace("/", "")
    if text in {"BTC", "ETH", "SOL", "XRP"}:
        return text + "USDT"
    return text


def vector(engine: str, features: dict[str, Any] | None, issued: dict[str, Any] | None, symbol: str, peers: list[float] | None) -> list[float]:
    """One causal row. Missing inputs are zero. Peer values are other coins' live values, known now."""
    names = feature_names(engine)
    raw = features if isinstance(features, dict) else {}
    econ = issued if isinstance(issued, dict) else {}
    own = _num(econ.get("policy_value"))
    others = [float(v) for v in (peers or []) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
    peer_mean = sum(others) / len(others) if others else 0.0
    coin = _canon(symbol)
    values = {
        "market_alpha": _num(econ.get("market_alpha")),
        "policy_gap": _num(econ.get("policy_gap")),
        "policy_calibration": _num(econ.get("policy_calibration")),
        "live_rank": _num(econ.get("rank_position")),
        "expected_cost": _num(econ.get("expected_cost") if econ.get("expected_cost") is not None else econ.get("roundtrip_cost")),
        "peer_mean": peer_mean,
        "own_minus_peer": own - peer_mean,
        "sym_btc": 1.0 if coin == "BTCUSDT" else 0.0,
        "sym_eth": 1.0 if coin == "ETHUSDT" else 0.0,
        "sym_sol": 1.0 if coin == "SOLUSDT" else 0.0,
        "sym_xrp": 1.0 if coin == "XRPUSDT" else 0.0,
    }
    for name in names:
        if name not in values:
            values[name] = _num(raw.get(name))
    return [values[name] for name in names]


def _standardize(train: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return np.clip((train - mu) / sd, -5.0, 5.0), np.clip((row - mu) / sd, -5.0, 5.0)


def _ridge(train: np.ndarray, y: np.ndarray, row: np.ndarray, *, drop: tuple[str, ...] = (), names: tuple[str, ...] = ()) -> tuple[float, np.ndarray]:
    if len(y) == 0:
        return 0.0, np.zeros(train.shape[1] if train.ndim == 2 and len(train) else row.shape[0])
    x = np.array(train, dtype=float, copy=True)
    zrow = np.array(row, dtype=float, copy=True)
    if drop and names:
        for i, name in enumerate(names):
            if name in drop:
                x[:, i] = 0.0
                zrow[i] = 0.0
    if len(y) == 1:
        return float(y[0]), np.zeros(x.shape[1])
    zt, zr = _standardize(x, zrow)
    mu = float(y.mean())
    gram = zt.T @ zt + RIDGE_ALPHA * np.eye(zt.shape[1])
    coef = np.linalg.solve(gram, zt.T @ (y - mu))
    return float(zr @ coef + mu), coef


def _huber(train: np.ndarray, y: np.ndarray, row: np.ndarray) -> float:
    if len(y) < 8:
        pred, _coef = _ridge(train, y, row)
        return pred
    zt, zr = _standardize(train, row)
    mu = float(y.mean())
    coef = np.zeros(zt.shape[1])
    target = y - mu
    for _ in range(12):
        resid = target - zt @ coef
        scale = float(np.median(np.abs(resid))) / 0.6745
        if scale < 1e-12:
            return mu
        limit = 1.35 * scale
        weight = np.ones(len(resid))
        big = np.abs(resid) > limit
        weight[big] = limit / np.abs(resid[big])
        sw = np.sqrt(weight)
        gram = (zt * sw[:, None]).T @ (zt * sw[:, None]) + RIDGE_ALPHA * np.eye(zt.shape[1])
        coef = np.linalg.solve(gram, (zt * sw[:, None]).T @ (target * sw))
    return float(zr @ coef + mu)


def _sgd(train: np.ndarray, y: np.ndarray, row: np.ndarray) -> float:
    """Replay prior rows in order. Standardization at each step uses only earlier rows."""
    dim = int(row.shape[0])
    w = np.zeros(dim)
    bias = 0.0
    mu = np.zeros(dim)
    var = np.ones(dim)
    seen = 0
    for i in range(len(y)):
        sd = np.sqrt(var + 1e-9)
        z = np.clip((train[i] - mu) / sd, -5.0, 5.0)
        err = float(z @ w + bias - y[i])
        step = 0.05 / (1.0 + float(z @ z))
        w -= step * err * z + 0.05 * 1e-3 * w
        bias -= 0.05 * err
        seen += 1
        rate = 0.05
        mu = (1.0 - rate) * mu + rate * train[i]
        var = np.maximum((1.0 - rate) * var + rate * (train[i] - mu) ** 2, 1e-9)
    sd = np.sqrt(var + 1e-9)
    zrow = np.clip((row - mu) / sd, -5.0, 5.0)
    if seen == 0:
        return 0.0
    return float(zrow @ w + bias)


def _tree(train: np.ndarray, y: np.ndarray, row: np.ndarray) -> float:
    if len(y) < 8:
        return float(y.mean()) if len(y) else 0.0
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(1, min(4, len(y) // 5)), random_state=0)
    model.fit(train, y)
    return float(model.predict(row.reshape(1, -1))[0])


def _predict_row(train: np.ndarray, y: np.ndarray, row: np.ndarray, names: tuple[str, ...], relative_train: tuple[np.ndarray, np.ndarray] | None) -> dict[str, float]:
    ridge, _coef = _ridge(train, y, row)
    no_alpha, _c = _ridge(train, y, row, drop=("market_alpha",), names=names)
    no_gap, _c = _ridge(train, y, row, drop=("policy_gap",), names=names)
    neither, _c = _ridge(train, y, row, drop=("market_alpha", "policy_gap"), names=names)
    relative = 0.0
    if relative_train is not None and len(relative_train[1]):
        relative, _c = _ridge(relative_train[0], relative_train[1], row)
    return {
        "ridge": ridge,
        "huber": _huber(train, y, row),
        "sgd": _sgd(train, y, row),
        "tree": _tree(train, y, row),
        "relative": relative,
        "ridge_no_alpha": no_alpha,
        "ridge_no_gap": no_gap,
        "ridge_no_both": neither,
    }


def _matrix(rows: list[list[float]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1))
    return np.array(rows, dtype=float)


def _relative_training(groups: dict[str, list[tuple[list[float], float, str]]]) -> tuple[np.ndarray, np.ndarray] | None:
    xs: list[list[float]] = []
    ys: list[float] = []
    for members in groups.values():
        coins = {_canon(sym) for _x, _y, sym in members}
        if any(coin not in coins for coin in COINS):
            continue
        nets = [net for _x, net, _sym in members]
        center = sum(nets) / len(nets)
        for feat, net, _sym in members:
            xs.append(feat)
            ys.append(net - center)
    if not xs:
        return None
    return _matrix(xs), np.array(ys, dtype=float)


def _closed_rows(db_path: str, engine: str) -> list[sqlite3.Row]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        return list(
            conn.execute(
                """
                SELECT e.symbol, e.decided_at, e.exit_at, e.net, e.decision_group_id, e.snapshot_json, m.features_json
                FROM policy_episodes e
                LEFT JOIN adaptive_candidate_markouts m ON m.id = e.candidate_id
                WHERE e.engine_id=? AND e.status='CLOSED' AND e.net IS NOT NULL
                ORDER BY e.exit_at DESC, e.id DESC
                LIMIT ?
                """,
                (str(engine), MAX_TRAIN),
            ).fetchall()
        )
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _training_vector(row: sqlite3.Row, engine: str) -> list[float] | None:
    """Feature vector known at the decision. The resolved net is not an input."""
    names = feature_names(engine)
    stored = _stored_x(str(row["snapshot_json"]))
    if stored is not None and len(stored) == len(names):
        return stored
    try:
        snap = json.loads(row["snapshot_json"] or "{}")
        try:
            raw_feats = row["features_json"]
        except IndexError:
            raw_feats = "{}"
        feats = json.loads(raw_feats or "{}")
    except (json.JSONDecodeError, IndexError):
        return None
    issued = snap.get("issued") if isinstance(snap, dict) and isinstance(snap.get("issued"), dict) else {}
    issued = dict(issued)
    if isinstance(snap, dict):
        issued.setdefault("expected_cost", snap.get("expected_cost"))
    return vector(engine, feats if isinstance(feats, dict) else {}, issued, str(row["symbol"]), None)


def _stored_x(raw: str) -> list[float] | None:
    try:
        snap = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    block = snap.get("state_features") if isinstance(snap, dict) else None
    if not isinstance(block, dict) or block.get("schema") != SCHEMA:
        return None
    values = block.get("x")
    if not isinstance(values, list) or not values:
        return None
    return [float(v) for v in values]


def freeze_prediction(
    db_path: str,
    *,
    engine: str,
    features: dict[str, Any] | None,
    issued: dict[str, Any] | None,
    symbol: str,
    peers: list[float] | None,
) -> dict[str, Any]:
    """Prediction from resolved episodes only. This call does not write."""
    names = feature_names(engine)
    current = vector(engine, features, issued, symbol, peers)
    prior = list(reversed(_closed_rows(db_path, engine)))
    xs: list[list[float]] = []
    ys: list[float] = []
    groups: dict[str, list[tuple[list[float], float, str]]] = {}
    for row in prior:
        feat = _training_vector(row, engine)
        if feat is None or len(feat) != len(names):
            continue
        xs.append(feat)
        ys.append(float(row["net"]))
        groups.setdefault(str(row["decision_group_id"] or ""), []).append((feat, float(row["net"]), str(row["symbol"])))
    train = _matrix(xs)
    target = np.array(ys, dtype=float)
    row = np.array(current, dtype=float)
    relative = _relative_training(groups)
    pred = _predict_row(train, target, row, names, relative)
    pred["n"] = float(len(ys))
    return {"features": {"schema": SCHEMA, "x": current}, "predictions": pred}


def ridge_families(engine: str, xs: list[list[float]], ys: list[float]) -> dict[str, float]:
    """Absolute ridge weight by feature family, from a causal training matrix."""
    names = feature_names(engine)
    if not xs:
        return {}
    _pred, coef = _ridge(_matrix(xs), np.array(ys, dtype=float), np.zeros(len(names)))
    weight = {name: abs(float(coef[i])) for i, name in enumerate(names)}
    out = {}
    for family, members in FAMILIES.items():
        out[family] = sum(weight.get(name, 0.0) for name in members)
    return out


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    xs, ys = ranks(left), ranks(right)
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    dx = sum((a - mx) ** 2 for a in xs)
    dy = sum((b - my) ** 2 for b in ys)
    if dx <= 0.0 or dy <= 0.0:
        return None
    return num / ((dx**0.5) * (dy**0.5))


def score_predictions(preds: list[float | None], nets: list[float]) -> dict[str, Any]:
    """Score one four-coin group from predictions written before the outcomes."""
    usable = [p is not None for p in preds]
    complete = all(usable) and len(preds) == len(nets) and len(nets) >= 2
    best = max(nets) if nets else 0.0
    oracle_positive = best if best > 0.0 else 0.0
    if not complete:
        return {
            "complete": 0,
            "spearman": None,
            "selected_net": None,
            "regret": None,
            "oracle_net": best,
            "oracle_positive": oracle_positive,
            "capture": None,
            "fp_count": 0,
            "fp_net": 0.0,
            "missed_count": 0,
            "missed_net": 0.0,
            "abstained": None,
        }
    values = [float(p) for p in preds if p is not None]
    order = max(range(len(values)), key=lambda i: values[i])
    selected = float(nets[order]) if values[order] > 0.0 else 0.0
    abstained = 1 if values[order] <= 0.0 else 0
    fp_net = sum(nets[i] for i, pred in enumerate(values) if pred > 0.0 and nets[i] <= 0.0)
    fp_count = sum(1 for i, pred in enumerate(values) if pred > 0.0 and nets[i] <= 0.0)
    missed_net = sum(nets[i] for i, pred in enumerate(values) if pred <= 0.0 and nets[i] > 0.0)
    missed_count = sum(1 for i, pred in enumerate(values) if pred <= 0.0 and nets[i] > 0.0)
    return {
        "complete": 1,
        "spearman": _spearman(values, nets),
        "selected_net": selected,
        "regret": best - selected,
        "oracle_net": best,
        "oracle_positive": oracle_positive,
        "capture": None if oracle_positive <= 0.0 else selected / oracle_positive,
        "fp_count": fp_count,
        "fp_net": fp_net,
        "missed_count": missed_count,
        "missed_net": missed_net,
        "abstained": abstained,
    }


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CHALLENGER_DDL)
    conn.execute(HORIZON_DDL)


def score_group(conn: sqlite3.Connection, group_id: str, primaries: list[dict[str, Any]], moment: float) -> None:
    """Write challenger metrics from the predictions stored on the episodes."""
    ensure_schema(conn)
    if not primaries:
        return
    engine = str(primaries[0].get("engine") or "")
    decided = float(primaries[0].get("decided_at") or moment)
    nets = [float(row["net"]) for row in primaries]
    models = {
        "live": [row.get("live") for row in primaries],
        "hierarchical": [row.get("direct") for row in primaries],
    }
    for name in (*VALUE_MODELS, "relative"):
        models[name] = [(row.get("state") or {}).get(name) for row in primaries]
    for name, preds in models.items():
        scored = score_predictions(preds, nets)
        if not scored["complete"]:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO policy_challenger_scores (
                decision_group_id, model, engine_id, decided_at, spearman, selected_net, regret,
                oracle_net, oracle_positive, capture, fp_count, fp_net, missed_count, missed_net, abstained
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                group_id,
                name,
                engine,
                decided,
                scored["spearman"],
                scored["selected_net"],
                scored["regret"],
                scored["oracle_net"],
                scored["oracle_positive"],
                scored["capture"],
                scored["fp_count"],
                scored["fp_net"],
                scored["missed_count"],
                scored["missed_net"],
                scored["abstained"],
            ),
        )


def record_due_horizons(conn: sqlite3.Connection, bid_at: Any, now: float) -> None:
    """After an exit, record a later executable bid. This does not change the exit."""
    from backend.services.adaptive_learning import executable_bid_net

    ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT id, symbol, entry_ask, roundtrip_cost, exit_at, net
        FROM policy_episodes
        WHERE status='CLOSED' AND exit_at IS NOT NULL AND net IS NOT NULL AND ? - exit_at BETWEEN 60 AND 3600
        ORDER BY exit_at DESC LIMIT 20
        """,
        (float(now),),
    ).fetchall()
    for row in rows:
        for horizon in HORIZONS:
            due = float(row["exit_at"]) + horizon
            if due > float(now):
                continue
            exists = conn.execute("SELECT 1 FROM policy_horizon_marks WHERE episode_id=? AND horizon_sec=?", (int(row["id"]), horizon)).fetchone()
            if exists is not None:
                continue
            bid = bid_at(str(row["symbol"]))
            if bid is None or float(bid) <= 0.0:
                continue
            net = executable_bid_net(float(row["entry_ask"]), float(bid), float(row["roundtrip_cost"] or 0.0))
            conn.execute(
                "INSERT OR IGNORE INTO policy_horizon_marks (episode_id, horizon_sec, marked_at, bid, net, exit_net) VALUES (?,?,?,?,?,?)",
                (int(row["id"]), horizon, float(now), float(bid), float(net), float(row["net"])),
            )


def walk_rows(db_path: str) -> list[dict[str, Any]]:
    """Closed episodes with a reconstructed causal vector. Read only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            """
            SELECT e.engine_id, e.symbol, e.decided_at, e.exit_at, e.net, e.decision_group_id, e.snapshot_json, m.features_json
            FROM policy_episodes e
            LEFT JOIN adaptive_candidate_markouts m ON m.id = e.candidate_id
            WHERE e.status='CLOSED' AND e.net IS NOT NULL
            ORDER BY e.exit_at, e.id
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for row in found:
        try:
            snap = json.loads(row["snapshot_json"] or "{}")
            feats = json.loads(row["features_json"] or "{}")
        except json.JSONDecodeError:
            continue
        issued = snap.get("issued") if isinstance(snap, dict) else {}
        if not isinstance(issued, dict):
            issued = {}
        issued = dict(issued)
        issued.setdefault("expected_cost", snap.get("expected_cost") if isinstance(snap, dict) else None)
        direct = snap.get("direct_policy") if isinstance(snap, dict) else {}
        names = feature_names(str(row["engine_id"]))
        stored = _stored_x(row["snapshot_json"])
        feat = stored if stored is not None and len(stored) == len(names) else vector(str(row["engine_id"]), feats if isinstance(feats, dict) else {}, issued, str(row["symbol"]), None)
        out.append(
            {
                "engine": str(row["engine_id"]),
                "symbol": _canon(str(row["symbol"])),
                "decided_at": float(row["decided_at"]),
                "exit_at": float(row["exit_at"] or row["decided_at"]),
                "net": float(row["net"]),
                "group": str(row["decision_group_id"] or ""),
                "x": feat,
                "live": issued.get("policy_value"),
                "direct": direct.get("mean") if isinstance(direct, dict) else None,
            }
        )
    return out


def _group_primaries(members: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in members:
        by_symbol.setdefault(row["symbol"], []).append(row)
    if any(coin not in by_symbol for coin in COINS):
        return None

    def _live(row: dict[str, Any]) -> float:
        try:
            return float(row["live"])
        except (TypeError, ValueError):
            return -1e99

    chosen = []
    for coin in COINS:
        best = max(by_symbol[coin], key=lambda row: (_live(row), -row["decided_at"]))
        chosen.append(best)
    return chosen


def walk_forward(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Chronological replay. Each group is predicted from episodes that had already exited."""
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], []).append(row)
    report: dict[str, dict[str, Any]] = {}
    for engine, engine_rows in by_engine.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in engine_rows:
            groups.setdefault(row["group"], []).append(row)
        ordered = sorted(groups, key=lambda gid: min(r["decided_at"] for r in groups[gid]))
        scores: dict[str, list[dict[str, Any]]] = {}
        last_coef: dict[str, float] = {}
        names = feature_names(engine)
        for gid in ordered:
            primaries = _group_primaries(groups[gid])
            if primaries is None:
                continue
            decided = min(r["decided_at"] for r in primaries)
            train = [r for r in engine_rows if r["exit_at"] < decided]
            xs = [r["x"] for r in train if len(r["x"]) == len(names)]
            ys = [r["net"] for r in train if len(r["x"]) == len(names)]
            grouped: dict[str, list[tuple[list[float], float, str]]] = {}
            for r in train:
                if len(r["x"]) == len(names):
                    grouped.setdefault(r["group"], []).append((r["x"], r["net"], r["symbol"]))
            matrix = _matrix(xs)
            target = np.array(ys, dtype=float)
            relative = _relative_training(grouped)
            pred_rows = []
            for primary in primaries:
                pred = _predict_row(matrix, target, np.array(primary["x"], dtype=float), names, relative)
                pred_rows.append(pred)
            if xs:
                last_coef = ridge_families(engine, xs, ys)
            nets = [p["net"] for p in primaries]
            bundle = {
                "live": [p["live"] if isinstance(p["live"], (int, float)) and not isinstance(p["live"], bool) else None for p in primaries],
                "hierarchical": [p["direct"] if isinstance(p["direct"], (int, float)) and not isinstance(p["direct"], bool) else None for p in primaries],
            }
            for name in (*VALUE_MODELS, "relative"):
                bundle[name] = [row[name] for row in pred_rows]
            for name, preds in bundle.items():
                scored = score_predictions(preds, nets)
                if scored["complete"]:
                    scores.setdefault(name, []).append(scored)
        report[engine] = {"models": {name: _aggregate(items) for name, items in scores.items()}, "families": last_coef, "groups": len(ordered)}
    return report


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [float(row[key]) for row in items if row.get(key) is not None]
        return None if not vals else sum(vals) / len(vals)

    selected = [float(row["selected_net"]) for row in items if row.get("selected_net") is not None]
    gains = sum(v for v in selected if v > 0.0)
    losses = -sum(v for v in selected if v < 0.0)
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in selected:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    oracle_pos = [float(row["oracle_positive"]) for row in items]
    return {
        "groups": len(items),
        "spearman": avg("spearman"),
        "selected_net": avg("selected_net"),
        "regret": avg("regret"),
        "oracle_net": avg("oracle_net"),
        "oracle_positive": avg("oracle_positive"),
        "capture": None if sum(oracle_pos) <= 0.0 else sum(selected) / sum(oracle_pos),
        "pf": None if losses <= 0.0 else gains / losses,
        "drawdown": drawdown,
        "fp_count": sum(int(row["fp_count"]) for row in items),
        "fp_net": sum(float(row["fp_net"]) for row in items),
        "missed_count": sum(int(row["missed_count"]) for row in items),
        "missed_net": sum(float(row["missed_net"]) for row in items),
        "abstained": sum(int(row["abstained"] or 0) for row in items),
    }


def opportunity(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Best policy net in each complete four-coin group, by engine."""
    out: dict[str, dict[str, float]] = {}
    by_engine: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], {}).setdefault(row["group"], []).append(row)
    for engine, groups in by_engine.items():
        bests = []
        for members in groups.values():
            primaries = _group_primaries(members)
            if primaries is None:
                continue
            bests.append(max(r["net"] for r in members))
        if not bests:
            continue
        ordered = sorted(bests)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        positive = [v for v in bests if v > 0.0]
        equity = 0.0
        peak = 0.0
        drawdown = 0.0
        gains = sum(v for v in positive)
        losses = -sum(v for v in bests if v <= 0.0)
        no_trade = [v if v > 0.0 else 0.0 for v in bests]
        for value in no_trade:
            equity += value
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        out[engine] = {
            "groups": float(len(bests)),
            "positive_groups": float(len(positive)),
            "positive_rate": len(positive) / len(bests),
            "mean_best": sum(bests) / len(bests),
            "median_best": median,
            "p90_best": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
            "oracle_sum": sum(bests),
            "oracle_avg": sum(bests) / len(bests),
            "oracle_pf": None if losses <= 0.0 else gains / losses,
            "no_trade_avg": sum(no_trade) / len(no_trade),
            "no_trade_sum": sum(no_trade),
            "no_trade_drawdown": drawdown,
        }
    return out
