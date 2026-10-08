"""Research-only group opportunity and coin residual. Not live entry authority.

A four-coin policy group has one shared outcome and four coin residuals.
Predictions are fit on groups that have already exited. They are written onto
episodes that are still open. Closing the group does not rewrite them.

There is no probability cutoff. A coin is economically positive only when its
learned expected net is above zero, the same boundary live funding already uses.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import numpy as np

from backend.services.policy_state_learner import (
    COINS,
    _huber,
    _matrix,
    _ridge,
    _sgd,
    _spearman,
    _training_vector,
    _tree,
    feature_names,
)

SCHEMA = "GROUP_V1"
DAY_KEYS = ("h1_ret", "bar_return", "atr15_pct", "range_location", "sma20_dist", "expected_cost", "market_alpha", "policy_gap")
SCALP_KEYS = (
    "ofi_1s",
    "ofi_5s",
    "ofi_30s",
    "obi_l1",
    "obi_l5",
    "microprice_pressure",
    "microprice_accel",
    "spread_pct",
    "agg_flow_imbalance_5s",
    "adverse_selection_score",
    "depth_fragility",
    "bid_absorption_score",
    "ask_absorption_score",
    "cancel_imbalance_5s",
    "expected_cost",
)
SCORE_DDL = """
CREATE TABLE IF NOT EXISTS policy_group_model_scores (
    decision_group_id TEXT NOT NULL,
    model TEXT NOT NULL,
    engine_id TEXT NOT NULL,
    decided_at REAL NOT NULL,
    predicted_best REAL,
    selected_net REAL,
    regret REAL,
    spearman REAL,
    fp_group INTEGER,
    fn_group INTEGER,
    fp_loss REAL,
    fn_missed REAL,
    positive_group INTEGER,
    PRIMARY KEY (decision_group_id, model)
)
"""


def shared_keys(engine: str) -> tuple[str, ...]:
    return DAY_KEYS if str(engine).upper().startswith("DAY") else SCALP_KEYS


def group_feature_names(engine: str) -> tuple[str, ...]:
    keys = shared_keys(engine)
    names = [item for key in keys for item in (f"mean_{key}", f"std_{key}", f"btc_gap_{key}")]
    names += [f"d_mean_{key}" for key in keys[:4]]
    names += ["btc_lead", "d_btc_lead"]
    return tuple(names)


def residual_names(engine: str) -> tuple[str, ...]:
    return (*(f"dev_{key}" for key in shared_keys(engine)), "sym_btc", "sym_eth", "sym_sol", "sym_xrp")


def targets(nets: dict[str, float]) -> dict[str, Any]:
    """Research targets from four resolved policy nets. Not a live input."""
    ordered = [float(nets[coin]) for coin in COINS]
    center = sum(ordered) / len(ordered)
    best = max(ordered)
    worst = min(ordered)
    return {
        "mean": center,
        "best": best,
        "worst": worst,
        "spread": best - worst,
        "positive_value": max(best, 0.0),
        "residual": {coin: float(nets[coin]) - center for coin in COINS},
    }


def reconstructs(nets: dict[str, float]) -> bool:
    """Group mean plus coin residual is the coin's policy net."""
    made = targets(nets)
    return all(math.isclose(made["mean"] + made["residual"][coin], float(nets[coin])) for coin in COINS)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    center = sum(values) / len(values)
    var = sum((v - center) ** 2 for v in values) / len(values)
    return center, math.sqrt(var)


def _lead_key(engine: str) -> str:
    return "bar_return" if str(engine).upper().startswith("DAY") else "ofi_5s"


def cross_section(named: dict[str, dict[str, float]], engine: str) -> dict[str, float]:
    keys = shared_keys(engine)
    out: dict[str, float] = {}
    btc = named.get("BTCUSDT", {})
    for key in keys:
        values = [float(named.get(coin, {}).get(key, 0.0)) for coin in COINS]
        center, scale = _mean_std(values)
        others = [float(named.get(coin, {}).get(key, 0.0)) for coin in COINS if coin != "BTCUSDT"]
        other_mean = sum(others) / len(others) if others else 0.0
        out[f"mean_{key}"] = center
        out[f"std_{key}"] = scale
        out[f"btc_gap_{key}"] = float(btc.get(key, 0.0)) - other_mean
    return out


def group_vector(named: dict[str, dict[str, float]], engine: str, previous: dict[str, float] | None) -> list[float]:
    """Causal group state. ``previous`` is an earlier decision's state, not an outcome."""
    current = cross_section(named, engine)
    prev = previous or {}
    lead = _lead_key(engine)
    current["btc_lead"] = float(named.get("BTCUSDT", {}).get(lead, 0.0))
    current["d_btc_lead"] = current["btc_lead"] - float(prev.get("btc_lead", 0.0))
    for key in shared_keys(engine)[:4]:
        current[f"d_mean_{key}"] = current[f"mean_{key}"] - float(prev.get(f"mean_{key}", 0.0))
    names = group_feature_names(engine)
    return [float(current.get(name, 0.0)) for name in names]


def residual_vector(named_coin: dict[str, float], means: dict[str, float], engine: str, symbol: str) -> list[float]:
    keys = shared_keys(engine)
    dev = [float(named_coin.get(key, 0.0)) - float(means.get(f"mean_{key}", 0.0)) for key in keys]
    coin = str(symbol)
    flags = [1.0 if coin == "BTCUSDT" else 0.0, 1.0 if coin == "ETHUSDT" else 0.0, 1.0 if coin == "SOLUSDT" else 0.0, 1.0 if coin == "XRPUSDT" else 0.0]
    return dev + flags


def _named_from_x(engine: str, values: list[float]) -> dict[str, float]:
    names = feature_names(engine)
    if len(values) != len(names):
        return {}
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def variance_split(groups: list[dict[str, Any]]) -> dict[str, float]:
    """Share of coin-net variance that is common to the group versus coin-specific."""
    rows = []
    for group in groups:
        nets = group.get("nets") or {}
        if any(coin not in nets for coin in COINS):
            continue
        center = sum(float(nets[coin]) for coin in COINS) / 4.0
        for coin in COINS:
            rows.append((center, float(nets[coin])))
    if len(rows) < 4:
        return {"shared": 0.0, "residual": 0.0, "correlation": 0.0}
    grand = sum(net for _center, net in rows) / len(rows)
    ss_total = sum((net - grand) ** 2 for _center, net in rows)
    ss_shared = sum((center - grand) ** 2 for center, _net in rows)
    shared = 0.0 if ss_total <= 0.0 else ss_shared / ss_total
    pair = []
    for a, b in (("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT"), ("BTCUSDT", "XRPUSDT"), ("ETHUSDT", "SOLUSDT"), ("ETHUSDT", "XRPUSDT"), ("SOLUSDT", "XRPUSDT")):
        xs = [float(g["nets"][a]) for g in groups if a in g.get("nets", {}) and b in g.get("nets", {})]
        ys = [float(g["nets"][b]) for g in groups if a in g.get("nets", {}) and b in g.get("nets", {})]
        if len(xs) >= 3:
            corr = _pearson(xs, ys)
            if corr is not None:
                pair.append(corr)
    return {"shared": shared, "residual": 1.0 - shared, "correlation": sum(pair) / len(pair) if pair else 0.0}


def sign_structure(groups: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {"all_negative": [], "all_positive": [], "mixed": []}
    for group in groups:
        nets = [float(group["nets"][coin]) for coin in COINS if coin in group.get("nets", {})]
        if len(nets) != 4:
            continue
        if all(v <= 0.0 for v in nets):
            buckets["all_negative"].append(nets)
        elif all(v > 0.0 for v in nets):
            buckets["all_positive"].append(nets)
        else:
            buckets["mixed"].append(nets)
    out = {}
    for name, rows in buckets.items():
        out[name] = {
            "n": len(rows),
            "mean_best": None if not rows else sum(max(v) for v in rows) / len(rows),
            "mean_average": None if not rows else sum(sum(v) / 4.0 for v in rows) / len(rows),
            "mean_worst": None if not rows else sum(min(v) for v in rows) / len(rows),
        }
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    dx = sum((a - mx) ** 2 for a in xs)
    dy = sum((b - my) ** 2 for b in ys)
    if dx <= 0.0 or dy <= 0.0:
        return None
    return num / math.sqrt(dx * dy)


def _fit_value(model: str, train: np.ndarray, y: np.ndarray, row: np.ndarray) -> float:
    if model == "huber":
        return _huber(train, y, row)
    if model == "sgd":
        return _sgd(train, y, row)
    if model == "tree":
        return _tree(train, y, row)
    pred, _coef = _ridge(train, y, row)
    return pred


def _predict_group(train: list[dict[str, Any]], current: dict[str, Any], model: str) -> dict[str, Any]:
    """Predict this group from earlier resolved groups only."""
    feat = np.array(current["feat"], dtype=float)
    if not train:
        zeros = dict.fromkeys(COINS, 0.0)
        return {"n": 0, "best": 0.0, "mean": 0.0, "positive_value": 0.0, "residual": zeros, "joint": dict(zeros), "expected_net": dict(zeros)}
    x_best = _matrix([g["feat"] for g in train])
    y_best = np.array([g["best"] for g in train], dtype=float)
    y_mean = np.array([g["mean"] for g in train], dtype=float)
    y_pos = np.array([g["positive_value"] for g in train], dtype=float)
    best = _fit_value(model, x_best, y_best, feat)
    center = _fit_value("ridge", x_best, y_mean, feat)
    positive = _fit_value("ridge", x_best, y_pos, feat)
    dev_rows = []
    dev_y = []
    joint_rows = []
    joint_y = []
    for group in train:
        for coin in COINS:
            dev_rows.append(group["dev"][coin])
            dev_y.append(group["residual"][coin])
            joint_rows.append(list(group["feat"]) + list(group["dev"][coin]))
            joint_y.append(group["nets"][coin])
    residual = {}
    joint = {}
    expected = {}
    dev_x = _matrix(dev_rows)
    joint_x = _matrix(joint_rows)
    for coin in COINS:
        dev = np.array(current["dev"][coin], dtype=float)
        res = _fit_value("ridge", dev_x, np.array(dev_y, dtype=float), dev)
        joined = _fit_value(model, joint_x, np.array(joint_y, dtype=float), np.array(list(feat) + list(dev), dtype=float))
        residual[coin] = res
        joint[coin] = joined
        expected[coin] = center + res
    return {"n": len(train), "best": best, "mean": center, "positive_value": positive, "residual": residual, "joint": joint, "expected_net": expected}


def _selection(pred: dict[str, Any], key: str) -> tuple[float, str | None]:
    values = pred[key]
    coin = max(COINS, key=lambda name: float(values[name]))
    if float(values[coin]) <= 0.0:
        return 0.0, None
    return float(values[coin]), coin


def _score_one(pred: dict[str, Any], group: dict[str, Any], key: str) -> dict[str, Any]:
    nets = [float(group["nets"][coin]) for coin in COINS]
    _chosen, coin = _selection(pred, key)
    selected = 0.0 if coin is None else float(group["nets"][coin])
    best = max(nets)
    positive = best > 0.0
    predicted_positive = float(pred["best"]) > 0.0 or coin is not None
    fp = predicted_positive and not positive
    fn = (not predicted_positive) and positive
    return {
        "predicted_best": float(pred["best"]),
        "selected_net": selected,
        "regret": best - selected,
        "spearman": _spearman([float(pred[key][c]) for c in COINS], nets),
        "fp_group": 1 if fp else 0,
        "fn_group": 1 if fn else 0,
        "fp_loss": selected if fp and coin is not None else 0.0,
        "fn_missed": best if fn else 0.0,
        "positive_group": 1 if positive else 0,
        "abstained": 1 if coin is None else 0,
    }


def prepare(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach causal features in decision order. Lag uses the prior decision, never its result."""
    ordered = sorted(groups, key=lambda g: (float(g["decided_at"]), str(g["group"])))
    previous: dict[str, float] | None = None
    for group in ordered:
        named = group["named"]
        group["feat"] = group_vector(named, group["engine"], previous)
        means = cross_section(named, group["engine"])
        group["dev"] = {coin: residual_vector(named.get(coin, {}), means, group["engine"], coin) for coin in COINS}
        made = targets(group["nets"]) if all(coin in group.get("nets", {}) for coin in COINS) else None
        if made is not None:
            group.update(made)
        current = cross_section(named, group["engine"])
        lead = _lead_key(group["engine"])
        current["btc_lead"] = float(named.get("BTCUSDT", {}).get(lead, 0.0))
        previous = current
    return ordered


def walk(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Score each group from groups that had already exited."""
    ordered = prepare(groups)
    scored: dict[str, list[dict[str, Any]]] = {"group_residual": [], "joint": []}
    conditional: list[float | None] = []
    last_coef: dict[str, float] = {}
    for index, group in enumerate(ordered):
        if "best" not in group:
            continue
        train = [earlier for earlier in ordered[:index] if "best" in earlier and float(earlier["exit_at"]) < float(group["decided_at"])]
        pred = _predict_group(train, group, "ridge")
        for key, label in (("expected_net", "group_residual"), ("joint", "joint")):
            scored[label].append(_score_one(pred, group, key))
        if group["best"] > 0.0 and pred["n"]:
            conditional.append(_spearman([pred["expected_net"][c] for c in COINS], [group["nets"][c] for c in COINS]))
        if train:
            _pred, coef = _ridge(_matrix([g["feat"] for g in train]), np.array([g["best"] for g in train]), np.array(group["feat"], dtype=float))
            names = group_feature_names(group["engine"])
            last_coef = {name: abs(float(coef[i])) for i, name in enumerate(names)} if len(coef) == len(names) else {}
    return {"models": {name: _aggregate(rows) for name, rows in scored.items()}, "conditional_rank": _avg(conditional), "families": last_coef}


def _avg(values: list[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return None if not kept else sum(kept) / len(kept)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"groups": 0}
    selected = [float(r["selected_net"]) for r in rows]
    gains = sum(v for v in selected if v > 0.0)
    losses = -sum(v for v in selected if v < 0.0)
    equity = peak = drawdown = 0.0
    for value in selected:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "groups": len(rows),
        "selected_net": sum(selected) / len(selected),
        "regret": sum(float(r["regret"]) for r in rows) / len(rows),
        "spearman": _avg([r["spearman"] for r in rows]),
        "fp_groups": sum(int(r["fp_group"]) for r in rows),
        "fn_groups": sum(int(r["fn_group"]) for r in rows),
        "fp_loss": sum(float(r["fp_loss"]) for r in rows),
        "fn_missed": sum(float(r["fn_missed"]) for r in rows),
        "pf": None if losses <= 0.0 else gains / losses,
        "drawdown": drawdown,
        "abstained": sum(int(r["abstained"]) for r in rows),
    }


def _live_summary(groups: list[dict[str, Any]]) -> dict[str, float | None]:
    bests, means, live_max, live_mean = [], [], [], []
    for group in groups:
        nets = group.get("nets") or {}
        lives = group.get("live") or {}
        if any(coin not in nets or lives.get(coin) is None for coin in COINS):
            continue
        values = [float(lives[coin]) for coin in COINS]
        bests.append(max(float(nets[coin]) for coin in COINS))
        means.append(sum(float(nets[coin]) for coin in COINS) / 4.0)
        live_max.append(max(values))
        live_mean.append(sum(values) / 4.0)
    return {"groups": float(len(bests)), "corr_max_best": _pearson(live_max, bests), "corr_mean_mean": _pearson(live_mean, means)}


def _cost_summary(groups: list[dict[str, Any]]) -> dict[str, float | None]:
    gross_best, net_best, drag = [], [], []
    disappeared = 0
    for group in groups:
        nets = group.get("nets") or {}
        gross = group.get("gross") or {}
        if any(coin not in nets or gross.get(coin) is None for coin in COINS):
            continue
        coin = max(COINS, key=lambda name: float(nets[name]))
        gross_best.append(max(float(gross[c]) for c in COINS))
        net_best.append(float(nets[coin]))
        drag.append(float(gross[coin]) - float(nets[coin]))
        if max(float(gross[c]) for c in COINS) > 0.0 and max(float(nets[c]) for c in COINS) <= 0.0:
            disappeared += 1
    n = len(net_best)
    return {
        "groups": float(n),
        "gross_oracle": None if not n else sum(gross_best) / n,
        "net_oracle": None if not n else sum(net_best) / n,
        "cost_drag": None if not n else sum(drag) / n,
        "gross_positive_net_negative": float(disappeared),
    }


def _families(weights: dict[str, float]) -> dict[str, float]:
    buckets = {
        "flow": ("ofi_", "agg_flow"),
        "book": ("obi_", "microprice", "depth_", "absorption", "adverse", "cancel"),
        "cost": ("spread", "expected_cost"),
        "trend": ("h1_ret", "bar_return", "sma20"),
        "volatility": ("atr",),
        "range": ("range_location",),
        "temporal": ("d_mean_", "d_btc"),
        "lead": ("btc_lead", "btc_gap"),
    }
    out = dict.fromkeys(buckets, 0.0)
    for feature, weight in weights.items():
        for name, parts in buckets.items():
            if any(part in feature for part in parts):
                out[name] += float(weight)
                break
    return out


def build_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One primary episode per coin. A coin without a resolved net is not invented."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["engine"], row["group"]), []).append(row)
    out = []
    for (engine, gid), members in grouped.items():
        by_coin: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            if row["symbol"] in COINS and row.get("named"):
                by_coin.setdefault(row["symbol"], []).append(row)
        if any(coin not in by_coin for coin in COINS):
            continue
        chosen = {}
        for coin, options in by_coin.items():
            chosen[coin] = max(options, key=lambda row: (row.get("live") is not None, row["live"] if row.get("live") is not None else -1e99))
        if any(chosen[coin].get("net") is None for coin in COINS):
            continue
        out.append(
            {
                "engine": engine,
                "group": gid,
                "decided_at": min(float(chosen[c]["decided_at"]) for c in COINS),
                "exit_at": max(float(chosen[c]["exit_at"]) for c in COINS),
                "named": {coin: chosen[coin]["named"] for coin in COINS},
                "nets": {coin: float(chosen[coin]["net"]) for coin in COINS},
                "gross": {coin: chosen[coin].get("gross") for coin in COINS},
                "live": {coin: chosen[coin].get("live") for coin in COINS},
            }
        )
    return out


def load_rows(db_path: str) -> list[dict[str, Any]]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        found = conn.execute(
            """
            SELECT e.engine_id, e.symbol, e.decided_at, e.exit_at, e.net, e.gross, e.decision_group_id, e.snapshot_json, m.features_json
            FROM policy_episodes e
            LEFT JOIN adaptive_candidate_markouts m ON m.id = e.candidate_id
            WHERE e.status='CLOSED' AND e.net IS NOT NULL AND e.decision_group_id IS NOT NULL
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
    rows = []
    for row in found:
        values = _training_vector(row, str(row["engine_id"]))
        named = _named_from_x(str(row["engine_id"]), values or [])
        try:
            snap = json.loads(row["snapshot_json"] or "{}")
        except json.JSONDecodeError:
            snap = {}
        issued = snap.get("issued") if isinstance(snap.get("issued"), dict) else {}
        live = issued.get("policy_value") if isinstance(issued, dict) else None
        try:
            live_value = None if live is None or isinstance(live, bool) else float(live)
        except (TypeError, ValueError):
            live_value = None
        gross = row["gross"]
        rows.append(
            {
                "engine": str(row["engine_id"]),
                "symbol": str(row["symbol"]).upper().replace("-", "").replace("/", ""),
                "group": str(row["decision_group_id"]),
                "decided_at": float(row["decided_at"]),
                "exit_at": float(row["exit_at"] or row["decided_at"]),
                "net": float(row["net"]),
                "gross": None if gross is None else float(gross),
                "named": named,
                "live": live_value,
            }
        )
    return rows


def research_report(db_path: str) -> dict[str, Any]:
    """Read-only walk-forward. Does not write predictions or change authority."""
    groups = build_groups(load_rows(db_path))
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        by_engine.setdefault(group["engine"], []).append(group)
    out = {}
    for engine, rows in by_engine.items():
        walked = walk(rows)
        positive = [g for g in rows if max(g["nets"].values()) > 0.0]
        out[engine] = {
            "structure": sign_structure(rows),
            "variance": variance_split(rows),
            "walk": walked,
            "live": _live_summary(rows),
            "cost": _cost_summary(rows),
            "families": _families(walked.get("families") or {}),
            "positive_groups": len(positive),
            "groups": len(rows),
            "positive_only_oracle": None if not rows else sum(max(*g["nets"].values(), 0.0) for g in rows) / len(rows),
        }
    return out


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SCORE_DDL)


def _open_named(row: sqlite3.Row, engine: str) -> dict[str, float]:
    values = _training_vector(row, engine)
    return _named_from_x(engine, values or [])


def attach_group_prediction(db_path: str, group_id: str, engine: str) -> None:
    """Write the group prediction once, while every coin in the group is still open."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        ensure_schema(conn)
        current_rows = conn.execute("SELECT * FROM policy_episodes WHERE decision_group_id=? AND engine_id=?", (group_id, engine)).fetchall()
        by_coin: dict[str, sqlite3.Row] = {}
        for row in current_rows:
            symbol = str(row["symbol"]).upper().replace("-", "").replace("/", "")
            if symbol in COINS and symbol not in by_coin:
                by_coin[symbol] = row
        if any(coin not in by_coin for coin in COINS):
            return
        if any(str(by_coin[coin]["status"]) != "OPEN" for coin in COINS):
            return
        snaps = {coin: json.loads(by_coin[coin]["snapshot_json"] or "{}") for coin in COINS}
        if all(isinstance(snaps[coin], dict) and snaps[coin].get("group_schema") == SCHEMA for coin in COINS):
            return
        history = conn.execute(
            """
            SELECT e.engine_id, e.symbol, e.decided_at, e.exit_at, e.net, e.gross, e.decision_group_id, e.snapshot_json, e.status, m.features_json
            FROM policy_episodes e
            LEFT JOIN adaptive_candidate_markouts m ON m.id = e.candidate_id
            WHERE e.engine_id=? AND e.status='CLOSED' AND e.net IS NOT NULL AND e.decision_group_id IS NOT NULL
            """,
            (engine,),
        ).fetchall()
        past = build_groups(_public_rows(history))
        current = {
            "engine": engine,
            "group": group_id,
            "decided_at": min(float(by_coin[c]["decided_at"]) for c in COINS),
            "exit_at": min(float(by_coin[c]["decided_at"]) for c in COINS),
            "named": {coin: _open_named(by_coin[coin], engine) for coin in COINS},
            "nets": {},
        }
        ordered = prepare([*past, current])
        prepared = next(g for g in ordered if g["group"] == group_id)
        train = [g for g in ordered if g["group"] != group_id and "best" in g and float(g["exit_at"]) < float(prepared["decided_at"])]
        pred = _predict_group(train, prepared, "ridge")
        for coin in COINS:
            snap = snaps[coin] if isinstance(snaps[coin], dict) else {}
            snap["group_schema"] = SCHEMA
            snap["group_predictions"] = {
                "n": pred["n"],
                "predicted_best": pred["best"],
                "predicted_mean": pred["mean"],
                "predicted_positive_value": pred["positive_value"],
                "expected_net": pred["expected_net"][coin],
                "residual": pred["residual"][coin],
                "joint_net": pred["joint"][coin],
            }
            conn.execute(
                "UPDATE policy_episodes SET snapshot_json=? WHERE id=? AND status='OPEN' AND exit_at IS NULL",
                (json.dumps(snap, separators=(",", ":"), default=str), int(by_coin[coin]["id"])),
            )
        conn.commit()
    finally:
        conn.close()


def _public_rows(found: list[sqlite3.Row]) -> list[dict[str, Any]]:
    rows = []
    for row in found:
        values = _training_vector(row, str(row["engine_id"]))
        named = _named_from_x(str(row["engine_id"]), values or [])
        symbol = str(row["symbol"]).upper().replace("-", "").replace("/", "")
        rows.append(
            {
                "engine": str(row["engine_id"]),
                "symbol": symbol,
                "group": str(row["decision_group_id"]),
                "decided_at": float(row["decided_at"]),
                "exit_at": float(row["exit_at"] or row["decided_at"]),
                "net": float(row["net"]),
                "gross": None if row["gross"] is None else float(row["gross"]),
                "named": named,
                "live": None,
            }
        )
    return rows


def score_stored_group(conn: sqlite3.Connection, group_id: str, primaries: list[dict[str, Any]], moment: float) -> None:
    """Score predictions that were stored on the open episodes. Missing predictions are not filled in."""
    ensure_schema(conn)
    if len(primaries) != 4:
        return
    preds = []
    nets = []
    for row in primaries:
        state = row.get("group_predictions") if isinstance(row.get("group_predictions"), dict) else {}
        if not state:
            return
        preds.append(state)
        nets.append(float(row["net"]))
    by_coin = {str(row["symbol"]): (preds[i], nets[i]) for i, row in enumerate(primaries)}
    if any(coin not in by_coin for coin in COINS):
        return
    expected = {coin: float(by_coin[coin][0].get("expected_net") or 0.0) for coin in COINS}
    joint = {coin: float(by_coin[coin][0].get("joint_net") or 0.0) for coin in COINS}
    best_pred = float(preds[0].get("predicted_best") or 0.0)
    actual = {coin: float(by_coin[coin][1]) for coin in COINS}
    made = targets(actual)
    engine = str(primaries[0].get("engine") or "")
    for label, values in (("group_residual", expected), ("joint", joint)):
        coin = max(COINS, key=lambda name: values[name])
        selected = 0.0 if values[coin] <= 0.0 else actual[coin]
        positive = made["best"] > 0.0
        predicted_positive = best_pred > 0.0 or values[coin] > 0.0
        fp = predicted_positive and not positive
        fn = (not predicted_positive) and positive
        conn.execute(
            """
            INSERT OR IGNORE INTO policy_group_model_scores (
                decision_group_id, model, engine_id, decided_at, predicted_best, selected_net, regret, spearman,
                fp_group, fn_group, fp_loss, fn_missed, positive_group
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                group_id,
                label,
                engine,
                moment,
                best_pred,
                selected,
                made["best"] - selected,
                _spearman([values[c] for c in COINS], [actual[c] for c in COINS]),
                1 if fp else 0,
                1 if fn else 0,
                selected if fp and values[coin] > 0.0 else 0.0,
                made["best"] if fn else 0.0,
                1 if positive else 0,
            ),
        )
