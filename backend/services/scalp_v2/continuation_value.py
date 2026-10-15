"""SCALP V2 learned continuation value.

One learned function prices a SCALP position on both sides of zero: the
expected change in executable net from keeping the current state for a
horizon, against selling it now. The entry reads the same function at the
state just after a fill, so an admitted entry is one the continuation keeps.

Training states are not limited to positions Mystic held. Every recorded SCALP
candidate is a virtual entry at its decision ask, and its later path in stored
1m bars supplies states and horizon labels whether or not anything was bought
or sold, so what follows an exit is observed as well. A path that reaches
catastrophic protection is labeled at the stop price and has no later states.
A label is learned only after its horizon has elapsed.

The state is continuous: net on each side of zero, adverse depth, favorable
excursion, giveback and age, with the entry's edge terms, the order book and
tape at entry (fading with age), and the candidate's symbol, setup and regime. Keep while the best informed horizon has positive
expected advantage; sell when it is negative. No sample floor, no threshold.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.services.adaptive_learning import PRIOR_STRENGTH, SCALP_ENGINE, _connect, _norm_symbol, current_economic_version, half_life_days
from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT

logger = logging.getLogger(__name__)

VALUE_VERSION = "SCALP_VALUE_V1"
MODEL = "scalp_value"
HORIZONS_SEC = (120, 300, 600, 1200)
STATE_AGES_MIN = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 180)
# A 1m bar is in the table shortly after it closes; labels wait this long past their horizon.
LABEL_LAG_SEC = 120.0
FOLD_INTERVAL_SEC = 60.0
BAR_SEC = 60.0
SPAN_SEC = STATE_AGES_MIN[-1] * 60.0 + HORIZONS_SEC[-1] + BAR_SEC
# 10 bps -> 1.0, so the ridge prior is comparable across terms.
_SCALE = 1000.0
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
SETUPS = (
    "BREAKOUT_MOMENTUM",
    "COMPRESSION_BREAKOUT",
    "FAILED_BREAKDOWN_REVERSAL",
    "FAILED_BREAKOUT_REVERSAL",
    "ORDERBOOK_TAPE_SCALP",
    "RANGE_BOUNCE_SCALP",
    "TREND_PULLBACK_MICRO",
    "VOLUME_IMPULSE_CONTINUATION",
    "VWAP_EMA_RECLAIM",
)
TRENDS = ("up", "down", "flat")
ENTRY_KEYS = ("calibrated", "adaptive", "micro", "raw_claim", "cost")
_ENTRY_SOURCES = {"calibrated": "calibrated_move", "adaptive": "adaptive_residual", "micro": "micro_residual", "raw_claim": "raw_claim", "cost": "expected_cost"}
# Order book and tape at entry, each scaled to roughly unit range.
MICRO_KEYS = ("obi", "ofi", "flow", "adverse", "pressure", "spread")
_MICRO_ENCODE = {
    "obi": ("obi_l5", lambda v: v),
    "ofi": ("ofi_5s", lambda v: math.tanh(v / 2.0)),
    "flow": ("agg_flow_imbalance_5s", lambda v: v),
    "adverse": ("adverse_selection_score", lambda v: 5.0 * v),
    "pressure": ("microprice_pressure", lambda v: 1e4 * v),
    "spread": ("spread_pct", lambda v: 1e4 * v),
}
MICRO_SOURCES = tuple(src for src, _ in _MICRO_ENCODE.values())
# Knots of the continuous piecewise-linear net term, in bps.
NET_KNOTS_BPS = (-130, -100, -60, -30, -15, -8, -4, 0, 4, 10, 30)
FEATURES: tuple[str, ...] = (
    "bias",
    "net",
    *(f"net_over_{knot}" for knot in NET_KNOTS_BPS),
    "mfe",
    "mae",
    "giveback",
    "age_log",
    *(f"entry_{key}" for key in ENTRY_KEYS),
    *(f"entry_{key}_fresh" for key in ENTRY_KEYS),
    *(f"micro_{key}_fresh" for key in MICRO_KEYS),
    *(f"sym_{sym}" for sym in SYMBOLS),
    *(f"setup_{name}" for name in SETUPS),
    *(f"btc_{trend}" for trend in TRENDS),
    "vol_hi",
)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _regime_parts(regime: str) -> tuple[str, str]:
    trend, vol = "", ""
    for part in str(regime or "").lower().split("_"):
        if part.startswith("btc"):
            trend = part[3:]
        elif part.startswith("vol"):
            vol = part[3:]
    return trend, vol


def micro_state(features: dict | None) -> dict[str, float]:
    """Encoded order book and tape at entry. Missing features are 0."""
    feats = features if isinstance(features, dict) else {}
    return {f"m_{key}": encode(_finite(feats.get(src))) for key, (src, encode) in _MICRO_ENCODE.items()}


def micro_raw(features: dict | None) -> dict[str, float]:
    """The stored subset of live microstructure features the value model reads."""
    feats = features if isinstance(features, dict) else {}
    return {src: _finite(feats.get(src)) for src in MICRO_SOURCES}


def entry_state(economic: dict | None) -> dict[str, float]:
    """Entry terms from a stored or live ``economic`` block, with its ``entry_micro``. Missing terms are 0."""
    econ = economic if isinstance(economic, dict) else {}
    out = {key: _finite(econ.get(src)) for key, src in _ENTRY_SOURCES.items()}
    out.update(micro_state(econ.get("entry_micro")))
    return out


def entry_position_state(cost: float) -> dict[str, float]:
    """State of a lot just filled at the ask: it is down its round-trip cost and nothing else."""
    return {"net": -float(cost), "mfe": 0.0, "mae": 0.0, "giveback": 0.0, "age_min": 0.0}


def vector(state: dict, entry: dict, symbol: str, setup: str, regime: str) -> np.ndarray:
    net = _finite(state.get("net")) * _SCALE
    age = max(0.0, _finite(state.get("age_min")))
    trend, vol = _regime_parts(regime)
    sym = _norm_symbol(symbol)
    name = str(setup or "").upper()
    values = [
        1.0,
        net,
        *(max(0.0, net - knot / 10.0) for knot in NET_KNOTS_BPS),
        _finite(state.get("mfe")) * _SCALE,
        _finite(state.get("mae")) * _SCALE,
        _finite(state.get("giveback")) * _SCALE,
        math.log1p(age),
    ]
    terms = [_finite(entry.get(key)) * _SCALE for key in ENTRY_KEYS]
    values += terms
    values += [term / (1.0 + age) for term in terms]
    values += [_finite(entry.get(f"m_{key}")) / (1.0 + age) for key in MICRO_KEYS]
    values += [1.0 if sym == s else 0.0 for s in SYMBOLS]
    values += [1.0 if name == s else 0.0 for s in SETUPS]
    values += [1.0 if trend == t else 0.0 for t in TRENDS]
    values.append(1.0 if vol == "hi" else 0.0)
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class Bars:
    opened: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @property
    def closes_at(self) -> np.ndarray:
        return self.opened + BAR_SEC


def bars_from(rows: list[tuple[float, float, float, float, float]]) -> Bars:
    arr = np.asarray(rows, dtype=float).reshape(-1, 5)
    return Bars(opened=arr[:, 0], high=arr[:, 2], low=arr[:, 3], close=arr[:, 4])


def load_bars(db_path: str, symbol: str, start: float, end: float) -> Bars:
    from backend.services.day_v2.lifecycle_sim import ohlcv_bars_1m

    return bars_from(ohlcv_bars_1m(db_path, symbol, start, end))


@dataclass(frozen=True)
class Candidate:
    id: int
    symbol: str
    setup: str
    regime: str
    t0: float
    ref: float
    cost: float
    entry: dict[str, float] = field(default_factory=dict)


def _json_dict(text: Any) -> dict:
    try:
        parsed = json.loads(text) if text else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_candidates(db_path: str, start: float, end: float) -> list[Candidate]:
    """Current-version SCALP candidates evaluated in [start, end), oldest first."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            """
            SELECT id, symbol, setup, regime, evaluated_at, ref_price, roundtrip_cost, economic_json, features_json
            FROM adaptive_candidate_markouts
            WHERE engine_id=? AND economic_version=? AND evaluated_at>=? AND evaluated_at<? AND ref_price>0
            ORDER BY evaluated_at, id
            """,
            (SCALP_ENGINE, current_economic_version(SCALP_ENGINE), float(start), float(end)),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out: list[Candidate] = []
    for cid, symbol, setup, regime, evaluated, ref, cost, econ, feats in rows:
        parsed, micro = _json_dict(econ), _json_dict(feats)
        entry = {**entry_state(parsed), **micro_state(micro)}
        out.append(Candidate(int(cid), _norm_symbol(symbol), str(setup or ""), str(regime or ""), float(evaluated), float(ref), _finite(cost), entry))
    return out


def path_state(candidate: Candidate, bars: Bars, age_min: float) -> tuple[int, float, dict[str, float]] | None:
    """(last closed bar index, gross, state) of the virtual lot ``age_min`` after entry.

    Uses only bars closed by then. None when that bar is missing or the path
    already reached catastrophic protection.
    """
    closes_at = bars.closes_at
    moment = candidate.t0 + 60.0 * age_min
    last = int(np.searchsorted(closes_at, moment, side="right")) - 1
    if age_min <= 0:
        return last, 0.0, entry_position_state(candidate.cost)
    if last < 0 or closes_at[last] <= moment - BAR_SEC or closes_at[last] <= candidate.t0:
        return None
    gross = float(bars.close[last]) / candidate.ref - 1.0
    first = int(np.searchsorted(bars.opened, candidate.t0, side="left"))
    mfe, mae = max(0.0, gross), max(0.0, -gross)
    if last >= first:
        low = float(bars.low[first : last + 1].min())
        if low <= candidate.ref * (1.0 - SCALP_V2_CATASTROPHIC_PCT):
            return None
        mfe = max(mfe, float(bars.high[first : last + 1].max()) / candidate.ref - 1.0)
        mae = max(mae, 1.0 - low / candidate.ref)
    state = {"net": gross - candidate.cost, "mfe": mfe, "mae": mae, "giveback": max(0.0, mfe - max(gross, 0.0)), "age_min": float(age_min)}
    return last, gross, state


def horizon_gross(candidate: Candidate, bars: Bars, last: int, end: float) -> float | None:
    """Gross of the virtual lot at ``end``: the stop price if the path reached it after bar ``last``."""
    closes_at = bars.closes_at
    k = int(np.searchsorted(closes_at, end, side="right")) - 1
    if k < 0 or closes_at[k] <= end - BAR_SEC:
        return None
    start = max(last + 1, int(np.searchsorted(bars.opened, candidate.t0, side="left")))
    if k >= start and float(bars.low[start : k + 1].min()) <= candidate.ref * (1.0 - SCALP_V2_CATASTROPHIC_PCT):
        return -SCALP_V2_CATASTROPHIC_PCT
    return float(bars.close[k]) / candidate.ref - 1.0


Label = tuple[int, np.ndarray, float, float, float]


def density(candidates: list[Candidate]) -> dict[str, dict[int, int]]:
    """Candidates per symbol per entry minute."""
    out: dict[str, dict[int, int]] = {}
    for candidate in candidates:
        per = out.setdefault(candidate.symbol, {})
        minute = int(candidate.t0 // BAR_SEC)
        per[minute] = per.get(minute, 0) + 1
    return out


def _label_weight(per_minute: dict[int, int] | None, candidate: Candidate, age: float, horizon: int) -> float:
    """One bar of information per symbol, horizon and label end bar.

    Candidates fire seconds apart and are sampled at many ages, so thousands of
    labels end on the same bar and share one price path. They split that bar's
    weight, and a horizon of k bars overlaps k label bars, so each carries 1/k.
    """
    sharing = 1
    if per_minute:
        minute = int(candidate.t0 // BAR_SEC)
        sharing = sum(per_minute.get(minute + round(age - a), 0) for a in STATE_AGES_MIN)
    return (BAR_SEC / horizon) / max(1, sharing)


def candidate_labels(candidate: Candidate, bars: Bars, lo: float, hi: float, per_minute: dict[int, int] | None = None) -> list[Label]:
    """(horizon, state vector, advantage, label time, weight) for horizons ending in (lo, hi]."""
    out: list[Label] = []
    if bars.opened.size == 0:
        return out
    for age in STATE_AGES_MIN:
        moment = candidate.t0 + 60.0 * age
        due = [h for h in HORIZONS_SEC if lo < moment + h <= hi]
        if not due:
            continue
        found = path_state(candidate, bars, age)
        if found is None:
            continue
        last, gross, state = found
        x = vector(state, candidate.entry, candidate.symbol, candidate.setup, candidate.regime)
        for horizon in due:
            later = horizon_gross(candidate, bars, last, moment + horizon)
            if later is None:
                continue
            out.append((horizon, x, later - gross, moment + horizon, _label_weight(per_minute, candidate, age, horizon)))
    return out


def _half_life_sec() -> float:
    return half_life_days(SCALP_ENGINE) * 86400.0


def _group_of(name: str) -> str:
    if name == "net" or name.startswith("net_over_"):
        return "net"
    if name in ("mfe", "mae", "giveback", "age_log"):
        return "path"
    if name.startswith("entry_"):
        return "entry_fresh" if name.endswith("_fresh") else "entry"
    for prefix, group in (("micro_", "micro"), ("sym_", "symbol"), ("setup_", "setup"), ("btc_", "regime"), ("vol_", "regime")):
        if name.startswith(prefix):
            return group
    return name


GROUPS: tuple[str, ...] = tuple(dict.fromkeys(_group_of(name) for name in FEATURES[1:]))
_GROUP_INDEX = np.asarray([GROUPS.index(_group_of(name)) for name in FEATURES[1:]], dtype=int)
_STRENGTH_BOUNDS = (1e-8, 1e12)


def _evidence_ridge(a: np.ndarray, b: np.ndarray, yy: float, n: float, iterations: int = 100) -> np.ndarray | None:
    """Ridge with an unpenalised mean and prior 0 on every state term.

    Labels share price paths and are weighted to their information content, so
    ``n`` is an effective count. The mean advantage is estimated freely. Each
    feature family (net, path, entry terms, fading entry terms, order book and
    tape, symbol, setup, regime) has its own prior variance, and those and the
    noise variance are re-estimated from the sufficient statistics (evidence
    maximisation). A family counts only as far as the data separates it from
    noise, and one informative family is not drowned by uninformative ones.
    ``FEATURES[0]`` is the constant 1, so ``a[0, 0]`` is the weight total.
    """
    total = float(a[0, 0])
    if n <= 0 or total <= 0:
        return None
    mean_x = a[1:, 0] / total
    a_c = a[1:, 1:] - total * np.outer(mean_x, mean_x)
    b_c = b[1:] - mean_x * float(b[0])
    yy_c = yy - float(b[0]) ** 2 / total
    rest = _shrunk(a_c, b_c, yy_c, n - 1.0, iterations)
    if rest is None:
        return None
    intercept = (float(b[0]) - float(a[0, 1:] @ rest)) / total
    beta = np.concatenate([[intercept], rest])
    return beta if np.all(np.isfinite(beta)) else None


def _shrunk(a: np.ndarray, b: np.ndarray, yy: float, n: float, iterations: int) -> np.ndarray | None:
    lo, hi = _STRENGTH_BOUNDS
    groups = _GROUP_INDEX if len(_GROUP_INDEX) == len(b) else np.zeros(len(b), dtype=int)
    strength = np.full(int(groups.max()) + 1, float(PRIOR_STRENGTH))
    beta = np.zeros_like(b)
    for _ in range(iterations):
        try:
            cov = np.linalg.inv(a + np.diag(strength[groups]))
        except np.linalg.LinAlgError:
            return None
        beta = cov @ b
        resid = max(yy - 2.0 * float(beta @ b) + float(beta @ a @ beta), 1e-18)
        gamma = 1.0 - strength[groups] * np.diag(cov)
        noise = resid / max(n - float(gamma.sum()), 1.0)
        updated = strength.copy()
        for g in range(len(strength)):
            mask = groups == g
            size = float(beta[mask] @ beta[mask])
            used = float(gamma[mask].sum())
            updated[g] = hi if size <= 0 or used <= 0 else min(hi, max(lo, noise * used / size))
        converged = np.all(np.abs(np.log(updated) - np.log(strength)) <= 1e-5)
        strength = updated
        if converged:
            break
    try:
        beta = np.linalg.solve(a + np.diag(strength[groups]), b)
    except np.linalg.LinAlgError:
        return None
    return beta if np.all(np.isfinite(beta)) else None


class ValueModel:
    """Per-horizon evidence ridge on the state vector, with time decay. Prior 0 for every coefficient."""

    def __init__(self) -> None:
        width = len(FEATURES)
        self.clock = 0.0
        self.meta: dict[str, Any] = {}
        self.stats: dict[int, dict[str, Any]] = {h: {"a": np.zeros((width, width)), "b": np.zeros(width), "yy": 0.0, "n": 0.0, "t": 0.0} for h in HORIZONS_SEC}
        self._beta: dict[int, np.ndarray] = {}

    def _decay(self, horizon: int, now: float) -> None:
        st = self.stats[horizon]
        if st["t"] and now > st["t"]:
            factor = 0.5 ** ((now - st["t"]) / _half_life_sec())
            st["a"] *= factor
            st["b"] *= factor
            st["yy"] *= factor
            st["n"] *= factor
        st["t"] = max(float(st["t"]), float(now))

    def fold(self, labels: list[Label], now: float) -> int:
        folded = 0
        for horizon in HORIZONS_SEC:
            rows = [item for item in labels if item[0] == horizon and math.isfinite(item[2])]
            if not rows:
                continue
            self._decay(horizon, now)
            x = np.vstack([item[1] for item in rows])
            y = np.asarray([item[2] for item in rows], dtype=float)
            w = np.asarray([item[4] for item in rows], dtype=float)
            w = w * 0.5 ** (np.maximum(0.0, now - np.asarray([item[3] for item in rows], dtype=float)) / _half_life_sec())
            st = self.stats[horizon]
            st["a"] += x.T @ (w[:, None] * x)
            st["b"] += x.T @ (w * y)
            st["yy"] += float(w @ (y * y))
            st["n"] += float(w.sum())
            self._beta.pop(horizon, None)
            folded += len(rows)
        return folded

    def beta(self, horizon: int) -> np.ndarray | None:
        st = self.stats[horizon]
        if float(st["n"]) <= 0:
            return None
        if horizon not in self._beta:
            beta = _evidence_ridge(st["a"], st["b"], float(st["yy"]), float(st["n"]))
            if beta is None:
                return None
            self._beta[horizon] = beta
        return self._beta[horizon]

    def horizon_values(self, x: np.ndarray) -> dict[int, float]:
        out: dict[int, float] = {}
        for horizon in HORIZONS_SEC:
            beta = self.beta(horizon)
            if beta is not None:
                value = float(x @ beta)
                if math.isfinite(value):
                    out[horizon] = value
        return out

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": VALUE_VERSION,
            "features": list(FEATURES),
            "clock": self.clock,
            "meta": self.meta,
            "horizons": {str(h): {"a": st["a"].tolist(), "b": st["b"].tolist(), "yy": st["yy"], "n": st["n"], "t": st["t"]} for h, st in self.stats.items()},
        }

    @classmethod
    def from_payload(cls, payload: Any) -> ValueModel | None:
        if not isinstance(payload, dict) or payload.get("version") != VALUE_VERSION or list(payload.get("features") or []) != list(FEATURES):
            return None
        model = cls()
        model.clock = _finite(payload.get("clock"))
        model.meta = dict(payload.get("meta") or {})
        for key, raw in (payload.get("horizons") or {}).items():
            horizon = int(key)
            if horizon not in model.stats:
                continue
            model.stats[horizon] = {
                "a": np.asarray(raw["a"], dtype=float),
                "b": np.asarray(raw["b"], dtype=float),
                "yy": _finite(raw.get("yy")),
                "n": _finite(raw.get("n")),
                "t": _finite(raw.get("t")),
            }
        return model


def model_key() -> str:
    return f"{MODEL}@{current_economic_version(SCALP_ENGINE)}"


_CACHE: dict[str, tuple[float, str, ValueModel | None]] = {}
_CACHE_TTL_SEC = 15.0


def load_model(db_path: str, *, fresh: bool = False) -> ValueModel | None:
    now = time.monotonic()
    hit = _CACHE.get(db_path)
    if hit and not fresh and now - hit[0] < _CACHE_TTL_SEC:
        return hit[2]
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT payload, updated_at FROM adaptive_linear_model WHERE engine_id=? AND model=?", (SCALP_ENGINE, model_key())).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        conn.close()
    stamp = str(row[1]) if row else ""
    if hit and hit[1] == stamp and stamp:
        _CACHE[db_path] = (now, stamp, hit[2])
        return hit[2]
    model = None
    if row and row[0]:
        try:
            model = ValueModel.from_payload(json.loads(row[0]))
        except (TypeError, ValueError, KeyError):
            model = None
    _CACHE[db_path] = (now, stamp, model)
    return model


def save_model(db_path: str, model: ValueModel) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(engine_id, model) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (SCALP_ENGINE, model_key(), json.dumps(model.to_payload(), separators=(",", ":")), f"{model.clock:.3f}"),
        )
        conn.commit()
    finally:
        conn.close()
    _CACHE.pop(db_path, None)


def authority(db_path: str) -> bool:
    """True once a value model of this version is installed for the current SCALP economics."""
    return load_model(db_path) is not None


def value(db_path: str, symbol: str, setup: str, regime: str, state: dict, entry: dict | None, *, model: ValueModel | None = None) -> dict[str, Any] | None:
    """Expected advantage of keeping ``state`` at each informed horizon, and the best of them.

    None when no value model is installed. An installed model with no informed
    horizon reads 0, which keeps a held lot and prices an entry at minus its cost.
    """
    use = model if model is not None else load_model(db_path)
    if use is None:
        return None
    per = use.horizon_values(vector(state, entry or {}, symbol, setup, regime))
    if not per:
        return {"advantage": 0.0, "horizon": None, "horizons": {}, "version": VALUE_VERSION}
    best = max(per, key=per.__getitem__)
    return {"advantage": per[best], "horizon": best, "horizons": per, "version": VALUE_VERSION}


def entry_value(db_path: str, symbol: str, setup: str, regime: str, entry: dict, *, model: ValueModel | None = None) -> float | None:
    """Advantage of keeping a lot filled now at the ask. None when no model is installed."""
    read = value(db_path, symbol, setup, regime, entry_position_state(_finite(entry.get("cost"))), entry, model=model)
    return None if read is None else float(read["advantage"])


def _labels_for(
    db_path: str,
    candidates: list[Candidate],
    lo: float,
    hi: float,
    bars: dict[str, Bars] | None = None,
    counts: dict[str, dict[int, int]] | None = None,
) -> list[Label]:
    if not candidates:
        return []
    start = min(c.t0 for c in candidates) - BAR_SEC
    bars = bars if bars is not None else {sym: load_bars(db_path, sym, start, hi + BAR_SEC) for sym in {c.symbol for c in candidates}}
    counts = counts if counts is not None else density(candidates)
    out: list[Label] = []
    for candidate in candidates:
        found = bars.get(candidate.symbol)
        if found is not None:
            out.extend(candidate_labels(candidate, found, lo, hi, counts.get(candidate.symbol)))
    return out


def build(db_path: str, *, as_of: float | None = None, state_db: str | None = None, chunk: int = 500) -> dict[str, Any]:
    """Train from every current-version candidate whose labels have elapsed, and install it."""
    clock = float(as_of if as_of is not None else time.time()) - LABEL_LAG_SEC
    candidates = load_candidates(db_path, 0.0, clock)
    model = ValueModel()
    counts = dict.fromkeys(HORIZONS_SEC, 0)
    if candidates:
        start = candidates[0].t0 - BAR_SEC
        bars = {sym: load_bars(db_path, sym, start, clock + BAR_SEC) for sym in {c.symbol for c in candidates}}
        per_minute = density(candidates)
        for index in range(0, len(candidates), chunk):
            labels = _labels_for(db_path, candidates[index : index + chunk], -math.inf, clock, bars, per_minute)
            for item in labels:
                counts[item[0]] += 1
            model.fold(labels, clock)
    model.clock = clock
    model.meta = {"built_at": time.time(), "build_candidates": len(candidates), "folded_at": time.time(), "last_folded": sum(counts.values())}
    save_model(state_db or db_path, model)
    return {"version": VALUE_VERSION, "clock": clock, "candidates": len(candidates), "labels": {str(h): n for h, n in counts.items()}}


_FOLD_LOCK = threading.Lock()
_LAST_FOLD: dict[str, float] = {}
_BOOTSTRAP: dict[str, subprocess.Popen] = {}
BUILD_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "scalp_value_build.py")


def bootstrap(db_path: str) -> bool:
    """Start one background build for the current economics. True while it runs.

    The build runs in its own low-priority process so its memory is returned
    when it finishes; nothing is folded until it has installed the model.
    """
    running = _BOOTSTRAP.get(db_path)
    if running is not None:
        if running.poll() is None:
            return True
        _BOOTSTRAP.pop(db_path, None)
        logger.warning("SCALP_VALUE_BOOTSTRAP_FINISHED rc=%s installed=%s", running.returncode, load_model(db_path, fresh=True) is not None)
        return False
    cmd = [sys.executable, os.path.normpath(BUILD_SCRIPT), "--db", db_path]
    try:
        _BOOTSTRAP[db_path] = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=(lambda: os.nice(10)) if hasattr(os, "nice") else None,  # noqa: PLW1509
        )
    except OSError:
        logger.warning("SCALP_VALUE_BOOTSTRAP_FAILED", exc_info=True)
        return False
    logger.warning("SCALP_VALUE_BOOTSTRAP_STARTED key=%s", model_key())
    return True


def fold_due(db_path: str, *, now: float | None = None, start_build: bool = True) -> int:
    """Fold labels whose horizon elapsed since the model's clock.

    Without a model for the current economics a background build is started
    instead. The clock and the statistics are written together, so a label is
    folded once.
    """
    moment = float(now if now is not None else time.time())
    if moment - _LAST_FOLD.get(db_path, 0.0) < FOLD_INTERVAL_SEC:
        return 0
    _LAST_FOLD[db_path] = moment
    with _FOLD_LOCK:
        if db_path in _BOOTSTRAP and bootstrap(db_path):
            return 0
        model = load_model(db_path, fresh=True)
        if model is None:
            if start_build:
                bootstrap(db_path)
            return 0
        hi = moment - LABEL_LAG_SEC
        lo = model.clock
        if hi <= lo:
            return 0
        started = time.monotonic()
        labels = _labels_for(db_path, load_candidates(db_path, lo - SPAN_SEC, hi), lo, hi)
        folded = model.fold(labels, hi)
        model.clock = hi
        model.meta.update({"folded_at": time.time(), "last_folded": folded, "last_fold_ms": round((time.monotonic() - started) * 1000.0, 1)})
        save_model(db_path, model)
        return folded


def status(db_path: str, *, now: float | None = None) -> dict[str, Any]:
    """Installed state of the SCALP value model, for operators. Reads only."""
    moment = float(now if now is not None else time.time())
    model = load_model(db_path, fresh=True)
    out: dict[str, Any] = {"key": model_key(), "version": VALUE_VERSION, "installed": model is not None}
    if model is None:
        return out
    horizons = {}
    for horizon in HORIZONS_SEC:
        beta = model.beta(horizon)
        horizons[str(horizon)] = {
            "effective_n": round(float(model.stats[horizon]["n"]), 1),
            "mean_advantage_bps": None if beta is None else round(float(beta[0]) * 1e4, 3),
            "state_terms_norm_bps": None if beta is None else round(float(np.linalg.norm(beta[1:])) * 1e4, 3),
        }
    out.update(
        {
            "clock": model.clock,
            "label_lag_sec": round(moment - model.clock, 1),
            "stalled": moment - model.clock > LABEL_LAG_SEC + 10.0 * FOLD_INTERVAL_SEC,
            "horizons": horizons,
            **model.meta,
        }
    )
    return out


__all__ = [
    "FEATURES",
    "HORIZONS_SEC",
    "STATE_AGES_MIN",
    "VALUE_VERSION",
    "Bars",
    "Candidate",
    "ValueModel",
    "authority",
    "bars_from",
    "bootstrap",
    "build",
    "candidate_labels",
    "density",
    "entry_position_state",
    "entry_state",
    "entry_value",
    "fold_due",
    "horizon_gross",
    "load_candidates",
    "load_model",
    "micro_raw",
    "micro_state",
    "path_state",
    "save_model",
    "status",
    "value",
    "vector",
]
