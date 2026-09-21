"""Supportable DAY net-return challengers on grouped labels.

Does not fit a 145-d return model. Does not deploy. Uses expanding-window
chronological folds with purge + embargo. Locked test is the last
LOCKED_TEST_FRAC of groups and is not used for feature or model choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from backend.services.binance_scalp.reconstructable_features import reconstructable_features
from backend.services.day_experiment_registry import (
    EMBARGO_SEC,
    LOCKED_TEST_FRAC,
    PREDECLARED_FEATURE_FAMILIES,
    registry,
)
from backend.services.day_grouped_decision_ledger import COINS, HOLD_SYMBOL, DecisionGroup
from backend.services.day_profit_attribution import (
    GroupAttribution,
    champion_picker,
    isolated_eligible,
    oracle_picker,
    portfolio_replay_from_policy,
)
from backend.services.feature_mapping import get_feature_name

CHALLENGER_DIM = 8
RIDGE_ALPHA = 10.0
MIN_TRAIN = 24
SYMBOL_INDEX = {s: i for i, s in enumerate(COINS)}


def expanding_folds(n: int, *, n_folds: int = 4) -> list[tuple[int, int, int]]:
    """Return (train_end, val_start, val_end) exclusive indices on a time-sorted list."""
    if n < 8:
        return []
    lock = max(1, int(n * LOCKED_TEST_FRAC))
    usable = n - lock
    if usable < 6:
        return []
    width = max(1, usable // n_folds)
    out = []
    for i in range(n_folds):
        val_end = min(usable, (i + 1) * width)
        val_start = i * width
        train_end = val_start
        if val_end <= val_start or train_end < 2:
            continue
        out.append((train_end, val_start, val_end))
    return out


def lock_slice(n: int) -> tuple[int, int]:
    lock = max(1, int(n * LOCKED_TEST_FRAC))
    return n - lock, n


def purge_mask(
    attributions: list[GroupAttribution],
    train_idx: list[int],
    val_start_epoch: float,
) -> list[int]:
    kept = []
    for i in train_idx:
        attr = attributions[i]
        overlap = False
        for lab in attr.labels.values():
            if not lab.labeled or lab.interval_start is None or lab.interval_end is None:
                continue
            if lab.interval_end > val_start_epoch - 1e-9:
                overlap = True
                break
        if not overlap:
            kept.append(i)
    return kept


def embargo_start(val_start_epoch: float) -> float:
    return float(val_start_epoch) + float(EMBARGO_SEC)


def _asof_low_dim(
    group: DecisionGroup,
    symbol: str,
    bars: dict[str, list[tuple[int, float, ...]]],
) -> list[float] | None:
    row = next((c for c in group.candidates if c.symbol == symbol), None)
    if row is None:
        return None
    path_ev = float(row.path_ev)
    spread = float(row.spread_bps or 0.0)
    ret5 = vol = rel_vol = 0.0
    series = bars.get(symbol) or []
    window = [b for b in series if int(b[0]) <= int(group.epoch)][-40:]
    if len(window) >= 8:
        dicts = [{"open": float(b[1]), "high": float(b[2]), "low": float(b[3]), "close": float(b[4]), "volume": float(b[5]) if len(b) > 5 else 0.0, "ts": int(b[0])} for b in window]
        btc = [b for b in (bars.get("BTCUSDT") or []) if int(b[0]) <= int(group.epoch)][-6:]
        btc_ret = 0.0
        if len(btc) >= 6 and float(btc[0][4]) > 0:
            btc_ret = (float(btc[-1][4]) - float(btc[0][4])) / float(btc[0][4])
        from datetime import datetime, timezone

        feats = reconstructable_features(
            dicts,
            btc_ret_5=btc_ret,
            market_vol_5=abs(btc_ret),
            ts=datetime.fromtimestamp(float(window[-1][0]), tz=timezone.utc),
        )
        ret5 = float(feats.get("ret_5") or 0.0)
        vol = float(feats.get("realized_vol_10") or 0.0)
        rel_vol = float(feats.get("rel_volume") or 0.0)
    dummy = [0.0, 0.0, 0.0, 0.0]
    dummy[SYMBOL_INDEX[symbol]] = 1.0
    return [path_ev, ret5, vol, rel_vol, spread / 1e4, *dummy[:3]]


def _candidate_xy(
    groups: list[DecisionGroup],
    attributions: list[GroupAttribution],
    bars: dict[str, list[tuple[int, float, ...]]],
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    xs: list[list[float]] = []
    ys: list[float] = []
    keys: list[tuple[int, str]] = []
    by_id = {a.decision_group_id: a for a in attributions}
    for i in indices:
        group = groups[i]
        attr = by_id[group.decision_group_id]
        for row in group.coin_rows():
            if not isolated_eligible(row):
                continue
            lab = attr.labels.get(row.symbol)
            if lab is None or not lab.labeled:
                continue
            feat = _asof_low_dim(group, row.symbol, bars)
            if feat is None:
                continue
            xs.append(feat)
            ys.append(lab.net_bps)
            keys.append((i, row.symbol))
    if not xs:
        return np.zeros((0, CHALLENGER_DIM)), np.zeros((0,)), []
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), keys


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> Ridge | None:
    if len(y) < MIN_TRAIN:
        return None
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x, y)
    return model


def _pick_from_scores(scores: dict[str, float]) -> str:
    best_sym, best = HOLD_SYMBOL, 0.0
    for sym, val in scores.items():
        if val > best:
            best_sym, best = sym, val
    return best_sym


def evaluate_challengers(
    groups: list[DecisionGroup],
    attributions: list[GroupAttribution],
    bars: dict[str, list[tuple[int, float, ...]]],
) -> dict[str, Any]:
    n = len(groups)
    lock_a, lock_b = lock_slice(n)
    folds = expanding_folds(n)
    by_id = {a.decision_group_id: a for a in attributions}

    def _calibrated_picker(model: Ridge | None):
        def _pick(group: DecisionGroup, attr: GroupAttribution, open_set: set[str]) -> str:
            if model is None:
                return champion_picker(group, attr, open_set)
            scores: dict[str, float] = {}
            for row in group.coin_rows():
                if row.symbol in open_set or not isolated_eligible(row):
                    continue
                x = np.asarray([[float(row.path_ev)]], dtype=float)
                scores[row.symbol] = float(model.predict(x)[0])
            return _pick_from_scores(scores)

        return _pick

    def _pooled_picker(model: Ridge | None):
        def _pick(group: DecisionGroup, attr: GroupAttribution, open_set: set[str]) -> str:
            if model is None:
                return champion_picker(group, attr, open_set)
            scores: dict[str, float] = {}
            for row in group.coin_rows():
                if row.symbol in open_set or not isolated_eligible(row):
                    continue
                feat = _asof_low_dim(group, row.symbol, bars)
                if feat is None:
                    continue
                scores[row.symbol] = float(model.predict(np.asarray([feat], dtype=float))[0])
            return _pick_from_scores(scores)

        return _pick

    fold_reports = []
    for fold_i, (train_end, val_start, val_end) in enumerate(folds):
        val_epoch = float(groups[val_start].epoch) if val_start < n else 0.0
        raw_train = list(range(0, train_end))
        train = [i for i in purge_mask(attributions, raw_train, val_epoch) if groups[i].epoch + EMBARGO_SEC <= val_epoch]
        x_cal, y_cal, _ = _candidate_xy(groups, attributions, bars, train)
        x_pool, y_pool, _ = _candidate_xy(groups, attributions, bars, train)
        cal_x = x_cal[:, :1] if len(x_cal) else x_cal
        cal = _fit_ridge(cal_x, y_cal) if len(y_cal) else None
        if cal is not None and len(cal_x):
            cal.fit(cal_x, y_cal)
        pooled = _fit_ridge(x_pool, y_pool)
        val_groups = groups[val_start:val_end]
        val_attr = [by_id[g.decision_group_id] for g in val_groups]
        fold_reports.append(
            {
                "fold": fold_i,
                "train_n": len(train),
                "val_n": len(val_groups),
                "champion": portfolio_replay_from_policy(val_groups, val_attr, picker=champion_picker),
                "calibrated_score": portfolio_replay_from_policy(val_groups, val_attr, picker=_calibrated_picker(cal)),
                "pooled_ridge": portfolio_replay_from_policy(val_groups, val_attr, picker=_pooled_picker(pooled)),
                "oracle": portfolio_replay_from_policy(val_groups, val_attr, picker=oracle_picker),
            }
        )

    train_all = list(range(0, lock_a))
    lock_epoch = float(groups[lock_a].epoch) if lock_a < n else 0.0
    train_all = [i for i in purge_mask(attributions, train_all, lock_epoch) if groups[i].epoch + EMBARGO_SEC <= lock_epoch]
    x_cal, y_cal, _ = _candidate_xy(groups, attributions, bars, train_all)
    x_pool, y_pool, _ = _candidate_xy(groups, attributions, bars, train_all)
    cal = None
    if len(y_cal) >= MIN_TRAIN:
        cal = Ridge(alpha=RIDGE_ALPHA)
        cal.fit(x_cal[:, :1], y_cal)
    pooled = _fit_ridge(x_pool, y_pool)
    lock_groups = groups[lock_a:lock_b]
    lock_attr = [by_id[g.decision_group_id] for g in lock_groups]
    locked = {
        "n_groups": len(lock_groups),
        "champion": portfolio_replay_from_policy(lock_groups, lock_attr, picker=champion_picker),
        "calibrated_score": portfolio_replay_from_policy(lock_groups, lock_attr, picker=_calibrated_picker(cal)),
        "pooled_ridge": portfolio_replay_from_policy(lock_groups, lock_attr, picker=_pooled_picker(pooled)),
        "oracle": portfolio_replay_from_policy(lock_groups, lock_attr, picker=oracle_picker),
    }
    return {
        "predeclared_feature_families": list(PREDECLARED_FEATURE_FAMILIES),
        "ridge_alpha": RIDGE_ALPHA,
        "embargo_sec": EMBARGO_SEC,
        "locked_test_frac": LOCKED_TEST_FRAC,
        "folds": fold_reports,
        "locked_test": locked,
        "grouped_ranker": {"status": "deferred", "reason": "independent_sample_size_unsupported"},
        "registry": registry(),
    }


def audit_stored_145(groups: list[DecisionGroup]) -> dict[str, Any]:
    """Read-only audit of stored inference vectors. Does not fit a 145-d model."""
    mats: list[list[float]] = []
    missing = 0
    for group in groups:
        for row in group.coin_rows():
            if row.features and len(row.features) == 145:
                mats.append(row.features)
            else:
                missing += 1
    if not mats:
        return {"n": 0, "missing": missing, "note": "no_stored_145_vectors"}
    arr = np.asarray(mats, dtype=float)
    n, d = arr.shape
    std = arr.std(axis=0)
    constants = [i for i, s in enumerate(std) if float(s) < 1e-12]
    nan_frac = float(np.mean(~np.isfinite(arr)))
    dup_pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            if float(std[i]) < 1e-12 or float(std[j]) < 1e-12:
                continue
            corr = float(np.corrcoef(arr[:, i], arr[:, j])[0, 1])
            if abs(corr) > 0.999:
                dup_pairs.append((i, j))
            if len(dup_pairs) >= 25:
                break
        if len(dup_pairs) >= 25:
            break
    names = [get_feature_name(i + 1) if i < 124 else f"ctx_{i + 1}" for i in range(d)]
    return {
        "n_vectors": n,
        "dim": d,
        "missing_candidate_vectors": missing,
        "observations_per_feature": round(n / d, 4) if d else 0.0,
        "constant_indices": constants,
        "constant_names": [names[i] for i in constants[:20]],
        "nan_frac": nan_frac,
        "near_duplicate_pairs": dup_pairs[:20],
        "note": "Preserve full 145-d schema. Do not fit a return model on execute-only rows.",
    }


def acceptance_test(locked: dict[str, Any], folds: list[dict[str, Any]]) -> dict[str, Any]:
    champ = locked.get("champion") or {}
    results = {}
    for name in ("calibrated_score", "pooled_ridge"):
        arm = locked.get(name) or {}
        fold_wins = 0
        fold_n = 0
        for fold in folds:
            c = (fold.get("champion") or {}).get("net_usd")
            a = (fold.get(name) or {}).get("net_usd")
            if c is None or a is None:
                continue
            fold_n += 1
            if float(a) > float(c):
                fold_wins += 1
        pf = arm.get("profit_factor")
        pf_ok = pf is not None and pf != float("inf") and float(pf) > 1.0
        net_ok = float(arm.get("net_usd") or 0.0) > 0.0
        beat_pnl = float(arm.get("net_usd") or 0.0) > float(champ.get("net_usd") or 0.0)
        beat_bps = float(arm.get("net_bps") or 0.0) > float(champ.get("net_bps") or 0.0)
        dd_ok = float(arm.get("max_drawdown_bps") or 0.0) <= float(champ.get("max_drawdown_bps") or 0.0) * 1.10 + 1e-9
        majority = fold_n > 0 and fold_wins > fold_n / 2.0
        passed = bool(net_ok and pf_ok and beat_pnl and beat_bps and majority and dd_ok)
        results[name] = {
            "passed": passed,
            "net_positive": net_ok,
            "profit_factor_gt_1": pf_ok,
            "beats_champion_usd": beat_pnl,
            "beats_champion_bps": beat_bps,
            "majority_folds": majority,
            "drawdown_ok": dd_ok,
            "promote": False,
        }
    return results


def block_bootstrap_ci(values: list[float], *, block: int = 8, n_boot: int = 200) -> dict[str, float] | None:
    if len(values) < block * 2:
        return None
    rng = np.random.default_rng(7)
    arr = np.asarray(values, dtype=float)
    starts = list(range(0, len(arr) - block + 1, block))
    if not starts:
        return None
    means = []
    for _ in range(n_boot):
        picks = rng.choice(starts, size=len(starts), replace=True)
        chunk = np.concatenate([arr[s : s + block] for s in picks])
        means.append(float(np.mean(chunk)))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(np.mean(arr)), "lo": float(lo), "hi": float(hi), "block": block, "n": len(values)}


def multiple_testing_note(n_candidates: int, n_arms: int) -> dict[str, Any]:
    if n_candidates < 80:
        return {"method": "pbo_dsr", "status": "sample_size_insufficient", "n_candidates": n_candidates, "n_arms": n_arms}
    return {"method": "pbo_dsr", "status": "not_computed_point_estimate_only", "n_candidates": n_candidates, "n_arms": n_arms}


def calibration_table(pred: list[float], realized: list[float], *, n_bins: int = 5) -> dict[str, Any]:
    if len(pred) < 10:
        return {"status": "insufficient", "n": len(pred)}
    order = np.argsort(np.asarray(pred, dtype=float))
    p = np.asarray(pred, dtype=float)[order]
    y = np.asarray(realized, dtype=float)[order]
    size = max(1, len(p) // n_bins)
    bins = []
    for i in range(n_bins):
        a = i * size
        b = len(p) if i == n_bins - 1 else (i + 1) * size
        if a >= b:
            continue
        bins.append(
            {
                "bin": i,
                "n": int(b - a),
                "mean_pred_bps": round(float(np.mean(p[a:b])), 3),
                "mean_realized_bps": round(float(np.mean(y[a:b])), 3),
            }
        )
    resid = y - p
    return {
        "status": "ok",
        "n": len(pred),
        "bins": bins,
        "mae_bps": round(float(np.mean(np.abs(resid))), 3),
        "note": "isotonic skipped: tiny-fold support not demonstrated",
    }
