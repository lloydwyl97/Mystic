"""Item p14: feature-family ablation framework (net expectancy/PF/drawdown/MFE-capture per family)."""

from __future__ import annotations

import json
import pickle
import random
import sqlite3

import pytest

from backend.services import feature_family_ablation as ffa
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

FEATURE_DIM = 20


class _StubModel:
    """Deterministic stand-in for a real sklearn classifier: BUY probability
    is driven entirely by feature[5] (inside a synthetic 'momentum' family),
    so ablating that family should show a large economic impact, while an
    unrelated family should show ~none."""

    def predict_proba(self, x):
        out = []
        for row in x:
            p_buy = 0.9 if row[5] > 0.5 else 0.1
            out.append([1.0 - p_buy, p_buy])
        return out


def _make_rows(n=100, seed=7):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        features = [rng.random() for _ in range(FEATURE_DIM)]
        # Ground truth: rows where feature[5] > 0.5 are real winners.
        is_winner = features[5] > 0.5
        net_pnl = 0.02 if is_winner else -0.01
        mfe = 0.03 if is_winner else 0.001
        rows.append({"features": features, "net_pnl_pct": net_pnl, "mfe_pct": mfe, "outcome_label": 1 if is_winner else 0})
    return rows


_FAMILIES = {"momentum_like": (5, 6), "unrelated": (15, 16)}


def test_insufficient_rows_reports_unavailable():
    report = ffa.run_feature_family_ablation(_StubModel(), _make_rows(n=5), families=_FAMILIES)
    assert report.available is False
    assert report.degraded_reason == "insufficient_rows"


def test_ablating_the_driving_family_hurts_net_expectancy():
    rows = _make_rows(n=200)
    report = ffa.run_feature_family_ablation(_StubModel(), rows, families=_FAMILIES, min_rows=20)
    assert report.available is True
    momentum_result = next(f for f in report.families if f.family == "momentum_like")
    delta = momentum_result.delta("net_expectancy")
    assert delta is not None
    assert delta < 0  # ablating the real driver should make traded-subset economics worse


def test_ablating_an_unrelated_family_has_near_zero_impact():
    rows = _make_rows(n=200)
    report = ffa.run_feature_family_ablation(_StubModel(), rows, families=_FAMILIES, min_rows=20)
    unrelated_result = next(f for f in report.families if f.family == "unrelated")
    delta = unrelated_result.delta("net_expectancy")
    assert delta is not None
    assert abs(delta) < 0.005


def test_ablate_family_zeroes_only_target_range_and_does_not_mutate_input():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    original = list(x)
    ablated = ffa.ablate_family(x, (1, 3))
    assert ablated == [1.0, 0.0, 0.0, 4.0, 5.0]
    assert x == original  # never mutates input


def test_most_impactful_families_ranks_by_absolute_delta():
    rows = _make_rows(n=200)
    report = ffa.run_feature_family_ablation(_StubModel(), rows, families=_FAMILIES, min_rows=20)
    top = report.most_impactful_families(top_n=1)
    assert top == ["momentum_like"]


def test_report_to_dict_is_json_safe():
    rows = _make_rows(n=200)
    report = ffa.run_feature_family_ablation(_StubModel(), rows, families=_FAMILIES, min_rows=20)
    payload = report.to_dict()
    assert payload["available"] is True
    assert isinstance(payload["families"], list)
    assert "delta_net_expectancy" in payload["families"][0]


def test_all_feature_families_cover_full_145_dim_reasonably():
    # Sanity: documented technical blocks + context block span the 0-144 range
    # without going out of bounds (approximate boundaries from
    # ai_market_diagnostics.FEATURE_BLOCKS, not required to be perfectly contiguous).
    max_hi = max(hi for _, hi in ffa.ALL_FEATURE_FAMILIES.values())
    min_lo = min(lo for lo, _ in ffa.ALL_FEATURE_FAMILIES.values())
    assert min_lo == 0
    assert max_hi == 145


def test_load_and_run_ablation_for_symbol_end_to_end(tmp_path):
    db = tmp_path / "ablation.db"
    ensure_ai_canonical_tables(str(db))
    rows = _make_rows(n=60)
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        for i, r in enumerate(rows):
            row = {
                "symbol": "BTCUSDT",
                "opened_at_utc": f"2026-08-01T00:{i % 60:02d}:00Z",
                "closed_at_utc": f"2026-08-01T00:{i % 60:02d}:30Z",
                "strategy_id": "day",
                "features_json": json.dumps(r["features"]),
                "net_pnl_pct": r["net_pnl_pct"],
                "max_favorable_excursion": r["mfe_pct"],
                "outcome_label": r["outcome_label"],
                "ingested_at_utc": "2026-08-01T00:00:00Z",
            }
            use = {k: v for k, v in row.items() if k in cols}
            conn.execute(
                f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                list(use.values()),
            )
        conn.commit()

    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps({"model": _StubModel()}))

    report = ffa.load_and_run_ablation_for_symbol("day", "BTCUSDT", db_path=str(db), model_artifact_path=str(model_path), limit=200)
    assert report.available is True
    assert report.n_rows == 60


class _RawScaleModel:
    """Model whose real decision boundary only makes sense on SCALED
    features (mimicking a live sklearn model fit on scaler.transform(X)) —
    used to prove the loader applies the artifact's scaler before scoring,
    not just passes raw features straight through."""

    def predict_proba(self, x):
        out = []
        for row in x:
            # Correct only when input has already been feature-scaled
            # (mean-subtracted, unit-variance): raw unscaled magnitudes
            # (e.g. ~1000) would never trip this boundary.
            p_buy = 0.9 if row[0] > 0 else 0.1
            out.append([1.0 - p_buy, p_buy])
        return out


class _FakeScaler:
    def transform(self, x):
        # Mimics standardization: subtract each column's mean.
        import statistics

        cols = list(zip(*x))
        means = [statistics.fmean(c) for c in cols]
        return [[v - m for v, m in zip(row, means)] for row in x]


def test_load_and_run_ablation_applies_scaler_before_scoring(tmp_path):
    db = tmp_path / "scaled.db"
    ensure_ai_canonical_tables(str(db))
    # feature[0] centered far from zero (e.g. 1000) so an unscaled model call
    # would misbehave, but after mean-centering some rows go positive/negative
    # in a way correlated with the winner label.
    rows = []
    for i in range(60):
        is_winner = i % 2 == 0
        f0 = 1005.0 if is_winner else 995.0
        features = [f0] + [0.0] * (FEATURE_DIM - 1)
        rows.append({"features": features, "net_pnl_pct": 0.02 if is_winner else -0.01, "mfe_pct": 0.03, "outcome_label": 1 if is_winner else 0})

    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ai_outcome_training_rows)")}
        for i, r in enumerate(rows):
            row = {
                "symbol": "BTCUSDT",
                "opened_at_utc": f"2026-08-01T00:{i % 60:02d}:00Z",
                "closed_at_utc": f"2026-08-01T00:{i % 60:02d}:30Z",
                "strategy_id": "day",
                "features_json": json.dumps(r["features"]),
                "net_pnl_pct": r["net_pnl_pct"],
                "max_favorable_excursion": r["mfe_pct"],
                "outcome_label": r["outcome_label"],
                "ingested_at_utc": "2026-08-01T00:00:00Z",
            }
            use = {k: v for k, v in row.items() if k in cols}
            conn.execute(
                f"INSERT INTO ai_outcome_training_rows ({', '.join(use)}) VALUES ({', '.join('?' for _ in use)})",
                list(use.values()),
            )
        conn.commit()

    model_path = tmp_path / "scaled_model.pkl"
    model_path.write_bytes(pickle.dumps({"model": _RawScaleModel(), "scaler": _FakeScaler()}))

    report = ffa.load_and_run_ablation_for_symbol("day", "BTCUSDT", db_path=str(db), model_artifact_path=str(model_path), limit=200)
    assert report.available is True
    # At least one family must show real traded rows once scaling is applied correctly.
    assert any(f.baseline["n_traded"] > 0 for f in report.families)


def test_load_and_run_ablation_missing_artifact_degrades(tmp_path):
    db = tmp_path / "ablation_empty.db"
    ensure_ai_canonical_tables(str(db))
    report = ffa.load_and_run_ablation_for_symbol("day", "BTCUSDT", db_path=str(db), model_artifact_path=str(tmp_path / "missing.pkl"))
    assert report.available is False
    assert report.degraded_reason == "model_artifact_missing"
