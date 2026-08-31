"""
Feature-family ablation framework (item p14).

Measures the real economic impact of each named feature family (the same
blocks documented in ``ai_market_diagnostics.FEATURE_BLOCKS`` plus the
21-dim DAY context block) by zero-ablating that family's dimensions in a
held-out, already-labeled row set and comparing the trading decisions the
model *would have made* with vs without that family — reported as after-cost
net expectancy, profit factor, max drawdown, and MFE-capture deltas, per the
architecture's "accuracy is diagnostic only" stance (item p23): accuracy
delta is reported too, but is not the headline number.

This is an offline analysis tool: given an already-trained model (any object
exposing ``predict_proba(X) -> array of shape (n, 2)``, e.g. the live RF/GBM
artifact's ``model``) and a set of already-closed, already-labeled rows
(features + realized net_pnl_pct + realized MFE), it:

  1. Computes each row's baseline buy/no-buy decision from the *full*
     feature vector.
  2. Zero-ablates one family's dimensions and recomputes the decision.
  3. Restricts to rows where each version says "buy" and computes real
     after-cost performance on that traded subset for both versions.
  4. Reports the delta (ablated - baseline) per family — a large negative
     net-expectancy delta means removing that family hurts real trading
     performance; a delta near zero means the model isn't economically
     relying on that family for its buy decisions today.

Never touches the live decision path — this is a reporting/analysis
function only, wired to an offline API endpoint
(``/portfolio-engine/feature-ablation/{symbol}``), matching item p13's
walk-forward validation wiring pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Reuses the real documented block boundaries (ai_market_diagnostics.FEATURE_BLOCKS,
# 1-indexed-inclusive over the 124-dim technical block) rather than inventing
# new ones, plus the 21-dim DAY context tail (dims 124-144).
TECHNICAL_FEATURE_FAMILIES: dict[str, tuple[int, int]] = {
    "basic_price": (0, 10),
    "technical_indicators": (10, 34),
    "volatility": (34, 44),
    "momentum": (44, 59),
    "trend": (59, 69),
    "volume_profile": (69, 77),
    "market_sentiment": (77, 87),
    "time_based": (87, 97),
    "advanced_ta": (97, 105),
    "advanced_volume": (105, 113),
    "microstructure": (113, 121),
}
CONTEXT_FEATURE_FAMILY: dict[str, tuple[int, int]] = {"day_context": (124, 145)}

ALL_FEATURE_FAMILIES: dict[str, tuple[int, int]] = {**TECHNICAL_FEATURE_FAMILIES, **CONTEXT_FEATURE_FAMILY}


class PredictProbaModel(Protocol):
    def predict_proba(self, x: list[list[float]]) -> Any: ...


class _ScaledModel:
    """Adapts a raw (model, scaler) pair from a live artifact into a single
    ``predict_proba``-only object, so `run_feature_family_ablation` (which
    only ever calls `model.predict_proba(rows)`) sees the same scaled input
    the live decision path uses. I/O-loader-only wrapper — never used on the
    pure `run_feature_family_ablation` code path directly by callers."""

    def __init__(self, model: Any, scaler: Any) -> None:
        self._model = model
        self._scaler = scaler

    def predict_proba(self, x: list[list[float]]) -> Any:
        return self._model.predict_proba(self._scaler.transform(x))


def ablate_family(x: list[float], index_range: tuple[int, int]) -> list[float]:
    """Zero out `x`'s dimensions in [lo, hi); returns a new list, never mutates input."""
    lo, hi = index_range
    out = list(x)
    for i in range(lo, min(hi, len(out))):
        out[i] = 0.0
    return out


def _buy_probabilities(model: PredictProbaModel, feature_rows: list[list[float]]) -> list[float]:
    proba = model.predict_proba(feature_rows)
    out = []
    for p in proba:
        # sklearn binary classifiers: column 1 is the positive (BUY) class.
        out.append(float(p[1]) if len(p) > 1 else float(p[0]))
    return out


def _max_drawdown(net_pnl_sequence: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in net_pnl_sequence:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _traded_subset_metrics(rows: list[dict[str, Any]], buy_probs: list[float], *, buy_threshold: float) -> dict[str, Any]:
    traded = [(r, p) for r, p in zip(rows, buy_probs, strict=True) if p >= buy_threshold]
    n = len(traded)
    if n == 0:
        return {"n_traded": 0, "net_expectancy": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0, "mfe_capture": None, "accuracy": None}

    net_pnls = [float(r.get("net_pnl_pct") or 0.0) for r, _ in traded]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.99 if gross_profit > 0 else 0.0)

    # MFE-capture: of the favorable excursion actually available on traded
    # rows, what fraction did the realized net PnL capture? (only computed
    # on winners, where "capture" is a meaningful concept.)
    capture_ratios = []
    for r, _ in traded:
        mfe = r.get("mfe_pct")
        net = r.get("net_pnl_pct")
        if mfe is None or net is None:
            continue
        try:
            mfe_f, net_f = float(mfe), float(net)
        except (TypeError, ValueError):
            continue
        if mfe_f > 0 and net_f > 0:
            capture_ratios.append(min(1.0, net_f / mfe_f))
    mfe_capture = sum(capture_ratios) / len(capture_ratios) if capture_ratios else None

    correct, labeled_n = 0, 0
    for r, _ in traded:
        actual = r.get("outcome_label")
        if actual is not None:
            labeled_n += 1
            if int(actual) == 1:
                correct += 1
    accuracy = correct / labeled_n if labeled_n else None

    return {
        "n_traded": n,
        "net_expectancy": sum(net_pnls) / n,
        "profit_factor": profit_factor,
        "max_drawdown_pct": _max_drawdown(net_pnls),
        "mfe_capture": mfe_capture,
        "accuracy": accuracy,
    }


@dataclass(frozen=True)
class FamilyAblationResult:
    family: str
    index_range: tuple[int, int]
    baseline: dict[str, Any]
    ablated: dict[str, Any]

    def delta(self, key: str) -> float | None:
        a, b = self.ablated.get(key), self.baseline.get(key)
        if a is None or b is None:
            return None
        return a - b

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "index_range": list(self.index_range),
            "baseline": self.baseline,
            "ablated": self.ablated,
            "delta_net_expectancy": self.delta("net_expectancy"),
            "delta_profit_factor": self.delta("profit_factor"),
            "delta_max_drawdown_pct": self.delta("max_drawdown_pct"),
            "delta_mfe_capture": self.delta("mfe_capture"),
            "delta_accuracy": self.delta("accuracy"),
        }


@dataclass(frozen=True)
class AblationReport:
    available: bool
    n_rows: int
    families: tuple[FamilyAblationResult, ...] = field(default_factory=tuple)
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "n_rows": self.n_rows,
            "families": [f.to_dict() for f in self.families],
            "degraded_reason": self.degraded_reason,
        }

    def most_impactful_families(self, *, top_n: int = 5) -> list[str]:
        """Families ranked by |delta_net_expectancy| descending — the
        families the live model's buy decisions most depend on today."""
        ranked = sorted(self.families, key=lambda f: abs(f.delta("net_expectancy") or 0.0), reverse=True)
        return [f.family for f in ranked[:top_n]]


def run_feature_family_ablation(
    model: PredictProbaModel,
    rows: list[dict[str, Any]],
    *,
    families: dict[str, tuple[int, int]] | None = None,
    buy_threshold: float = 0.5,
    min_rows: int = 20,
) -> AblationReport:
    """Pure — no I/O. `rows` each need a `features` key (list[float], already
    the exact model input dim) plus `net_pnl_pct`/`mfe_pct`/`outcome_label`
    (realized ground truth)."""
    usable_rows = [r for r in rows if isinstance(r.get("features"), list) and r["features"]]
    n = len(usable_rows)
    if n < min_rows:
        return AblationReport(available=False, n_rows=n, degraded_reason="insufficient_rows")

    families = families or ALL_FEATURE_FAMILIES
    baseline_features = [r["features"] for r in usable_rows]
    try:
        baseline_probs = _buy_probabilities(model, baseline_features)
    except Exception as exc:
        logger.debug("FEATURE_ABLATION_BASELINE_PREDICT_FAILED: %s", exc)
        return AblationReport(available=False, n_rows=n, degraded_reason="baseline_predict_failed")
    baseline_metrics = _traded_subset_metrics(usable_rows, baseline_probs, buy_threshold=buy_threshold)

    results: list[FamilyAblationResult] = []
    for family_name, index_range in families.items():
        ablated_features = [ablate_family(r["features"], index_range) for r in usable_rows]
        try:
            ablated_probs = _buy_probabilities(model, ablated_features)
        except Exception as exc:
            logger.debug("FEATURE_ABLATION_PREDICT_FAILED family=%s: %s", family_name, exc)
            continue
        ablated_metrics = _traded_subset_metrics(usable_rows, ablated_probs, buy_threshold=buy_threshold)
        results.append(FamilyAblationResult(family=family_name, index_range=index_range, baseline=baseline_metrics, ablated=ablated_metrics))

    if not results:
        return AblationReport(available=False, n_rows=n, degraded_reason="all_family_predictions_failed")

    return AblationReport(available=True, n_rows=n, families=tuple(results))


def load_and_run_ablation_for_symbol(
    strategy_id: str,
    symbol: str,
    *,
    db_path: str,
    model_artifact_path: str,
    limit: int = 2000,
    buy_threshold: float = 0.5,
) -> AblationReport:
    """Convenience I/O loader: loads the pickled active model artifact +
    real closed-trade rows, runs the ablation. Never called from the live
    decision path — used only by the offline
    `/portfolio-engine/feature-ablation/{symbol}` diagnostic endpoint."""
    import json
    import pickle
    from pathlib import Path

    path = Path(model_artifact_path)
    if not path.exists():
        return AblationReport(available=False, n_rows=0, degraded_reason="model_artifact_missing")
    try:
        artifact = pickle.loads(path.read_bytes())
        if isinstance(artifact, dict):
            model = artifact["model"]
            scaler = artifact.get("scaler")
        else:
            model, scaler = artifact, None
        # The live artifact's model was fit on scaler-transformed features
        # (see ai_model_promotion_holdout.py's scoring path) — predict_proba
        # on raw, unscaled features silently produces meaningless
        # probabilities (never truly "wrong", just economically useless),
        # which would make every family show zero traded rows regardless of
        # its real importance. Wrap so ablation sees exactly what live
        # inference sees.
        if scaler is not None:
            model = _ScaledModel(model, scaler)
    except Exception as exc:
        logger.debug("FEATURE_ABLATION_ARTIFACT_LOAD_FAILED %s: %s", path, exc)
        return AblationReport(available=False, n_rows=0, degraded_reason="model_artifact_unreadable")

    try:
        from backend.services.ai_canonical_storage import read_recent_outcome_training_rows

        raw_rows = read_recent_outcome_training_rows(symbol=symbol, strategy_id=strategy_id, limit=limit, db_path=db_path)
    except Exception as exc:
        logger.debug("FEATURE_ABLATION_ROWS_LOAD_FAILED symbol=%s: %s", symbol, exc)
        return AblationReport(available=False, n_rows=0, degraded_reason="outcome_rows_load_failed")

    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        raw_features = r.get("features_json")
        if not raw_features:
            continue
        try:
            features = json.loads(raw_features)
        except (TypeError, ValueError):
            continue
        if not isinstance(features, list):
            continue
        score_components = {}
        if r.get("score_components_json"):
            try:
                score_components = json.loads(r["score_components_json"])
            except (TypeError, ValueError):
                score_components = {}
        rows.append(
            {
                "features": [float(v) for v in features],
                "net_pnl_pct": r.get("net_pnl_pct"),
                "mfe_pct": score_components.get("mfe_pct") or r.get("max_favorable_excursion"),
                "outcome_label": r.get("outcome_label"),
            }
        )

    return run_feature_family_ablation(model, rows, buy_threshold=buy_threshold)


__all__ = [
    "ALL_FEATURE_FAMILIES",
    "CONTEXT_FEATURE_FAMILY",
    "TECHNICAL_FEATURE_FAMILIES",
    "AblationReport",
    "FamilyAblationResult",
    "ablate_family",
    "load_and_run_ablation_for_symbol",
    "run_feature_family_ablation",
]
