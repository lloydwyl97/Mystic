"""
True rolling walk-forward validation with purge/embargo (item p11).

Every chronological split found repo-wide (``ai_training_pipeline.py``'s
80/20 RF split, ``ltf_pattern_miner.walk_forward_split``, the 60/20/20 and
50/25/25 splits in ``scripts/replay_baselines/*``) trains on
``entry_ts < split_ts`` and tests on ``entry_ts >= split_ts`` with **no gap**
between them. That leaks information whenever a training sample's own
outcome window (open -> close) straddles the train/test boundary: its label
was determined partly using price action that occurred *during* the test
period, which the model would never have had access to live.

This module adds the missing purge + embargo mechanism (Lopez de Prado
"Advances in Financial Machine Learning" style, adapted for a single
expanding walk-forward series rather than full combinatorial CV):

  - PURGE: any training row whose ``close_time`` falls after the *earliest*
    ``open_time`` in the test fold is dropped — its label resolution window
    overlaps the test period, so keeping it would leak test-period price
    action into training.
  - EMBARGO: an additional contiguous slice of rows immediately preceding
    the test fold's start is also dropped from training, as a buffer against
    serial correlation in near-boundary samples (fixed fraction of total
    sample count, not (mis)modeled as a fixed time duration — same convention
    de Prado uses for the CPCV embargo).

This is an **offline validation/reporting utility** (used for model
evaluation, not a live per-tick decision) — pure functions operating on
already-fetched row lists, no I/O of its own beyond an optional convenience
loader. It reports genuine after-cost performance (win rate, profit factor,
total net PnL, max drawdown) per fold, not just accuracy — accuracy is
included only as one diagnostic field among several, consistent with item
p23's "accuracy becomes diagnostic only" shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    purged_count: int
    embargoed_count: int


def purged_walk_forward_splits(
    open_times: list[float],
    close_times: list[float],
    *,
    n_splits: int = 5,
    embargo_frac: float = 0.02,
) -> list[FoldSplit]:
    """Expanding-window walk-forward splits over `n_splits` contiguous,
    equal-sized test folds (chronologically ordered by open_time), each with
    a purge (drop train rows whose outcome window overlaps the test fold)
    and an embargo (drop a further embargo_frac-sized slice of rows
    immediately preceding the test fold)."""
    n = len(open_times)
    if n != len(close_times):
        raise ValueError("open_times and close_times must be the same length")
    if n_splits < 1 or n < (n_splits + 1) * 2:
        return []

    order = sorted(range(n), key=lambda i: open_times[i])
    fold_size = n // (n_splits + 1)
    embargo_n = max(0, round(embargo_frac * n))

    splits: list[FoldSplit] = []
    for fold in range(1, n_splits + 1):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits else n
        test_idx = order[test_start:test_end]
        if not test_idx:
            continue
        test_open_min = min(open_times[i] for i in test_idx)

        candidate_train_idx = order[:test_start]
        purged = [i for i in candidate_train_idx if close_times[i] > test_open_min]
        clean_train_idx = [i for i in candidate_train_idx if close_times[i] <= test_open_min]

        embargo_cut = max(0, len(clean_train_idx) - embargo_n)
        embargoed = clean_train_idx[embargo_cut:]
        train_idx = clean_train_idx[:embargo_cut]

        splits.append(
            FoldSplit(
                fold_index=fold,
                train_idx=tuple(train_idx),
                test_idx=tuple(test_idx),
                purged_count=len(purged),
                embargoed_count=len(embargoed),
            )
        )
    return splits


@dataclass(frozen=True)
class FoldPerformance:
    fold_index: int
    n_train: int
    n_test: int
    purged_count: int
    embargoed_count: int
    win_rate: float
    profit_factor: float
    total_net_pnl_pct: float
    avg_net_pnl_pct: float
    max_drawdown_pct: float
    accuracy: float | None  # diagnostic only, per item p23

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "purged_count": self.purged_count,
            "embargoed_count": self.embargoed_count,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_net_pnl_pct": round(self.total_net_pnl_pct, 6),
            "avg_net_pnl_pct": round(self.avg_net_pnl_pct, 6),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
        }


def _max_drawdown(net_pnl_sequence: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in net_pnl_sequence:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def compute_after_cost_fold_metrics(
    rows: list[dict[str, Any]],
    idx: tuple[int, ...],
    *,
    net_pnl_key: str = "net_pnl_pct",
    predicted_label_key: str | None = "predicted_label",
    actual_label_key: str | None = "outcome_label",
) -> dict[str, Any]:
    """Real after-cost performance for the rows at `idx` — win rate, profit
    factor, total/avg net PnL, max drawdown, and (diagnostic-only) accuracy
    when predicted/actual label keys are both present and populated."""
    subset = [rows[i] for i in idx]
    n = len(subset)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0, "total_net_pnl_pct": 0.0, "avg_net_pnl_pct": 0.0, "max_drawdown_pct": 0.0, "accuracy": None}

    net_pnls: list[float] = []
    for r in subset:
        v = r.get(net_pnl_key)
        try:
            net_pnls.append(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            net_pnls.append(0.0)

    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    win_rate = len(wins) / n
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 99.99 if gross_profit > 0 else 0.0

    accuracy: float | None = None
    if predicted_label_key and actual_label_key:
        labeled = [(r.get(predicted_label_key), r.get(actual_label_key)) for r in subset]
        labeled = [(p, a) for p, a in labeled if p is not None and a is not None]
        if labeled:
            correct = sum(1 for p, a in labeled if int(p) == int(a))
            accuracy = correct / len(labeled)

    return {
        "n": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_net_pnl_pct": sum(net_pnls),
        "avg_net_pnl_pct": sum(net_pnls) / n,
        "max_drawdown_pct": _max_drawdown(net_pnls),
        "accuracy": accuracy,
    }


@dataclass(frozen=True)
class WalkForwardReport:
    available: bool
    n_rows: int
    n_splits_requested: int
    folds: tuple[FoldPerformance, ...] = field(default_factory=tuple)
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "n_rows": self.n_rows,
            "n_splits_requested": self.n_splits_requested,
            "folds": [f.to_dict() for f in self.folds],
            "degraded_reason": self.degraded_reason,
        }

    def mean_across_folds(self, field_name: str) -> float | None:
        vals = [getattr(f, field_name) for f in self.folds if getattr(f, field_name) is not None]
        return sum(vals) / len(vals) if vals else None


def run_purged_walk_forward_report(
    rows: list[dict[str, Any]],
    *,
    open_time_key: str = "opened_at_epoch",
    close_time_key: str = "closed_at_epoch",
    net_pnl_key: str = "net_pnl_pct",
    predicted_label_key: str | None = "predicted_label",
    actual_label_key: str | None = "outcome_label",
    n_splits: int = 5,
    embargo_frac: float = 0.02,
) -> WalkForwardReport:
    """Pure aggregation — no I/O. `rows` must already carry numeric
    open/close timestamps (epoch seconds) alongside the outcome fields."""
    n = len(rows)
    try:
        open_times = [float(r[open_time_key]) for r in rows]
        close_times = [float(r[close_time_key]) for r in rows]
    except (KeyError, TypeError, ValueError) as exc:
        return WalkForwardReport(available=False, n_rows=n, n_splits_requested=n_splits, degraded_reason=f"missing_or_invalid_timestamps:{exc}")

    splits = purged_walk_forward_splits(open_times, close_times, n_splits=n_splits, embargo_frac=embargo_frac)
    if not splits:
        return WalkForwardReport(available=False, n_rows=n, n_splits_requested=n_splits, degraded_reason="insufficient_rows_for_requested_splits")

    folds: list[FoldPerformance] = []
    for split in splits:
        metrics = compute_after_cost_fold_metrics(rows, split.test_idx, net_pnl_key=net_pnl_key, predicted_label_key=predicted_label_key, actual_label_key=actual_label_key)
        folds.append(
            FoldPerformance(
                fold_index=split.fold_index,
                n_train=len(split.train_idx),
                n_test=len(split.test_idx),
                purged_count=split.purged_count,
                embargoed_count=split.embargoed_count,
                win_rate=metrics["win_rate"],
                profit_factor=metrics["profit_factor"],
                total_net_pnl_pct=metrics["total_net_pnl_pct"],
                avg_net_pnl_pct=metrics["avg_net_pnl_pct"],
                max_drawdown_pct=metrics["max_drawdown_pct"],
                accuracy=metrics["accuracy"],
            )
        )

    return WalkForwardReport(available=True, n_rows=n, n_splits_requested=n_splits, folds=tuple(folds))


def _epoch(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


def load_and_report_for_symbol(
    strategy_id: str,
    symbol: str,
    *,
    db_path: str,
    n_splits: int = 5,
    embargo_frac: float = 0.02,
    limit: int = 3000,
) -> WalkForwardReport:
    """Convenience loader: reads real closed-trade rows from
    ``ai_outcome_training_rows`` and runs the purged walk-forward report
    against them. This is the offline reporting entrypoint used by the
    `/portfolio-engine/walk-forward/{symbol}` diagnostic endpoint — never
    called from the live per-tick decision path."""
    try:
        from backend.services.ai_canonical_storage import read_recent_outcome_training_rows

        raw_rows = read_recent_outcome_training_rows(symbol=symbol, strategy_id=strategy_id, limit=limit, db_path=db_path)
    except Exception as exc:
        logger.debug("WALK_FORWARD_LOAD_FAILED strategy=%s symbol=%s: %s", strategy_id, symbol, exc)
        return WalkForwardReport(available=False, n_rows=0, n_splits_requested=n_splits, degraded_reason="load_failed")

    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        opened = _epoch(r.get("opened_at_utc"))
        closed = _epoch(r.get("closed_at_utc"))
        if opened is None or closed is None:
            continue
        rows.append(
            {
                "opened_at_epoch": opened,
                "closed_at_epoch": closed,
                "net_pnl_pct": r.get("net_pnl_pct"),
                "outcome_label": r.get("outcome_label"),
            }
        )
    # Rows must be chronological for the purge/embargo split to be meaningful.
    rows.sort(key=lambda r: r["opened_at_epoch"])

    return run_purged_walk_forward_report(
        rows,
        predicted_label_key=None,
        actual_label_key=None,
        n_splits=n_splits,
        embargo_frac=embargo_frac,
    )


__all__ = [
    "FoldPerformance",
    "FoldSplit",
    "WalkForwardReport",
    "compute_after_cost_fold_metrics",
    "load_and_report_for_symbol",
    "purged_walk_forward_splits",
    "run_purged_walk_forward_report",
]
