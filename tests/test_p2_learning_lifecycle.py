"""P2 learning / model lifecycle repair regressions."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion import maybe_rollback_underperforming_model, register_candidate_and_maybe_promote


REPO = Path(__file__).resolve().parents[1]


def _write_artifact(path: Path, *, accuracy: float = 0.6, feature_version: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    payload = {
        "accuracy": accuracy,
        "feature_version": feature_version,
        "feature_dim": 145,
        "live_strategy_id": "day",
        "model": None,
    }
    path.write_bytes(pickle.dumps(payload))


def test_promote_archives_prior_active_and_keeps_candidate_path(tmp_path: Path):
    db = tmp_path / "lifecycle.db"
    ensure_ai_canonical_tables(str(db))
    cand1 = tmp_path / "versions" / "day_BTCUSDT_1.pkl"
    cand2 = tmp_path / "versions" / "day_BTCUSDT_2.pkl"
    active = tmp_path / "active" / "day_BTCUSDT_direction.pkl"
    _write_artifact(cand1, accuracy=0.55)
    _write_artifact(cand2, accuracy=0.70)
    _write_artifact(active, accuracy=0.50)

    metrics = {
        "holdout_status": "OK",
        "holdout_sample_count": 40,
        "candidate_accuracy": 0.70,
        "active_accuracy": 0.50,
        "candidate_profit_after_cost": 0.002,
        "active_profit_after_cost": 0.0005,
        "candidate_bad_trade_rate": 0.2,
        "active_bad_trade_rate": 0.3,
        "holdout_buy_label_count": 10,
        "promotion_path": "unit_test",
        "candidate_holdout": {
            "buy_signal_count": 12,
            "hold_signal_count": 28,
            "buy_precision_if_followed": 0.62,
        },
        "active_holdout": {
            "buy_signal_count": 8,
            "hold_signal_count": 32,
            "buy_precision_if_followed": 0.40,
        },
    }

    with patch(
        "backend.services.ai_model_promotion.evaluate_signal_hash_artifact_contract",
        return_value=(True, None, {}),
    ):
        ok1, _ = register_candidate_and_maybe_promote(
            strategy_id="day",
            symbol="BTCUSDT",
            candidate_path=cand1,
            active_path=active,
            validation_metrics=metrics,
            db_path=str(db),
        )
        ok2, _ = register_candidate_and_maybe_promote(
            strategy_id="day",
            symbol="BTCUSDT",
            candidate_path=cand2,
            active_path=active,
            validation_metrics=metrics,
            db_path=str(db),
        )

    assert ok1 is True and ok2 is True
    with sqlite3.connect(db) as conn:
        actives = conn.execute("SELECT path, status FROM ai_model_versions WHERE status='active'").fetchall()
        archived = conn.execute("SELECT path, status FROM ai_model_versions WHERE status='archived'").fetchall()
    assert len(actives) == 1
    assert str(cand2) in actives[0][0]
    assert len(archived) >= 1
    assert any(str(cand1) in row[0] for row in archived)


def test_rollback_matches_ccxt_outcome_symbol_and_restores_artifact(tmp_path: Path):
    db = tmp_path / "rollback.db"
    ensure_ai_canonical_tables(str(db))
    cand_prev = tmp_path / "versions" / "day_BTCUSDT_prev.pkl"
    cand_cur = tmp_path / "versions" / "day_BTCUSDT_cur.pkl"
    from backend.services.live_strategy_contracts import per_coin_artifact_file

    active = per_coin_artifact_file(tmp_path / "active", "day", "BTCUSDT")
    _write_artifact(cand_prev, accuracy=0.66)
    _write_artifact(cand_cur, accuracy=0.40)
    _write_artifact(active, accuracy=0.40)

    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO ai_model_versions
            (model_id, strategy_id, symbol, feature_version, artifact_hash, path, status, created_at, promoted_at, retired_at)
            VALUES
            ('day:BTCUSDT:prev', 'day', 'BTCUSDT', 5, 'prevhash', ?, 'archived', datetime('now'), datetime('now'), datetime('now')),
            ('day:BTCUSDT:cur', 'day', 'BTCUSDT', 5, 'curhash', ?, 'active', datetime('now'), datetime('now'), NULL)
            """,
            (str(cand_prev), str(cand_cur)),
        )
        for i in range(25):
            conn.execute(
                """
                INSERT INTO ai_outcome_training_rows
                (symbol, opened_at_utc, closed_at_utc, strategy_id, net_pnl_pct, ingested_at_utc)
                VALUES ('BTC/USDT', ?, ?, 'day', ?, datetime('now'))
                """,
                (f"2026-08-01T00:00:{i:02d}Z", f"2026-08-01T01:00:{i:02d}Z", -0.01),
            )
        conn.commit()

    with patch(
        "backend.services.live_strategy_contracts.per_coin_artifact_file",
        return_value=active,
    ):
        ok, reason = maybe_rollback_underperforming_model(
            strategy_id="day",
            symbol="BTCUSDT",
            min_samples=20,
            db_path=str(db),
        )
    assert ok is True, reason
    assert reason == "rollback_executed"
    with sqlite3.connect(db) as conn:
        statuses = dict(conn.execute("SELECT model_id, status FROM ai_model_versions").fetchall())
    assert statuses["day:BTCUSDT:prev"] == "active"
    assert statuses["day:BTCUSDT:cur"] == "rollback"
    # Restored bytes come from previous candidate artifact.
    import pickle

    restored = pickle.loads(active.read_bytes())
    assert float(restored.get("accuracy") or 0) == 0.66


def test_fail_open_fallback_removed_from_pipeline():
    src = (REPO / "backend/ai_training_pipeline.py").read_text()
    assert "fallback_direct_write" not in src
    assert "MODEL_PROMOTION_ERROR" in src
    assert "active unchanged" in src


def test_prune_retires_registry_candidates():
    src = (REPO / "backend/ai_training_pipeline.py").read_text()
    assert "status = 'retired'" in src
    assert "ai_model_versions" in src


def test_model_panel_reads_promote_reject_event_types():
    src = (REPO / "backend/endpoints/portfolio_engine_endpoints.py").read_text()
    assert '("promote", "promoted")' in src or '"promote", "promoted"' in src
    assert '("reject", "rejected")' in src or '"reject", "rejected"' in src


def test_scalp_learning_resolves_nested_rank_score():
    src = (REPO / "backend/services/binance_scalp/paper_engine.py").read_text()
    assert '_sr.get("best_rank_score")' in src
    assert '"rank_score": ranking_meta.get("rank_score")' in src or 'ranking_meta.get("rank_score")' in src


def test_feature_version_no_longer_invents_five_on_missing():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    assert 'ex_payload.get("feature_version") or 5' not in src
    assert 'signal.get("feature_version") or 5' not in src
    assert 'original_explain["feature_version"] = 5' not in src
    assert "do not invent HTF-v5" in src or "Do not invent feature_version" in src


def test_rollback_logger_bound():
    src = (REPO / "backend/services/ai_model_promotion.py").read_text()
    assert "logger = logging.getLogger(__name__)" in src
