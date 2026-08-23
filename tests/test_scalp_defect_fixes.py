from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services.binance_scalp.paper_engine import _round_trip_execution_costs
from backend.services.binance_scalp.scalp_position_lifecycle import _stale_review_due
from backend.services.binance_scalp import status_snapshot
from backend.services.binance_scalp.schema import init_scalp_schema


class _Signal:
    def __init__(self, symbol: str, score: float) -> None:
        self.symbol = symbol
        self.setup_name = "test_setup"
        self.score = score
        self.passed = True
        self.expected_move_pct = 0.0035
        self.spread_pct = 0.0002
        self.impact_pct = 0.0
        self.confidence = 0.7

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "setup_name": self.setup_name,
            "score": self.score,
            "passed": self.passed,
            "expected_move_pct": self.expected_move_pct,
            "spread_pct": self.spread_pct,
            "impact_pct": self.impact_pct,
            "confidence": self.confidence,
        }


class _Tracker:
    def record(self, *_args) -> None:
        pass

    def diagnostics(self, *_args):
        return SimpleNamespace(as_dict=lambda: {"mid_change_15s": 0.001})


def test_status_uses_enriched_global_pick_and_only_marks_selected_symbol(monkeypatch):
    monkeypatch.setenv("SCALP_FORWARD_NET_ARTIFACT", "/tmp/missing_scalp_artifact.json")
    from backend.services.binance_scalp.forward_net_predictor import reset_artifact_cache

    reset_artifact_cache()
    rows = []
    for symbol, score in (("BTCUSDT", 1.8), ("SOLUSDT", 1.7)):
        signal = _Signal(symbol, score)
        rows.append(
            {
                "symbol": symbol,
                "snap": SimpleNamespace(best_bid=100.0, mid=100.0, spread_pct=0.0002),
                "signal": signal,
                "all_signals": [signal.as_dict()],
                "rank_meta": {
                    "ranked": [{"setup_name": "test_setup", "rank_score": score}],
                    "regime": "range",
                    "reachability_surplus": 0.01,
                },
                "rank_score": score,
                "entry_eligible": True,
                "best_setup": "test_setup",
            }
        )

    class _Router:
        def __init__(self, **_kwargs) -> None:
            pass

        def evaluate_all(self, **_kwargs):
            return rows

        def _current_regime(self, *_args):
            return "range"

        def strategy_inventory(self):
            return {"enabled": ["test_setup"]}

    class _Klines:
        def get(self, _symbol):
            return []

    def _enrich(candidates, **_kwargs):
        enriched = [dict(row) for row in candidates]
        enriched[1]["rank_score_raw"] = enriched[1]["rank_score"]
        enriched[1]["rank_score"] = 2.0
        enriched[1]["intelligence"] = {"boost": 0.3, "unsafe": float("nan")}
        return enriched

    monkeypatch.setattr(status_snapshot, "ScalpStrategyRouter", _Router)
    monkeypatch.setattr(status_snapshot, "KlineCache", _Klines)
    monkeypatch.setattr(
        "backend.services.scalp_ai_rank_enrichment.enrich_scalp_ranked_candidates",
        _enrich,
    )

    result = status_snapshot._evaluate_strategy_router(
        SimpleNamespace(products=("BTCUSDT", "SOLUSDT"), max_notional_paper=150.0),
        SimpleNamespace(),
        SimpleNamespace(),
        _Tracker(),
        warm_rounds=0,
    )

    assert result["overall_entry_ready"] is True
    assert result["best_global_candidate"]["symbol"] == "SOLUSDT"
    assert result["symbols"]["SOLUSDT"]["router_entry_ready"] is True
    assert result["symbols"]["BTCUSDT"]["router_entry_ready"] is False
    assert result["symbols"]["BTCUSDT"]["per_symbol_entry_eligible"] is True
    assert result["symbols"]["SOLUSDT"]["intelligence"]["unsafe"] is None
    json.dumps(result, allow_nan=False)


def test_stale_exit_preview_waits_for_same_review_interval_as_engine():
    now = datetime.now(timezone.utc).timestamp()
    recent = datetime.fromtimestamp(now - 10, timezone.utc).isoformat()
    old = datetime.fromtimestamp(now - 31, timezone.utc).isoformat()

    assert not _stale_review_due(
        hold_sec=600,
        stale_timeout_sec=300,
        stale_review_count=1,
        last_review_ts=recent,
        now_epoch=now,
        review_interval_sec=30,
    )
    assert _stale_review_due(
        hold_sec=600,
        stale_timeout_sec=300,
        stale_review_count=1,
        last_review_ts=old,
        now_epoch=now,
        review_interval_sec=30,
    )
    assert _stale_review_due(
        hold_sec=300,
        stale_timeout_sec=300,
        stale_review_count=0,
        last_review_ts=recent,
        now_epoch=now,
        review_interval_sec=30,
    )


def test_learning_costs_use_persisted_entry_and_exit_economics():
    econ = SimpleNamespace(
        taker_fee_pct=0.001,
        slippage_buffer_pct=0.0005,
        entry_fee_pct=lambda: 0.001,
        exit_fee_pct=lambda: 0.001,
    )
    fees, slippage = _round_trip_execution_costs(
        entry_notional=150.0,
        exit_notional=153.0,
        econ=econ,
        persisted_entry_fee=0.17,
        persisted_entry_slippage=0.08,
    )

    assert fees == pytest.approx(0.17 + 153.0 * 0.001)
    assert slippage == pytest.approx(0.08 + 153.0 * 0.0005)


def test_scalp_entry_rests_as_maker_and_exit_crosses_as_taker():
    """Entries already post at limit_buy_price; exits must cross to guarantee the close."""
    from backend.services.binance_scalp.economics import ScalpEconomics

    econ = ScalpEconomics.from_env()
    assert econ.entry_is_maker is True
    assert econ.exit_is_maker is False
    assert econ.entry_fee_pct() == econ.maker_fee_pct
    assert econ.exit_fee_pct() == econ.taker_fee_pct
    assert econ.roundtrip_fee_pct == econ.maker_fee_pct + econ.taker_fee_pct


def test_scalp_costs_follow_fill_mode_not_hardcoded_taker():
    """The maker/taker mode flag was previously computed and then ignored."""
    from backend.services.binance_scalp.economics import ScalpEconomics

    econ = ScalpEconomics.from_env()
    taker_rt = econ.roundtrip_fee_for_mode(entry_maker=False, exit_maker=False)
    maker_rt = econ.roundtrip_fee_for_mode(entry_maker=True, exit_maker=True)
    assert maker_rt < taker_rt

    fees_taker, _ = _round_trip_execution_costs(
        entry_notional=1000.0,
        exit_notional=1000.0,
        econ=SimpleNamespace(
            taker_fee_pct=econ.taker_fee_pct,
            slippage_buffer_pct=econ.slippage_buffer_pct,
            entry_fee_pct=lambda: econ.taker_fee_pct,
            exit_fee_pct=lambda: econ.taker_fee_pct,
        ),
    )
    fees_maker, _ = _round_trip_execution_costs(
        entry_notional=1000.0,
        exit_notional=1000.0,
        econ=SimpleNamespace(
            taker_fee_pct=econ.taker_fee_pct,
            slippage_buffer_pct=econ.slippage_buffer_pct,
            entry_fee_pct=lambda: econ.maker_fee_pct,
            exit_fee_pct=lambda: econ.maker_fee_pct,
        ),
    )
    assert fees_maker < fees_taker


def _breaker_probe(
    db_path: Path,
    *,
    epoch: str,
    max_consec: int = 10,
    daily_limit_pct: float = 0.05,
    recovery_sec: int = 14400,
    now=None,
):
    """Run the real breaker against a real DB without booting the whole engine."""
    from datetime import datetime, timezone

    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = SimpleNamespace(
        circuit_breaker_epoch=epoch,
        max_consecutive_losses=max_consec,
        daily_loss_limit_pct=daily_limit_pct,
        breaker_recovery_sec=recovery_sec,
        database_path=str(db_path),
    )
    engine._conn = lambda: sqlite3.connect(str(db_path), timeout=10.0)
    engine._ledger = lambda conn: {"principal": 1000.0}
    engine._utcnow_override = now or datetime.now(timezone.utc)
    engine._last_breaker_reason = ""
    engine._last_breaker_recovery_until = ""
    engine._last_breaker_eval_after = ""
    open_ = BinanceScalpPaperEngine._check_scalp_circuit_breaker(engine)
    return open_, engine


def _seed_sells(db_path: Path, rows: list[tuple[float, str]]) -> None:
    init_scalp_schema(db_path, principal=1000.0)
    with sqlite3.connect(db_path) as conn:
        for idx, (pnl, created_at) in enumerate(rows):
            conn.execute(
                "INSERT INTO scalp_paper_trades (trade_id, symbol, side, quantity, price, notional, pnl_usd, created_at) "
                "VALUES (?,'XRPUSDT','SELL',1,1,1,?,?)",
                (f"t{idx}", pnl, created_at),
            )
        conn.commit()


def test_circuit_breaker_epoch_clears_a_stale_losing_run(tmp_path: Path):
    """Operator epoch still excludes a tripped streak from the consec window."""
    from datetime import datetime, timezone

    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 17, 8, 15, tzinfo=timezone.utc)
    _seed_sells(db, [(-0.05, f"2026-08-17 08:{i:02d}:00") for i in range(10)])

    open_, _ = _breaker_probe(db, epoch="", recovery_sec=3600, now=now)
    assert open_ is True
    open_, _ = _breaker_probe(db, epoch="2026-08-21 00:00:00", recovery_sec=3600, now=now)
    assert open_ is False


def test_circuit_breaker_rearms_on_new_losses_after_epoch(tmp_path: Path):
    """Moving the window must not disarm the breaker against fresh losses."""
    from datetime import datetime, timezone

    db = tmp_path / "scalp.db"
    now = datetime(2026, 8, 22, 1, 15, tzinfo=timezone.utc)
    _seed_sells(
        db,
        [(-0.05, f"2026-08-17 08:{i:02d}:00") for i in range(10)]
        + [(-0.05, f"2026-08-22 01:{i:02d}:00") for i in range(10)],
    )

    open_, _ = _breaker_probe(db, epoch="2026-08-21 00:00:00", recovery_sec=3600, now=now)
    assert open_ is True


def test_circuit_breaker_epoch_still_honours_daily_loss_limit(tmp_path: Path):
    """The epoch excludes old trades; it does not raise the daily loss threshold."""
    db = tmp_path / "scalp.db"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _seed_sells(db, [(-60.0, f"{today} 01:00:00")])

    open_, engine = _breaker_probe(db, epoch=f"{today} 00:00:00")
    assert open_ is True
    assert engine._last_breaker_reason == "DAILY_LOSS_LIMIT"


def test_circuit_breaker_epoch_unset_preserves_legacy_behaviour(tmp_path: Path):
    """A mixed run must not trip: the guard is consecutive losses, not any loss."""
    db = tmp_path / "scalp.db"
    _seed_sells(
        db,
        [(-0.05, f"2026-08-17 08:{i:02d}:00") for i in range(9)] + [(0.02, "2026-08-17 09:00:00")],
    )

    open_, _ = _breaker_probe(db, epoch="")
    assert open_ is False


def test_empty_scalp_ledger_repairs_cash_basis_mismatch(tmp_path: Path):
    db_path = tmp_path / "scalp.db"
    init_scalp_schema(db_path, principal=1000.0)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE scalp_paper_ledger SET cash_balance=10000, total_equity=10000 WHERE id=1")
        conn.commit()

    applied = init_scalp_schema(db_path, principal=1000.0)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT principal, cash_balance, total_equity FROM scalp_paper_ledger WHERE id=1").fetchone()
    assert "repair_empty_ledger_basis" in applied
    assert row == (1000.0, 1000.0, 1000.0)
