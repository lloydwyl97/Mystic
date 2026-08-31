from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine
from backend.services.binance_scalp.structural_mode import StructuralModeError
from backend.services.binance_scalp.structural_thesis import (
    STRUCTURAL_NOT_EXECUTABLE,
    new_entry_block_reason,
    prediction_circuit_breaker_applies,
    prediction_entries_permitted,
    ranking_eval_permitted,
    status_fields,
    structural_lp_executable,
)


def test_default_env_is_structural_and_not_live(monkeypatch):
    monkeypatch.delenv("SCALP_THESIS", raising=False)
    monkeypatch.delenv("SCALP_LEGACY_PREDICTION_ENTRIES", raising=False)
    monkeypatch.setenv("SCALP_LIVE", "false")
    cfg = ScalpConfig.from_env()
    assert cfg.scalp_thesis == "structural"
    assert cfg.legacy_prediction_entries is False
    assert cfg.scalp_live is False
    assert prediction_entries_permitted(cfg) is False
    assert ranking_eval_permitted(cfg) is False
    assert prediction_circuit_breaker_applies(cfg) is False
    assert new_entry_block_reason(cfg) == STRUCTURAL_NOT_EXECUTABLE
    assert cfg.resolved_structural_mode() == "STRUCTURAL_PAPER"


def test_legacy_flags_refuse_structural_startup(monkeypatch):
    monkeypatch.setenv("SCALP_THESIS", "legacy_prediction")
    monkeypatch.setenv("SCALP_LEGACY_PREDICTION_ENTRIES", "true")
    monkeypatch.setenv("SCALP_LIVE", "false")
    cfg = ScalpConfig.from_env()
    assert prediction_entries_permitted(cfg) is False
    with pytest.raises(StructuralModeError):
        cfg.assert_structural_startup()


def test_status_fields_paper_lp_not_live_or_arb():
    cfg = SimpleNamespace(
        scalp_thesis="structural",
        legacy_prediction_entries=False,
        scalp_paper_enabled=True,
        scalp_live=False,
        structural_mode="STRUCTURAL_PAPER",
    )
    fields = status_fields(cfg)
    assert fields["structural_entries_executable"] is True
    assert fields["structural_arb_executable"] is False
    assert fields["prediction_entries_permitted"] is False
    assert fields["prediction_circuit_breaker_applies"] is False
    assert structural_lp_executable(cfg) is True


def test_entry_candidates_blocked_on_structural_thesis(monkeypatch):
    monkeypatch.delenv("SCALP_THESIS", raising=False)
    monkeypatch.setenv("SCALP_PAPER_ENABLED", "true")
    monkeypatch.setenv("SCALP_LIVE", "false")
    engine = object.__new__(BinanceScalpPaperEngine)
    engine.config = ScalpConfig.from_env()
    engine._pending_opportunity_rows = None
    engine._pending_opportunity_epoch = None
    engine._publish_last_decision = lambda **_k: None
    engine._record_gate = lambda **_k: None
    assert BinanceScalpPaperEngine._entry_candidates(engine, None) == []
