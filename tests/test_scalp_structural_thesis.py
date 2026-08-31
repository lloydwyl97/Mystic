from __future__ import annotations

from types import SimpleNamespace

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine
from backend.services.binance_scalp.structural_thesis import (
    PREDICTION_RETIRED,
    STRUCTURAL_NOT_EXECUTABLE,
    new_entry_block_reason,
    prediction_entries_permitted,
    status_fields,
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
    assert new_entry_block_reason(cfg) == STRUCTURAL_NOT_EXECUTABLE


def test_legacy_requires_both_flags(monkeypatch):
    monkeypatch.setenv("SCALP_THESIS", "legacy_prediction")
    monkeypatch.setenv("SCALP_LEGACY_PREDICTION_ENTRIES", "false")
    cfg = ScalpConfig.from_env()
    assert prediction_entries_permitted(cfg) is False
    assert new_entry_block_reason(cfg) == PREDICTION_RETIRED
    monkeypatch.setenv("SCALP_LEGACY_PREDICTION_ENTRIES", "true")
    cfg2 = ScalpConfig.from_env()
    assert prediction_entries_permitted(cfg2) is True
    assert new_entry_block_reason(cfg2) is None


def test_status_fields_never_claim_structural_or_live_execution():
    cfg = SimpleNamespace(scalp_thesis="structural", legacy_prediction_entries=False)
    fields = status_fields(cfg)
    assert fields["structural_entries_executable"] is False
    assert fields["prediction_entries_permitted"] is False


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
