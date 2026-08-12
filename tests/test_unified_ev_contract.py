"""Item p22: unified ranking/EV contract manifest + snapshot aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services import unified_ev_contract as uec


def test_manifest_returns_nonempty_list():
    manifest = uec.unified_contract_manifest()
    assert len(manifest) > 10
    names = [f.name for f in manifest]
    assert len(names) == len(set(names))  # no duplicate family names


def test_manifest_never_marks_a_family_as_gating():
    # Architecture rule: trade-opinion evidence never becomes a hard gate.
    manifest = uec.unified_contract_manifest()
    assert all(f.gating is False for f in manifest)


def test_manifest_covers_both_engines():
    manifest = uec.unified_contract_manifest()
    engines = {f.engine for f in manifest}
    assert "day" in engines
    assert "scalp" in engines
    assert "shared" in engines


def test_manifest_as_dict_matches_manifest_length():
    assert len(uec.manifest_as_dict()) == len(uec.unified_contract_manifest())


def test_manifest_weights_are_pulled_live_not_hardcoded(monkeypatch):
    monkeypatch.setenv("HOLDEV_WEIGHT_MOMENTUM", "0.99")
    manifest = uec.unified_contract_manifest()
    hev_mom = next(f for f in manifest if f.name == "hold_ev_momentum")
    assert hev_mom.weight_or_cap == 0.99


def test_snapshot_handles_empty_payload():
    result = uec.compute_unified_ranking_snapshot("BTCUSDT")
    assert result["symbol"] == "BTCUSDT"
    assert result["families"]["ctx_multiplier"] == 0.0
    assert result["families"]["feature_stack"] is None


def test_snapshot_parses_json_subfields():
    ctx = {
        "ctx_multiplier": "1.02",
        "ctx_rs_btc": 0.3,
        "ctx_feature_stack_json": json.dumps({"momentum": {"1h": 0.01}}),
        "ctx_derivatives_json": json.dumps({"available": True, "basis_pct": 0.001}),
    }
    result = uec.compute_unified_ranking_snapshot("ETHUSDT", ctx_payload=ctx)
    families = result["families"]
    assert families["ctx_multiplier"] == 1.02
    assert families["feature_stack"]["momentum"]["1h"] == 0.01
    assert families["derivatives_reference"]["available"] is True


def test_snapshot_degrades_gracefully_on_malformed_json():
    ctx = {"ctx_feature_stack_json": "{not valid json"}
    result = uec.compute_unified_ranking_snapshot("SOLUSDT", ctx_payload=ctx)
    assert result["families"]["feature_stack"] == {"parse_error": True}


def test_snapshot_includes_scalp_ranking_meta_when_provided():
    scalp_meta = {
        "arm_penalty_mult": 0.8,
        "mtf_penalty_mult": 0.4,
        "regime_mismatch": True,
        "symbol_stall_risk": False,
        "strategy_passed": False,
        "entry_owner": "ranking_ev",
    }
    result = uec.compute_unified_ranking_snapshot("XRPUSDT", scalp_ranking_meta=scalp_meta)
    families = result["families"]
    assert families["scalp_arm_penalty_mult"] == 0.8
    assert families["scalp_regime_mismatch"] is True
    assert families["scalp_entry_owner"] == "ranking_ev"


def test_snapshot_omits_scalp_fields_when_no_meta_given():
    result = uec.compute_unified_ranking_snapshot("BTCUSDT")
    assert "scalp_arm_penalty_mult" not in result["families"]


def test_snapshot_includes_calibration_when_provided():
    result = uec.compute_unified_ranking_snapshot("BTCUSDT", calibration={"mult": 0.85, "reason": "calibration_degraded:brier=0.4>0.28"})
    assert result["families"]["calibration_confidence_mult"] == 0.85
    assert "calibration_degraded" in result["families"]["calibration_reason"]


def test_snapshot_omits_calibration_when_not_given():
    result = uec.compute_unified_ranking_snapshot("BTCUSDT")
    assert "calibration_confidence_mult" not in result["families"]


def test_manifest_includes_calibration_family():
    manifest = uec.unified_contract_manifest()
    names = [f.name for f in manifest]
    assert "calibration_confidence_mult" in names
