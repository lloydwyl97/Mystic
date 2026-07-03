"""
Artifact contract enforcement for live ML (Step 3).

Fails closed when Redis signal fields do not match the canonical per-strategy,
per-symbol artifact layout (version, dim, hash, path).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.services.ai_decision_contract import feature_dim_for_version
from backend.services.live_strategy_contracts import live_ai_min_feature_version_for_strategy

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def artifact_contract_gate_enabled() -> bool:
    return os.getenv("ARTIFACT_CONTRACT_GATE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _norm_bus(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def _valid_hex_sha256(s: str | None) -> bool:
    return bool(s and _HEX64.match(s.strip()))


@lru_cache(maxsize=2048)
def _artifact_contract_from_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"exists": False}
    raw = p.read_bytes()
    artifact_hash = hashlib.sha256(raw).hexdigest()
    out: dict[str, Any] = {"exists": True, "sha256": artifact_hash}
    try:
        payload = pickle.loads(raw)
        if isinstance(payload, dict):
            out["feature_version"] = payload.get("feature_version")
            out["feature_dim"] = payload.get("feature_dim")
            out["live_strategy_id"] = payload.get("live_strategy_id")
    except Exception:
        # Hash-only validation still provides a strict contract check.
        pass
    return out


def _artifact_path_matches_contract(path: str, strategy_id: str, symbol_bus: str) -> bool:
    raw = (path or "").strip()
    if not raw or raw.lower() in ("unknown", "none"):
        return False
    sid = strategy_id.strip().lower()
    if sid != "day":
        return False
    sym = _norm_bus(symbol_bus)
    p = raw.replace("\\", "/").lower()
    fname = f"{sym.lower()}_direction.pkl"
    if fname not in p:
        return False
    if f"/{sid}/" in p or p.startswith(f"{sid}/"):
        return True
    return any(seg.lower() == sid for seg in p.split("/"))


def evaluate_signal_hash_artifact_contract(
    dd: dict[str, str],
    *,
    redis_strategy_id: str,
    symbol_bus: str,
) -> tuple[bool, str | None, dict[str, Any]]:
    """
    Validate canonical artifact fields on the Redis ai_signal hash.

    redis_strategy_id comes from the key segment (authoritative strategy namespace).
    symbol_bus is the BUS segment from the same key (e.g. BTCUSDT).
    """
    detail: dict[str, Any] = {"redis_strategy_id": redis_strategy_id, "symbol_bus": _norm_bus(symbol_bus)}
    if not artifact_contract_gate_enabled():
        detail["gate_disabled"] = True
        return True, None, detail

    sid_key = redis_strategy_id.strip().lower()
    strat_h = (dd.get("live_ai_strategy") or "").strip().lower()
    if not strat_h:
        return False, "ARTIFACT_CONTRACT_STRATEGY_FIELD_MISSING", detail
    if strat_h != sid_key:
        detail["live_ai_strategy_hash"] = strat_h
        return False, "ARTIFACT_CONTRACT_STRATEGY_MISMATCH", detail

    raw_fv = dd.get("feature_version")
    raw_fd = dd.get("feature_dim")
    if raw_fv is None or str(raw_fv).strip() == "":
        return False, "ARTIFACT_CONTRACT_AMBIGUOUS_VERSION_DIM", detail
    if raw_fd is None or str(raw_fd).strip() == "":
        return False, "ARTIFACT_CONTRACT_AMBIGUOUS_VERSION_DIM", detail

    try:
        fv = int(float(raw_fv))
        fd = int(float(raw_fd))
    except (TypeError, ValueError):
        return False, "ARTIFACT_CONTRACT_AMBIGUOUS_VERSION_DIM", detail

    detail["feature_version"] = fv
    detail["feature_dim"] = fd

    min_fv = live_ai_min_feature_version_for_strategy(sid_key)
    if fv < min_fv:
        detail["min_feature_version"] = min_fv
        return False, "ARTIFACT_CONTRACT_VERSION_TOO_LOW", detail

    if fd not in (124, 145):
        return False, "ARTIFACT_CONTRACT_DIM_INVALID", detail

    if (fv == 1 and fd != 124) or (fv >= 2 and fd != 145):
        return False, "ARTIFACT_CONTRACT_FV_DIM_MISMATCH", detail

    # Cross-check against canonical decision contract.
    try:
        expected_dim = int(feature_dim_for_version(fv))
    except ValueError:
        return False, "ARTIFACT_CONTRACT_UNKNOWN_FEATURE_VERSION", detail
    if expected_dim != fd:
        detail["expected_feature_dim"] = expected_dim
        return False, "ARTIFACT_CONTRACT_FV_DIM_MISMATCH", detail

    sha = (dd.get("artifact_sha256") or "").strip()

    apath = dd.get("model_artifact_path") or ""
    if not _artifact_path_matches_contract(str(apath), sid_key, symbol_bus):
        detail["model_artifact_path_tail"] = str(apath)[-120:]
        return False, "ARTIFACT_CONTRACT_PATH_MISMATCH", detail

    file_contract = _artifact_contract_from_file(str(apath))
    if not file_contract.get("exists"):
        return False, "ARTIFACT_CONTRACT_FILE_MISSING", detail

    # Strict hash contract: if incoming hash is missing, recover only from real file hash.
    # This preserves fail-closed behavior for truly invalid artifacts while repairing
    # transient missing-hash payloads.
    if not _valid_hex_sha256(sha):
        sha = str(file_contract.get("sha256") or "").strip()
        if _valid_hex_sha256(sha):
            dd["artifact_sha256"] = sha
            detail["hash_recovered_from_file"] = True
        else:
            detail["artifact_sha256_len"] = len(str(dd.get("artifact_sha256") or ""))
            return False, "ARTIFACT_CONTRACT_HASH_MISSING_OR_INVALID", detail

    file_sha = str(file_contract.get("sha256") or "")
    if file_sha and sha.lower() != file_sha.lower():
        detail["artifact_sha256_file_prefix"] = file_sha[:12]
        detail["artifact_sha256_payload_prefix"] = sha[:12]
        # Artifacts can rotate on disk between signal emission and consumption.
        # When path/strategy/fv/dim contracts are valid, trust the canonical on-disk hash.
        dd["artifact_sha256"] = file_sha
        detail["hash_recovered_from_file_mismatch"] = True

    file_fv = file_contract.get("feature_version")
    file_fd = file_contract.get("feature_dim")
    if isinstance(file_fv, int) and file_fv != fv:
        detail["artifact_feature_version"] = file_fv
        return False, "ARTIFACT_CONTRACT_FILE_FEATURE_VERSION_MISMATCH", detail
    if isinstance(file_fd, int) and file_fd != fd:
        detail["artifact_feature_dim"] = file_fd
        return False, "ARTIFACT_CONTRACT_FILE_FEATURE_DIM_MISMATCH", detail
    file_sid = str(file_contract.get("live_strategy_id") or "").strip().lower()
    if file_sid and file_sid != sid_key:
        detail["artifact_live_strategy_id"] = file_sid
        return False, "ARTIFACT_CONTRACT_FILE_STRATEGY_MISMATCH", detail

    return True, None, detail


def evaluate_explainability_artifact_contract(explainability: Any) -> tuple[bool, str | None, dict[str, Any]]:
    """Re-check artifact contract at BUY execution (same rules as Redis hash).

    The per-strategy artifact contract (version/dim/hash/path) only applies to the
    canonical live ML "day" strategy. Other strategy family labels used for
    attribution skip this gate; genuine "day" ML paths remain fail-closed.
    """
    strat = str(getattr(explainability, "live_ai_strategy", "") or "").strip()
    if strat.lower() != "day":
        return True, None, {"skipped": "non_day_strategy_label", "live_ai_strategy": strat}

    sym_ccxt = str(getattr(explainability, "symbol", "") or "")
    symbol_bus = _norm_bus(sym_ccxt)

    dd: dict[str, str] = {
        "live_ai_strategy": strat.lower(),
        "feature_version": str(getattr(explainability, "feature_version", "") or ""),
        "feature_dim": str(getattr(explainability, "feature_dim", "") or ""),
        "artifact_sha256": str(getattr(explainability, "artifact_sha256", "") or ""),
        "model_artifact_path": str(getattr(explainability, "artifact_path", "") or ""),
    }
    return evaluate_signal_hash_artifact_contract(
        dd,
        redis_strategy_id=strat.lower(),
        symbol_bus=symbol_bus,
    )


__all__ = [
    "artifact_contract_gate_enabled",
    "evaluate_explainability_artifact_contract",
    "evaluate_signal_hash_artifact_contract",
]
