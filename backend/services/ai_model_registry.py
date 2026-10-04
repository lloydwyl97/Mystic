"""Immutable model artifact registry, ACTIVE/PREVIOUS pointers, and atomic promotion.

Layout under ``models/registry/<strategy>/``::

    promotion_control.json          auto-promotion switch (missing => disabled)
    <SYMBOL>/<sha256>.pkl           write-once artifact (content addressed)
    <SYMBOL>/<sha256>.json          write-once metadata
    <SYMBOL>/ACTIVE.json            pointer to the artifact live inference serves
    <SYMBOL>/PREVIOUS.json          pointer to the incumbent the last switch replaced
    <SYMBOL>/events.jsonl           append-only promotion / rollback / reject log

Live inference keeps reading ``models/active/<strategy>/<SYMBOL>_direction.pkl``;
that file is only ever replaced by ``os.replace`` from a verified copy of a
registered artifact, so it is never half written. Artifacts that were ever ACTIVE
or PREVIOUS are never deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REGISTRY_DIRNAME = "registry"
CANDIDATE_RETENTION = int(os.getenv("MODEL_REGISTRY_CANDIDATE_RETENTION", "96") or "96")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_root(active_dir: Path | str | None = None) -> Path:
    if active_dir is None:
        from backend.utils.path_helpers import ensure_model_directories

        active_dir = ensure_model_directories()["active"]
    return Path(active_dir).resolve().parent / REGISTRY_DIRNAME


def _sym_dir(root: Path, strategy_id: str, symbol: str) -> Path:
    return root / strategy_id.strip().lower() / symbol.strip().upper()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(dst)
    _fsync_dir(dst.parent)


def atomic_write_json(dst: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(dst, json.dumps(payload, indent=2, sort_keys=True, default=str).encode())


def read_json(path: Path) -> dict[str, Any]:
    try:
        out = json.loads(Path(path).read_text())
        return out if isinstance(out, dict) else {}
    except (OSError, ValueError):
        return {}


def source_commit() -> str:
    try:
        repo = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------- promotion switch


def promotion_control_path(strategy_id: str, root: Path | None = None) -> Path:
    return (root or registry_root()) / strategy_id.strip().lower() / "promotion_control.json"


def promotion_enabled(strategy_id: str, root: Path | None = None) -> tuple[bool, str]:
    """Automatic replacement of the active model is off unless explicitly enabled."""
    ctl = read_json(promotion_control_path(strategy_id, root))
    if not ctl:
        return False, "promotion_control_missing"
    return bool(ctl.get("enabled")), str(ctl.get("reason") or "")


def set_promotion_enabled(strategy_id: str, enabled: bool, reason: str, root: Path | None = None) -> dict[str, Any]:
    payload = {"enabled": bool(enabled), "reason": reason, "set_at": _now(), "source_commit": source_commit()}
    atomic_write_json(promotion_control_path(strategy_id, root), payload)
    return payload


# --------------------------------------------------------------- artifacts


def reload_test(path: Path, feature_dim: int | None = None) -> tuple[bool, str]:
    """Load ``path`` from disk and run one prediction; any failure means not servable."""
    try:
        art = pickle.loads(Path(path).read_bytes())
    except Exception as exc:
        return False, f"unpickle_failed:{type(exc).__name__}"
    if not isinstance(art, dict) or art.get("model") is None or art.get("scaler") is None:
        return False, "missing_model_or_scaler"
    dim = int(feature_dim or art.get("feature_dim") or 0)
    if dim <= 0:
        return False, "unknown_feature_dim"
    try:
        x = art["scaler"].transform(np.zeros((1, dim), dtype=np.float64))
        pred = art["model"].predict(x)
        if len(np.asarray(pred).reshape(-1)) != 1:
            return False, "bad_prediction_shape"
    except Exception as exc:
        return False, f"predict_failed:{type(exc).__name__}"
    return True, "ok"


def artifact_path(strategy_id: str, symbol: str, sha: str, root: Path | None = None) -> Path:
    return _sym_dir(root or registry_root(), strategy_id, symbol) / f"{sha}.pkl"


def register_artifact(strategy_id: str, symbol: str, src: Path, meta: dict[str, Any] | None = None, root: Path | None = None) -> dict[str, Any]:
    """Copy ``src`` into the registry once (content addressed) and write its metadata once."""
    root = root or registry_root()
    data = Path(src).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    dst = artifact_path(strategy_id, symbol, sha, root)
    if not dst.exists():
        atomic_write_bytes(dst, data)
    if sha256_file(dst) != sha:
        raise OSError(f"registry artifact hash mismatch {dst}")
    meta_path = dst.with_suffix(".json")
    if not meta_path.exists():
        record = {"version": sha[:16], "sha256": sha, "strategy_id": strategy_id, "symbol": symbol.upper(), "registered_at": _now(), "source_path": str(src), "source_commit": source_commit()}
        record.update(meta or {})
        atomic_write_json(meta_path, record)
    return {"version": sha[:16], "sha256": sha, "path": str(dst), "meta_path": str(meta_path)}


def artifact_meta(art_path: Path) -> dict[str, Any]:
    try:
        art = pickle.loads(Path(art_path).read_bytes())
    except Exception:
        return {}
    if not isinstance(art, dict):
        return {}
    keep = (
        "trained_at",
        "feature_version",
        "feature_dim",
        "train_outcome_max_id",
        "holdout_window",
        "train_samples",
        "accuracy",
        "artifact_id",
        "live_strategy_id",
        "training_window",
        "training_fingerprint",
    )
    return {k: art.get(k) for k in keep if k in art}


def _pointer(root: Path, strategy_id: str, symbol: str, which: str) -> Path:
    return _sym_dir(root, strategy_id, symbol) / f"{which}.json"


def read_pointer(strategy_id: str, symbol: str, which: str, root: Path | None = None) -> dict[str, Any]:
    return read_json(_pointer(root or registry_root(), strategy_id, symbol, which))


def append_event(strategy_id: str, symbol: str, event: dict[str, Any], root: Path | None = None) -> None:
    path = _sym_dir(root or registry_root(), strategy_id, symbol) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"at": _now(), **event}, sort_keys=True, default=str)
    with path.open("a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_events(strategy_id: str, symbol: str, root: Path | None = None, limit: int = 200) -> list[dict[str, Any]]:
    path = _sym_dir(root or registry_root(), strategy_id, symbol) / "events.jsonl"
    try:
        lines = path.read_text().splitlines()[-limit:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _pointer_payload(reg: dict[str, Any], *, reason: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"version": reg["version"], "sha256": reg["sha256"], "path": reg["path"], "set_at": _now(), "reason": reason, "meta": meta or {}}


def ensure_incumbent_registered(strategy_id: str, symbol: str, active_path: Path, root: Path | None = None, *, reason: str = "incumbent_registered") -> dict[str, Any] | None:
    """Archive the bytes live inference serves and make ACTIVE point at them."""
    root = root or registry_root()
    active_path = Path(active_path)
    if not active_path.exists():
        return None
    reg = register_artifact(strategy_id, symbol, active_path, {"role": "incumbent", **artifact_meta(active_path)}, root)
    ptr = read_pointer(strategy_id, symbol, "ACTIVE", root)
    if ptr.get("sha256") != reg["sha256"]:
        atomic_write_json(_pointer(root, strategy_id, symbol, "ACTIVE"), _pointer_payload(reg, reason=reason, meta=artifact_meta(active_path)))
        append_event(strategy_id, symbol, {"event": "active_pointer_synced", "version": reg["version"], "reason": reason}, root)
    return reg


def _switch_active_file(src: Path, active_path: Path) -> None:
    atomic_write_bytes(active_path, Path(src).read_bytes())


def promote_atomic(
    strategy_id: str,
    symbol: str,
    candidate_path: Path,
    active_path: Path,
    *,
    decision: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Switch live serving to ``candidate_path``; any failure leaves the incumbent active.

    Order: candidate registered (written, hash-checked) -> reload-tested -> metadata ->
    incumbent archived -> active file replaced atomically -> post-switch reload verified
    -> pointers written. A failed post-switch check restores the incumbent bytes.
    """
    root = root or registry_root()
    active_path = Path(active_path)
    info: dict[str, Any] = {}
    try:
        cand = register_artifact(strategy_id, symbol, candidate_path, {"role": "candidate", **artifact_meta(candidate_path)}, root)
    except Exception as exc:
        return False, f"candidate_write_failed:{type(exc).__name__}", info
    info["candidate"] = cand
    ok, why = reload_test(Path(cand["path"]))
    if not ok:
        append_event(strategy_id, symbol, {"event": "promotion_aborted", "stage": "candidate_reload", "reason": why, "candidate": cand["version"]}, root)
        return False, f"candidate_reload_failed:{why}", info
    incumbent = None
    try:
        incumbent = ensure_incumbent_registered(strategy_id, symbol, active_path, root, reason="archived_before_switch")
    except Exception as exc:
        append_event(strategy_id, symbol, {"event": "promotion_aborted", "stage": "archive_incumbent", "reason": str(exc)}, root)
        return False, f"incumbent_archive_failed:{type(exc).__name__}", info
    info["incumbent"] = incumbent
    if incumbent is not None and incumbent["sha256"] == cand["sha256"]:
        return False, "candidate_identical_to_incumbent", info
    try:
        _switch_active_file(Path(cand["path"]), active_path)
        ok, why = reload_test(active_path)
        if ok and sha256_file(active_path) != cand["sha256"]:
            ok, why = False, "post_switch_hash_mismatch"
    except Exception as exc:
        ok, why = False, f"switch_failed:{type(exc).__name__}"
    if not ok:
        if incumbent is not None:
            _switch_active_file(Path(incumbent["path"]), active_path)
        append_event(strategy_id, symbol, {"event": "promotion_rolled_back", "stage": "post_switch", "reason": why, "candidate": cand["version"]}, root)
        return False, f"post_switch_verify_failed:{why}", info
    if incumbent is not None:
        atomic_write_json(_pointer(root, strategy_id, symbol, "PREVIOUS"), _pointer_payload(incumbent, reason="replaced_by_promotion", meta=artifact_meta(Path(incumbent["path"]))))
    atomic_write_json(_pointer(root, strategy_id, symbol, "ACTIVE"), _pointer_payload(cand, reason="promoted", meta={**artifact_meta(Path(cand["path"])), "decision": decision or {}}))
    append_event(
        strategy_id,
        symbol,
        {"event": "promoted", "version": cand["version"], "previous": incumbent["version"] if incumbent else None, "decision": decision or {}, "source_commit": source_commit()},
        root,
    )
    return True, "promoted", info


def was_rolled_back(strategy_id: str, symbol: str, version: str, root: Path | None = None) -> bool:
    return any(ev.get("event") == "rollback" and ev.get("replaced") == version for ev in read_events(strategy_id, symbol, root, limit=100_000))


def rollback_to_previous(strategy_id: str, symbol: str, active_path: Path, *, reason: str, root: Path | None = None) -> tuple[bool, str]:
    """Deterministically restore the PREVIOUS pointer's artifact; pointers swap."""
    root = root or registry_root()
    prev = read_pointer(strategy_id, symbol, "PREVIOUS", root)
    if not prev or not Path(str(prev.get("path") or "")).exists():
        return False, "previous_artifact_missing"
    if was_rolled_back(strategy_id, symbol, str(prev.get("version") or ""), root):
        return False, "previous_was_rolled_back"
    if sha256_file(Path(prev["path"])) != prev.get("sha256"):
        return False, "previous_artifact_corrupt"
    ok, why = reload_test(Path(prev["path"]))
    if not ok:
        return False, f"previous_reload_failed:{why}"
    current = ensure_incumbent_registered(strategy_id, symbol, Path(active_path), root, reason="archived_before_rollback")
    _switch_active_file(Path(prev["path"]), Path(active_path))
    if sha256_file(Path(active_path)) != prev["sha256"] or not reload_test(Path(active_path))[0]:
        if current is not None:
            _switch_active_file(Path(current["path"]), Path(active_path))
        return False, "post_rollback_verify_failed"
    reg = {"version": prev["version"], "sha256": prev["sha256"], "path": prev["path"]}
    if current is not None:
        atomic_write_json(_pointer(root, strategy_id, symbol, "PREVIOUS"), _pointer_payload(current, reason=f"rolled_back:{reason}"))
    atomic_write_json(_pointer(root, strategy_id, symbol, "ACTIVE"), _pointer_payload(reg, reason=f"rollback:{reason}", meta=prev.get("meta") or {}))
    append_event(strategy_id, symbol, {"event": "rollback", "version": prev["version"], "replaced": current["version"] if current else None, "reason": reason}, root)
    return True, "rollback_executed"


def protected_shas(strategy_id: str, symbol: str, root: Path | None = None) -> set[str]:
    root = root or registry_root()
    out = {read_pointer(strategy_id, symbol, w, root).get("sha256") for w in ("ACTIVE", "PREVIOUS")}
    for ev in read_events(strategy_id, symbol, root, limit=100_000):
        for key in ("version", "previous", "replaced"):
            if ev.get("event") in ("promoted", "rollback", "active_pointer_synced") and ev.get(key):
                out.add(str(ev[key]))
    return {s for s in out if s}


def prune_candidates(strategy_id: str, symbol: str, keep: int = CANDIDATE_RETENTION, root: Path | None = None) -> int:
    """Drop the oldest never-served candidates beyond ``keep``. Served artifacts are kept forever."""
    root = root or registry_root()
    d = _sym_dir(root, strategy_id, symbol)
    if not d.is_dir():
        return 0
    protected = protected_shas(strategy_id, symbol, root)
    arts = sorted(d.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    removable = [p for p in arts if p.stem not in protected and p.stem[:16] not in protected]
    removed = 0
    for p in removable[keep:]:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
        removed += 1
    return removed


def symbol_status(strategy_id: str, symbol: str, active_path: Path, root: Path | None = None) -> dict[str, Any]:
    root = root or registry_root()
    act = read_pointer(strategy_id, symbol, "ACTIVE", root)
    prev = read_pointer(strategy_id, symbol, "PREVIOUS", root)
    live_sha = sha256_file(Path(active_path)) if Path(active_path).exists() else None
    events = read_events(strategy_id, symbol, root, limit=200)
    last_eval = next((e for e in reversed(events) if e.get("event") in ("evaluated", "promoted", "promotion_rolled_back", "promotion_aborted")), {})
    last_promo = next((e for e in reversed(events) if e.get("event") in ("promoted", "rollback")), {})
    enabled, why = promotion_enabled(strategy_id, root)
    meta = act.get("meta") or {}
    hw = meta.get("holdout_window") or {}
    return {
        "active_model_version": act.get("version"),
        "active_matches_live_file": bool(live_sha and live_sha == act.get("sha256")),
        "previous_model_version": prev.get("version"),
        "trained_at": meta.get("trained_at"),
        "training_cutoff_outcome_id": meta.get("train_outcome_max_id"),
        "evaluation_window": {k: hw.get(k) for k in ("first_closed_at", "last_closed_at", "n") if k in hw},
        "last_candidate_version": last_eval.get("candidate"),
        "candidate_verdict": (last_eval.get("decision") or {}).get("verdict"),
        "promotion_enabled": enabled,
        "promotion_control_reason": why,
        "last_promotion_reason": last_promo.get("reason") or (last_promo.get("decision") or {}).get("verdict") or act.get("reason"),
        "rollback_available": bool(prev.get("path") and Path(str(prev["path"])).exists() and not was_rolled_back(strategy_id, symbol, str(prev.get("version") or ""), root)),
    }


__all__ = [
    "CANDIDATE_RETENTION",
    "append_event",
    "artifact_meta",
    "artifact_path",
    "atomic_write_json",
    "ensure_incumbent_registered",
    "promote_atomic",
    "promotion_enabled",
    "protected_shas",
    "prune_candidates",
    "read_events",
    "read_pointer",
    "register_artifact",
    "registry_root",
    "reload_test",
    "rollback_to_previous",
    "set_promotion_enabled",
    "sha256_file",
    "symbol_status",
    "was_rolled_back",
]
