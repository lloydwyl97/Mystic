#!/usr/bin/env python3
"""Run all SCALP finish verification scripts and summarize."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "venv" / "bin" / "python3"

SCRIPTS = (
    "scalp_book_hygiene.py",
    "scalp_indicator_truth_audit.py",
    "scalp_outcome_attribution_schema_verify.py",
    "scalp_candidate_simulation.py",
    "scalp_live_intelligence_verify.py",
)


def main() -> int:
    results: dict[str, dict] = {}
    failed = 0
    for name in SCRIPTS:
        proc = subprocess.run([str(PY), str(ROOT / "scripts" / name)], capture_output=True, text=True, check=False)
        ok = proc.returncode == 0
        if not ok:
            failed += 1
        results[name] = {
            "pass": ok,
            "exit_code": proc.returncode,
            "tail": (proc.stdout or proc.stderr or "").strip().splitlines()[-3:],
        }
    print(json.dumps({"pass": failed == 0, "scripts": results}, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
