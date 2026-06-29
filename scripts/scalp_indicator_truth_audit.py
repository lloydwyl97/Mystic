#!/usr/bin/env python3
"""SCALP indicator truth audit for top-4 symbols."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.scalp_indicator_truth_audit import run_scalp_indicator_truth_audit

OUT = ROOT / "scripts" / "replay_baselines" / "scalp_indicator_truth_audit_latest.json"


def main() -> int:
    report = run_scalp_indicator_truth_audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("=== SCALP INDICATOR TRUTH AUDIT ===")
    print(f"feature_dim={report.get('feature_dim')} pass={report.get('pass')}")
    print(f"needs_fix={report.get('needs_fix_count')} needs_adj={report.get('needs_adjustment_count')}")
    if report.get("fail_reasons"):
        for r in report["fail_reasons"][:10]:
            print(f"  FAIL: {r}")
    print(f"Wrote {OUT}")
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
