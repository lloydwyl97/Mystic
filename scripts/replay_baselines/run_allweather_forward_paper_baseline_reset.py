#!/usr/bin/env python3
"""Reset forward paper baseline to clean $25k (excludes synthetic smoke from forward equity)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB = ROOT / "mystic_trading.db"
BACKUP_DIR = ROOT / "var/paper_ledger_rebase"
OUT = ROOT / "scripts/replay_baselines/allweather_forward_paper_baseline_reset_latest.json"
API = "http://localhost:8000"


def _http(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=25) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    from backend.services.allweather_paper_accounting import compute_pnl_breakdown, reset_forward_paper_baseline

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUP_DIR / f"mystic_trading.db.pre_forward_baseline_{ts}"
    shutil.copy2(DB, backup)

    was_running = subprocess.run(["pgrep", "-f", "start_portfolio_engine_integration"], capture_output=True).returncode == 0
    stop_info = {}
    if was_running:
        r = subprocess.run([str(ROOT / "stop_mystic.sh")], cwd=str(ROOT), capture_output=True, text=True)
        time.sleep(3)
        stop_info = {"stopped": True, "exit_code": r.returncode}

    reset_result = reset_forward_paper_baseline(DB)
    breakdown_before = compute_pnl_breakdown(DB)

    start_info = {}
    if was_running:
        r = subprocess.run([str(ROOT / "start_mystic.sh"), "core"], cwd=str(ROOT), capture_output=True, text=True)
        time.sleep(10)
        start_info = {"started": True, "exit_code": r.returncode}

    api_status = {}
    try:
        api_status = _http("/api/portfolio-engine/status").get("data", {})
    except OSError as exc:
        api_status = {"error": str(exc)}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "python3 scripts/replay_baselines/run_allweather_forward_paper_baseline_reset.py",
        "backup_path": str(backup),
        "reset": reset_result,
        "pnl_breakdown": breakdown_before,
        "api_status_after": {
            k: api_status.get(k)
            for k in (
                "principal",
                "cash_balance",
                "total_equity",
                "realized_pnl",
                "realized_pnl_forward",
                "synthetic_smoke_pnl",
                "pre_rebase_history_pnl",
                "forward_equity",
                "unrealized_pnl",
                "positions_count",
            )
        },
        "stop_info": stop_info,
        "start_info": start_info,
        "synthetic_smoke_excluded": True,
        "not_strategy_performance": True,
        "not_forward_pnl": True,
        "not_live_trade": True,
        "not_real_money": True,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
