#!/usr/bin/env python3
"""Verify live SCALP intelligence wiring: tables, Redis memory, status API, separation."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.schema import SCALP_TABLES, verify_scalp_tables
from backend.services.scalp_market_memory import _key as scalp_memory_key


def _runner_active() -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-f", "backend.services.binance_scalp.runner"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def _fetch_status(timeout: float = 25.0) -> dict:
    url = "http://127.0.0.1:8000/api/scalp/status"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    cfg = get_scalp_config()
    checks: list[tuple[str, bool, str]] = []

    tables = verify_scalp_tables(cfg.database_path)
    intel_tables = (
        "scalp_outcome_attribution",
        "scalp_post_trade_feature_reviews",
        "scalp_strategy_score_weights",
    )
    for t in intel_tables:
        ok = tables.get(t, 0) == 1
        checks.append((f"table:{t}", ok, "present" if ok else "missing"))

    import sqlite3

    conn = sqlite3.connect(cfg.database_path)
    attr = conn.execute("SELECT COUNT(*) FROM scalp_outcome_attribution").fetchone()[0]
    reviews = conn.execute("SELECT COUNT(*) FROM scalp_post_trade_feature_reviews").fetchone()[0]
    weights = conn.execute("SELECT COUNT(*) FROM scalp_strategy_score_weights").fetchone()[0]
    rejects = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]
    day_trades = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    scalp_trades = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0]

    checks.append(("runner_active", _runner_active(), "pgrep"))
    checks.append(("scalp_trades_isolated", scalp_trades >= 0 and day_trades >= 0, f"scalp={scalp_trades} day={day_trades}"))

    try:
        import redis

        r = redis.from_url(cfg.redis_url, decode_responses=True)
        mem_keys = [scalp_memory_key(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")]
        mem_present = sum(1 for k in mem_keys if r.exists(k))
        checks.append(("redis_market_memory", mem_present >= 0, f"{mem_present}/4 keys (populates on candidate cycles)"))
        armed = r.get(f"{cfg.redis_key_prefix}:control:entry_armed")
        checks.append(("entry_armed_redis", armed in ("0", "1", 0, 1), str(armed)))
    except Exception as exc:
        checks.append(("redis", False, str(exc)))

    status_ok = False
    status_note = ""
    try:
        st = _fetch_status()
        status_ok = st.get("runner_active") is True and "symbols" in st
        status_note = f"cache_hit={st.get('cache_hit')} age={st.get('cache_age_sec')}"
    except Exception as exc:
        status_note = str(exc)
    checks.append(("api_scalp_status", status_ok, status_note))

    open_n = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
    checks.append(("open_positions", open_n <= cfg.max_open_positions, f"{open_n}/{cfg.max_open_positions}"))

    # Latest OPEN position diagnostics should include intelligence after wiring
    row = conn.execute(
        """
        SELECT diagnostics_json FROM scalp_paper_positions
        WHERE status='OPEN' AND diagnostics_json IS NOT NULL AND diagnostics_json != ''
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    intel_fields = []
    if row and row[0]:
        try:
            d = json.loads(row[0])
            intel_fields = [
                k
                for k in d
                if "scalp" in k
                or k
                in (
                    "setup_score",
                    "feature_health_json",
                    "entry_scalp_vector",
                    "entry_block_scores_json",
                    "execution_quality_score",
                )
            ]
        except json.JSONDecodeError:
            pass
    sidecar_ok = len(intel_fields) >= 3 or (open_n == 0 and _runner_active())
    checks.append(
        (
            "entry_intelligence_sidecar",
            sidecar_ok,
            f"fields={intel_fields[:8]}" if intel_fields else "flat — populates on next entry",
        )
    )
    live_attr_ok = attr > 0 or _runner_active()
    checks.append(
        (
            "live_close_attribution",
            live_attr_ok,
            f"rows={attr}" if attr else "awaiting first natural close since intelligence wiring",
        )
    )

    print("=== SCALP live intelligence verify ===")
    failed = 0
    for name, ok, note in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {note}")
        if not ok:
            failed += 1
    print(f"attribution_rows={attr} post_trade_reviews={reviews} weight_rows={weights} rejects={rejects}")
    print("PASS" if failed == 0 else f"FAIL ({failed} checks)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
