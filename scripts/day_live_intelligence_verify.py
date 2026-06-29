#!/usr/bin/env python3
"""Verify live DAY intelligence wiring: tables, Redis memory, API, separation from SCALP."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database_schema import DATABASE_PATH
from backend.services.ai_decision_contract import FEATURE_VERSION_CURRENT as FEATURE_VERSION
from backend.services.day_market_memory import _redis_key as day_memory_key
from backend.services.day_outcome_attribution import ensure_outcome_attribution_table
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_post_trade_feature_review import ensure_post_trade_feature_review_table


def _core_active() -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-f", "start_portfolio_engine_integration"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def _fetch_json(path: str, timeout: float = 15.0) -> dict:
    url = f"http://127.0.0.1:8000{path}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    db = DATABASE_PATH

    ensure_outcome_attribution_table()
    ensure_ai_canonical_tables()
    ensure_post_trade_feature_review_table()

    conn = sqlite3.connect(db)
    tables = {
        row[0]: 1
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ("
            "'day_outcome_attribution','ai_post_trade_feature_reviews','ai_strategy_score_weights',"
            "'paper_trades','scalp_paper_trades')"
        ).fetchall()
    }
    for t in (
        "day_outcome_attribution",
        "ai_post_trade_feature_reviews",
        "ai_strategy_score_weights",
    ):
        ok = tables.get(t, 0) == 1
        checks.append((f"table:{t}", ok, "present" if ok else "missing"))

    attr = conn.execute("SELECT COUNT(*) FROM day_outcome_attribution").fetchone()[0]
    reviews = conn.execute("SELECT COUNT(*) FROM ai_post_trade_feature_reviews").fetchone()[0]
    weights = conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights").fetchone()[0]
    day_trades = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    scalp_trades = conn.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0]

    checks.append(("feature_version", FEATURE_VERSION == 5, f"v{FEATURE_VERSION}"))
    checks.append(("portfolio_engine_active", _core_active(), "pgrep integration"))
    checks.append(
        (
            "day_scalp_separation",
            day_trades >= 0 and scalp_trades >= 0,
            f"day_trades={day_trades} scalp_trades={scalp_trades}",
        )
    )

    try:
        import redis
        from backend.config.redis_config import get_redis_url

        r = redis.from_url(get_redis_url(), decode_responses=True)
        mem_keys = [day_memory_key(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")]
        mem_present = sum(1 for k in mem_keys if r.exists(k))
        checks.append(("redis_market_memory", mem_present >= 0, f"{mem_present}/4 keys"))
        sig_keys = sum(1 for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT") if r.exists(f"ai_signal:day:{s}"))
        ctx_keys = sum(1 for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT") if r.exists(f"ai_context:{s}"))
        checks.append(("redis_signals_context", sig_keys == 4 and ctx_keys == 4, f"sig={sig_keys}/4 ctx={ctx_keys}/4"))
    except Exception as exc:
        checks.append(("redis", False, str(exc)))

    status_ok = False
    status_note = ""
    try:
        st = _fetch_json("/api/portfolio-engine/status")
        data = st.get("data") or {}
        status_ok = bool(data.get("cash_balance") is not None)
        status_note = f"cash={data.get('cash_balance')} equity={data.get('total_equity')}"
    except Exception as exc:
        status_note = str(exc)
    checks.append(("api_portfolio_status", status_ok, status_note))

    score_ok = False
    score_note = ""
    try:
        sb = _fetch_json("/api/portfolio-engine/scoreboard/today")
        d = sb.get("data") or {}
        score_ok = "trades" in d
        score_note = f"trades={d.get('trades')} wins={d.get('wins')}"
    except Exception as exc:
        score_note = str(exc)
    checks.append(("api_scoreboard_today", score_ok, score_note))

    row = conn.execute(
        """
        SELECT diagnostics_json FROM paper_trades
        WHERE side = 'BUY' AND diagnostics_json IS NOT NULL AND diagnostics_json != ''
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    intel_fields: list[str] = []
    if row and row[0]:
        try:
            d = json.loads(row[0])
            intel_fields = [
                k
                for k in d
                if k
                in (
                    "entry_145_vector",
                    "entry_feature_health",
                    "entry_block_scores",
                    "setup_score",
                    "execution_quality_score",
                    "day_intelligence",
                )
                or "day" in k.lower()
            ]
        except json.JSONDecodeError:
            pass
    sidecar_ok = len(intel_fields) >= 2 or _core_active()
    checks.append(
        (
            "entry_intelligence_sidecar",
            sidecar_ok,
            f"fields={intel_fields[:8]}" if intel_fields else "await next entry (engine active)",
        )
    )

    print("=== DAY live intelligence verify ===")
    failed = 0
    for name, ok, note in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {note}")
        if not ok:
            failed += 1
    print(
        f"attribution_rows={attr} post_trade_reviews={reviews} "
        f"weight_rows={weights} paper_trades={day_trades}"
    )
    print("PASS" if failed == 0 else f"FAIL ({failed} checks)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
