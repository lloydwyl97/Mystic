#!/usr/bin/env python3
"""Verify Mystic operator dashboard APIs and HTML wiring (CI/local smoke test)."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "backend/static/dashboard"


def get(path: str, timeout: float = 60.0) -> tuple[int, str]:
    req = urllib.request.Request(f"{BASE}{path}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def main() -> int:
    failures: list[str] = []

    html = (DASH / "index.html").read_text(encoding="utf-8")
    tabs = re.findall(r'data-tab="([^"]+)"', html)
    expected_tabs = [
        "command",
        "day",
        "scalp",
        "positions",
        "learning",
        "performance",
        "marketlens",
        "settings",
    ]
    for t in expected_tabs:
        if t not in tabs:
            failures.append(f"missing tab: {t}")

    for asset in ("assets/mystic-logo.svg", "assets/hero-banner.svg", "vendor/chart.umd.min.js", "app.js?v=45"):
        if "app.js" in asset and "app.js?v=" not in html:
            failures.append(f"index.html missing script app.js?v=")
        code, _ = get(f"/dashboard/{asset.split('?')[0]}")
        if code != 200:
            failures.append(f"asset {asset} HTTP {code}")

    required_ids = [
        "status-cash",
        "eng-day-status",
        "eng-scalp-status",
        "ph-uvicorn",
        "day-basket-tbody",
        "scalp-symbols-tbody",
        "scalp-positions-tbody",
        "ml-feed-pre",
        "operator-config-form",
    ]
    for rid in required_ids:
        if f'id="{rid}"' not in html:
            failures.append(f"missing DOM id: {rid}")

    endpoints = [
        ("/api/system/process-health", lambda d: d.get("processes", {}).get("uvicorn") is True),
        ("/api/portfolio-engine/dashboard-canonical", lambda d: d.get("success") is True),
        ("/api/portfolio-engine/day-health", lambda d: len((d.get("data") or {}).get("basket_signals") or []) >= 1),
        ("/api/scalp/status", lambda d: d.get("runner_active") is True and len(d.get("symbols") or {}) == 4),
        ("/api/scalp/positions", lambda d: "positions" in d),
        ("/api/scalp/trades?limit=5", lambda d: "trades" in d),
        ("/api/scalp/scoreboard?days=7", lambda d: "rows" in d),
        ("/api/scalp/learning-summary", lambda d: d.get("engine") == "scalp"),
        ("/api/scalp/strategies", lambda d: "enabled" in d),
        ("/api/public/mystic-marketlens-feed", lambda d: "summary" in d),
        ("/favicon.ico", lambda _: True),
    ]

    for path, check in endpoints:
        code, body = get(path, timeout=90.0 if "canonical" in path else 60.0)
        if code != 200:
            failures.append(f"{path} HTTP {code}")
            continue
        if path == "/favicon.ico":
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            failures.append(f"{path} not JSON")
            continue
        if not check(data):
            failures.append(f"{path} failed content check: {body[:200]}")

    scalp = json.loads(get("/api/scalp/status")[1])
    if scalp.get("status_error"):
        failures.append(f"scalp status_error: {scalp.get('status_error')}")

    print("=== Mystic Dashboard Live Verification ===")
    print(f"Tabs: {len(tabs)} found — {', '.join(tabs)}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    canon = json.loads(get("/api/portfolio-engine/dashboard-canonical", timeout=90)[1])
    cash = (canon.get("data") or {}).get("risk", {}).get("cash_balance")
    print(f"DAY cash: ${cash:,.2f}" if cash else "DAY cash: ok")
    print(f"SCALP decision: {scalp.get('overall_decision')} blocker: {scalp.get('top_blocker')}")
    print(f"SCALP open: {scalp.get('open_scalp_positions')}")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
