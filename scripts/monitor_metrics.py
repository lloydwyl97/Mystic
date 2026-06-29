"""Safe API fetch + snapshot metrics for Mystic monitor/report scripts."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any


def fetch_json(url: str, timeout: float = 20.0) -> tuple[dict[str, Any] | None, str | None]:
    """Return (parsed_json, error_message). Never raises."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
            if not raw:
                return None, f"empty response from {url}"
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                preview = raw[:120].replace("\n", " ")
                return None, f"invalid JSON from {url}: {exc}; preview={preview!r}"
            if not isinstance(data, dict):
                return None, f"expected JSON object from {url}, got {type(data).__name__}"
            return data, None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code} from {url}"
    except urllib.error.URLError as exc:
        return None, f"URL error for {url}: {exc.reason}"
    except TimeoutError:
        return None, f"timeout fetching {url}"
    except Exception as exc:
        return None, f"fetch failed for {url}: {exc}"


def build_snapshot_metrics(
    *,
    api_base: str,
    procs: int,
    dash_code: str,
    mem: str,
    disk: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "procs": int(procs),
        "dashboard_http": str(dash_code),
        "mem": mem,
        "disk": disk,
    }

    status, err = fetch_json(f"{api_base}/api/portfolio-engine/status")
    if err:
        out["status_fetch_error"] = err
    elif status:
        d = status.get("data") if isinstance(status.get("data"), dict) else {}
        out["infra_health"] = d.get("account_status")
        out["equity"] = round(float(d.get("total_equity") or 0), 2)
        out["cash"] = round(float(d.get("cash_balance") or 0), 2)
        out["ledger_realized_alltime"] = round(float(d.get("realized_pnl") or 0), 4)
        out["unrealized_pnl"] = round(float(d.get("unrealized_pnl") or 0), 2)
        out["positions"] = len(d.get("open_positions") or [])
        out["degraded"] = d.get("degraded")
        out["exit_blocked"] = len(d.get("exit_blocked_positions") or [])
        out["symbols"] = [p.get("symbol") for p in (d.get("open_positions") or []) if isinstance(p, dict)]

    score, err2 = fetch_json(f"{api_base}/api/portfolio-engine/scoreboard/today")
    if err2:
        out["scoreboard_fetch_error"] = err2
    elif score:
        s = score.get("data") if isinstance(score.get("data"), dict) else {}
        out["strategy_pass_fail"] = s.get("pass_fail")
        out["strategy_fail_reasons"] = s.get("fail_reasons")
        # Strategy scoreboard: audit-based closed AI SELLs today (excludes legacy/admin ops)
        out["strategy_today_realized_pnl"] = round(float(s.get("realized_pnl") or 0), 4)
        out["strategy_today_trades"] = s.get("trades")
        out["paper_sells_today_pnl"] = round(float(s.get("ai_realized_pnl_today") or s.get("ai_closed_pnl") or 0), 4)
        lr = s.get("ledger_realized_pnl")
        if lr is not None:
            out["ledger_realized_alltime_scoreboard"] = round(float(lr), 4)

    return out


def main() -> None:
    if len(sys.argv) < 6:
        print(json.dumps({"error": "usage: monitor_metrics.py API procs dash mem disk"}))
        sys.exit(1)
    api, procs, dash, mem, disk = sys.argv[1:6]
    print(json.dumps(build_snapshot_metrics(api_base=api, procs=int(procs), dash_code=dash, mem=mem, disk=disk), separators=(",", ":")))


if __name__ == "__main__":
    main()
