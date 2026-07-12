# Mystic Local Laptop Operations (canonical)

**Environment:** `mystic-Virtual-Machine` — repo `/home/mystic/mystic`.

## Start / stop

| Action | Command |
|--------|---------|
| **Start (use this)** | `cd /home/mystic/mystic && ./start_mystic.sh core` (DAY top-4 + scalp paper) |
| Same as core | `./start_mystic.sh full` |
| Stop | `cd /home/mystic/mystic && ./stop_mystic.sh` |
| Desktop shortcut | `~/Desktop/Start Mystic.desktop` → runs **core** |

**Do not use** `./start_mystic.sh all`, `ai`, `collector`, or `agents` — all exit with error (retired legacy stack).

See **CANONICAL_SYSTEM.md** for the full active path, API honesty table, and quarantined legacy list.

## Core stack processes (`./start_mystic.sh core`)

1. Backend API (uvicorn :8000)
2. `start_live_market_data.py`
3. `start_ai_signal_generator.py`
4. `start_portfolio_engine_integration.py`
5. `start_ai_market_context.py`
6. `start_ai_learning.py`
7. `backend.services.binance_scalp.runner` (scalp paper — isolated from DAY)

DAY and scalp are **separate engines**. Dashboard shows them separately; PnL is not mixed.
Scalp `NEAR_PASS` / spread blockers are scalp diagnostics only, not DAY faults.

Requires: Redis running, `.env` with `EXTERNAL_SUPERVISOR_MODE=true`, paper mode unless explicitly switched.

Note: `live_data_collector.py`, `start_ai_ml_trading.py`, and `start_agent_orchestrator.py` are retired stubs — OHLCV runs in `start_live_market_data.py`. Process-health reports `live_data_collector` under `optional_processes` as `retired_optional`; it is not required for a healthy core stack.

## Trading design (do not drift)

- **Paper mode** default; `LIVE_TRADES_ALLOWED=false` unless explicitly changed
- **Top-4 DAY only:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
- **Buy:** `PortfolioEngine.process_bar_candidates` → `execute_buy_fifo`
- **Sell:** exit loop → `monitor_all_positions` → `_check_exit_conditions` (net profit after costs only)
- **Learning:** `trade_learning_outcomes` from real closed trades
- **No:** scalping, stop loss, time exit, replacement churn, bridge BUY, fallback buys

## Dead services (removed / disabled — do not rebuild)

- `ai_position_tracker` — module missing; startup disabled
- `ai_outcome_bridge` — module missing; startup disabled
- Do not wire `ai_position:*` into trading or advisory exits
- DB tables `ai_position_state` / `ai_position_recommendation` kept empty (schema only)

## Redis signal keys

`ai_signal:day:{SYMBOL}` and `ai_context:{SYMBOL}` are **Redis hashes**, not strings. Use `HGETALL` / hash reads — `GET` returns WRONGTYPE.

Active keys to preserve: all four `ai_signal:day:*` and `ai_context:*` for top-4.

## Logs and dashboard

- Logs: `/tmp/mystic_*.log` (backend, live_md, signal, portfolio, learning, collector, ai_context)
- Dashboard: `http://127.0.0.1:8000/dashboard/`

## API timing

Shared cadence constants: `backend/config/mystic_api_schedule.py`

## Backup reference

Final cleanup backup (2026-05-24):  
`mystic_trading.db.backup_before_final_cleanup_20260524T235400Z`
