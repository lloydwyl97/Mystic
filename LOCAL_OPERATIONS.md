# Mystic Local Laptop Operations (canonical)

**Environment:** `mystic-Virtual-Machine` — repo `/home/mystic/mystic`.

## Start / stop

| Action | Command |
|--------|---------|
| **Start (use this)** | `cd /home/mystic/mystic && ./start_mystic.sh core` (DAY V2 + SCALP V2, top-4) |
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

That is the whole core stack: six processes. SCALP V2 runs inside the portfolio engine
(`start_portfolio_engine_integration.py`); the old paper `binance_scalp.runner` was retired 2026-09-22
and `start_mystic.sh` stops any lingering instance.

DAY and SCALP are **separate engines**. Dashboard shows them separately; PnL is not mixed.

Requires: Redis running, `.env` with `EXTERNAL_SUPERVISOR_MODE=true`. This VM runs `TRADING_MODE=paper`,
`SCALP_LIVE=false`; production (Ocean) runs live — see the Ocean operations rule.

Note: `live_data_collector.py`, `start_ai_ml_trading.py`, and `start_agent_orchestrator.py` are retired stubs — OHLCV runs in `start_live_market_data.py`. Process-health reports `live_data_collector` under `optional_processes` as `retired_optional`; it is not required for a healthy core stack.

## Trading design (do not drift)

- **Top-4 only:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
- **DAY V2 entry:** 15m bars → learned net-edge decision (`day_v2_decisions`) → direct entry
  (`backend/services/day_v2/live_entry.py`, `submit_day_v2_direct_entry`); status reports `day_entry_path=direct`
- **DAY exit:** exit loop → `monitor_all_positions` → `_check_exit_conditions` via `day_controlled_exits`
- **SCALP V2:** entry and keep-vs-sell both read the learned value model (below); orders go through
  `execute_buy_fifo` with the order-boundary repricing check (`EDGE_DECAYED_AT_ORDER`)
- **Learning:** `trade_learning_outcomes`, `adaptive_candidate_markouts` and the adaptive state
- **No:** soft-rank SCALP entries, replacement churn, bridge BUY, fallback buys, DAY adverse stop-loss
  (SCALP keeps only its catastrophic protection)

## Dead services (removed / disabled — do not rebuild)

- `ai_position_tracker` — module missing; startup disabled
- `ai_outcome_bridge` — module missing; startup disabled
- Do not wire `ai_position:*` into trading or advisory exits
- DB tables `ai_position_state` / `ai_position_recommendation` kept empty (schema only)

## Redis signal keys

`ai_signal:day:{SYMBOL}` and `ai_context:{SYMBOL}` are **Redis hashes**, not strings. Use `HGETALL` / hash reads — `GET` returns WRONGTYPE.

Active keys to preserve: all four `ai_signal:day:*` and `ai_context:*` for top-4.

## Logs and dashboard

- Logs: `/home/mystic/mystic/logs/mystic_*.log` (backend, live_md, signal, portfolio, learning, ai_context)
- Dashboard: `http://127.0.0.1:8000/dashboard/`

## API timing

Shared cadence constants: `backend/config/mystic_api_schedule.py`

## SCALP microstructure

See `docs/SCALP_MICROSTRUCTURE.md`. Exit review cadence `SCALP_EXIT_INTERVAL_SEC` (default 0.25s); rank cadence `SCALP_RANK_INTERVAL_SEC` (default 1s). Do not set `SCALP_LOOP_SEC=0.1`.

## SCALP learned value (SCALP_VALUE_V1)

- SCALP keep-vs-sell and entry read one learned model: `backend/services/scalp_v2/continuation_value.py`.
  Entry final edge = learned value of the just-filled lot − live cost; the claim's `market_edge` is kept for comparison only.
- No model for the current economics: entries price at −cost (none admitted), exits hold neutrally, and the
  integration starts a background build. The binary `hold_remaining_*` metric is not used for SCALP.
- Manual build (app stopped): `venv/bin/python3 scripts/scalp_value_build.py --db mystic_trading.db`
- Status: `curl -s http://localhost:8000/api/portfolio-engine/adaptive/scalp-value | jq .data`
  (`stalled: true` or `SCALP_VALUE_FOLD_FAILED` in the logs means the learner is not advancing).
- Causal replay: `venv/bin/python3 scripts/research/scalp_value_replay.py --db <copy> --start ... --end ...`
- 2026-10-15 finding: only entry-time order flow carries evidence (~+1.4 bps at 2m) against ~6 bps cost,
  so the model admits nothing. Do not add gates to change that.

## Backup reference

Final cleanup backup (2026-05-24):  
`mystic_trading.db.backup_before_final_cleanup_20260524T235400Z`
