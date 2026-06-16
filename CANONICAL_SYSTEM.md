# Mystic Canonical System

Single source of truth for what is **active**, **supported**, and **honest** in this repo.

## Startup (supported)

| Command | Purpose |
|---------|---------|
| `./start_mystic.sh core` | **Canonical 24/7** — DAY top-4 + scalp paper (separate engines) |
| `./start_mystic.sh full` | Same stack as core (DAY + scalp paper) |
| `./stop_mystic.sh` | Stop all Mystic processes |

**Retired (exit with error):** `./start_mystic.sh all`, `./start_mystic.sh ai`, `start_ai_ml_trading.py`, `start_ai_position_tracker.py`, `start_ai_outcome_bridge.py`

Requires: Redis, `.env` with `EXTERNAL_SUPERVISOR_MODE=true`, paper mode unless explicitly switched live.

## Active processes (`full` / `core`)

1. `uvicorn backend.main:app` — API + dashboard (read-only for orders)
2. `start_live_market_data.py` — market data → Redis/SQLite
3. `start_ai_signal_generator.py` — RF `.pkl` → Redis `ai_signal:day:{SYMBOL}`
4. `start_portfolio_engine_integration.py` — **sole DAY execution supervisor**
5. `start_ai_market_context.py` — context → Redis `ai_context:{SYMBOL}`
6. `start_ai_learning.py` — retrain/promote models (learning only; no runtime gates)
7. `backend.services.binance_scalp.runner` — isolated scalp paper (core + full)

**Engine separation:** DAY PnL lives in `portfolio_engine_ledger` / `paper_trades`.
Scalp PnL lives in `scalp_paper_trades` / `scalp_paper_ledger`. Combined account
equity on the dashboard is the DAY ledger; scalp PnL is shown separately.
Scalp diagnostics (`NEAR_PASS`, `SPREAD_TOO_WIDE`, etc.) are not DAY faults.

## Canonical AI execution path

```
start_live_market_data
  → start_ai_market_context → ai_context:{SYMBOL}
  → start_ai_signal_generator → models/active/day/{SYMBOL}_direction.pkl → ai_signal:day:{SYMBOL}
  → start_portfolio_engine_integration._signal_consumption_loop
      → add_buy_candidate (BUY side only; SELL side = ranking penalty telemetry)
  → _bar_processor_loop → process_bar_candidates → execute_buy_fifo
```

**AI controls:** direction/confidence ranking, context nudge, rule-based entry thesis (`day_trade_thesis`), expectancy trust read, optional adaptive weight read (`ADAPTIVE_SCORE_WEIGHT_ENABLED`, default **false**).

**AI does not control:** hard buy/sell gates, kill switch, max positions, net-profit exit floor, thesis `force_sell` emergency path.

**Diagnostics-only:** `ai_diagnostics_endpoints`, dashboard AI panels, `profit_system_diagnostics`.

**Learning-only:** `start_ai_learning.py`, `ai_outcome_training_writer`, `trade_learning_writer`, `ai_learning_ingestion` — no runtime gate changes.

## Continuous learning ingestion (execution stays selective; learning does not)

Trade execution and learning ingestion are **separate**. The engine trades
only the top-4 DAY symbols under strict gates, but learning rows are written
from every meaningful market decision (`backend/services/ai_learning_ingestion.py`):

| Stream | Table | Written by |
|--------|-------|-----------|
| Candidate snapshots (BUY/REJECT/BLOCK/NO_TRADE + reason, rank, thesis) | `ai_candidate_snapshots` | `process_bar_candidates` |
| Forward labels (15m/30m/1h/4h/24h returns, MFE/MAE, target/invalidation, verdict) | same rows | snapshot labeler in learning collection loop |
| Open-position heartbeats (MFE/MAE path, thesis validity, would-sell-now) | `ai_position_heartbeats` | `monitor_all_positions` (throttled 300s) |
| Closed-trade outcomes (strongest labels) | `ai_outcome_training_rows` | `ai_outcome_training_writer` (unchanged) |
| Missed-move stats → bounded rank delta (±0.03 max; reorder-only, never opens trades) | derived from labeled snapshots | `rank_score()` via `missed_move_rank_delta` |

**Tiered training mix** (per-coin RF, train-split only — validation and PnL
metrics stay on real data):

- Tier A: closed-trade outcomes (weight 5.0)
- Tier B: open-trade MFE/MAE path labels (weight 2.0)
- Tier C: rejected/no-trade forward-return labels (weight 0.8)
- Tier D: walk-forward candle anchors / self-supervised (weight 1.0)

**Promotion fallback:** when the real closed-trade holdout is below the
confidence minimum, a Tier C synthetic holdout (≥40 labeled snapshots) can
approve promotion (`tiered_holdout_pass`); synthetic rows are never mixed
into real PnL metrics.

**Backfill:** the learning collector sweeps all four coins per cycle across
the cached 4h bundle history (`DAY_HISTORICAL_TRAIN_BASES=BTC,ETH,SOL,XRP`,
tail 480 bars, stride 2 — set in `start_mystic.sh start_learning`).

**Readout:** `/api/ai-diagnostics/learning-health` + dashboard "Learning
health (DAY top-4)" panel. `/api/scalp/status` reports `runner_active=false`
when the scalp runner is not running (core mode).

## Canonical DAY buy path

```
_signal_consumption_loop → add_buy_candidate
_bar_processor_loop → process_bar_candidates → execute_buy_fifo
```

No HTTP buy. No Redis bridge buy. No `execute_buy_from_signal` (removed).

## Canonical DAY sell path

```
_position_monitor_loop → _monitor_positions_once
  → monitor_all_positions → _check_exit_conditions → execute_sell_fifo
```

Also on bar rank: open-position thesis invalidation → `execute_sell_fifo(force_sell=True)`.

No HTTP sell. No model-signal sell execution. No rotation-for-capacity.

## Dashboard / UI

- **Read-only** for positions, scoreboard, diagnostics, DAY health
- **Operator controls only:** PAPER/LIVE mode, kill switch, limits (ADMIN_TOKEN)
- **No buy/sell buttons**

## Supported API surface

| Route | Role |
|-------|------|
| `/api/portfolio-engine/*` | Canonical ledger, scoreboard, status |
| `/api/ai-diagnostics/*` | Read-only AI diagnostics |
| `/api/scalp/*` | Scalp paper status |
| `/api/paper-trading/*` GET | Legacy read mirrors (portfolio/positions) |
| `/api/trading/*` GET | Admin Binance.US account read |
| `/api/system/*` | Pause/resume trading flags |

**Retired (HTTP 410):**

- `POST /api/paper-trading/orders`
- `POST /api/paper-trading/process-signals`
- `POST /api/trading/execute_live_market_buy`
- `POST /api/trading/execute_live_market_sell`
- `POST /api/orders`, `POST /api/orders/place`, `POST /orders/advanced`

## Quarantined / removed legacy

- `execute_buy_from_signal`, `execute_sell_from_signal` — removed from engine
- `_check_quality_filters` — removed (unused); pacing uses `record_sell_cooldown` / `increment_buy_counters`
- `_rotate_weakest_position_for_capacity` — removed
- `ai_position_tracker`, `ai_outcome_bridge` — modules absent; launchers exit 1
- `live_data_collector.py` — deprecated; OHLCV in live market data
- Dead AI registry entries (LSTM/transformer/heuristics) — removed from registry

## Scalp (isolated)

`binance_scalp/paper_engine.py` — separate tables and runner; not routed through `portfolio_engine` FIFO.
