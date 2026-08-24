# SCALP microstructure (implemented)

Version stamps: `feature_version=1` (existing 40-dim vector unchanged),
`microstructure_version=scalp_micro_v1`, `model_version=scalp_micro_ev_v1`,
`selection_version=scalp_micro_select_v1`.

## Data streams

- Binance.US spot `depth20@100ms` via `order_book_collector` (full top-20 snapshots + `lastUpdateId`).
- Local sequenced book: `backend/services/binance_scalp/l2_book.py` (snapshot + optional incremental U/u diffs).
- `aggTrade` via existing `agg_trade_collector` → `microstructure_engine.record_agg_trade`.
- Book ticker / klines remain on the existing hydrator; not duplicated.
- Cross-market: Coinbase public REST via `cross_exchange_reference` (fail-open). Perp/basis is optional; unavailable venues never stop SCALP.

Stale book threshold: 2.0s (`l2_book.STALE_SEC`). Sequence gap / crossed / empty side / reconnect → not authoritative. A stale book is missing evidence, not a strategy hard-block.

## Local book

`LocalL2Book` maintains bids/asks, L1–L20, lastUpdateId, gap/duplicate/rebuild counters. Corrupt or out-of-sequence books cannot stay authoritative.

## Feature vector

Existing 40-dim `scalp_feature_contract` is unchanged. Parallel micro family is documented in `scalp_micro_contract.MICRO_FEATURE_SPEC`:

OFI windows (100ms–30s), queue imbalance L1/L5/L10/L20, weighted depth, microprice + displacement + accel, aggressive flow, trade intensity, inferred cancel/replenish, absorption, fragility, adverse-selection score, optional cross-venue dislocation / basis.

Honesty: cancel/replenish are inferred from snapshot deltas, not a literal exchange event log.

## Labels / model

Training target is future **executable net** (not up/down). Horizons: 1/5/10/30/60s EV plus markouts at 1/2/5/10/20/30/60/120s (mid and executable).

Until markout N ≥ 80, `scalp_micro_ev` uses a documented heuristic tilt blended with the existing path-net EV at 30s/60s. Calibration status is **INCONCLUSIVE** below that N. Brier/ECE helpers live in `calibration_report`.

## Ranking / learning / size

Microstructure, adverse selection, and markout-bucket learning adjust **rank and size only**. They are not eligibility gates. Universe remains BTC/ETH/SOL/XRP every cycle.

## Event loop

- Book updates on the existing 100ms depth stream (uvicorn collector).
- Exit/risk: `tick(rank=False)` on `SCALP_EXIT_INTERVAL_SEC` (default 0.25s), always before ranking.
- Entry rank: `SCALP_RANK_INTERVAL_SEC` (default 1s; `SCALP_LOOP_SEC` still honored as fallback).
- Persistence: in-memory + Redis hashes; markouts batched to `scalp_micro_markouts` on the SCALP DB (timeout skip on lock). Circuit breaker is not fail-opened by markout writes.

## Paper fills

Unchanged walk-book path: buy at executable ask/impact, sell at executable bid/impact, plus fee and slip model. Midpoint is not the fill price.

## Replay

`scalp_micro_replay.replay_events` applies time-sorted events only. Features at T use data ≤ T.

## Safety unchanged

Stale/corrupt data, executable spread/impact, net economics, paper/live flags, max-open, circuit breaker, fail-closed unknown state. `SCALP_LIVE` stays false. DAY strategy/sizing/4H/`same_4h_rise_no_rebuy` are not modified.
