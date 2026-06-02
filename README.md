# Mystic

Paper-trading crypto stack: AI signals, portfolio engine, learning loop, and dashboard on port 8000.

**Canonical local path:** `/home/mystic/mystic` (matches systemd units and `start_mystic.sh`).

## Quick start (fresh clone)

```bash
git clone https://github.com/lloydwyl97/Mystic.git /home/mystic/mystic
cd /home/mystic/mystic
bash scripts/setup_local.sh
# Edit .env — at minimum BINANCEUS_API_KEY / BINANCEUS_API_SECRET
./start_mystic.sh core
```

Dashboard: http://127.0.0.1:8000/dashboard/

## Prerequisites

- Ubuntu 22.04+ (tested on local VM `mystic-Virtual-Machine`)
- Python 3.11+
- Redis on `127.0.0.1:6379`
- Binance US API keys (read-only market data is sufficient for paper mode)

## Stack (core mode)

| Process | Role |
|---------|------|
| uvicorn :8000 | API + dashboard |
| `start_live_market_data.py` | OHLCV / ticker → SQLite + Redis |
| `start_ai_market_context.py` | Sentiment / context hashes |
| `start_ai_signal_generator.py` | `ai_signal:day:*` Redis signals |
| `start_portfolio_engine_integration.py` | Paper buys/sells |
| `start_ai_learning.py` | Retrain + promotion loop |

Stop: `./stop_mystic.sh` or `systemctl --user stop mystic.target`

## systemd (24/7 user units)

Units live in `deploy/systemd/user/`. `scripts/setup_local.sh` copies them to `~/.config/systemd/user/` and enables `mystic.target`.

```bash
systemctl --user start mystic.target
systemctl --user status mystic-uvicorn.service
```

## Config

| File | Purpose |
|------|---------|
| `.env` | Secrets and runtime flags (not in git) |
| `.env.example` | Template |
| `deploy/core_only_local.env` | Local paper profile layered on `.env` |

Paper mode defaults: `PAPER_TRADING=true`, `LIVE_TRADES_ALLOWED=false`, top-4 DAY symbols (BTC/ETH/SOL/XRP).

## Database and models

- SQLite DB is **local state** — not committed. Fresh install creates schema on first run or restore from backup.
- `models/active/` — promoted models (small, tracked in git when present).
- `models/versions/` and `models/training_data/` — large caches, gitignored; rebuilt by learning loop.

## Operations

See `LOCAL_OPERATIONS.md` for day-to-day commands, Redis key types, and design constraints.

## Production (Ocean)

Production droplet uses repo root `/home/mystic` on `mystic-prod`. Deploy by pull + `systemctl restart mystic` — see project ops docs.
