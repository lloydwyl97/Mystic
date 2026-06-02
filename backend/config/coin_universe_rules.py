"""
COIN UNIVERSE RULES — MANDATORY (NO AUTO-EXPANSION)

Scope: All trading / AI / portfolio / risk modules.
Goal: Optimize the current 10-coin universe. Do not add coins automatically.

1. Fixed Coin Universe
   - Active trading universe is exactly the configured symbols (e.g. 10).
   - Never auto-add or auto-discover new symbols.
   - Any change to the coin list must be explicit configuration, not code logic.

2. Per-Coin Metrics Required
   - For each active coin: win_rate, profit_factor, avg_pnl, trades_last_30d,
     confidence_to_pnl_correlation.
   - If trades_last_30d < MIN_TRADES_FOR_RISK_SCALE → do not scale risk up.

3. Dynamic Risk Scaling (Per Coin Only)
   - Risk multipliers are per coin, not global.
   - win_rate > 60% → small risk increase (bounded by global caps).
   - win_rate < 40% → small risk decrease.
   - Never exceed global position/notional limits.

4. Automatic Temporary Disable (Safety Net)
   - Temporarily disable coin if: profit_factor < 1.0 AND trades >= 20 AND lookback >= 30 days.
    - Disable duration: DISABLE_DAYS_MIN-DISABLE_DAYS_MAX, then re-evaluate.
   - Do not permanently delete coins automatically.

5. Metrics Table Integrity
   - Daily metrics persistence must contain exactly the active coins.
   - Fewer than expected → pipeline bug. More than expected → rogue symbol leak.
   - Fail verification if mismatch is detected.

6. Capital Distribution Guard
   - No single coin may exceed MAX_COIN_CONCENTRATION_PCT of total capital unless configured.
   - Log warnings when concentration thresholds are approached or breached.

7. Expansion Preconditions (Do Not Bypass)
    - Only consider adding coins if all true: 30-60 days stable profitability,
     overall profit_factor > 1.3, max drawdown < 25%, metrics persistence verified,
     no invariant breaches, positive confidence ↔ PnL correlation.
    - When expanding: add 1-2 coins at a time only. Expansion is config-only, never in code.

8. Enforcement
   - Treat the coin list as a fixed universe unless configuration changes.
   - Validate metrics existence per coin before any risk increase.
   - Apply risk scaling per coin only. Auto-disable underperformers temporarily.
   - Log and surface integrity or concentration violations.
   - Never auto-expand the universe in code. Quality over quantity.
"""

from typing import Final

# Fixed universe — no auto-expansion
COIN_UNIVERSE_FIXED: Final[bool] = True

# Per-coin: do not scale risk up below this many trades (30d)
MIN_TRADES_FOR_RISK_SCALE: Final[int] = 20

# Temporary disable: min/max days (re-evaluate after)
DISABLE_DAYS_MIN: Final[int] = 7
DISABLE_DAYS_MAX: Final[int] = 14

# Underperformer disable: need at least this many trades and 30d lookback
MIN_TRADES_FOR_DISABLE: Final[int] = 20
MIN_LOOKBACK_DAYS_FOR_DISABLE: Final[int] = 30
PROFIT_FACTOR_DISABLE_THRESHOLD: Final[float] = 1.0

# Capital concentration: max share of total capital per coin (0-1 or use pct)
MAX_COIN_CONCENTRATION_PCT: Final[float] = 30.0
CONCENTRATION_WARN_PCT: Final[float] = 25.0

# Risk scaling bounds (per coin)
PER_COIN_RISK_SCALE_MIN: Final[float] = 0.5
PER_COIN_RISK_SCALE_MAX: Final[float] = 1.2
WIN_RATE_INCREASE_THRESHOLD: Final[float] = 0.60
WIN_RATE_DECREASE_THRESHOLD: Final[float] = 0.40

# Expansion preconditions (documentation / validation only; no code expansion)
EXPANSION_MIN_PROFIT_FACTOR: Final[float] = 1.3
EXPANSION_MAX_DRAWDOWN_PCT: Final[float] = 25.0
EXPANSION_STABLE_DAYS_MIN: Final[int] = 30
EXPANSION_STABLE_DAYS_MAX: Final[int] = 60
EXPANSION_ADD_AT_A_TIME: Final[int] = 2
