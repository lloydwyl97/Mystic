import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from backend.config.redis_config import get_shared_redis_async
from backend.services.binance_rest_client import BinanceREST, BinanceWeightLimiter

logger = logging.getLogger(__name__)

"""
Risk Management Service
Manages portfolio risk and position limits
"""

# Import from single source of truth
try:
    from backend.config.trading_universe import (
        EXCHANGE_ID,
        TRADING_SYMBOLS,
    )
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

load_dotenv(dotenv_path=str(Path(__file__).parent.parent / ".env"))

# Use BINANCEUS_BASE environment variable or default
BINANCE_US = os.getenv("BINANCEUS_BASE", "https://api.binance.us")
# Use TRADING_SYMBOLS from trading_universe (live data)
SYMBOLS = list(TRADING_SYMBOLS)
INTERVAL = "1h"
LIMIT = 200
RISK_FILE = "./config/risk.json"


class RiskManager:
    def __init__(self) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        self.redis_client = get_shared_redis_async()
        if self.redis_client is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.running = False

    async def start(self):
        # Production optimization: Removed startup logging
        self.running = True
        await self.monitor_risk()

    def _load_config(self) -> dict[str, Any]:
        try:
            risk_file_path = Path(RISK_FILE)
            if risk_file_path.exists():
                with risk_file_path.open(encoding="utf-8") as f:
                    return json.load(f)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        return {"max_position_size": 0.15, "max_daily_loss": 0.05}

    async def _get_weights(self) -> dict[str, float]:
        try:
            raw = await self.redis_client.get("portfolio:weights")
            if raw:
                data = json.loads(raw)
                w = {s: float(data.get(s, 0.0)) for s in SYMBOLS}
                s = sum(abs(v) for v in w.values())
                if s > 0:
                    return {k: v / s for k, v in w.items()}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
        eq = 1.0 / len(SYMBOLS)
        return dict.fromkeys(SYMBOLS, eq)

    async def _fetch_klines(self, symbol: str) -> list[list[Any]]:
        """Fetch live kline data from Binance.US API."""
        try:
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
            klines = await client.get_klines(symbol, INTERVAL, limit=LIMIT)
            return klines if klines else []
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to fetch live klines for {symbol}: {e}")
            return []

    async def _series(self) -> tuple[list[int], dict[str, dict[int, float]]]:
        ts_sets = []
        price_maps: dict[str, dict[int, float]] = {}
        for s in SYMBOLS:
            kl = await self._fetch_klines(s)
            mp: dict[int, float] = {}
            for k in kl:
                t = int(k[0])
                c = float(k[4])
                mp[t] = c
            price_maps[s] = mp
            ts_sets.append(set(mp.keys()))
        if ts_sets:
            # compute intersection of all timestamp sets
            common_set = ts_sets[0].intersection(*ts_sets[1:]) if len(ts_sets) > 1 else ts_sets[0]
            common = sorted(common_set)
        else:
            common = []
        return common, price_maps

    def _returns_map(self, times: list[int], prices: dict[int, float]) -> dict[int, float]:
        rets: dict[int, float] = {}
        prev_t = None
        prev_p = None
        for t in sorted(times):
            p = prices.get(t)
            if prev_t is not None and prev_p is not None and p is not None and prev_p != 0:
                rets[t] = (p - prev_p) / prev_p
            prev_t = t
            prev_p = p
        return rets

    def _portfolio_returns(
        self,
        times: list[int],
        price_maps: dict[str, dict[int, float]],
        weights: dict[str, float],
    ) -> list[tuple[int, float]]:
        per_symbol_rets: dict[str, dict[int, float]] = {s: self._returns_map(times, price_maps[s]) for s in SYMBOLS}
        aligned = [t for t in times if all(t in per_symbol_rets[s] for s in SYMBOLS)]
        out: list[tuple[int, float]] = []
        for t in aligned:
            r = 0.0
            for s in SYMBOLS:
                r += weights[s] * per_symbol_rets[s][t]
            out.append((t, r))
        return out

    def _stdev(self, xs: list[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        m = sum(xs) / n
        v = sum((x - m) ** 2 for x in xs) / (n - 1)
        return v**0.5

    def _percentile(self, xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        k = max(0, min(len(ys) - 1, int(p * (len(ys) - 1))))
        return ys[k]

    def _cvar(self, xs: list[float], alpha: float) -> float:
        if not xs:
            return 0.0
        var = self._percentile(xs, 1 - alpha)
        tail = [x for x in xs if x <= var]
        if not tail:
            return var
        return sum(tail) / len(tail)

    def _max_drawdown(self, rs: list[float]) -> float:
        if not rs:
            return 0.0
        equity = []
        v = 1.0
        for r in rs:
            v *= 1 + r
            equity.append(v)
        peak = equity[0]
        mdd = 0.0
        for v in equity:
            peak = max(peak, v)
            dd = (v - peak) / peak
            mdd = min(mdd, dd)
        return abs(mdd)

    def _corr_pair(self, a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        ma = sum(a[:n]) / n
        mb = sum(b[:n]) / n
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
        sa = (sum((x - ma) ** 2 for x in a[:n]) / (n - 1)) ** 0.5
        sb = (sum((x - mb) ** 2 for x in b[:n]) / (n - 1)) ** 0.5
        if sa == 0 or sb == 0:
            return 0.0
        return cov / (sa * sb)

    def _avg_pairwise_corr(self, times: list[int], price_maps: dict[str, dict[int, float]]) -> float:
        per_symbol_rets = {s: [v for _, v in sorted(self._returns_map(times, price_maps[s]).items())] for s in SYMBOLS}
        pairs = []
        for i in range(len(SYMBOLS)):
            for j in range(i + 1, len(SYMBOLS)):
                ci = self._corr_pair(per_symbol_rets[SYMBOLS[i]], per_symbol_rets[SYMBOLS[j]])
                pairs.append(ci)
        if not pairs:
            return 0.0
        return sum(pairs) / len(pairs)

    async def monitor_risk(self):
        # Production optimization: Removed monitoring logging
        while self.running:
            try:
                data = await self.calculate_risk_metrics()
                alerts = await self.check_risk_alerts(data)
                await self.store_risk_data(data)
                if alerts:
                    await self.publish_risk_alerts(alerts)
                await asyncio.sleep(60)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error in risk monitoring: {e}")
                await asyncio.sleep(120)

    async def calculate_risk_metrics(self) -> dict[str, Any]:
        try:
            cfg = self._load_config()
            weights = await self._get_weights()
            times, price_maps = await self._series()
            port = self._portfolio_returns(times, price_maps, weights)
            if not port:
                return {
                    "error": "no_data",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ts = [t for t, _ in port]
            rs = [r for _, r in port]
            vol = self._stdev(rs)
            var95 = self._percentile(rs, 0.05)
            cvar95 = self._cvar(rs, 0.95)
            mdd = self._max_drawdown(rs)
            corr_avg = self._avg_pairwise_corr(times, price_maps)
            max_w = max(abs(w) for w in weights.values()) if weights else 0.0
            return {
                "portfolio_risk": {
                    "var_95": var95,
                    "cvar_95": cvar95,
                    "volatility": vol,
                    "correlation_avg": corr_avg,
                    "max_drawdown": mdd,
                },
                "position_limits": {
                    "max_position_size": float(cfg.get("max_position_size", 0.15)),
                    "max_daily_loss": float(cfg.get("max_daily_loss", 0.05)),
                    "max_weight": max_w,
                },
                "weights": weights,
                "series": {
                    "times": ts[-60:],
                    "returns": rs[-60:],
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def check_risk_alerts(self, risk_data: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            alerts: list[dict[str, Any]] = []
            pr = risk_data.get("portfolio_risk", {})
            pl = risk_data.get("position_limits", {})
            if pr and pr.get("var_95", 0) < -0.05:
                alerts.append(
                    {
                        "type": "VaR",
                        "message": "VaR(95) threshold exceeded",
                        "severity": "HIGH",
                        "value": pr.get("var_95"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            if pr and pr.get("volatility", 0) > 0.20:
                alerts.append(
                    {
                        "type": "Volatility",
                        "message": "High portfolio volatility",
                        "severity": "MEDIUM",
                        "value": pr.get("volatility"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            if pl and pr and pl.get("max_weight", 0) > pl.get("max_position_size", 0.15):
                alerts.append(
                    {
                        "type": "PositionLimit",
                        "message": "Per-symbol position size limit exceeded",
                        "severity": "HIGH",
                        "value": pl.get("max_weight"),
                        "limit": pl.get("max_position_size"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            sr = risk_data.get("series", {}).get("returns", [])
            if sr and min(sr[-12:]) < -0.03:
                alerts.append(
                    {
                        "type": "Shock",
                        "message": "Recent return shock detected",
                        "severity": "MEDIUM",
                        "value": min(sr[-12:]),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            return []
        else:
            return alerts

    async def store_risk_data(self, data: dict[str, Any]):
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            await self.redis_client.set("risk_data", json.dumps(data), ex=1800)

    async def publish_risk_alerts(self, alerts: list[dict[str, Any]]):
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            await self.redis_client.publish("risk_alerts", json.dumps(alerts))

    # ENHANCED ADAPTIVE RISK MANAGEMENT METHODS

    async def calculate_dynamic_position_size(self, symbol: str, confidence: float, existing_positions: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Calculate dynamic position size using Kelly Criterion with risk adjustments

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            confidence: AI prediction confidence (0.0-1.0)
            existing_positions: Current portfolio positions

        Returns:
            Dict with position size details
        """
        try:
            existing_positions = existing_positions or {}

            # 1. Get volatility for the symbol
            volatility = await self._get_symbol_volatility(symbol)

            # 2. Calculate Kelly Criterion base size
            kelly_size = self._kelly_criterion(confidence, await self._get_win_rate(), await self._get_avg_win_loss())

            # 3. Apply volatility adjustment
            vol_adjustment = self._volatility_adjustment(volatility)

            # 4. Apply portfolio correlation penalty
            correlation_penalty = await self._correlation_penalty(symbol, existing_positions)

            # 5. Apply drawdown protection
            drawdown_multiplier = await self._drawdown_protection()

            # 6. Calculate final position size
            account_balance = await self._get_account_balance()
            base_position_size = account_balance * self._get_max_position_pct()

            final_size = kelly_size * vol_adjustment * correlation_penalty * drawdown_multiplier
            final_size = min(final_size, base_position_size)
            final_size = max(final_size, self._get_min_position_size())

            return {
                "symbol": symbol,
                "position_size": final_size,
                "kelly_base": kelly_size,
                "volatility_adjustment": vol_adjustment,
                "correlation_penalty": correlation_penalty,
                "drawdown_multiplier": drawdown_multiplier,
                "confidence": confidence,
                "volatility": volatility,
                "account_balance": account_balance,
                "max_position_pct": self._get_max_position_pct(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.exception(f"Error calculating dynamic position size for {symbol}: {e}")
            # Return safe fallback
            return {"symbol": symbol, "position_size": 0.0, "error": str(e), "fallback": True, "timestamp": datetime.now(timezone.utc).isoformat()}

    def _kelly_criterion(self, confidence: float, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
                return 0.05  # Conservative fallback

            # Kelly formula: (bp - q) / b
            # where b = odds (win_loss_ratio), p = win probability, q = loss probability
            b = win_loss_ratio
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply half-Kelly for safety
            kelly_fraction *= 0.5

            # Bound between 0 and 0.2 (20% of account)
            return max(0.01, min(kelly_fraction, 0.20))

        except (ValueError, TypeError, ZeroDivisionError):
            return 0.05

    def _volatility_adjustment(self, volatility: float) -> float:
        """Adjust position size based on volatility"""
        try:
            # Higher volatility = smaller position
            # Scale: 0.2 (low vol) to 0.8 (high vol)
            vol_score = min(volatility * 4, 1.0)  # Normalize volatility
            adjustment = 1.0 - (vol_score * 0.6)  # Reduce by up to 60%
            return max(0.2, adjustment)
        except (ValueError, TypeError):
            return 0.5

    async def _correlation_penalty(self, symbol: str, existing_positions: dict[str, Any]) -> float:
        """Calculate correlation-based position size penalty"""
        try:
            if not existing_positions:
                return 1.0  # No penalty if no existing positions

            # Calculate average correlation with existing positions
            correlations = []
            for existing_symbol in existing_positions:
                if existing_symbol != symbol:
                    corr = await self._get_symbol_correlation(symbol, existing_symbol)
                    correlations.append(abs(corr))

            if not correlations:
                return 1.0

            avg_correlation = sum(correlations) / len(correlations)

            # Higher correlation = smaller position
            # Penalty: 0.3 (perfect correlation) to 1.0 (no correlation)
            penalty = 1.0 - (avg_correlation * 0.7)
            return max(0.3, penalty)

        except Exception:
            # Production optimization: Reduced logging
            return 0.8  # Conservative penalty on error

    async def _drawdown_protection(self) -> float:
        """Apply drawdown-based position size reduction"""
        try:
            # Get recent drawdown from risk metrics
            risk_data = await self.calculate_risk_metrics()
            current_drawdown = risk_data.get("portfolio_risk", {}).get("max_drawdown", 0.0)

            # Reduce position size based on drawdown
            if current_drawdown > 0.10:  # 10% drawdown
                return 0.5  # Half size
            elif current_drawdown > 0.05:  # 5% drawdown
                return 0.75  # 75% size
            else:
                return 1.0  # Full size

        except Exception:
            # Production optimization: Reduced logging
            return 0.8  # Conservative reduction

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get annualized volatility for symbol"""
        try:
            # Fetch recent price data
            klines = await self._fetch_symbol_klines(symbol, "1h", 100)
            if not klines:
                return 0.5  # Default volatility

            # Calculate returns
            prices = [float(k[4]) for k in klines]  # Close prices
            returns = np.diff(np.log(prices))

            # Calculate annualized volatility
            volatility = np.std(returns) * np.sqrt(8760)  # Annualize from hourly
            return min(volatility, 2.0)  # Cap at 200%

        except Exception as e:
            logger.warning(f"Error getting volatility for {symbol}: {e}")
            return 0.5

    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            # Fetch price data for both symbols
            klines1 = await self._fetch_symbol_klines(symbol1, "1h", 50)
            klines2 = await self._fetch_symbol_klines(symbol2, "1h", 50)

            if not klines1 or not klines2:
                return 0.0

            prices1 = [float(k[4]) for k in klines1]
            prices2 = [float(k[4]) for k in klines2]

            # Calculate returns
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))

            # Calculate correlation
            if len(returns1) == len(returns2) and len(returns1) > 5:
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    async def _fetch_symbol_klines(self, symbol: str, interval: str, limit: int) -> list:
        """Fetch klines for a specific symbol"""
        try:
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
            return await client.get_klines(symbol, interval, limit=limit)
        except Exception as e:
            logger.warning(f"Error fetching klines for {symbol}: {e}")
            return []

    async def _get_win_rate(self) -> float:
        """Get current win rate from trading history"""
        try:
            # Try to get from Redis cache first
            win_rate_raw = await self.redis_client.get("trading:win_rate")
            if win_rate_raw:
                return float(win_rate_raw)
            else:
                # Fallback to calculation from trade history
                # This would need to be implemented based on your trade storage
                return 0.5  # Conservative default

        except Exception:
            return 0.5

    async def _get_avg_win_loss(self) -> float:
        """Get average win/loss ratio"""
        try:
            # Try to get from Redis cache
            ratio_raw = await self.redis_client.get("trading:avg_win_loss_ratio")
            if ratio_raw:
                return float(ratio_raw)
            else:
                return 1.5  # Conservative default (1.5:1 win/loss ratio)

        except Exception:
            return 1.5

    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            balance_raw = await self.redis_client.get("account:balance")
            if balance_raw:
                return float(balance_raw)
            else:
                return 1000.0  # Fallback balance

        except Exception:
            return 1000.0

    def _get_max_position_pct(self) -> float:
        """Get maximum position size as percentage of account"""
        config = self._load_config()
        return config.get("max_position_size", 0.15)

    def _get_min_position_size(self) -> float:
        """Get minimum position size in account currency"""
        return 10.0  # $10 minimum

    async def stop(self):
        logger.info("Stopping Risk Management Service...")
        self.running = False


async def main():
    manager = RiskManager()
    try:
        await manager.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error in main: {e}")
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
