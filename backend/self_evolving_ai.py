import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from backend.config.redis_config import get_shared_redis_sync

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self_evolving_ai")

# All Live Data, No Fallback/Hardcoded Data
BINANCE_US_BASE = os.getenv("BINANCEUS_BASE", "https://api.binance.us")
ALLOWED_SYMBOLS = list(TRADING_SYMBOLS)


@dataclass
class ModelGenome:
    ema_fast: int
    ema_slow: int
    rsi_period: int
    rsi_buy: float
    rsi_sell: float
    tp: float
    sl: float
    hold_max: int
    score: float = 0.0
    generation: int = 0
    parents: list[str] | None = None
    mutations: list[str] | None = None


class SelfEvolvingAI:
    def __init__(self) -> None:
        self.population: dict[str, ModelGenome] = {}
        self.history: list[dict[str, Any]] = []
        self.population_size = int(os.getenv("SEA_POP_SIZE", "14"))
        self.generation = 1
        self.mutation_rate = float(os.getenv("SEA_MUTATION_RATE", "0.2"))
        self.crossover_rate = float(os.getenv("SEA_CROSSOVER_RATE", "0.7"))
        self.interval_sec = int(os.getenv("SEA_INTERVAL_SEC", "1800"))
        self.klines_limit = int(os.getenv("SEA_KLINES_LIMIT", "500"))
        self.redis = get_shared_redis_sync()
        if self.redis is None:
            msg = "Shared Redis client unavailable"
            raise RuntimeError(msg)
        self.client: Any = None  # httpx.AsyncClient
        self.running = False

    async def start(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=10))
        await self._init_population()
        self.running = True
        while self.running:
            try:
                await self._evolve_once()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"evolution error: {e}")
            await asyncio.sleep(self.interval_sec)

    async def stop(self) -> None:
        self.running = False
        if self.client:
            await self.client.aclose()

    async def _evolve_once(self) -> None:
        data = await self._fetch_all_klines()
        await self._evaluate_population(data)
        survivors = self._select_survivors()
        new_gen = await self._spawn_generation(survivors)
        await self._replace_population(new_gen)
        if not self.population:
            return
        best_id, best = max(self.population.items(), key=lambda kv: kv[1].score)
        await self._deploy_best(best_id, best)

    async def _init_population(self) -> None:
        seeds: list[ModelGenome] = []
        seeds.append(
            ModelGenome(
                ema_fast=8,
                ema_slow=21,
                rsi_period=14,
                rsi_buy=35.0,
                rsi_sell=65.0,
                tp=0.015,
                sl=0.008,
                hold_max=240,
            )
        )
        seeds.append(
            ModelGenome(
                ema_fast=12,
                ema_slow=26,
                rsi_period=14,
                rsi_buy=30.0,
                rsi_sell=70.0,
                tp=0.02,
                sl=0.01,
                hold_max=180,
            )
        )
        seeds.append(
            ModelGenome(
                ema_fast=5,
                ema_slow=20,
                rsi_period=10,
                rsi_buy=38.0,
                rsi_sell=62.0,
                tp=0.012,
                sl=0.007,
                hold_max=120,
            )
        )
        seeds.append(
            ModelGenome(
                ema_fast=10,
                ema_slow=30,
                rsi_period=12,
                rsi_buy=32.0,
                rsi_sell=68.0,
                tp=0.018,
                sl=0.009,
                hold_max=300,
            )
        )
        while len(seeds) < self.population_size:
            seeds.append(
                ModelGenome(
                    ema_fast=_rand_int(5, 14),
                    ema_slow=_rand_int(20, 40),
                    rsi_period=_rand_int(8, 20),
                    rsi_buy=_rand_float(28.0, 40.0),
                    rsi_sell=_rand_float(60.0, 75.0),
                    tp=_rand_float(0.01, 0.025),
                    sl=_rand_float(0.006, 0.012),
                    hold_max=_rand_int(100, 360),
                ),
            )
        self.population = {f"m_{i}": seeds[i] for i in range(len(seeds))}

    async def _fetch_all_klines(self) -> dict[str, list[float]]:
        assert self.client is not None

        async def fetch(symbol: str) -> tuple[str, list[float]]:
            url = f"{BINANCE_US_BASE}/api/v3/klines"
            params = {"symbol": symbol, "interval": "1m", "limit": self.klines_limit}
            try:
                r = await self.client.get(url, params=params)
                if r.status_code != 200:
                    return symbol, []
                raw = r.json()
                closes = [float(k[4]) for k in raw]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                return symbol, []
            else:
                return symbol, closes

        tasks = [fetch(sym) for sym in ALLOWED_SYMBOLS]
        results = await asyncio.gather(*tasks)
        return {s: c for s, c in results if c}

    async def _evaluate_population(self, data: dict[str, list[float]]) -> None:
        for _mid, g in self.population.items():
            g.score = await self._score_genome(g, data)

    async def _score_genome(self, g: ModelGenome, data: dict[str, list[float]]) -> float:
        if not data:
            return 0.0
        perfs: list[float] = []
        for _sym, closes in data.items():
            perf = _backtest(
                closes,
                g.ema_fast,
                g.ema_slow,
                g.rsi_period,
                g.rsi_buy,
                g.rsi_sell,
                g.tp,
                g.sl,
                g.hold_max,
            )
            perfs.append(perf)
        if not perfs:
            return 0.0
        avg = sum(perfs) / len(perfs)
        vol = _stdev(perfs)
        if vol == 0:
            return max(0.0, avg)
        sharpe_like = avg / vol
        return float(
            max(
                0.0,
                min(1.0, 0.5 * _sigmoid(3.0 * sharpe_like) + 0.5 * _sigmoid(50.0 * avg)),
            )
        )

    def _select_survivors(self) -> list[tuple[str, ModelGenome]]:
        ranked = sorted(self.population.items(), key=lambda kv: kv[1].score, reverse=True)
        keep = max(2, self.population_size // 2)
        survivors = ranked[:keep]
        tail = ranked[keep:]
        if tail:
            k = min(2, len(tail))
            survivors.extend(_random_sample(tail, k))
        return survivors

    async def _spawn_generation(self, survivors: list[tuple[str, ModelGenome]]) -> dict[str, ModelGenome]:
        new_pop: dict[str, ModelGenome] = {}
        elite = min(2, len(survivors))
        for i in range(elite):
            _mid, g = survivors[i]
            clone = _clone(g)
            clone.generation = self.generation
            new_pop[f"gen{self.generation}_elite_{i}"] = clone
        needed = self.population_size - len(new_pop)
        for i in range(needed):
            if len(survivors) >= 2 and _rand_float(0.0, 1.0) < self.crossover_rate:
                p1 = _random_choice(survivors)[1]
                p2 = _random_choice(survivors)[1]
                child = _crossover(p1, p2)
            else:
                p = _random_choice(survivors)[1]
                child = _mutate(p, self.mutation_rate)
            child.generation = self.generation
            new_pop[f"gen{self.generation}_offspring_{i}"] = child
        return new_pop

    async def _replace_population(self, new_gen: dict[str, ModelGenome]) -> None:
        self.population = new_gen
        scores = [g.score for g in new_gen.values()]
        avg = sum(scores) / len(scores) if scores else 0.0
        mx = max(scores) if scores else 0.0
        self.history.append(
            {
                "generation": self.generation,
                "population_size": len(new_gen),
                "avg_score": round(avg, 6),
                "max_score": round(mx, 6),
                "timestamp": _now_iso(),
            },
        )
        self.generation += 1

    async def _deploy_best(self, model_id: str, model: ModelGenome) -> None:
        payload = {
            "model_id": model_id,
            "generation": model.generation,
            "score": round(model.score, 6),
            "params": {
                "ema_fast": model.ema_fast,
                "ema_slow": model.ema_slow,
                "rsi_period": model.rsi_period,
                "rsi_buy": model.rsi_buy,
                "rsi_sell": model.rsi_sell,
                "tp": model.tp,
                "sl": model.sl,
                "hold_max": model.hold_max,
            },
            "timestamp": _now_iso(),
        }
        try:
            if self.redis:

                def _sync_redis_publish():
                    self.redis.set("evolving_ai:best_model", json.dumps(payload), ex=3600)
                    self.redis.publish("evolving_ai_updates", json.dumps(payload))

                await asyncio.to_thread(_sync_redis_publish)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"redis error: {e}")
        try:
            out = Path("models/evolved_signal_params.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"file write error: {e}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ema(values: list[float], period: int) -> list[float]:
    if not values or period <= 1:
        return values[:]
    k = 2.0 / (period + 1.0)
    out: list[float] = [values[0]]
    ema = values[0]
    for v in values[1:]:
        ema = (v - ema) * k + ema
        out.append(ema)
    return out


def _rsi(values: list[float], period: int) -> list[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    out: list[float] = []
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    out.extend([50.0] * period)
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rs = math.inf
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        out.append(float(max(0.0, min(100.0, rsi))))
    while len(out) < len(values):
        out.insert(0, 50.0)
    return out


def _backtest(
    closes: list[float],
    ema_fast: int,
    ema_slow: int,
    rsi_period: int,
    rsi_buy: float,
    rsi_sell: float,
    tp: float,
    sl: float,
    hold_max: int,
) -> float:
    if len(closes) < max(ema_slow + 2, rsi_period + 2, 50):
        return 0.0
    f = _ema(closes, ema_fast)
    s = _ema(closes, ema_slow)
    rsi = _rsi(closes, rsi_period)
    pos = 0
    entry = 0.0
    bars = 0
    pnl: list[float] = []
    fee = 0.0005
    for i in range(1, len(closes)):
        price = closes[i]
        prev_cross = f[i - 1] - s[i - 1]
        cross = f[i] - s[i]
        buy_sig = (prev_cross <= 0 and cross > 0) or (rsi[i] <= rsi_buy)
        sell_sig = (prev_cross >= 0 and cross < 0) or (rsi[i] >= rsi_sell)
        if pos == 0 and buy_sig:
            pos = 1
            entry = price * (1 + fee)
            bars = 0
            continue
        if pos == 1:
            bars += 1
            ret = (price * (1 - fee) - entry) / entry
            stop_hit = ret <= -sl
            take_hit = ret >= tp
            time_hit = bars >= hold_max
            if sell_sig or stop_hit or take_hit or time_hit:
                pnl.append(ret)
                pos = 0
                entry = 0.0
                bars = 0
    if pos == 1 and entry > 0:
        last_ret = (closes[-1] * (1 - fee) - entry) / entry
        pnl.append(last_ret)
    if not pnl:
        return 0.0
    avg = sum(pnl) / len(pnl)
    vol = _stdev(pnl)
    dd = _max_drawdown_from_returns(pnl)
    score = 0.0
    score += 0.6 * _sigmoid(60.0 * avg)
    score += 0.3 * (0.0 if vol == 0 else _sigmoid(5.0 * (avg / vol)))
    score += 0.1 * (1.0 - min(1.0, dd))
    return float(max(0.0, min(1.0, score)))


def _stdev(x: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    var = sum((v - m) * (v - m) for v in x) / (n - 1)
    return math.sqrt(max(0.0, var))


def _max_drawdown_from_returns(rets: list[float]) -> float:
    equity = 1.0
    peaks = 1.0
    max_dd = 0.0
    for r in rets:
        equity *= 1.0 + r
        peaks = max(peaks, equity)
        dd = (peaks - equity) / peaks if peaks > 0 else 0.0
        max_dd = max(max_dd, dd)
    return float(max_dd)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _clone(g: ModelGenome) -> ModelGenome:
    return ModelGenome(
        ema_fast=g.ema_fast,
        ema_slow=g.ema_slow,
        rsi_period=g.rsi_period,
        rsi_buy=g.rsi_buy,
        rsi_sell=g.rsi_sell,
        tp=g.tp,
        sl=g.sl,
        hold_max=g.hold_max,
        score=g.score,
        generation=g.generation,
        parents=(g.parents[:] if g.parents else []),
        mutations=(g.mutations[:] if g.mutations else []),
    )


def _mutate(g: ModelGenome, rate: float) -> ModelGenome:
    c = _clone(g)
    muts: list[str] = []
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_int(-2, 2)
        c.ema_fast = int(max(3, min(c.ema_slow - 1, c.ema_fast + d)))
        muts.append("ema_fast")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_int(-4, 4)
        c.ema_slow = int(max(c.ema_fast + 1, min(60, c.ema_slow + d)))
        muts.append("ema_slow")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_int(-3, 3)
        c.rsi_period = int(max(6, min(30, c.rsi_period + d)))
        muts.append("rsi_period")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_float(-3.0, 3.0)
        c.rsi_buy = float(max(20.0, min(45.0, c.rsi_buy + d)))
        muts.append("rsi_buy")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_float(-3.0, 3.0)
        c.rsi_sell = float(max(55.0, min(85.0, c.rsi_sell + d)))
        muts.append("rsi_sell")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_float(-0.003, 0.003)
        c.tp = float(max(0.006, min(0.035, c.tp + d)))
        muts.append("tp")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_float(-0.003, 0.003)
        c.sl = float(max(0.003, min(0.02, c.sl + d)))
        muts.append("sl")
    if _rand_float(0.0, 1.0) < rate:
        d = _rand_int(-40, 40)
        c.hold_max = int(max(60, min(600, c.hold_max + d)))
        muts.append("hold_max")
    c.mutations = (c.mutations or []) + muts
    return c


def _crossover(a: ModelGenome, b: ModelGenome) -> ModelGenome:
    def pick(x, y):
        return x if _rand_float(0.0, 1.0) < 0.5 else y

    child = ModelGenome(
        ema_fast=pick(a.ema_fast, b.ema_fast),
        ema_slow=max(
            pick(a.ema_slow, b.ema_slow),
            1 + min(pick(a.ema_fast, b.ema_fast), pick(b.ema_fast, a.ema_fast)),
        ),
        rsi_period=pick(a.rsi_period, b.rsi_period),
        rsi_buy=pick(a.rsi_buy, b.rsi_buy),
        rsi_sell=pick(a.rsi_sell, b.rsi_sell),
        tp=pick(a.tp, b.tp),
        sl=pick(a.sl, b.sl),
        hold_max=pick(a.hold_max, b.hold_max),
        score=0.0,
        generation=0,
        parents=[hex(id(a)), hex(id(b))],
        mutations=["crossover"],
    )
    return _mutate(child, 0.3)


# Random state - using dict to avoid global keyword
_rand_state_dict: dict[str, int] = {"state": 1234567}


def _rand_float(lo: float, hi: float) -> float:
    _rand_state_dict["state"] = (1103515245 * _rand_state_dict["state"] + 12345) & 0x7FFFFFFF
    u = _rand_state_dict["state"] / 0x7FFFFFFF
    return lo + (hi - lo) * u


def _rand_int(lo: int, hi: int) -> int:
    return round(_rand_float(lo, hi))


def _random_choice(seq: list[Any]) -> Any | None:
    if not seq:
        return None
    idx = _rand_int(0, len(seq) - 1)
    return seq[idx]


def _random_sample(seq: list[Any], k: int) -> list[Any]:
    if k >= len(seq):
        return seq[:]
    chosen: set[int] = set()
    out: list[Any] = []
    while len(out) < k:
        i = _rand_int(0, len(seq) - 1)
        if i not in chosen:
            chosen.add(i)
            out.append(seq[i])
    return out


async def start_self_evolution() -> None:
    ai = SelfEvolvingAI()
    try:
        await ai.start()
    except KeyboardInterrupt:
        pass
    finally:
        await ai.stop()


if __name__ == "__main__":
    asyncio.run(start_self_evolution())
