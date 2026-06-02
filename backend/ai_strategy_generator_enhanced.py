import ast
import hashlib
import importlib.util
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from openai import OpenAI

import redis
from backend.config.redis_config import get_shared_redis_sync

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID, TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.services.binance_rest_client import BinanceREST
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = tuple(TRADING_SYMBOLS)

STRATEGY_DIR = "./generated_modules"
logger = logging.getLogger(__name__)

STRATEGY_TEMPLATES: dict[str, str] = {
    "rsi_strategy": (
        "def rsi_strategy(df):\n"
        "    import pandas as pd\n"
        "    delta = df['close'].diff()\n"
        "    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()\n"
        "    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()\n"
        "    rs = gain / loss\n"
        "    df['rsi'] = 100 - (100 / (1 + rs))\n"
        "    df['signal'] = 0\n"
        "    df.loc[df['rsi'] < 30, 'signal'] = 1\n"
        "    df.loc[df['rsi'] > 70, 'signal'] = -1\n"
        "    return df\n"
    ),
    "macd_strategy": (
        "def macd_strategy(df):\n"
        "    import pandas as pd\n"
        "    ema_fast = df['close'].ewm(span=12).mean()\n"
        "    ema_slow = df['close'].ewm(span=26).mean()\n"
        "    df['macd'] = ema_fast - ema_slow\n"
        "    df['macd_signal'] = df['macd'].ewm(span=9).mean()\n"
        "    df['signal'] = 0\n"
        "    df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1\n"
        "    df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1\n"
        "    return df\n"
    ),
    "bollinger_strategy": (
        "def bollinger_strategy(df):\n"
        "    import pandas as pd\n"
        "    sma = df['close'].rolling(window=20).mean()\n"
        "    std = df['close'].rolling(window=20).std()\n"
        "    df['bb_upper'] = sma + (std * 2)\n"
        "    df['bb_lower'] = sma - (std * 2)\n"
        "    df['signal'] = 0\n"
        "    df.loc[df['close'] < df['bb_lower'], 'signal'] = 1\n"
        "    df.loc[df['close'] > df['bb_upper'], 'signal'] = -1\n"
        "    return df\n"
    ),
}


def _ensure_dirs() -> None:
    try:
        Path(STRATEGY_DIR).mkdir(parents=True, exist_ok=True)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"{EXCHANGE_ID} ensure dir failed: {e}")


def _redis_client() -> redis.Redis | None:
    try:
        client = get_shared_redis_sync()
        if client is None:
            logger.warning(f"{EXCHANGE_ID} shared Redis unavailable")
            return None
        else:
            return client
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.info(f"{EXCHANGE_ID} redis unavailable: {e}")
        return None


def get_live_market_data() -> dict[str, dict[str, float]]:
    try:
        r = _redis_client()
        if not r:
            return {}
        out: dict[str, dict[str, float]] = {}
        for s in ALLOWED_SYMBOLS:
            raw = r.hget(f"price:{s}", "v")
            if raw is None:
                continue
            try:
                value = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                price = float(value)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                continue
            out[s] = {"price": price, "change": 0.0, "volume": 0.0}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"{EXCHANGE_ID} market data fetch error: {e}")
        return {}
    else:
        return out


def validate_strategy_code(code: str) -> bool:
    try:
        ast.parse(code)
        result = False if "import pandas" not in code or "def " not in code else "signal" in code
    except SyntaxError:
        return False
    else:
        return result


def _extract_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                result = node.name
                break
        else:
            result = None
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None
    else:
        return result


def _strip_code_fences(code: str) -> str:
    s = code.strip()
    # If code is fenced with ``` optionally followed by a language, extract the inner content
    if s.startswith("```"):
        first = s.find("```")
        last = s.rfind("```")
        if first != -1 and last != -1 and last > first:
            inner = s[first + 3 : last].lstrip()
            # If a language tag is present on the first line, strip it
            if inner.startswith(("python", "py")):
                # remove the first line (the language tag)
                parts = inner.split("\n", 1)
                if len(parts) == 2:
                    return parts[1].strip()
                return ""
            return inner.strip()
    return s


def generate_strategy_hash(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()[:8]


def save_strategy_version(code: str, filename: str, metadata: dict[str, Any]) -> None:
    _ensure_dirs()
    fp = Path(STRATEGY_DIR) / filename
    with fp.open("w", encoding="utf-8") as f:
        f.write(code)
    meta_fp = fp.with_suffix("_metadata.json")
    metadata["hash"] = generate_strategy_hash(code)
    metadata["created_at"] = datetime.now(timezone.utc).isoformat()
    metadata["filename"] = filename
    with meta_fp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_strategy_callable(path: str, fn_name: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("strategy_mod", path)
    if not spec or not spec.loader:
        msg = "could not load strategy module spec"
        raise ValueError(msg)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)  # type: ignore[attr-defined]
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        msg = "strategy function not found"
        raise TypeError(msg)
    return fn


def run_basic_backtest(filename: str, symbol: str | None = None) -> dict[str, Any]:
    # All Live Data, No Fallback/Hardcoded Data
    # Validate symbol before entering try block
    if not symbol:
        if not TRADING_SYMBOLS:
            msg = "No trading symbols available - symbol required for backtest"
            raise ValueError(msg)
        symbol = TRADING_SYMBOLS[0]

    try:

        async def _kl():
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
            return await client.klines(symbol, interval="1h", limit=500)

        klines = anyio.run(_kl) or []
        closes = [float(k[4]) for k in klines if isinstance(k, (list, tuple)) and len(k) > 4]
        if len(closes) < 50:
            return {"error": "insufficient_data"}
        df = pd.DataFrame({"close": pd.Series(closes, dtype="float64")})
        path = Path(STRATEGY_DIR) / filename
        with path.open(encoding="utf-8") as f:
            code = f.read()
        fn_name = _extract_function_name(code) or "rsi_strategy"
        strat_fn = _load_strategy_callable(path, fn_name)
        df = strat_fn(df)
        if "signal" not in df.columns:
            return {"error": "no_signal"}
        ret = df["close"].pct_change().fillna(0.0)
        pos = df["signal"].clip(-1, 1).fillna(0.0)
        # Replace pandas missing markers and coerce to float
        pos = pos.replace([pd.NA, pd.NaT], 0).astype(float)
        pos_shift = pos.shift(1).fillna(0.0)
        strat_ret = pos_shift * ret
        mu = float(strat_ret.mean())
        sigma = float(strat_ret.std())
        period_per_year = 24 * 365
        sharpe = (mu / sigma) * (period_per_year**0.5) if sigma > 0 else 0.0
        trades = df["signal"].diff().fillna(0.0)
        buy_idx = trades[trades > 0].index.tolist()
        sell_idx = trades[trades < 0].index.tolist()
        total_trades = min(len(buy_idx), len(sell_idx))
        profit = 0.0
        for i in range(total_trades):
            b = buy_idx[i]
            s = sell_idx[i] if i < len(sell_idx) else None
            if s is None or s <= b:
                continue
            profit += float(df["close"].iloc[s] - df["close"].iloc[b])
        total_nonzero_trades = int((trades != 0).sum())
        winrate = float(total_trades) / max(total_nonzero_trades, 1)
        return {
            "winrate": winrate,
            "total_trades": total_trades,
            "profit": float(profit),
            "max_drawdown": float(df["close"].max() - df["close"].min()),
            "sharpe_ratio": float(sharpe),
            "backtest_date": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        return {"error": str(e)}


def generate_prompt(template: str | None = None) -> str:
    market = get_live_market_data()
    ctx = []
    for sym, d in market.items():
        ctx.append(f"{sym}: ${d['price']:.2f} (+0.00%)")
    market_context = ", ".join(ctx) if ctx else ""
    base = (
        "You are an advanced crypto quant trader. Create a new Python trading strategy function.\n\n"
        f"Market Context: {market_context}\n\n"
        "Requirements:\n"
        "- Use pandas for indicators and implement logic inline\n"
        "- Return buy/sell signals in a column named 'signal' with values in {-1,0,1}\n"
        "- Include basic position sizing, stop-loss, and take-profit variables within the function scope\n"
        "- Use robust error handling for empty or short DataFrames\n"
        "- Use Python 3.12 syntax with type hints\n"
        "Return only the complete Python function with no explanations."
    )
    if template:
        base = f"{base}\n\nTemplate:\n{template}"
    return base


def generate_strategy_enhanced() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.error(f"{EXCHANGE_ID} OPENAI_API_KEY missing")
        return
    try:
        # Pass api key explicitly to client if available
        client = OpenAI(api_key=key) if key else OpenAI()
        template = None
        # Use deterministic template selection instead of random
        if len(STRATEGY_TEMPLATES) > 0:
            template = next(iter(STRATEGY_TEMPLATES.values()))  # Use first template
        prompt = generate_prompt(template)
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        # Robust extraction of text content from different possible response shapes
        code_raw = ""
        try:
            # Newer SDK shape
            code_raw = res.choices[0].message.content  # type: ignore[attr-defined]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            try:
                code_raw = res.choices[0].message["content"]  # type: ignore[index]
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                try:
                    code_raw = res.choices[0].text  # type: ignore[attr-defined]
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    code_raw = str(res)
        code = _strip_code_fences(code_raw or "")
        if not validate_strategy_code(code):
            logger.error(f"{EXCHANGE_ID} generated code failed validation")
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_llm_{ts}.py"
        meta: dict[str, Any] = {
            "model": "gpt-4",
            "market_context": get_live_market_data(),
            "template_used": bool(template),
            "validation_passed": True,
            "prompt_length": len(prompt),
        }
        save_strategy_version(code, filename, meta)
        # Use first symbol from trading_universe for backtest (live data)
        backtest_symbol = TRADING_SYMBOLS[0] if TRADING_SYMBOLS else None
        if backtest_symbol:
            results = run_basic_backtest(filename, symbol=backtest_symbol)
            meta["backtest_results"] = results
        else:
            logger.warning(f"{EXCHANGE_ID} No trading symbols available for backtest")
            results = {"error": "no_symbols_available"}
            meta["backtest_results"] = results
        meta_fp = Path(STRATEGY_DIR) / filename.replace(".py", "_metadata.json")
        with meta_fp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"{EXCHANGE_ID} strategy saved {filename}")
        logger.info(f"{EXCHANGE_ID} backtest {results}")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"{EXCHANGE_ID} generation error: {e}")


def main() -> None:
    _ensure_dirs()
    generate_strategy_enhanced()


if __name__ == "__main__":
    main()
