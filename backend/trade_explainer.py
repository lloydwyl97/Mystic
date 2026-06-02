#!/usr/bin/env python3
"""
trade_explainer.py
Hardened JSONL-based Trade Explainer with SQLite persistence.

- Reads trades from ./logs/trade_log.jsonl (append-only, not rewritten)
- Generates explanations via OpenAI (optional if API key present)
- Appends explanations to ./data/trade_explanations.jsonl
- Upserts each explanation into ./data/trade_explanations.db
- Creates a ping file for dashboard monitoring
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ----------------------------
# OpenAI client (safe import)
# ----------------------------
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # Modern SDK

    _OPENAI_AVAILABLE = True
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    # Fallback for environments without the new SDK
    OpenAI = None  # type: ignore[assignment]

try:
    from backend.services.confidence_normalizer import ConfidenceNormalizer
except (ImportError, ModuleNotFoundError):
    ConfidenceNormalizer = None  # type: ignore[assignment, misc]

# ----------------------------
# Configuration
# ----------------------------
LOG_FILE = "./logs/trade_log.jsonl"  # source trades (append-only)
EXPLANATIONS_JSONL = "./data/trade_explanations.jsonl"  # explanations store (append-only)
DB_PATH = os.getenv("TRADE_EXPLANATIONS_DB", "./data/trade_explanations.db")  # SQLite persistence
PING_FILE = "./logs/trade_explainer.ping"  # liveness/status
SLEEP_SECONDS = int(os.getenv("EXPLANATION_INTERVAL", "300"))  # default: 5 minutes
MAX_EXPLANATIONS_PER_CYCLE = int(os.getenv("MAX_EXPLANATIONS_PER_CYCLE", "5"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _ensure_dir_for_path(path: str) -> None:
    d = Path(path).parent
    d.mkdir(parents=True, exist_ok=True)


# Ensure directories exist for configured paths (guard empty dirname)
_ensure_dir_for_path(LOG_FILE)
_ensure_dir_for_path(EXPLANATIONS_JSONL)
_ensure_dir_for_path(DB_PATH)
_ensure_dir_for_path(PING_FILE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("trade_explainer")


# ----------------------------
# Utilities
# ----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_trade_id(trade: dict[str, Any]) -> str:
    """
    Produce a stable ID for a trade. Uses explicit 'id' when present; otherwise
    hashes a subset of keys to remain deterministic across runs.
    """
    if "id" in trade and trade["id"] is not None:
        return str(trade["id"])
    bases = {
        "symbol": trade.get("symbol"),
        "side": trade.get("side") or trade.get("trade_type"),
        "entry_price": trade.get("entry_price"),
        "exit_price": trade.get("exit_price"),
        "timestamp": trade.get("timestamp"),
        "qty": trade.get("quantity"),
    }
    raw = "|".join([str(bases.get(k, "")) for k in ("symbol", "side", "entry_price", "exit_price", "timestamp", "qty")])
    return "t_" + hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    with path_obj.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# SQLite persistence
# ----------------------------
def ensure_db() -> None:
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                pnl REAL,
                explanation_text TEXT NOT NULL,
                confidence REAL,
                risk_assessment TEXT,
                factors_json TEXT,
                market_context TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_explanations_trade_id ON trade_explanations(trade_id)")
        con.commit()
    finally:
        con.close()


def get_explained_ids_from_db() -> set[str]:
    con = None
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT trade_id FROM trade_explanations")
        rows = cur.fetchall()
        return {row[0] for row in rows if row and row[0]}
    except sqlite3.OperationalError:
        return set()
    finally:
        if con is not None:
            con.close()


def upsert_explanation_db(row: dict[str, Any]) -> None:
    raw_conf = row.get("confidence")
    if raw_conf is not None and ConfidenceNormalizer is not None:
        try:
            conf_val = ConfidenceNormalizer.normalize(float(raw_conf))
        except (TypeError, ValueError):
            conf_val = raw_conf
    else:
        conf_val = raw_conf
    con = None
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO trade_explanations (
                trade_id, timestamp, symbol, side, entry_price, exit_price, quantity, pnl,
                explanation_text, confidence, risk_assessment, factors_json, market_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_id) DO UPDATE SET
                timestamp=excluded.timestamp,
                symbol=excluded.symbol,
                side=excluded.side,
                entry_price=excluded.entry_price,
                exit_price=excluded.exit_price,
                quantity=excluded.quantity,
                pnl=excluded.pnl,
                explanation_text=excluded.explanation_text,
                confidence=excluded.confidence,
                risk_assessment=excluded.risk_assessment,
                factors_json=excluded.factors_json,
                market_context=excluded.market_context
            """,
            (
                row.get("trade_id"),
                row.get("timestamp"),
                row.get("symbol"),
                row.get("side"),
                row.get("entry_price"),
                row.get("exit_price"),
                row.get("quantity"),
                row.get("pnl"),
                row.get("explanation_text"),
                conf_val,
                row.get("risk_assessment"),
                row.get("factors_json"),
                row.get("market_context"),
            ),
        )
        con.commit()
    finally:
        if con is not None:
            con.close()


# ----------------------------
# Explanation logic
# ----------------------------
def collect_new_trades() -> list[dict[str, Any]]:
    """Return trades that do NOT yet have an explanation (checked against JSONL + DB)."""
    explained_ids_jsonl: set[str] = set()
    for rec in read_jsonl(EXPLANATIONS_JSONL):
        tid = rec.get("trade_id")
        if tid:
            explained_ids_jsonl.add(tid)

    explained_ids_db = get_explained_ids_from_db()
    explained = explained_ids_jsonl | explained_ids_db

    new_trades: list[dict[str, Any]] = []
    for trade in read_jsonl(LOG_FILE):
        tid = stable_trade_id(trade)
        if tid in explained:
            continue
        # Ensure minimum fields
        trade["_trade_id"] = tid
        trade.setdefault("symbol", "UNKNOWN")
        trade.setdefault("side", trade.get("trade_type", "UNKNOWN"))
        trade.setdefault("timestamp", utc_now())
        trade.setdefault("entry_price", None)
        trade.setdefault("exit_price", None)
        trade.setdefault("quantity", None)
        trade.setdefault("pnl", None)
        new_trades.append(trade)

    return new_trades


def build_prompt(trade: dict[str, Any]) -> str:
    summary = {
        "symbol": trade.get("symbol"),
        "side": trade.get("side"),
        "entry_price": trade.get("entry_price"),
        "exit_price": trade.get("exit_price"),
        "quantity": trade.get("quantity"),
        "pnl": trade.get("pnl"),
        "timestamp": trade.get("timestamp"),
    }
    return (
        "Explain in 5-8 sentences why this trade could make sense to a professional crypto trader. "
        "Be concrete about risk, alternative scenarios, and what invalidates the thesis. "
        "Avoid hype; be clear and concise.\n\n"
        f"TRADE: {json.dumps(summary, ensure_ascii=False)}"
    )


def generate_explanation_text(trade: dict[str, Any]) -> tuple[str, float, str]:
    """
    Returns (explanation_text, confidence, risk_assessment).
    Falls back to a deterministic template if OpenAI isn't configured.
    """
    # Fallback baseline confidence based on minimal info
    base_conf = 0.6

    if not (_OPENAI_AVAILABLE and OPENAI_API_KEY):
        explanation = (
            f"{trade.get('side', 'TRADE')} {trade.get('symbol', 'ASSET')} based on recent price structure and "
            "expected momentum/mean reversion. Risk managed via tight invalidation and asymmetric reward/risk. "
            "This is a generic explanation (OpenAI not configured)."
        )
        return explanation, base_conf, "Medium risk (no LLM analysis)"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = build_prompt(trade)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=350,
        )
        # Best-effort extraction; different SDKs may vary shape
        text = ""
        try:
            text = (resp.choices[0].message.content or "").strip()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            try:
                # older/newer shapes
                text = (resp.choices[0].text or "").strip()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                text = str(resp).strip()
        # Mild heuristic: longer = a bit more confidence (capped)
        conf = min(base_conf + len(text) / 2000.0, 0.9)
        risk = "Medium risk (balanced thesis with clear invalidation)"
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"OpenAI call failed: {e}")
        explanation = f"{trade.get('side', 'TRADE')} {trade.get('symbol', 'ASSET')} using trend + structure. Fallback explanation due to API error."
        return explanation, base_conf, "Medium risk (LLM fallback)"
    else:
        return text, conf, risk


def write_ping(explanations_generated: int, total_candidates: int) -> None:
    try:
        ping_path = Path(PING_FILE)
        ping_path.parent.mkdir(parents=True, exist_ok=True)
        with ping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": utc_now(),
                    "explanations_generated": explanations_generated,
                    "pending_candidates": total_candidates,
                },
                f,
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.debug(f"Ping write failed: {e}")


# ----------------------------
# One processing cycle
# ----------------------------
def process_cycle() -> tuple[int, int]:
    ensure_db()
    candidates = collect_new_trades()
    if not candidates:
        write_ping(0, 0)
        return 0, 0

    to_process = candidates[:MAX_EXPLANATIONS_PER_CYCLE]
    generated = 0

    for trade in to_process:
        trade_id = trade["_trade_id"]
        explanation_text, confidence, risk = generate_explanation_text(trade)

        record = {
            "trade_id": trade_id,
            "timestamp": utc_now(),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "quantity": trade.get("quantity"),
            "pnl": trade.get("pnl"),
            "explanation_text": explanation_text,
            "confidence": confidence,
            "risk_assessment": risk,
            "factors_json": "[]",  # kept simple; can be populated later
            "market_context": "",  # kept simple; can be populated later
        }

        # Append to JSONL explanations
        append_jsonl(EXPLANATIONS_JSONL, record)

        # Upsert into SQLite
        upsert_explanation_db(record)

        generated += 1
        # Gentle pacing to avoid rate spikes (especially without a paid tier)
        time.sleep(0.5)

    write_ping(generated, len(candidates))
    return generated, len(candidates)


# ----------------------------
# Main loop
# ----------------------------
async def main():
    logger.info("Trade Explainer started")
    logger.info(f"Interval: {SLEEP_SECONDS}s | Max per cycle: {MAX_EXPLANATIONS_PER_CYCLE} | Model: {OPENAI_MODEL}")
    while True:
        try:
            generated, total = process_cycle()
            if generated:
                logger.info(f"Generated {generated} explanation(s) this cycle ({total} candidate(s) total).")
            else:
                logger.info("No new trades need explanations.")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Cycle error: {e}")
        await asyncio.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
