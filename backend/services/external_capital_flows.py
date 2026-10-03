"""External capital flows: venue deposits and withdrawals into the capital baseline.

The capital baseline is the 9039923 forward cash baseline plus every external
flow the venue reports after that baseline was adopted. Each venue flow is
recorded once, keyed by its venue id, so a flow can never be applied twice and
a restart or a stale in-memory principal cannot drop one.

Flows are never trading P&L and never learning input. Flows at or before the
baseline adoption are recorded for audit with ``applies_to_baseline=0``: the
adopted cash already contains them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from backend.services.day_entry_spendable import money
from backend.services.live_account_basis import TRAILING_BUY_ANCHOR_EQUITY

logger = logging.getLogger(__name__)

FLOW_TABLE = "external_capital_flows"
# Commit time of 9039923 (2026-09-15T14:50:43Z), when 228.06746265 was adopted.
BASELINE_ADOPTED_MS = 1789483843000
STABLE_USD_ASSETS = frozenset({"USD", "USDT", "USDC"})
DEPOSIT = "DEPOSIT"
WITHDRAWAL = "WITHDRAWAL"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {FLOW_TABLE} (
    flow_id TEXT PRIMARY KEY,
    direction TEXT NOT NULL,
    asset TEXT NOT NULL,
    amount TEXT NOT NULL,
    usd_value TEXT NOT NULL,
    venue_time_ms INTEGER NOT NULL,
    source TEXT NOT NULL,
    applies_to_baseline INTEGER NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{{}}',
    recorded_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class CapitalFlow:
    flow_id: str
    direction: str
    asset: str
    amount: Decimal
    usd_value: Decimal
    venue_time_ms: int
    source: str
    evidence: dict[str, Any]

    @property
    def signed_usd(self) -> Decimal:
        return self.usd_value if self.direction == DEPOSIT else -self.usd_value

    @property
    def applies_to_baseline(self) -> bool:
        return self.venue_time_ms > BASELINE_ADOPTED_MS


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)


def _ms(raw: Any) -> int:
    try:
        return int(float(raw or 0))
    except (TypeError, ValueError):
        return 0


def parse_fiat_history(payload: Any, *, direction: str) -> list[CapitalFlow]:
    """Binance.US ``/sapi/v1/fiatpayment/query/{deposit,withdraw}/history``.

    Deposits credit ``amount`` (the card/bank fee is charged outside the
    account). Withdrawals debit ``amount`` plus venue fees.
    """
    rows = (payload or {}).get("assetLogRecordList") if isinstance(payload, dict) else None
    out: list[CapitalFlow] = []
    for row in rows or []:
        if not isinstance(row, dict) or str(row.get("orderStatus") or "").strip().lower() != "successful":
            continue
        asset = str(row.get("fiatCurrency") or "").upper()
        amount = money(row.get("amount"))
        oid = str(row.get("orderId") or "").strip()
        if not oid or amount <= 0 or asset not in STABLE_USD_ASSETS:
            continue
        value = amount if direction == DEPOSIT else amount + money(row.get("transactionFee")) + money(row.get("platformFee"))
        out.append(CapitalFlow(f"fiat:{direction}:{oid}", direction, asset, amount, value, _ms(row.get("createTime")), "binanceus_fiat", dict(row)))
    return out


def parse_crypto_history(payload: Any, *, direction: str) -> list[CapitalFlow]:
    """Binance.US ``/sapi/v1/capital/deposit/hisrec`` and ``/capital/withdraw/history``.

    Only completed flows of USD-stable assets are valued here (deposit status 1,
    withdrawal status 6). A non-stable crypto flow has no executable USD value
    in this record and is left for operator classification, never guessed.
    """
    out: list[CapitalFlow] = []
    done = 1 if direction == DEPOSIT else 6
    for row in payload or []:
        if not isinstance(row, dict) or _ms(row.get("status")) != done:
            continue
        asset = str(row.get("coin") or "").upper()
        amount = money(row.get("amount"))
        ref = str(row.get("txId") or row.get("id") or "").strip()
        if not ref or amount <= 0 or asset not in STABLE_USD_ASSETS:
            continue
        value = amount if direction == DEPOSIT else amount + money(row.get("transactionFee"))
        when = _ms(row.get("insertTime")) or _ms(row.get("completeTime")) or _ms(row.get("applyTime"))
        if not when and row.get("applyTime"):
            try:
                when = int(datetime.strptime(str(row["applyTime"]), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError:
                when = 0
        out.append(CapitalFlow(f"crypto:{direction}:{asset}:{ref}", direction, asset, amount, value, when, "binanceus_capital", dict(row)))
    return out


def record_flows(conn: sqlite3.Connection, flows: list[CapitalFlow]) -> list[CapitalFlow]:
    """Insert flows not seen before. Returns only the newly recorded ones."""
    ensure_schema(conn)
    now = datetime.now(timezone.utc).isoformat()
    new: list[CapitalFlow] = []
    for f in flows:
        cur = conn.execute(
            f"""
            INSERT OR IGNORE INTO {FLOW_TABLE}
            (flow_id, direction, asset, amount, usd_value, venue_time_ms, source, applies_to_baseline, evidence_json, recorded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (f.flow_id, f.direction, f.asset, str(f.amount), str(f.usd_value), f.venue_time_ms, f.source, 1 if f.applies_to_baseline else 0, json.dumps(f.evidence, default=str), now),
        )
        if cur.rowcount:
            new.append(f)
    return new


def capital_baseline(conn: sqlite3.Connection) -> Decimal | None:
    """Anchor plus applied flows, or None when no flow has ever been recorded."""
    try:
        rows = conn.execute(f"SELECT direction, usd_value, applies_to_baseline FROM {FLOW_TABLE}").fetchall()
    except sqlite3.OperationalError:
        return None
    if not rows:
        return None
    total = TRAILING_BUY_ANCHOR_EQUITY
    for direction, usd_value, applies in rows:
        if int(applies or 0):
            total += money(usd_value) if str(direction) == DEPOSIT else -money(usd_value)
    return total


def resolve_principal(db_path: str, stored_principal: float) -> float:
    """Capital baseline when flows are recorded, else the stored principal."""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as conn:
            base = capital_baseline(conn)
    except sqlite3.Error:
        return float(stored_principal)
    return float(base) if base is not None else float(stored_principal)


def sync_flows(db_path: str, flows: list[CapitalFlow]) -> tuple[list[CapitalFlow], float | None]:
    """Record venue flows once and persist the resulting baseline as ledger principal.

    Returns (newly recorded flows, baseline). The baseline is None until the
    first flow exists.
    """
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        new = record_flows(conn, flows)
        base = capital_baseline(conn)
        if base is not None:
            conn.execute("UPDATE portfolio_engine_ledger SET principal=? WHERE id=1", (float(base),))
        conn.commit()
    for f in new:
        logger.warning(
            "EXTERNAL_CAPITAL_FLOW_RECORDED id=%s direction=%s asset=%s amount=%s usd=%s venue_time_ms=%s applies=%s",
            f.flow_id,
            f.direction,
            f.asset,
            f.amount,
            f.usd_value,
            f.venue_time_ms,
            f.applies_to_baseline,
        )
    return new, (float(base) if base is not None else None)


VENUE_FLOW_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("/sapi/v1/fiatpayment/query/deposit/history", DEPOSIT, "fiat"),
    ("/sapi/v1/fiatpayment/query/withdraw/history", WITHDRAWAL, "fiat"),
    ("/sapi/v1/capital/deposit/hisrec", DEPOSIT, "crypto"),
    ("/sapi/v1/capital/withdraw/history", WITHDRAWAL, "crypto"),
)


def parse_endpoint(kind: str, direction: str, payload: Any) -> list[CapitalFlow]:
    return parse_fiat_history(payload, direction=direction) if kind == "fiat" else parse_crypto_history(payload, direction=direction)
