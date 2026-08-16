"""Decision-time book/tape collector. Persistence only.

Does not select trades. Does not change EV, rank, HOLD, or exits.
Does not call Binance REST. Reads Redis hashes written by the live book/tape
streams. Fail-open: any error is swallowed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)

HOLD_EV = 0.0
THROTTLE_SEC = float(os.getenv("DECISION_BOOK_TAPE_THROTTLE_SEC", "15") or "15")
TABLE = "decision_book_tape"

_TABLE_READY = False
_LAST_QUIET: dict[str, float] = {}
_REDIS = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    if s.endswith("USDT"):
        s = s[:-4]
    return s or str(symbol or "")


def _bus(symbol: str) -> str:
    base = _base(symbol)
    return f"{base}USDT" if base else ""


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _redis_client(existing: Any = None) -> Any:
    if existing is not None:
        return existing
    global _REDIS
    if _REDIS is not None:
        return _REDIS
    try:
        import redis

        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        _REDIS = redis.from_url(url, decode_responses=True)
        return _REDIS
    except Exception:
        return None


def _hgetall(r: Any, key: str) -> dict[str, Any]:
    if r is None:
        return {}
    try:
        raw = r.hgetall(key)
    except Exception:
        return {}
    return dict(raw or {})


def _get(r: Any, key: str) -> Any:
    if r is None:
        return None
    try:
        return r.get(key)
    except Exception:
        return None


def snapshot_book(symbol: str, redis_client: Any = None) -> dict[str, Any]:
    """Best-effort Redis book/tape. Never fetches REST depth."""
    r = _redis_client(redis_client)
    base = _base(symbol)
    bus = _bus(symbol)
    book = {
        "best_bid": None,
        "best_ask": None,
        "mid": None,
        "spread_pct": None,
        "bid_qty_top5": None,
        "ask_qty_top5": None,
        "imbalance_top5": None,
        "ofi_5s": None,
        "agg_flow_imbalance_5s": None,
        "microprice_pressure": None,
        "book_source": "missing",
        "book_age_sec": None,
    }
    if not base:
        return book
    depth_raw = _get(r, f"scalp:depth_cache:{bus}")
    if depth_raw:
        try:
            payload = json.loads(depth_raw)
            bids = payload.get("bids") or []
            asks = payload.get("asks") or []
            fetched_at = float(payload.get("fetched_at") or 0)
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
                bid_qty = sum(float(q) for _, q in bids[:5])
                ask_qty = sum(float(q) for _, q in asks[:5])
                book.update(
                    {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid": mid or None,
                        "spread_pct": ((best_ask - best_bid) / mid) if mid > 0 else None,
                        "bid_qty_top5": bid_qty,
                        "ask_qty_top5": ask_qty,
                        "imbalance_top5": ((bid_qty - ask_qty) / (bid_qty + ask_qty)) if (bid_qty + ask_qty) > 0 else None,
                        "book_source": "redis_depth_cache",
                        "book_age_sec": (time.time() - fetched_at) if fetched_at else None,
                    }
                )
        except Exception:
            pass
    ob = _hgetall(r, f"orderbook:{base}")
    micro = _hgetall(r, f"microstructure:{base}")
    if book["best_bid"] is None:
        bid = _num(ob.get("bid") if ob.get("bid") not in (None, "") else ob.get("bid_price"))
        ask = _num(ob.get("ask") if ob.get("ask") not in (None, "") else ob.get("ask_price"))
        if bid and ask:
            mid = (bid + ask) / 2.0
            book["best_bid"] = bid
            book["best_ask"] = ask
            book["mid"] = mid
            book["spread_pct"] = _num(ob.get("spread_pct")) or ((ask - bid) / mid if mid else None)
            book["book_source"] = "redis_orderbook"
            ts = _num(ob.get("ts_utc") if ob.get("ts_utc") not in (None, "") else ob.get("updated_at"))
            if ts and ts > 1e12:
                ts = ts / 1000.0
            if ts and ts > 1e9:
                book["book_age_sec"] = time.time() - ts
    if book["imbalance_top5"] is None:
        book["imbalance_top5"] = _num(ob.get("order_book_imbalance"))
    book["ofi_5s"] = _num(micro.get("ofi_5s") if micro.get("ofi_5s") not in (None, "") else ob.get("ofi_5s"))
    book["agg_flow_imbalance_5s"] = _num(
        micro.get("agg_flow_imbalance_5s") if micro.get("agg_flow_imbalance_5s") not in (None, "") else ob.get("agg_flow_imbalance_5s")
    )
    book["microprice_pressure"] = _num(
        micro.get("microprice_pressure") if micro.get("microprice_pressure") not in (None, "") else ob.get("microprice_pressure")
    )
    if book["book_source"] == "missing" and (book["ofi_5s"] is not None or book["microprice_pressure"] is not None):
        book["book_source"] = "redis_microstructure"
    return book


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return

    def _create() -> None:
        with connect_rw(DATABASE_PATH) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    selected_action TEXT,
                    selection_reason TEXT,
                    buy_ev REAL,
                    hold_ev REAL,
                    model_version TEXT,
                    best_bid REAL,
                    best_ask REAL,
                    mid REAL,
                    spread_pct REAL,
                    bid_qty_top5 REAL,
                    ask_qty_top5 REAL,
                    imbalance_top5 REAL,
                    ofi_5s REAL,
                    agg_flow_imbalance_5s REAL,
                    microprice_pressure REAL,
                    book_source TEXT,
                    book_age_sec REAL,
                    extras_json TEXT,
                    realized_close_net REAL,
                    realized_fill_net REAL,
                    labeled_at TEXT
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_engine_ts ON {TABLE}(engine, ts_utc)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_symbol_ts ON {TABLE}(symbol, ts_utc)")

    run_locked_retry(_create)
    _TABLE_READY = True


def _should_write(engine: str, force: bool) -> bool:
    if force:
        return True
    now = time.time()
    last = _LAST_QUIET.get(engine, 0.0)
    if (now - last) < THROTTLE_SEC:
        return False
    _LAST_QUIET[engine] = now
    return True


def record_rows(rows: list[dict[str, Any]], *, force: bool = False) -> int:
    """Insert decision+book rows. Returns inserted count. Never raises."""
    try:
        if not rows:
            return 0
        engine = str(rows[0].get("engine") or "")
        any_buy = any(
            str(r.get("selected_action") or "").upper().startswith("BUY") or ((_num(r.get("buy_ev")) or 0.0) > HOLD_EV)
            for r in rows
        )
        if not _should_write(engine, force or any_buy):
            return 0
        _ensure_table()

        def _insert() -> int:
            n = 0
            with connect_rw(DATABASE_PATH) as conn:
                for r in rows:
                    conn.execute(
                        f"""
                        INSERT INTO {TABLE} (
                            ts_utc, engine, symbol, selected_action, selection_reason,
                            buy_ev, hold_ev, model_version,
                            best_bid, best_ask, mid, spread_pct,
                            bid_qty_top5, ask_qty_top5, imbalance_top5,
                            ofi_5s, agg_flow_imbalance_5s, microprice_pressure,
                            book_source, book_age_sec, extras_json
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            r.get("ts_utc") or _now_iso(),
                            str(r.get("engine") or ""),
                            str(r.get("symbol") or ""),
                            r.get("selected_action"),
                            r.get("selection_reason"),
                            _num(r.get("buy_ev")),
                            _num(r.get("hold_ev")) if r.get("hold_ev") not in (None, "") else HOLD_EV,
                            r.get("model_version"),
                            _num(r.get("best_bid")),
                            _num(r.get("best_ask")),
                            _num(r.get("mid")),
                            _num(r.get("spread_pct")),
                            _num(r.get("bid_qty_top5")),
                            _num(r.get("ask_qty_top5")),
                            _num(r.get("imbalance_top5")),
                            _num(r.get("ofi_5s")),
                            _num(r.get("agg_flow_imbalance_5s")),
                            _num(r.get("microprice_pressure")),
                            r.get("book_source"),
                            _num(r.get("book_age_sec")),
                            r.get("extras_json"),
                        ),
                    )
                    n += 1
            return n

        return int(run_locked_retry(_insert) or 0)
    except Exception as exc:
        logger.debug("decision_book_tape record failed: %s", exc)
        return 0


def record_day_decision(
    *,
    symbol: str,
    provenance: dict[str, Any] | None,
    redis_client: Any = None,
    extras: dict[str, Any] | None = None,
) -> int:
    """Stamp one DAY authority decision. Does not change the decision."""
    try:
        prov = dict(provenance or {})
        book = snapshot_book(symbol, redis_client)
        row = {
            "ts_utc": prov.get("prediction_timestamp") or _now_iso(),
            "engine": "day",
            "symbol": _bus(symbol),
            "selected_action": prov.get("selected_action"),
            "selection_reason": prov.get("selection_reason"),
            "buy_ev": prov.get("buy_ev") if prov.get("buy_ev") not in (None, "") else prov.get("predicted_net_return"),
            "hold_ev": prov.get("hold_ev") if prov.get("hold_ev") not in (None, "") else HOLD_EV,
            "model_version": prov.get("model_version") or prov.get("forward_net_model_version"),
            **book,
            "extras_json": json.dumps(extras, default=str) if extras else None,
        }
        return record_rows([row], force=True)
    except Exception as exc:
        logger.debug("decision_book_tape day failed: %s", exc)
        return 0


def record_scalp_cycle(
    *,
    ranked: list[dict[str, Any]] | None,
    chosen: dict[str, Any] | None,
    redis_client: Any = None,
    hold_ev: float = HOLD_EV,
    model_version: str = "",
) -> int:
    """Stamp one SCALP cycle (all four coins). HOLD and BUY. Does not change the pick."""
    try:
        rows_in = list(ranked or [])
        chosen_sym = _bus(str((chosen or {}).get("symbol") or ""))
        version = model_version or str((chosen or {}).get("forward_net_model_version") or "")
        if not version and rows_in:
            version = str(rows_in[0].get("forward_net_model_version") or "")
        selected_action = f"BUY_{chosen_sym}" if chosen_sym else "HOLD"
        selection_reason = "PATH_NET_BEATS_HOLD" if chosen_sym else "HOLD_WINS_ACTION_RANK"
        out = []
        for item in rows_in:
            sym = _bus(str(item.get("symbol") or ""))
            if not sym:
                continue
            ev = item.get("expected_net_ev")
            if ev in (None, ""):
                ev = item.get("predicted_net_return")
            book = snapshot_book(sym, redis_client)
            action = selected_action if sym == chosen_sym else "HOLD"
            out.append(
                {
                    "ts_utc": _now_iso(),
                    "engine": "scalp",
                    "symbol": sym,
                    "selected_action": action,
                    "selection_reason": selection_reason if action.startswith("BUY") else "HOLD_WINS_ACTION_RANK",
                    "buy_ev": ev,
                    "hold_ev": hold_ev,
                    "model_version": version or item.get("forward_net_model_version"),
                    **book,
                    "extras_json": None,
                }
            )
        if not out:
            book = snapshot_book(chosen_sym or "BTCUSDT", redis_client)
            out.append(
                {
                    "ts_utc": _now_iso(),
                    "engine": "scalp",
                    "symbol": chosen_sym or "HOLD",
                    "selected_action": selected_action,
                    "selection_reason": selection_reason,
                    "buy_ev": (chosen or {}).get("expected_net_ev"),
                    "hold_ev": hold_ev,
                    "model_version": version,
                    **book,
                    "extras_json": None,
                }
            )
        return record_rows(out)
    except Exception as exc:
        logger.debug("decision_book_tape scalp failed: %s", exc)
        return 0
