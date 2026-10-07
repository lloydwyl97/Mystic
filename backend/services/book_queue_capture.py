"""Compressed L1-L20 queue history from the existing depth stream.

The order-book collector already receives a ``depth20@100ms`` partial
snapshot per symbol and passes it to ``order_book_service.process_order_book``.
This module keeps what that path used to discard: price/quantity levels and
``lastUpdateId``. No socket or daemon is added.

Per symbol, one chunk per UTC minute: a keyframe of the first snapshot and,
for every later snapshot, the receive-time offset, the update-id delta and the
levels whose quantity changed (quantity 0 = the level left the top 20).
Chunks are zlib-compressed and written by a single background thread, never on
the event loop. Each chunk carries its own quality counters:

* ``n_dup``: update id equal to the previous one (same book re-sent)
* ``n_out_of_order``: update id lower than the previous one
* ``n_crossed``: best bid >= best ask
* ``max_gap_ms``: longest receive gap between snapshots

A chunk is ``usable`` only when ids never went backwards, no book was
crossed, no gap exceeded ``STALE_GAP_MS`` and the chunk covers most of its
minute. Partial-depth ids are exchange-global, so a positive jump is normal
and is not a gap; the receive-time gap is the continuity test. Measured on
Binance.US: ~200-300 snapshots/min/symbol, ~27 KB/min compressed for four
symbols (~1.6 MB/h, ~38 MB/day), zlib ratio ~4.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BOOK_QUEUE_VERSION = 1
LEVELS = 20
CHUNK_SEC = 60
# Binance.US partial depth is change-driven: a quiet book sends nothing for a
# few seconds (5.3 s observed inside healthy minutes), so only a longer silence
# is treated as a stall.
STALE_GAP_MS = 10_000
MIN_COVER_FRAC = 0.8
TABLE = "book_queue_chunks"
ENABLED = os.getenv("BOOK_QUEUE_CAPTURE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    chunk_start REAL NOT NULL,
    first_ts REAL NOT NULL,
    last_ts REAL NOT NULL,
    first_update_id INTEGER,
    last_update_id INTEGER,
    n_updates INTEGER NOT NULL,
    n_dup INTEGER NOT NULL,
    n_out_of_order INTEGER NOT NULL,
    n_crossed INTEGER NOT NULL,
    max_gap_ms INTEGER NOT NULL,
    usable INTEGER NOT NULL,
    version INTEGER NOT NULL,
    raw_bytes INTEGER NOT NULL,
    payload BLOB NOT NULL,
    created_at REAL NOT NULL
)
"""
INDEX = f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE}_sym_start ON {TABLE}(symbol, chunk_start)"


def _levels(side: Any) -> list[tuple[float, float]]:
    out = []
    for lvl in list(side or [])[:LEVELS]:
        try:
            p, q = float(lvl[0]), float(lvl[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(p) and math.isfinite(q) and p > 0 and q >= 0:
            out.append((p, q))
    return out


def _diff(prev: dict[float, float], cur: dict[float, float]) -> list[list[float]]:
    out = [[p, q] for p, q in cur.items() if prev.get(p) != q]
    out.extend([p, 0.0] for p in prev if p not in cur)
    return out


@dataclass
class _Chunk:
    symbol: str
    start: float
    first_ts: float
    first_uid: int | None
    key_bids: list[tuple[float, float]]
    key_asks: list[tuple[float, float]]
    bids: dict[float, float]
    asks: dict[float, float]
    last_ts: float
    last_uid: int | None
    diffs: list[list[Any]] = field(default_factory=list)
    n_dup: int = 0
    n_out_of_order: int = 0
    n_crossed: int = 0
    max_gap_ms: int = 0

    @property
    def n_updates(self) -> int:
        return 1 + len(self.diffs)

    def usable(self) -> bool:
        cover = (self.last_ts - self.first_ts) + 0.1
        return self.n_out_of_order == 0 and self.n_crossed == 0 and self.max_gap_ms <= STALE_GAP_MS and cover >= MIN_COVER_FRAC * CHUNK_SEC

    def encode(self) -> tuple[bytes, int]:
        body = {
            "v": BOOK_QUEUE_VERSION,
            "sym": self.symbol,
            "t0": round(self.first_ts, 3),
            "u0": self.first_uid,
            "b": [[p, q] for p, q in self.key_bids],
            "a": [[p, q] for p, q in self.key_asks],
            "d": self.diffs,
        }
        raw = json.dumps(body, separators=(",", ":")).encode()
        return zlib.compress(raw, 6), len(raw)


def decode(payload: bytes) -> list[dict[str, Any]]:
    """Every snapshot of one chunk: ``{"ts", "update_id", "bids", "asks"}``, best first."""
    body = json.loads(zlib.decompress(payload))
    bids = {float(p): float(q) for p, q in body["b"]}
    asks = {float(p): float(q) for p, q in body["a"]}
    ts = float(body["t0"])
    uid = body.get("u0")

    def snap() -> dict[str, Any]:
        return {
            "ts": ts,
            "update_id": uid,
            "bids": sorted(bids.items(), key=lambda x: -x[0]),
            "asks": sorted(asks.items(), key=lambda x: x[0]),
        }

    out = [snap()]
    for dt_ms, du, db, da in body["d"]:
        ts = float(body["t0"]) + float(dt_ms) / 1000.0
        uid = (uid + int(du)) if uid is not None and du is not None else None
        for side, changes in ((bids, db), (asks, da)):
            for p, q in changes:
                if float(q) == 0.0:
                    side.pop(float(p), None)
                else:
                    side[float(p)] = float(q)
        out.append(snap())
    return out


class BookQueueCapture:
    def __init__(self, db_path: str | None = None, *, max_pending: int = 512) -> None:
        self.db_path = db_path
        self._chunks: dict[str, _Chunk] = {}
        self._q: queue.Queue = queue.Queue(maxsize=max_pending)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.stats = {"written": 0, "dropped_full": 0, "write_errors": 0, "rejected_empty": 0}

    # ---------------------------------------------------------------- hot path
    def record(self, symbol: str, bids: Any, asks: Any, update_id: int | None, ts: float | None = None) -> None:
        now = float(ts if ts is not None else time.time())
        sym = str(symbol or "").upper().replace("USDT", "").replace("-", "").replace("/", "")
        b, a = _levels(bids), _levels(asks)
        if not sym or not b or not a:
            self.stats["rejected_empty"] += 1
            return
        uid = int(update_id) if update_id is not None else None
        start = math.floor(now / CHUNK_SEC) * CHUNK_SEC
        ch = self._chunks.get(sym)
        if ch is not None and ch.start != start:
            self._flush(ch)
            ch = None
        crossed = b[0][0] >= a[0][0]
        if ch is None:
            self._chunks[sym] = _Chunk(sym, start, now, uid, b, a, dict(b), dict(a), now, uid, n_crossed=int(crossed))
            return
        if uid is not None and ch.last_uid is not None:
            if uid == ch.last_uid:
                ch.n_dup += 1
                ch.last_ts = now
                return
            if uid < ch.last_uid:
                ch.n_out_of_order += 1
                return
        gap = round((now - ch.last_ts) * 1000.0)
        ch.max_gap_ms = max(ch.max_gap_ms, gap)
        cur_b, cur_a = dict(b), dict(a)
        du = (uid - ch.last_uid) if uid is not None and ch.last_uid is not None else None
        ch.diffs.append([round((now - ch.first_ts) * 1000.0), du, _diff(ch.bids, cur_b), _diff(ch.asks, cur_a)])
        ch.bids, ch.asks, ch.last_ts, ch.last_uid = cur_b, cur_a, now, uid
        ch.n_crossed += int(crossed)

    def flush_all(self) -> None:
        for ch in list(self._chunks.values()):
            self._flush(ch)
        self._chunks.clear()

    def _flush(self, ch: _Chunk) -> None:
        try:
            payload, raw = ch.encode()
        except (TypeError, ValueError):
            self.stats["write_errors"] += 1
            return
        row = (
            ch.symbol,
            ch.start,
            ch.first_ts,
            ch.last_ts,
            ch.first_uid,
            ch.last_uid,
            ch.n_updates,
            ch.n_dup,
            ch.n_out_of_order,
            ch.n_crossed,
            ch.max_gap_ms,
            1 if ch.usable() else 0,
            BOOK_QUEUE_VERSION,
            raw,
            sqlite3.Binary(payload),
            time.time(),
        )
        self._ensure_writer()
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self.stats["dropped_full"] += 1

    # ---------------------------------------------------------------- writer
    def _ensure_writer(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="book-queue-writer", daemon=True)
                self._thread.start()

    def _path(self) -> str:
        if self.db_path:
            return self.db_path
        from backend.database_schema import DATABASE_PATH

        return str(DATABASE_PATH)

    def _run(self) -> None:
        ready = False
        while True:
            row = self._q.get()
            if row is None:
                return
            try:
                with contextlib.closing(sqlite3.connect(self._path(), timeout=10)) as conn:
                    if not ready:
                        conn.execute(DDL)
                        conn.execute(INDEX)
                        ready = True
                    conn.execute(
                        f"INSERT OR IGNORE INTO {TABLE} (symbol, chunk_start, first_ts, last_ts, first_update_id, last_update_id, n_updates, n_dup, n_out_of_order, n_crossed, "
                        "max_gap_ms, usable, version, raw_bytes, payload, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        row,
                    )
                    conn.commit()
                self.stats["written"] += 1
            except sqlite3.Error:
                self.stats["write_errors"] += 1
                logger.debug("BOOK_QUEUE_WRITE_FAILED", exc_info=True)

    def drain(self, timeout: float = 5.0) -> None:
        """Wait until queued chunks are written (tests, shutdown)."""
        deadline = time.time() + timeout
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)


_capture: BookQueueCapture | None = None


def capture() -> BookQueueCapture:
    global _capture
    if _capture is None:
        _capture = BookQueueCapture()
    return _capture


def record_depth(symbol: str, bids: Any, asks: Any, update_id: int | None) -> None:
    """Hot-path hook for ``order_book_service.process_order_book``. Never raises."""
    if not ENABLED:
        return
    with contextlib.suppress(Exception):
        capture().record(symbol, bids, asks, update_id)


# --------------------------------------------------------------------------- features


def queue_features(snaps: list[dict[str, Any]], *, depth: int = 5) -> dict[str, float]:
    """Causal queue dynamics over a run of decoded snapshots (oldest first).

    Additions/removals are quantity changes at prices present in both
    consecutive top-``depth`` books; a vanished best level is a depletion, a
    restored one after a decrease is a replenishment. Slope is qty-weighted
    distance from mid (bps), concentration the HHI of top-``depth`` quantity.
    """
    out = dict.fromkeys(
        (
            "bid_add",
            "bid_remove",
            "ask_add",
            "ask_remove",
            "bid_depletions",
            "ask_depletions",
            "bid_replenish",
            "ask_replenish",
            "micro_disp_mean_bps",
            "micro_disp_last_bps",
            "bid_slope_bps",
            "ask_slope_bps",
            "bid_concentration",
            "ask_concentration",
            "n",
        ),
        0.0,
    )
    if not snaps:
        return out
    disp: list[float] = []
    last_down = {"bid": False, "ask": False}
    prev = None
    for s in snaps:
        bids, asks = s["bids"][:depth], s["asks"][:depth]
        if not bids or not asks:
            continue
        (bp, bq), (ap, aq) = bids[0], asks[0]
        mid = (bp + ap) / 2.0
        micro = (bp * aq + ap * bq) / (bq + aq) if (bq + aq) > 0 else mid
        disp.append((micro / mid - 1.0) * 1e4)
        if prev is not None:
            for name, cur, old in (("bid", bids, prev[0]), ("ask", asks, prev[1])):
                old_map, cur_map = dict(old), dict(cur)
                for p, q in cur_map.items():
                    if p in old_map:
                        d = q - old_map[p]
                        if d > 0:
                            out[f"{name}_add"] += d
                        elif d < 0:
                            out[f"{name}_remove"] += -d
                best_old, best_cur = old[0], cur[0]
                if best_cur[0] != best_old[0] and best_old[0] not in cur_map:
                    out[f"{name}_depletions"] += 1
                if best_cur[0] == best_old[0]:
                    if best_cur[1] < best_old[1]:
                        last_down[name] = True
                    elif best_cur[1] > best_old[1] and last_down[name]:
                        out[f"{name}_replenish"] += 1
                        last_down[name] = False
        prev = (bids, asks)
    last = snaps[-1]
    bids, asks = last["bids"][:depth], last["asks"][:depth]
    if bids and asks:
        mid = (bids[0][0] + asks[0][0]) / 2.0
        for name, side in (("bid", bids), ("ask", asks)):
            qty = sum(q for _, q in side)
            if qty > 0 and mid > 0:
                out[f"{name}_slope_bps"] = sum(abs(p / mid - 1.0) * 1e4 * q for p, q in side) / qty
                out[f"{name}_concentration"] = sum((q / qty) ** 2 for _, q in side)
    if disp:
        out["micro_disp_mean_bps"] = sum(disp) / len(disp)
        out["micro_disp_last_bps"] = disp[-1]
    out["n"] = float(len(snaps))
    return out


def load_chunks(db_path: str, symbol: str, start: float, end: float, *, usable_only: bool = True) -> list[dict[str, Any]]:
    """Decoded snapshots of ``symbol`` in [start, end), skipping unusable chunks."""
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            f"SELECT payload, usable FROM {TABLE} WHERE symbol=? AND chunk_start>=? AND chunk_start<? ORDER BY chunk_start",
            (str(symbol).upper().replace("USDT", ""), math.floor(start / CHUNK_SEC) * CHUNK_SEC, end),
        ).fetchall()
    out = []
    for payload, usable in rows:
        if usable_only and not int(usable):
            continue
        out.extend(s for s in decode(payload) if start <= s["ts"] < end)
    return out


__all__ = ["BOOK_QUEUE_VERSION", "STALE_GAP_MS", "TABLE", "BookQueueCapture", "capture", "decode", "load_chunks", "queue_features", "record_depth"]
