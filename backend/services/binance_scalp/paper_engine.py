"""Binance.US scalp paper engine — paper fills only, no live orders."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import redis
from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.exit_manager import (
    DECISION_SELL,
    EXIT_MAX_HOLD_HARD_LIMIT,
    ExitReviewResult,
    PositionTrack,
    STATE_MAX_HOLD_REVIEW,
    _max_hold_hard_sec,
    evaluate_exit,
    track_from_row,
)
from backend.services.binance_scalp.market_reader import MarketSnapshot, ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.protected_preflight import (
    FEE_MODEL_UNVERIFIED,
    NET_PROFIT_TARGET_NOT_MET,
    SCALP_PAPER_DISABLED,
    run_scalp_preflight,
)
from backend.services.binance_scalp.redis_keys import (
    assert_key_allowed,
    market_key,
    position_key,
    runner_state_key,
    scan_key,
    signal_key,
)
from backend.services.binance_scalp.scalp_control import (
    clear_entry_armed,
    is_entry_armed,
    set_entry_armed,
)
from backend.services.binance_scalp.scalp_reject_throttle import (
    ScalpRejectThrottle,
    maybe_run_scalp_reject_retention,
)
from backend.services.binance_scalp.scalp_position_retention import maybe_run_scalp_position_housekeeping
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.schema import init_scalp_schema
from backend.services.binance_scalp.strategies.kline_cache import KlineCache
from backend.utils.sqlite_runtime import connect_rw, is_locked_error, run_locked_retry

logger = logging.getLogger(__name__)
WOULD_ENTER_NOT_ARMED = "WOULD_ENTER_NOT_ARMED"


def _round_trip_execution_costs(
    *,
    entry_notional: float,
    exit_notional: float,
    econ: ScalpEconomics,
    persisted_entry_fee: float | None = None,
    persisted_entry_slippage: float | None = None,
) -> tuple[float, float]:
    entry_fee = float(persisted_entry_fee) if persisted_entry_fee is not None else entry_notional * econ.entry_fee_pct()
    entry_slippage = float(persisted_entry_slippage) if persisted_entry_slippage is not None else entry_notional * econ.slippage_buffer_pct
    fees = entry_fee + exit_notional * econ.exit_fee_pct()
    slippage = entry_slippage + exit_notional * econ.slippage_buffer_pct
    return fees, slippage


class BinanceScalpPaperEngine:
    """Paper-only scalp loop — isolated from Mystic DAY portfolio_engine."""

    def __init__(
        self,
        config: ScalpConfig | None = None,
        econ: ScalpEconomics | None = None,
    ) -> None:
        self.config = config or get_scalp_config()
        self.config.assert_no_live_trading()
        if self.config.allow_repair_add:
            raise RuntimeError("SCALP_ALLOW_REPAIR_ADD must remain false")
        self.econ = econ or economics_for_config(self.config)
        self.reader = ScalpMarketReader(self.config)
        self._momentum = MomentumTracker()
        self._klines = KlineCache()
        self._router = ScalpStrategyRouter(
            config=self.config,
            econ=self.econ,
            reader=self.reader,
            momentum=self._momentum,
            klines=self._klines,
        )
        try:
            from backend.services.binance_scalp import scalp_signal_engine as _se

            _se.bind_paper_engine(router=self._router, momentum=self._momentum, klines=self._klines)
        except Exception:
            pass
        self._reject_throttle = ScalpRejectThrottle()
        self._redis = redis.from_url(self.config.redis_url, decode_responses=True)
        self._entry_reservations: dict[str, dict] = {}
        self._heartbeat_stop = False
        self._heartbeat_thread = None
        self._cached_open_symbols: list[str] = []
        self._last_circuit_breaker_open = False
        self._last_breaker_reason = ""
        self._last_breaker_recovery_until = ""
        self._last_breaker_eval_after = ""
        try:
            from backend.database_schema import DATABASE_PATH as _DAY_DB
            from backend.services.atomic_execution_book import migrate_scalp_money_database

            migrate_scalp_money_database(_DAY_DB, self.config.database_path)
        except Exception:
            logger.exception("SCALP_MONEY_DB_MIGRATE_SKIPPED")
        init_scalp_schema(self.config.database_path)
        with contextlib.suppress(Exception):
            from backend.services.scalp_gate_telemetry import ensure_scalp_gate_schema

            ensure_scalp_gate_schema(self.config.database_path)
        with contextlib.suppress(Exception):
            from backend.services.validation_cutoff import ensure_validation_cutoff

            ensure_validation_cutoff(self.config.database_path, engine="scalp", repo_root=self.config.repo_root)
        if self.config.scalp_paper_enabled and self.config.scalp_paper_auto_arm:
            set_entry_armed(
                self._redis,
                prefix=self.config.redis_key_prefix,
                armed=True,
                persistent=True,
            )
        else:
            set_entry_armed(self._redis, prefix=self.config.redis_key_prefix, armed=False)
        try:
            if self._check_scalp_circuit_breaker():
                self._last_circuit_breaker_open = True
                self._publish_last_decision(decision="BLOCKED", reason="SCALP_CIRCUIT_BREAKER_OPEN")
            else:
                self._last_circuit_breaker_open = False
                self._publish_last_decision(decision="SCAN", reason="")
        except Exception:
            logger.debug("SCALP_STARTUP_CB_STATUS_SKIPPED", exc_info=True)

    def _conn(self) -> sqlite3.Connection:
        conn = connect_rw(self.config.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _now(self) -> tuple[str, float]:
        dt = self._utcnow()
        return dt.isoformat(), dt.timestamp()

    def _utcnow(self) -> datetime:
        override = getattr(self, "_utcnow_override", None)
        if override is not None:
            return override
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_utc_ts(raw: str | None) -> datetime | None:
        s = str(raw or "").strip()
        if not s:
            return None
        s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _consec_eval_floor(self, epoch: str, eval_after: str) -> str:
        candidates = [s.strip() for s in (epoch, eval_after) if str(s or "").strip()]
        if not candidates:
            return ""
        parsed: list[tuple[datetime, str]] = []
        for item in candidates:
            dt = self._parse_utc_ts(item)
            if dt is not None:
                parsed.append((dt, item))
        if not parsed:
            return max(candidates)
        return max(parsed, key=lambda item: item[0])[1]

    def _load_consec_breaker_state(self, conn: sqlite3.Connection) -> dict[str, str]:
        try:
            row = conn.execute(
                """
                SELECT consec_breaker_tripped_at, consec_breaker_recovery_until,
                       consec_breaker_eval_after, consec_breaker_reason
                FROM scalp_meta WHERE id = 1
                """
            ).fetchone()
        except sqlite3.OperationalError:
            return {}
        if row is None:
            return {}
        return {
            "tripped_at": str(row[0] or ""),
            "recovery_until": str(row[1] or ""),
            "eval_after": str(row[2] or ""),
            "reason": str(row[3] or ""),
        }

    def _save_consec_breaker_state(
        self,
        conn: sqlite3.Connection,
        *,
        tripped_at: str = "",
        recovery_until: str = "",
        eval_after: str | None = None,
        reason: str = "",
    ) -> None:
        if eval_after is None:
            conn.execute(
                """
                UPDATE scalp_meta
                SET consec_breaker_tripped_at = ?,
                    consec_breaker_recovery_until = ?,
                    consec_breaker_reason = ?
                WHERE id = 1
                """,
                (tripped_at or None, recovery_until or None, reason or None),
            )
            return
        conn.execute(
            """
            UPDATE scalp_meta
            SET consec_breaker_tripped_at = ?,
                consec_breaker_recovery_until = ?,
                consec_breaker_eval_after = ?,
                consec_breaker_reason = ?
            WHERE id = 1
            """,
            (tripped_at or None, recovery_until or None, eval_after or None, reason or None),
        )

    def _ledger(self, conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM scalp_paper_ledger WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError("scalp_paper_ledger missing")
        return row

    def _open_positions(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(conn.execute("SELECT * FROM scalp_paper_positions WHERE status = 'OPEN'"))

    def _position_strategy_id(self, row: sqlite3.Row) -> str:
        try:
            sid = row["strategy_id"]
            if sid:
                return str(sid)
        except (KeyError, IndexError, TypeError):
            pass
        return self.config.strategy_id

    def _record_gate(
        self,
        *,
        reason: str = "",
        gate_id: str = "",
        symbol: str = "",
        outcome: str = "hard_blocked",
        setup: str = "",
        entry_price: float = 0.0,
        stop_price: float = 0.0,
        target_price: float = 0.0,
        shadow: bool = True,
        detail: str = "",
        diag: dict | None = None,
    ) -> None:
        """Always count gate events (independent of reject throttle)."""
        with contextlib.suppress(Exception):
            from backend.services.scalp_gate_telemetry import record_gate_event, record_shadow_reject

            db = self.config.database_path
            record_gate_event(
                db,
                gate_id=gate_id,
                reason=reason,
                symbol=symbol,
                outcome=outcome,
                setup=setup,
                detail=detail or reason,
            )
            if shadow and outcome == "hard_blocked" and symbol:
                record_shadow_reject(
                    db,
                    symbol=symbol,
                    gate_id=gate_id,
                    reason=reason,
                    setup=setup,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    detail=detail or reason,
                    diag=diag,
                )

    def _record_reject(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        side: str,
        reason: str,
        detail: str,
    ) -> None:
        if str(side).upper() == "BUY":
            self._record_gate(reason=reason, symbol=symbol, detail=detail[:500] if detail else "")
        if not self._reject_throttle.should_log(symbol, side, reason):
            return
        conn.execute(
            """
            INSERT INTO scalp_rejects (symbol, exchange, strategy_id, side, reason, detail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                self.config.exchange,
                self.config.strategy_id,
                side,
                reason,
                detail[:2000] if detail else "",
            ),
        )

    def _write_signal(self, symbol_bus: str, payload: dict) -> None:
        key = signal_key(self.config.redis_key_prefix, symbol_bus)
        assert_key_allowed(key, prefix=self.config.redis_key_prefix)
        self._redis.setex(key, 120, json.dumps(payload))

    def _write_market_cache(self, snap: MarketSnapshot) -> None:
        key = market_key(self.config.redis_key_prefix, snap.symbol_bus)
        assert_key_allowed(key, prefix=self.config.redis_key_prefix)
        self._redis.setex(
            key,
            120,
            json.dumps(
                {
                    "symbol": snap.symbol_bus,
                    "best_bid": snap.best_bid,
                    "best_ask": snap.best_ask,
                    "mid": snap.mid,
                    "spread_pct": snap.spread_pct,
                    "order_book_imbalance": snap.order_book_imbalance,
                    "source": "scalp_paper_engine",
                    "strategy_id": self.config.strategy_id,
                }
            ),
        )

    def _seed_scalp_market_memory(self, sym: str, snap: MarketSnapshot, *, micro_regime: str = "") -> None:
        try:
            from backend.services.scalp_market_memory import update_scalp_market_memory_on_candidate

            update_scalp_market_memory_on_candidate(
                self._redis,
                sym,
                {
                    "spread_pct": snap.spread_pct,
                    "orderbook_age_sec": snap.orderbook_age_sec,
                    "micro_regime": micro_regime or "range",
                    "mid_change_15s": 0.0,
                },
            )
        except Exception:
            pass

    def _publish_scan_snapshot(
        self,
        sym: str,
        snap: MarketSnapshot,
        *,
        micro_regime: str,
        epoch: float,
    ) -> None:
        try:
            mom = self._momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
            key = scan_key(self.config.redis_key_prefix, sym)
            self._redis.setex(
                key,
                90,
                json.dumps(
                    {
                        "symbol": sym,
                        "micro_regime": micro_regime,
                        "momentum_confirmed": bool(mom.momentum_confirmed),
                        "momentum_sample_count": int(mom.sample_count),
                        "momentum_history_sec": float(mom.history_sec),
                        "spread_pct": float(snap.spread_pct),
                        "updated_at_epoch": epoch,
                    },
                    separators=(",", ":"),
                ),
            )
        except Exception:
            pass

    def _publish_api_status_snapshot(
        self,
        *,
        open_rows: list[sqlite3.Row] | None = None,
        epoch: float | None = None,
        entry_blocked_reason: str | None = None,
        snapshot_source: str = "runner_tick",
        include_pnl: bool = False,
    ) -> None:
        """Publish lightweight Redis snapshot for GET /api/scalp/status (no REST rebuild)."""
        try:
            from backend.services.binance_scalp.redis_keys import last_decision_key, runner_state_key
            from backend.services.binance_scalp.scalp_status_cache import (
                build_runner_api_status_payload,
                publish_status_snapshot,
                status_cache_ttl_sec,
            )

            epoch = float(epoch if epoch is not None else time.time())
            if open_rows is None:
                open_syms = list(self._cached_open_symbols)
                open_count = len(open_syms)
            else:
                open_syms = [str(r["symbol"]) for r in open_rows]
                open_count = len(open_syms)
                self._cached_open_symbols = list(open_syms)

            rs_raw = self._redis.get(runner_state_key(self.config.redis_key_prefix))
            ld_raw = self._redis.get(last_decision_key(self.config.redis_key_prefix))
            runner_state = (
                json.loads(rs_raw)
                if rs_raw
                else {
                    "updated_at_epoch": epoch,
                    "operational_mode": ("max_open_positions_reached" if entry_blocked_reason == "MAX_OPEN_POSITIONS" else "entry_scan_active"),
                    "open_count": open_count,
                    "max_open_positions": int(self.config.max_open_positions),
                    "open_symbols": open_syms,
                    "entry_blocked_reason": entry_blocked_reason,
                }
            )
            if isinstance(runner_state, dict):
                runner_state = dict(runner_state)
                runner_state["updated_at_epoch"] = epoch
                runner_state["open_count"] = open_count
                runner_state["open_symbols"] = open_syms
                if entry_blocked_reason:
                    runner_state["entry_blocked_reason"] = entry_blocked_reason
            last_decision = json.loads(ld_raw) if ld_raw else {}
            # Never block heartbeat on SQLite PnL — optional and skippable.
            pnl: dict = {"engine": "scalp", "note": "pnl_omitted_on_fast_path"}
            if include_pnl:
                try:
                    from backend.services.binance_scalp.pnl_summary import build_scalp_pnl_summary

                    pnl = build_scalp_pnl_summary(self.config.database_path)
                except Exception:
                    pnl = {"engine": "scalp", "note": "pnl_unavailable"}
            payload = build_runner_api_status_payload(
                runner_state=runner_state if isinstance(runner_state, dict) else {},
                last_decision=last_decision if isinstance(last_decision, dict) else {},
                entry_armed=bool(self._entry_armed_ok()),
                open_count=open_count,
                products=list(self.config.products),
                scalp_live=bool(self.config.scalp_live),
                scalp_paper_enabled=bool(self.config.scalp_paper_enabled),
                pnl_summary=pnl,
                snapshot_source=snapshot_source,
                open_symbols=open_syms,
            )
            publish_status_snapshot(payload, ttl_sec=status_cache_ttl_sec())
            with contextlib.suppress(Exception):
                from backend.services.binance_scalp.scalp_entry_telemetry import (
                    touch_rolling_telemetry_heartbeat,
                )

                touch_rolling_telemetry_heartbeat(self._redis, prefix=self.config.redis_key_prefix)
        except Exception as exc:
            logger.warning("SCALP_API_STATUS_PUBLISH_SKIPPED %s", exc)

    def _start_status_heartbeat(self, interval_sec: float = 30.0) -> None:
        """Keep snapshot TTL alive even when tick is blocked on REST/klines."""
        import threading

        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop = False

        def _loop() -> None:
            while not self._heartbeat_stop:
                try:
                    from backend.services.binance_scalp.scalp_status_cache import (
                        refresh_status_snapshot_heartbeat,
                    )

                    refresh_status_snapshot_heartbeat(reason="runner_heartbeat")
                    with contextlib.suppress(Exception):
                        from backend.services.binance_scalp.scalp_entry_telemetry import (
                            touch_rolling_telemetry_heartbeat,
                        )

                        touch_rolling_telemetry_heartbeat(self._redis, prefix=self.config.redis_key_prefix)
                except Exception as exc:
                    logger.debug("SCALP_STATUS_HEARTBEAT_SKIPPED %s", exc)
                for _ in range(int(max(5.0, interval_sec) * 10)):
                    if self._heartbeat_stop:
                        break
                    time.sleep(0.1)

        self._heartbeat_thread = threading.Thread(target=_loop, name="scalp-status-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _publish_runner_state(
        self,
        conn: sqlite3.Connection,
        *,
        open_rows: list[sqlite3.Row],
        epoch: float,
        entry_blocked_reason: str | None = None,
    ) -> None:
        try:
            open_count = len(open_rows)
            max_open = int(self.config.max_open_positions)
            open_symbols = [str(r["symbol"]) for r in open_rows]
            if open_count >= max_open:
                mode = "max_open_positions_reached"
            elif open_count > 0:
                mode = "entry_scan_active"
            elif entry_blocked_reason:
                mode = "entry_rejected_by_strategy"
            else:
                mode = "entry_scan_active"
            payload = {
                "updated_at_epoch": epoch,
                "operational_mode": mode,
                "open_count": open_count,
                "max_open_positions": max_open,
                "open_symbols": open_symbols,
                "products_scanned": list(self.config.products),
                "entry_blocked_reason": entry_blocked_reason,
                "momentum_warmed": True,
                "circuit_breaker_open": bool(self._last_circuit_breaker_open),
                "circuit_breaker_reason": getattr(self, "_last_breaker_reason", "") or "",
                "breaker_recovery_until": getattr(self, "_last_breaker_recovery_until", "") or "",
                "breaker_eval_after": getattr(self, "_last_breaker_eval_after", "") or "",
            }
            key = runner_state_key(self.config.redis_key_prefix)
            self._redis.setex(key, 90, json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    def _publish_last_decision(
        self,
        *,
        decision: str,
        reason: str = "",
        selected_symbol: str | None = None,
        rank_score: float | None = None,
        entry_armed: bool | None = None,
        ranked_summary: list[dict] | None = None,
    ) -> None:
        """
        Publish the ACTUAL canonical pre-order decision from this tick.

        This is the single source of truth for "would the engine enter right
        now" — status_snapshot.py reads this instead of running its own
        independent ranking/momentum simulation, so the dashboard can never
        disagree with what the paper engine itself just decided.
        """
        try:
            from backend.services.binance_scalp.redis_keys import last_decision_key

            payload = {
                "updated_at_epoch": time.time(),
                "decision": decision,
                "reason": reason,
                "selected_symbol": selected_symbol,
                "rank_score": rank_score,
                "entry_armed": entry_armed,
                "ranked_summary": ranked_summary or [],
            }
            key = last_decision_key(self.config.redis_key_prefix)
            self._redis.setex(key, 90, json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    def _write_position_cache(self, symbol_bus: str, payload: dict | None) -> None:
        key = position_key(self.config.redis_key_prefix, symbol_bus)
        assert_key_allowed(key, prefix=self.config.redis_key_prefix)
        if payload is None:
            self._redis.delete(key)
        else:
            self._redis.setex(key, 300, json.dumps(payload))

    def _audit(
        self,
        conn: sqlite3.Connection,
        *,
        trade_id: str,
        action: str,
        symbol: str,
        qty: float | None,
        price: float | None,
        pre: dict,
        post: dict,
        reason: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO scalp_trade_audit
            (trade_id, action, symbol, exchange, strategy_id, qty, price,
             pre_ledger_json, post_ledger_json, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                action,
                symbol,
                self.config.exchange,
                self.config.strategy_id,
                qty,
                price,
                json.dumps(pre),
                json.dumps(post),
                reason,
            ),
        )

    def _record_momentum(self, snap: MarketSnapshot) -> None:
        _, epoch = self._now()
        self._momentum.record(snap.symbol_bus, epoch, snap.best_bid, snap.mid)

    def _entry_armed_ok(self) -> bool:
        """Calibration mode auto-arms paper entries; normal mode requires operator arm."""
        if self.config.calibration_mode:
            return True
        if self.config.scalp_paper_enabled and self.config.scalp_paper_auto_arm:
            return True
        return is_entry_armed(self._redis, prefix=self.config.redis_key_prefix)

    def _check_scalp_circuit_breaker(self) -> bool:
        """Check SCALP-specific circuit breaker. Returns True if circuit is open (halt new entries).

        Two independent conditions trigger the breaker:
        1. Today's closed-trade PnL is worse than -SCALP_DAILY_LOSS_LIMIT_PCT * principal.
        2. The last SCALP_MAX_CONSECUTIVE_LOSSES closed trades (after the eval floor)
           are all losses.

        Consecutive-loss trips persist a cooldown of SCALP_BREAKER_RECOVERY_SEC.
        After that window the engine starts a fresh consec evaluation; historical
        SELL rows stay in the book. A restart reads the persisted cooldown so an
        active trip cannot be erased. Daily-loss protection is not merged into
        this cooldown. SCALP_CIRCUIT_BREAKER_EPOCH remains an operator floor.
        """
        self._last_breaker_reason = ""
        self._last_breaker_recovery_until = ""
        self._last_breaker_eval_after = ""
        try:
            return run_locked_retry(self._evaluate_scalp_circuit_breaker)
        except sqlite3.OperationalError as e:
            self._last_breaker_reason = "SCALP_BREAKER_STATE_UNAVAILABLE"
            kind = "locked" if is_locked_error(e) else "operational"
            logger.error("[SCALP_CIRCUIT_BREAKER] state unavailable fail-closed (%s): %s", kind, e)
            return True
        except Exception as e:
            self._last_breaker_reason = "SCALP_BREAKER_STATE_UNAVAILABLE"
            logger.error("[SCALP_CIRCUIT_BREAKER] Check failed (fail-closed): %s", e)
            return True

    def _evaluate_scalp_circuit_breaker(self) -> bool:
        try:
            now = self._utcnow()
            today = now.strftime("%Y-%m-%d")
            epoch = (self.config.circuit_breaker_epoch or "").strip()
            recovery_sec = int(getattr(self.config, "breaker_recovery_sec", 14400) or 14400)
            if recovery_sec < 0:
                recovery_sec = 0
            with self._conn() as conn:
                ledger = self._ledger(conn)
                principal = float(ledger["principal"])

                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(pnl_usd), 0) AS today_pnl
                    FROM scalp_paper_trades
                    WHERE upper(side) = 'SELL'
                      AND pnl_usd IS NOT NULL
                      AND date(created_at) = ?
                      AND (? = '' OR created_at >= ?)
                    """,
                    (today, epoch, epoch),
                ).fetchone()
                today_pnl = float(row[0]) if row else 0.0
                daily_limit = self.config.daily_loss_limit_pct * principal
                if today_pnl <= -daily_limit:
                    self._last_breaker_reason = "DAILY_LOSS_LIMIT"
                    logger.warning(
                        "[SCALP_CIRCUIT_BREAKER] Daily loss limit hit: today_pnl=%.4f limit=-%.4f principal=%.2f halt=True",
                        today_pnl,
                        daily_limit,
                        principal,
                    )
                    return True

                state = self._load_consec_breaker_state(conn)
                eval_after = str(state.get("eval_after") or "").strip()
                recovery_until_raw = str(state.get("recovery_until") or "").strip()
                recovery_until_dt = self._parse_utc_ts(recovery_until_raw)
                trip_dt = self._parse_utc_ts(str(state.get("tripped_at") or ""))
                epoch_dt = self._parse_utc_ts(epoch)
                if (
                    recovery_until_dt is not None
                    and now < recovery_until_dt
                    and epoch_dt is not None
                    and trip_dt is not None
                    and epoch_dt > trip_dt
                ):
                    self._save_consec_breaker_state(
                        conn,
                        tripped_at="",
                        recovery_until="",
                        eval_after=eval_after,
                        reason="",
                    )
                    recovery_until_dt = None
                    recovery_until_raw = ""
                    logger.info(
                        "[SCALP_CIRCUIT_BREAKER] operator epoch=%s cleared trip_at=%s",
                        epoch,
                        state.get("tripped_at"),
                    )

                if recovery_until_dt is not None and now < recovery_until_dt:
                    self._last_breaker_reason = "CONSECUTIVE_LOSSES_COOLDOWN"
                    self._last_breaker_recovery_until = recovery_until_raw
                    self._last_breaker_eval_after = eval_after
                    logger.warning(
                        "[SCALP_CIRCUIT_BREAKER] consec cooldown until=%s halt=True",
                        recovery_until_raw,
                    )
                    return True

                if recovery_until_dt is not None and now >= recovery_until_dt:
                    eval_after = recovery_until_raw
                    self._save_consec_breaker_state(
                        conn,
                        tripped_at="",
                        recovery_until="",
                        eval_after=eval_after,
                        reason="",
                    )
                    logger.info(
                        "[SCALP_CIRCUIT_BREAKER] consec cooldown expired eval_after=%s",
                        eval_after,
                    )

                floor = self._consec_eval_floor(epoch, eval_after)
                max_consec = int(self.config.max_consecutive_losses)
                recent_rows = conn.execute(
                    """
                    SELECT pnl_usd, created_at FROM scalp_paper_trades
                    WHERE upper(side) = 'SELL' AND pnl_usd IS NOT NULL
                      AND (? = '' OR created_at >= ?)
                    ORDER BY id DESC LIMIT ?
                    """,
                    (floor, floor, max_consec),
                ).fetchall()
                if len(recent_rows) >= max_consec and all(float(r[0]) <= 0.0 for r in recent_rows):
                    newest_loss_at = str(recent_rows[0][1] or "")
                    trip_dt = self._parse_utc_ts(newest_loss_at) or now
                    rec_until_dt = trip_dt + timedelta(seconds=recovery_sec)
                    rec_until_iso = rec_until_dt.strftime("%Y-%m-%d %H:%M:%S")
                    if now < rec_until_dt:
                        self._save_consec_breaker_state(
                            conn,
                            tripped_at=newest_loss_at,
                            recovery_until=rec_until_iso,
                            eval_after=eval_after,
                            reason="CONSECUTIVE_LOSSES",
                        )
                        self._last_breaker_reason = "CONSECUTIVE_LOSSES_COOLDOWN"
                        self._last_breaker_recovery_until = rec_until_iso
                        self._last_breaker_eval_after = eval_after
                        logger.warning(
                            "[SCALP_CIRCUIT_BREAKER] %d consecutive losses trip_at=%s until=%s halt=True",
                            max_consec,
                            newest_loss_at,
                            rec_until_iso,
                        )
                        return True
                    self._save_consec_breaker_state(
                        conn,
                        tripped_at="",
                        recovery_until="",
                        eval_after=rec_until_iso,
                        reason="",
                    )
                    self._last_breaker_eval_after = rec_until_iso
                    logger.info(
                        "[SCALP_CIRCUIT_BREAKER] stale consec streak recovered eval_after=%s",
                        rec_until_iso,
                    )
                    return False

                self._last_breaker_eval_after = eval_after
                return False
        except Exception:
            raise

    def _entry_candidates(
        self,
        conn: sqlite3.Connection,
        *,
        pre_ranked: list[dict] | None = None,
    ) -> list[tuple[str, MarketSnapshot, object]]:
        if not self.config.scalp_paper_enabled:
            self._publish_last_decision(decision="BLOCKED", reason=SCALP_PAPER_DISABLED)
            return []

        open_symbols = {str(r[0]).upper() for r in conn.execute("SELECT symbol FROM scalp_paper_positions WHERE status='OPEN'")}

        _, epoch = self._now()
        notional = min(self.config.max_notional_paper, float(self._ledger(conn)["cash_balance"]))

        # When the signal engine flag is on (default for paper), go through the
        # clean facade. For v1 this still uses the router + existing gate/strategies
        # but gives us a single place to swap in future lab-validated entry logic
        # (exactly parallel to how the day all-weather engine replaced the old router
        # when ALLWEATHER_ENGINE_ENABLED=true).
        ranked: list[dict] = []
        if pre_ranked is not None:
            ranked = list(pre_ranked)
        else:
            try:
                from backend.services.binance_scalp import scalp_signal_engine as _se

                if _se.scalp_signal_engine_enabled():
                    _se.bind_paper_engine(router=self._router, momentum=self._momentum, klines=self._klines)
                    ranked = self._router.evaluate_all(epoch=epoch, notional_usd=notional) or []
            except Exception:
                ranked = []

            if not ranked:
                ranked = self._router.evaluate_all(epoch=epoch, notional_usd=notional) or []

        self._pending_opportunity_rows = ranked
        self._pending_opportunity_epoch = epoch

        if open_symbols:
            ranked = [r for r in ranked if str(r.get("symbol") or "").upper() not in open_symbols]

        for row in ranked:
            sym = row["symbol"]
            snap = row["snap"]
            self._record_momentum(snap)
            mom = self._momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
            row["mom"] = mom
            bars = self._klines.get(sym)
            row["micro_regime"] = self._router._current_regime(sym, epoch, bars)

        try:
            from backend.services.scalp_ai_rank_enrichment import enrich_scalp_ranked_candidates

            ranked = enrich_scalp_ranked_candidates(ranked, redis_client=self._redis) or ranked
        except Exception as exc:
            logger.debug("SCALP intelligence enrich skipped: %s", exc)

        try:
            from backend.services.binance_scalp.scalp_entry_telemetry import publish_entry_telemetry

            publish_entry_telemetry(self._redis, ranked, prefix=self.config.redis_key_prefix)
        except Exception as exc:
            logger.debug("SCALP entry telemetry skipped: %s", exc)

        for row in ranked:
            sym = row["symbol"]
            snap = row["snap"]
            self._write_market_cache(snap)
            self._record_momentum(snap)
            sig = row["signal"]
            self._write_signal(
                sym,
                {
                    "symbol": sym,
                    "strategy_id": self.config.strategy_id,
                    "exchange": self.config.exchange,
                    "side": "BUY",
                    "setup": sig.as_dict(),
                    "all_setups": row["all_signals"],
                    "paper_only": True,
                    "live_blocked": True,
                },
            )

        if not ranked:
            for sym in self.config.products:
                snap = self.reader.read(sym)
                if snap is None:
                    continue
                _, all_sigs, meta = self._router.evaluate_symbol(sym, epoch=epoch, notional_usd=notional, snap=snap)
                reason = meta.get("hard_block") or meta.get("soft_reason") or f"RANK_BELOW_MIN:{meta.get('best_rank_score')}"
                self._record_reject(
                    conn,
                    sym,
                    "BUY",
                    reason,
                    json.dumps({"rank_meta": meta, "signals": [s.as_dict() for s in all_sigs]}),
                )
            self._publish_last_decision(decision="NO_SIGNAL", reason="NO_RANKED_CANDIDATE", entry_armed=self._entry_armed_ok())
            return []

        from backend.services.binance_scalp.scalp_candidate_ranking import (
            HOLD_ACTION_EV,
            attach_action_predictions,
            pick_best_global_candidate,
            rank_actions_with_hold,
        )

        for row in ranked:
            attach_action_predictions(row)
        best = pick_best_global_candidate(ranked)
        with contextlib.suppress(Exception):
            from backend.services.decision_book_tape import record_scalp_cycle

            record_scalp_cycle(
                ranked=ranked,
                chosen=best,
                redis_client=self._redis,
                hold_ev=HOLD_ACTION_EV,
            )
        if best is None:
            eligible_rows = [r for r in ranked if r.get("entry_eligible")]
            # Prefer an eligible row for reject reason; fall back to rank leader.
            top = eligible_rows[0] if eligible_rows else (ranked[0] if ranked else {})
            meta = top.get("rank_meta") or {}
            top_score = float(top.get("rank_score") or 0)
            second_score = float(eligible_rows[1].get("rank_score") or 0) if len(eligible_rows) > 1 else 0.0
            actions = rank_actions_with_hold(eligible_rows) if eligible_rows else []
            best_buy_ev = max((float(r.get("expected_net_ev") or 0) for r in eligible_rows), default=None)
            hold_won = bool(eligible_rows) and (best_buy_ev is None or best_buy_ev <= HOLD_ACTION_EV)
            reason = (
                "HOLD_WINS_ACTION_RANK"
                if hold_won
                else (
                    "RANK_LOW_CONFIDENCE_TIE"
                    if eligible_rows and top_score < float(os.getenv("SCALP_MIN_CONFIDENT_RANK", "1.55"))
                    else top.get("hard_block") or meta.get("hard_block") or top.get("soft_reason") or meta.get("soft_reason") or f"RANK_BELOW_MIN:{top.get('rank_score')}"
                )
            )
            if not eligible_rows and ranked:
                reason = top.get("hard_block") or meta.get("hard_block") or top.get("soft_reason") or meta.get("soft_reason") or "NO_ENTRY_ELIGIBLE"
            self._record_reject(
                conn,
                top.get("symbol") or (self.config.products[0] if self.config.products else "UNKNOWN"),
                "BUY",
                reason,
                json.dumps(
                    {
                        "rank_meta": meta,
                        "best_global": {
                            "symbol": top.get("symbol"),
                            "setup": top.get("best_setup"),
                            "rank_score": top.get("rank_score"),
                            "hard_block": top.get("hard_block"),
                            "entry_eligible": bool(top.get("entry_eligible")),
                            "rank_margin": round(top_score - second_score, 4),
                        },
                        "action_rank": [
                            {
                                "action": r.get("action_name"),
                                "symbol": r.get("symbol"),
                                "expected_net_ev": r.get("expected_net_ev"),
                                "predicted_prob_positive_net": r.get("predicted_prob_positive_net"),
                                "expected_mfe": r.get("expected_mfe"),
                                "expected_mae": r.get("expected_mae"),
                                "rank_score": r.get("rank_score"),
                            }
                            for r in actions
                        ],
                        "hold_action_ev": HOLD_ACTION_EV,
                        "all_symbols": [
                            {
                                "symbol": r["symbol"],
                                "rank_score": r.get("rank_score"),
                                "entry_eligible": r.get("entry_eligible"),
                                "best_setup": r.get("best_setup"),
                                "hard_block": r.get("hard_block"),
                                "soft_reason": r.get("soft_reason"),
                                "expected_net_ev": r.get("expected_net_ev"),
                                "predicted_net_return": r.get("predicted_net_return"),
                                "predicted_prob_positive_net": r.get("predicted_prob_positive_net"),
                                "expected_mfe": r.get("expected_mfe"),
                                "expected_mae": r.get("expected_mae"),
                                "expected_hold": r.get("expected_hold"),
                            }
                            for r in ranked
                        ],
                    }
                ),
            )
            ranked_summary = [
                {
                    "symbol": r["symbol"],
                    "rank_score": r.get("rank_score"),
                    "entry_eligible": r.get("entry_eligible"),
                    "hard_block": r.get("hard_block"),
                    "expected_net_ev": r.get("expected_net_ev"),
                    "predicted_net_return": r.get("predicted_net_return"),
                    "predicted_prob_positive_net": r.get("predicted_prob_positive_net"),
                    "expected_mfe": r.get("expected_mfe"),
                    "expected_mae": r.get("expected_mae"),
                    "expected_hold": r.get("expected_hold"),
                }
                for r in ranked
            ]
            self._publish_last_decision(
                decision="NO_SIGNAL",
                reason=reason,
                selected_symbol=top.get("symbol"),
                rank_score=top.get("rank_score"),
                entry_armed=self._entry_armed_ok(),
                ranked_summary=ranked_summary,
            )
            return []
        sym, snap, sig = best["symbol"], best["snap"], best["signal"]
        ranked_summary = [
            {
                "symbol": r["symbol"],
                "rank_score": r.get("rank_score"),
                "entry_eligible": r.get("entry_eligible"),
                "hard_block": r.get("hard_block"),
                "expected_net_ev": r.get("expected_net_ev"),
                "predicted_net_return": r.get("predicted_net_return"),
                "predicted_prob_positive_net": r.get("predicted_prob_positive_net"),
                "expected_mfe": r.get("expected_mfe"),
                "expected_mae": r.get("expected_mae"),
                "expected_hold": r.get("expected_hold"),
            }
            for r in ranked
        ]
        # Architecture v2 (2026-08-11): sig.passed is no longer a promotion
        # requirement here — pick_best_global_candidate() already only
        # returns candidates with entry_eligible=True, meaning every
        # mechanical safety hard_block (stale data, bad spread/impact, no
        # net edge, duplicate position, exposure cap) already cleared for
        # this candidate. A strategy-rejected (soft-rank) pick is executable;
        # scalp_dynamic_sizing.py is what protects capital on it, not a
        # second permission check here.
        if not bool(best.get("entry_eligible")):
            self._record_reject(
                conn,
                sym,
                "BUY",
                best.get("hard_block") or "RANKED_NOT_ELIGIBLE",
                json.dumps({"setup": sig.as_dict(), "rank_score": best.get("rank_score")}),
            )
            self._publish_last_decision(
                decision="BLOCKED",
                reason=str(best.get("hard_block") or "RANKED_NOT_ELIGIBLE"),
                selected_symbol=sym,
                rank_score=best.get("rank_score"),
                entry_armed=self._entry_armed_ok(),
                ranked_summary=ranked_summary,
            )
            return []
        entry_intel = dict(best.get("intelligence") or {})
        self._last_ranking_meta = {
            "selection_reason": f"{sig.setup_name} rank={best.get('rank_score')} score={sig.score:.2f} {sig.entry_reason}",
            "selected_symbol": sym,
            "rank_score": best.get("rank_score"),
            "entry_eligible": True,
            "ranking": [r["signal"].as_dict() for r in ranked],
            # Selected symbol's full per-symbol diagnostics (regime, mtf_5m_trend_pct,
            # mtf_5m_aligned, etc.) so these survive into the persisted trade's
            # symbol_ranking for later analysis — previously only the summary below
            # (rank_score/entry_eligible/best_setup/hard_block) was kept.
            "rank_meta": best.get("rank_meta"),
            "ranked_summary": [
                {
                    "symbol": r["symbol"],
                    "rank_score": r.get("rank_score"),
                    "entry_eligible": r.get("entry_eligible"),
                    "best_setup": r.get("best_setup"),
                    "hard_block": r.get("hard_block"),
                    "expected_net_ev": r.get("expected_net_ev"),
                    "predicted_net_return": r.get("predicted_net_return"),
                    "predicted_prob_positive_net": r.get("predicted_prob_positive_net"),
                    "expected_mfe": r.get("expected_mfe"),
                    "expected_mae": r.get("expected_mae"),
                    "expected_hold": r.get("expected_hold"),
                }
                for r in ranked
            ],
            "selected_action": f"BUY_{sym}",
            "hold_action_ev": HOLD_ACTION_EV,
            "selected_expected_net_ev": best.get("expected_net_ev"),
            "selected_predicted_prob_positive_net": best.get("predicted_prob_positive_net"),
            "selected_expected_mfe": best.get("expected_mfe"),
            "selected_expected_mae": best.get("expected_mae"),
            "selected_expected_hold": best.get("expected_hold"),
            "forward_net_model_version": best.get("forward_net_model_version") or "",
            "scalp_intelligence": entry_intel,
        }
        logger.info("SCALP_STRATEGY_PICK %s", self._last_ranking_meta["selection_reason"])

        if not self._entry_armed_ok():
            self._record_reject(
                conn,
                sym,
                "BUY",
                WOULD_ENTER_NOT_ARMED,
                json.dumps({"setup": sig.as_dict(), "entry_armed": False}),
            )
            self._publish_last_decision(
                decision="PASS_NOT_ARMED",
                reason=WOULD_ENTER_NOT_ARMED,
                selected_symbol=sym,
                rank_score=best.get("rank_score"),
                entry_armed=False,
                ranked_summary=ranked_summary,
            )
            return []

        self._publish_last_decision(
            decision="WOULD_ENTER",
            reason="",
            selected_symbol=sym,
            rank_score=best.get("rank_score"),
            entry_armed=True,
            ranked_summary=ranked_summary,
        )
        return [(sym, snap, sig)]

    def _try_entry(self, conn: sqlite3.Connection, *, pre_ranked: list[dict] | None = None) -> None:
        open_count = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
        pending_slots = len(self._entry_reservations)
        if open_count + pending_slots >= self.config.max_open_positions:
            self._record_gate(gate_id="CASH_OR_SLOTS", reason="MAX_OPEN_POSITIONS", detail=f"open={open_count} reserved={pending_slots}")
            return

        breaker_open = self._check_scalp_circuit_breaker()
        self._last_circuit_breaker_open = bool(breaker_open)
        # Measure and store every cycle even when the breaker blocks execution.
        candidates = self._entry_candidates(conn, pre_ranked=pre_ranked)
        if breaker_open:
            self._publish_last_decision(decision="BLOCKED", reason="SCALP_CIRCUIT_BREAKER_OPEN")
            self._record_gate(gate_id="SCALP_CIRCUIT_BREAKER", reason="SCALP_CIRCUIT_BREAKER_OPEN")
            return
        if not candidates:
            return

        sym, snap, sig = candidates[0]
        # Architecture v2 (2026-08-11): entry_eligible (mechanical hard_block
        # only) already gated this candidate in _entry_candidates(). A
        # soft-rank / opinion-conflicted pick is executable — it must size
        # small via scalp_dynamic_sizing.py below, not be refused here.

        if sym.upper() in self._entry_reservations:
            self._record_reject(conn, sym, "BUY", "ENTRY_RESERVED", json.dumps({"symbol": sym}))
            return

        if conn.execute(
            "SELECT 1 FROM scalp_paper_positions WHERE symbol = ? AND status = 'OPEN' LIMIT 1",
            (sym,),
        ).fetchone():
            self._record_reject(conn, sym, "BUY", "SYMBOL_ALREADY_OPEN", json.dumps({"symbol": sym}))
            return

        ledger = self._ledger(conn)
        reserved_n = sum(float((r or {}).get("notional") or 0.0) for r in self._entry_reservations.values())
        free_cash = float(ledger["cash_balance"]) - reserved_n
        # Per-symbol caps (e.g. SOLUSDT:50) remain the mechanical exposure
        # ceiling. scalp_dynamic_sizing.py scales *down* from this ceiling
        # using confidence/arm/MTF/regime/volatility/liquidity evidence —
        # it never raises above it and never re-blocks the trade.
        base_cap = min(self.config.notional_cap_for_symbol(sym), free_cash)
        if base_cap < 1.0:
            self._record_reject(conn, sym, "BUY", "INSUFFICIENT_CASH", f"cash={ledger['cash_balance']} reserved={reserved_n}")
            return

        ranking_meta_for_size = getattr(self, "_last_ranking_meta", {}) or {}
        try:
            _, epoch_for_size = self._now()
            realized_vol = self._momentum.diagnostics(sym, epoch_for_size, snap.best_bid, snap.mid).realized_volatility_pct
        except Exception:
            realized_vol = None
        from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size

        try:
            from backend.services.ai_calibration_tracker import calibration_confidence_multiplier

            cal_mult, _cal_reason = calibration_confidence_multiplier(sym)
        except Exception:
            cal_mult = 1.0

        sizing = compute_scalp_position_size(
            base_cap=base_cap,
            free_cash=free_cash,
            min_notional=5.0,
            strategy_passed=bool(getattr(sig, "passed", False)),
            arm_penalty_mult=float(ranking_meta_for_size.get("arm_penalty_mult", 1.0) or 1.0),
            mtf_penalty_mult=float(ranking_meta_for_size.get("mtf_penalty_mult", 1.0) or 1.0),
            regime_mismatch=bool(ranking_meta_for_size.get("regime_mismatch", False)),
            symbol_stall_risk=bool(ranking_meta_for_size.get("symbol_stall_risk", False)),
            spread_pct=float(getattr(sig, "spread_pct", 0.0) or 0.0),
            impact_pct=float(getattr(sig, "impact_pct", 0.0) or 0.0),
            realized_volatility_pct=realized_vol,
            calibration_mult=cal_mult,
            micro_quality_mult=float(ranking_meta_for_size.get("micro_quality_mult", 1.0) or 1.0),
        )
        notional = sizing.notional
        if notional < 1.0:
            self._record_reject(conn, sym, "BUY", "INSUFFICIENT_CASH", f"cash={ledger['cash_balance']} reserved={reserved_n} sizing={sizing.reasoning}")
            return
        with contextlib.suppress(Exception):
            self._record_gate(
                gate_id="DYNAMIC_SIZE_APPLIED",
                symbol=sym,
                outcome="ranked",
                setup=sig.setup_name,
                entry_price=float(getattr(sig, "limit_buy_price", 0.0) or 0.0),
                detail=sizing.reasoning,
            )

        limit_buy = sig.limit_buy_price
        qty = notional / limit_buy
        fee = notional * self.econ.entry_fee_pct()
        slip = notional * self.econ.slippage_buffer_pct
        trade_id = f"scalp_paper_{sym}_{int(time.time() * 1000)}"
        ts, epoch = self._now()
        pre = dict(ledger)

        # Atomic cash/slot/symbol reservation before paper fill
        self._entry_reservations[sym.upper()] = {"notional": float(notional + fee + slip), "ts": time.time(), "trade_id": trade_id}

        ranking_meta = getattr(self, "_last_ranking_meta", {}) or {}
        entry_intel = dict(ranking_meta.get("scalp_intelligence") or {})
        from backend.services.day_trade_thesis import scalp_strategy_to_thesis

        thesis_fields = scalp_strategy_to_thesis(sig.setup_name, sig.setup_context or {})
        strategy_passed = bool(getattr(sig, "passed", False))
        soft_rank_entry = bool((sig.setup_context or {}).get("soft_rank_entry", not strategy_passed))
        entry_diag = {
            "setup_name": sig.setup_name,
            "setup_context": sig.setup_context,
            "setup_signal": sig.as_dict(),
            "passed": strategy_passed,
            "soft_rank_entry": soft_rank_entry,
            "entry_eligible": True,
            "entry_owner": "strategy" if strategy_passed else "ranking_ev",
            "ml_role": "rank_size",
            "decision_policy_version": "scalp_path_aware_v1",
            "bar_closed": True,
            "selected_symbol": sym,
            "symbol_ranking": ranking_meta,
            # Top-level rank_score for learning writer (not only nested under symbol_ranking).
            "rank_score": ranking_meta.get("rank_score") if ranking_meta.get("rank_score") is not None else ranking_meta.get("best_rank_score"),
            "selection_confidence": ranking_meta.get("selection_confidence"),
            "review_lows": [],
            "session_low_bid": limit_buy,
            "entry_time": ts,
            "spread_at_entry": float(snap.spread_pct),
            "dynamic_sizing": sizing.reasoning,
            "dynamic_sizing_multiplier": sizing.combined_multiplier,
            "base_cap": base_cap,
            **thesis_fields,
            **entry_intel,
        }
        if entry_intel.get("feature_health_json") and "entry_scalp_vector" not in entry_diag:
            entry_diag["entry_scalp_vector"] = entry_intel.get("entry_scalp_vector") or []
        if entry_intel.get("scalp_setup"):
            entry_diag["scalp_setup"] = entry_intel.get("scalp_setup")
        if entry_intel.get("micro_regime"):
            entry_diag["micro_regime"] = entry_intel.get("micro_regime")
        with contextlib.suppress(Exception):
            from backend.services.binance_scalp.scalp_micro_contract import version_stamps

            entry_diag.update(version_stamps())
            ctx = sig.setup_context or {}
            for k in (
                "microstructure_features",
                "EV_1s",
                "EV_5s",
                "EV_10s",
                "EV_30s",
                "EV_60s",
                "p_positive_executable_net_5s",
                "p_positive_executable_net_10s",
                "p_positive_executable_net_30s",
                "p_adverse_move",
                "adverse_selection_score",
                "selection_micro_score",
                "calibration_status",
            ):
                if ctx.get(k) is not None:
                    entry_diag[k] = ctx.get(k)
            if ctx.get("microstructure_features"):
                entry_diag.update({f"micro_{kk}": vv for kk, vv in (ctx.get("microstructure_features") or {}).items()})

        # Entry-time market-role context (Redis — cross-process; for learning attribution)
        _role_ctx_snap = "{}"
        try:
            from backend.services.market_role_intelligence import get_role_context_snapshot_json

            _role_ctx_snap = get_role_context_snapshot_json(sym) or "{}"
        except Exception:
            _role_ctx_snap = "{}"
        entry_diag["context_snapshot_json"] = _role_ctx_snap
        _scalp_prov: dict = {}
        with contextlib.suppress(Exception):
            from backend.services.entry_decision_authority import build_scalp_entry_provenance

            _feat_fp = ""
            _vec = (ranking_meta.get("rank_meta") or {}).get("feature_vector") or entry_intel.get("entry_scalp_vector") or []
            if _vec:
                _feat_fp = ",".join(str(round(float(x), 6)) for x in list(_vec)[:12])
            _scalp_prov = build_scalp_entry_provenance(
                ranking_meta=ranking_meta,
                symbol=sym,
                setup_name=sig.setup_name,
                strategy_passed=strategy_passed,
                epoch=epoch,
                feature_fingerprint=_feat_fp,
            )
            entry_diag.update(_scalp_prov)

        try:
            conn.execute(
                """
                INSERT INTO scalp_paper_trades
                (trade_id, symbol, exchange, strategy_id, side, quantity, price, notional,
                 fee_usd, slippage_usd, diagnostics_json)
                VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    sym,
                    self.config.exchange,
                    sig.setup_name,
                    qty,
                    limit_buy,
                    notional,
                    fee,
                    slip,
                    json.dumps(
                        {
                            "setup": sig.as_dict(),
                            "setup_signal": sig.as_dict(),
                            "paper_limit": True,
                            "selected_symbol": sym,
                            "passed": strategy_passed,
                            "soft_rank_entry": soft_rank_entry,
                            "entry_eligible": True,
                            "entry_owner": "strategy" if strategy_passed else "ranking_ev",
                            "ml_role": "rank_size",
                            "decision_policy_version": _scalp_prov.get("entry_policy_version") or "scalp_path_aware_v1",
                            "bar_closed": True,
                            "setup_name": sig.setup_name,
                            "rank_score": ranking_meta.get("rank_score"),
                            "selection_confidence": ranking_meta.get("selection_confidence"),
                            "context_snapshot_json": _role_ctx_snap,
                            "dynamic_sizing": sizing.reasoning,
                            "predicted_net_ev": ranking_meta.get("selected_expected_net_ev"),
                            "predicted_net_return": ranking_meta.get("selected_expected_net_ev"),
                            "predicted_prob_positive_net": ranking_meta.get("selected_predicted_prob_positive_net"),
                            "predicted_mfe": ranking_meta.get("selected_expected_mfe"),
                            "predicted_mae": ranking_meta.get("selected_expected_mae"),
                            "predicted_horizon": ranking_meta.get("selected_expected_hold"),
                            "forward_net_model_version": ranking_meta.get("forward_net_model_version") or "",
                            **_scalp_prov,
                            **{
                                k: entry_diag[k]
                                for k in (
                                    "feature_version",
                                    "microstructure_version",
                                    "selection_version",
                                    "model_version",
                                    "feature_set_version",
                                    "microstructure_features",
                                    "EV_1s",
                                    "EV_5s",
                                    "EV_10s",
                                    "EV_30s",
                                    "EV_60s",
                                    "p_positive_executable_net_5s",
                                    "p_positive_executable_net_10s",
                                    "p_positive_executable_net_30s",
                                    "p_adverse_move",
                                    "selection_micro_score",
                                    "calibration_status",
                                )
                                if k in entry_diag
                            },
                        }
                    ),
                ),
            )
            conn.execute(
                """
                INSERT INTO scalp_paper_positions
                (symbol, exchange, strategy_id, quantity, entry_price, entry_time,
                 entry_time_epoch, trade_id, paper_order_id, status, state,
                 max_favorable_pct, max_adverse_pct, stale_review_count,
                 session_low_bid, diagnostics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 'OPEN', 0, 0, 0, ?, ?)
                """,
                (
                    sym,
                    self.config.exchange,
                    sig.setup_name,
                    qty,
                    limit_buy,
                    ts,
                    epoch,
                    trade_id,
                    f"paper_order_{trade_id}",
                    limit_buy,
                    json.dumps(entry_diag),
                ),
            )
            new_cash = float(ledger["cash_balance"]) - notional - fee - slip
            pos_val = qty * limit_buy
            conn.execute(
                """
                UPDATE scalp_paper_ledger SET
                  cash_balance = ?,
                  positions_value = positions_value + ?,
                  total_equity = ? + positions_value + ?,
                  updated_at = datetime('now')
                WHERE id = 1
                """,
                (new_cash, pos_val, new_cash, pos_val),
            )
            post = dict(self._ledger(conn))
            self._audit(
                conn,
                trade_id=trade_id,
                action="BUY",
                symbol=sym,
                qty=qty,
                price=limit_buy,
                pre=pre,
                post=post,
                reason=f"SCALP_PAPER_ENTRY_{sig.setup_name}",
            )
            self._write_position_cache(
                sym,
                {
                    "symbol": sym,
                    "quantity": qty,
                    "entry_price": limit_buy,
                    "setup_name": sig.setup_name,
                    "trade_id": trade_id,
                    "strategy_id": sig.setup_name,
                    "paper_only": True,
                },
            )
            logger.info("SCALP_PAPER_BUY %s setup=%s qty=%.8f price=%.4f", sym, sig.setup_name, qty, limit_buy)
            with contextlib.suppress(Exception):
                from backend.services.binance_scalp.scalp_markout import schedule_markout

                schedule_markout(
                    kind="entry",
                    symbol=sym,
                    side="BUY",
                    mid=float(snap.mid or limit_buy),
                    entry_px=float(limit_buy),
                    qty=float(qty),
                    notional=float(notional),
                    fee_pct=float(self.econ.entry_fee_pct() + self.econ.exit_fee_pct()),
                    slip_pct=float(self.econ.slippage_buffer_pct),
                    extra=dict((sig.setup_context or {}).get("microstructure_features") or {}),
                )
            set_entry_armed(self._redis, prefix=self.config.redis_key_prefix, armed=False)
            with contextlib.suppress(Exception):
                from backend.services.scalp_gate_telemetry import record_gate_event

                record_gate_event(
                    self.config.database_path,
                    gate_id="STRATEGY_PASS",
                    symbol=sym,
                    outcome="passed",
                    setup=sig.setup_name,
                    detail="paper_fill",
                )
        finally:
            self._entry_reservations.pop(sym.upper(), None)

    def shutdown(self) -> None:
        clear_entry_armed(self._redis, prefix=self.config.redis_key_prefix)

    def _record_position_review(
        self,
        conn: sqlite3.Connection,
        *,
        trade_id: str,
        sym: str,
        review_diag: dict,
    ) -> None:
        d = review_diag
        conn.execute(
            """
            INSERT INTO scalp_position_reviews
            (trade_id, symbol, hold_seconds, current_bid, entry_price,
             executable_net_pct, max_favorable_pct, max_adverse_pct,
             recovery_from_low_pct, bid_change_15s, bid_change_30s, bid_change_60s,
             higher_lows, spread_pct, decision, state, reason, diagnostics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                sym,
                float(d.get("hold_seconds") or 0),
                float(d.get("current_bid") or 0),
                float(d.get("entry_price") or 0),
                float(d.get("executable_net_pct") or 0),
                float(d.get("max_favorable_pct") or 0),
                float(d.get("max_adverse_pct") or 0),
                float(d.get("recovery_from_low_pct") or 0),
                float(d.get("bid_change_15s") or 0),
                float(d.get("bid_change_30s") or 0),
                float(d.get("bid_change_60s") or 0),
                1 if d.get("higher_lows") else 0,
                float(d.get("spread_pct") or 0),
                str(d.get("decision") or ""),
                str(d.get("state") or ""),
                str(d.get("reason") or ""),
                json.dumps(d),
            ),
        )

    def _persist_position_track(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        track: PositionTrack,
        *,
        ts: str,
        reason: str,
        review_lows: tuple[float, ...],
        touch_review_ts: bool = False,
    ) -> None:
        raw_diag = row["diagnostics_json"]
        diag = json.loads(raw_diag) if raw_diag else {}
        diag["review_lows"] = list(review_lows)
        diag["session_low_bid"] = track.session_low_bid
        # Only advance last_review_ts when a timed review actually ran; updating every
        # tick prevented SCALP_REVIEW_INTERVAL_SEC from ever elapsing (stale_review_count stuck at 1).
        last_review_ts = ts if touch_review_ts else (row["last_review_ts"] or ts)
        conn.execute(
            """
            UPDATE scalp_paper_positions SET
              state = ?,
              max_favorable_pct = ?,
              max_adverse_pct = ?,
              stale_review_count = ?,
              session_low_bid = ?,
              last_review_ts = ?,
              last_state_reason = ?,
              diagnostics_json = ?
            WHERE id = ?
            """,
            (
                track.state,
                track.max_favorable_pct,
                track.max_adverse_pct,
                track.stale_review_count,
                track.session_low_bid,
                last_review_ts,
                reason,
                json.dumps(diag),
                row["id"],
            ),
        )

    def _execute_sell(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        sym: str,
        entry: float,
        qty: float,
        exit_price: float,
        net_pct: float,
        net_usd: float,
        reason: str,
        pf_dict: dict,
        exit_gate: dict,
        hold_seconds: float = 0.0,
        spread_at_exit: float = 0.0,
    ) -> None:
        trade_id = str(row["trade_id"])
        sell_tid = f"{trade_id}_SELL"
        strategy_id = self._position_strategy_id(row)
        notional = qty * exit_price
        fee = notional * self.econ.exit_fee_pct()
        slip = notional * self.econ.slippage_buffer_pct
        ledger = self._ledger(conn)
        pre = dict(ledger)

        entry_diag: dict = {}
        try:
            raw_entry_diag = row["diagnostics_json"]
            entry_diag = json.loads(raw_entry_diag) if raw_entry_diag else {}
        except Exception:
            entry_diag = {}
        setup_signal = entry_diag.get("setup_signal") if isinstance(entry_diag, dict) else None
        if not isinstance(setup_signal, dict):
            setup_signal = entry_diag.get("setup") if isinstance(entry_diag, dict) else None
        if not isinstance(setup_signal, dict):
            setup_signal = {}
        # Backfill tags from matching BUY row when older position diags omit them.
        buy_diag: dict = {}
        try:
            buy_row = conn.execute(
                "SELECT diagnostics_json FROM scalp_paper_trades WHERE trade_id = ? AND upper(side) = 'BUY' LIMIT 1",
                (trade_id,),
            ).fetchone()
            if buy_row and buy_row[0]:
                buy_diag = json.loads(buy_row[0]) if isinstance(buy_row[0], str) else {}
        except Exception:
            buy_diag = {}
        if not isinstance(buy_diag, dict):
            buy_diag = {}
        passed = bool(setup_signal.get("passed", entry_diag.get("passed", buy_diag.get("passed", False))))
        soft_rank_entry = bool(
            setup_signal.get(
                "soft_rank_entry",
                entry_diag.get("soft_rank_entry", buy_diag.get("soft_rank_entry", False)),
            )
        )
        sell_diag = {
            "preflight": pf_dict,
            "paper_limit": True,
            "exit_gate": exit_gate,
            "passed": passed,
            "soft_rank_entry": soft_rank_entry,
            "entry_eligible": bool(passed) and not soft_rank_entry,
            "setup_name": entry_diag.get("setup_name") or buy_diag.get("setup_name") or strategy_id,
            "entry_setup_signal": setup_signal,
            "scalp_setup": entry_diag.get("scalp_setup"),
            "micro_regime": entry_diag.get("micro_regime"),
        }
        from backend.services.entry_decision_authority import copy_entry_provenance

        sell_diag = copy_entry_provenance(entry_diag, sell_diag)
        sell_diag = copy_entry_provenance(buy_diag, sell_diag)

        conn.execute(
            """
            INSERT INTO scalp_paper_trades
            (trade_id, symbol, exchange, strategy_id, side, quantity, price, notional,
             fee_usd, slippage_usd, pnl_usd, pnl_pct, entry_price, exit_reason, diagnostics_json)
            VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sell_tid,
                sym,
                self.config.exchange,
                strategy_id,
                qty,
                exit_price,
                notional,
                fee,
                slip,
                net_usd,
                net_pct,
                entry,
                reason,
                json.dumps(sell_diag),
            ),
        )
        conn.execute(
            "UPDATE scalp_paper_positions SET status='CLOSED', state=? WHERE id=?",
            (reason, row["id"]),
        )
        pos_cost = entry * qty
        new_cash = float(ledger["cash_balance"]) + notional - fee - slip
        new_pos_val = max(0.0, float(ledger["positions_value"]) - pos_cost)
        new_equity = new_cash + new_pos_val
        conn.execute(
            """
            UPDATE scalp_paper_ledger SET
              cash_balance = ?,
              positions_value = ?,
              realized_pnl = realized_pnl + ?,
              total_equity = ?,
              updated_at = datetime('now')
            WHERE id = 1
            """,
            (new_cash, new_pos_val, net_usd, new_equity),
        )
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        win = 1 if net_usd > 0 else 0
        loss = 0 if net_usd > 0 else 1
        conn.execute(
            """
            INSERT INTO scalp_scoreboard_daily (day, trades, wins, losses, net_pnl)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
              trades = trades + 1,
              wins = wins + ?,
              losses = losses + ?,
              net_pnl = net_pnl + ?,
              updated_at = datetime('now')
            """,
            (day, win, loss, net_usd, win, loss, net_usd),
        )
        post = dict(self._ledger(conn))
        self._audit(
            conn,
            trade_id=sell_tid,
            action="SELL",
            symbol=sym,
            qty=qty,
            price=exit_price,
            pre=pre,
            post=post,
            reason=reason,
        )
        self._write_position_cache(sym, None)
        # Log only after caller commits — avoid false SCALP_PAPER_SELL when txn rolls back.
        self._pending_sell_log = (sym, float(net_usd), str(reason))

    def _record_scalp_close_intelligence(
        self,
        *,
        trade_id: str,
        sym: str,
        entry_diag: dict,
        entry: float,
        qty: float,
        exit_price: float,
        net_usd: float,
        net_pct: float,
        hold_seconds: float,
        spread_at_exit: float,
        reason: str,
        ts: str,
        exit_diag: dict | None = None,
    ) -> None:
        """Post-close SCALP attribution, review, learning, and market memory (isolated from DAY)."""
        notional = qty * exit_price
        sell_fee = notional * self.econ.exit_fee_pct()
        buy_fee = entry * qty * self.econ.entry_fee_pct()
        fees = sell_fee + buy_fee
        gross_pnl = (exit_price - entry) * qty
        slip_usd = notional * self.econ.slippage_buffer_pct
        realized_slip = slip_usd / notional if notional > 0 else 0.0

        intel = dict(entry_diag or {})
        ed = dict(exit_diag or {})
        if ed:
            intel["exit_diagnostics"] = ed
            intel["exit_state"] = ed.get("state")
            intel["exit_trigger_detail"] = ed.get("reason")
            intel["exit_stale_review_count"] = ed.get("stale_review_count")
            intel["exit_max_favorable_pct"] = ed.get("max_favorable_pct")
            intel["exit_recovery_from_low_pct"] = ed.get("recovery_from_low_pct")
            intel["exit_bid_change_15s"] = ed.get("bid_change_15s")
            intel["exit_bid_change_30s"] = ed.get("bid_change_30s")
            intel["exit_bid_change_60s"] = ed.get("bid_change_60s")
            intel["exit_higher_lows"] = ed.get("higher_lows")
            # Scratch-specific fields only present when EARLY_SCRATCH_EXIT fired.
            intel["scratch_target_progress_pct"] = ed.get("scratch_target_progress_pct")
            intel["scratch_min_reviews"] = ed.get("scratch_min_reviews")
            intel["scratch_momentum_weak"] = ed.get("scratch_momentum_weak")
            intel["scratch_flat_or_slight_neg"] = ed.get("scratch_flat_or_slight_neg")
        setup_name = str(intel.get("setup_name") or intel.get("scalp_setup") or "")
        if setup_name and not intel.get("scalp_setup"):
            from backend.services.scalp_feature_contract import STRATEGY_TO_SCALP_SETUP

            intel["scalp_setup"] = STRATEGY_TO_SCALP_SETUP.get(setup_name, setup_name)
        if setup_name and not intel.get("setup_name"):
            intel["setup_name"] = setup_name
        intel.setdefault("spread_at_entry", float(entry_diag.get("spread_at_entry") or entry_diag.get("spread_pct") or 0.0))
        intel["spread_at_exit"] = float(spread_at_exit)
        intel["realized_slippage"] = float(realized_slip)
        intel["entry_time"] = str(entry_diag.get("entry_time") or "")
        intel["hold_seconds"] = float(hold_seconds)

        try:
            from backend.services.scalp_outcome_attribution import classify_scalp_outcome, record_scalp_outcome_attribution

            intel["outcome_reason"] = classify_scalp_outcome(
                intelligence=intel,
                net_pnl=net_usd,
                hold_seconds=hold_seconds,
                exit_reason=reason,
            )
            record_scalp_outcome_attribution(
                trade_id=trade_id,
                symbol=sym,
                intelligence=intel,
                gross_pnl=gross_pnl,
                fees=fees,
                net_pnl=net_usd,
                hold_seconds=hold_seconds,
                exit_reason=reason,
                db_path=self.config.database_path,
            )
        except Exception as exc:
            logger.debug("SCALP_OUTCOME_ATTRIBUTION_SKIPPED %s", exc)

        try:
            from backend.services.scalp_post_trade_feature_review import record_scalp_post_trade_review

            record_scalp_post_trade_review(
                trade_id=trade_id,
                symbol=sym,
                closed_at_utc=ts,
                intelligence=intel,
                net_pnl=net_usd,
                hold_seconds=hold_seconds,
                db_path=self.config.database_path,
            )
        except Exception as exc:
            logger.debug("SCALP_POST_TRADE_REVIEW_SKIPPED %s", exc)

        try:
            from backend.services.scalp_strategy_score_weight_writer import propagate_scalp_adaptive_weights_for_close

            propagate_scalp_adaptive_weights_for_close(
                symbol=sym,
                intelligence=intel,
                net_pnl=net_usd,
                db_path=self.config.database_path,
            )
        except Exception as exc:
            logger.debug("SCALP_ADAPTIVE_WEIGHTS_SKIPPED %s", exc)

        try:
            from backend.services.binance_scalp.scalp_post_exit_path import schedule_post_exit_path

            schedule_post_exit_path(
                self.config.database_path,
                trade_id=trade_id,
                symbol=sym,
                setup=setup_name,
                exit_reason=reason,
                exit_ts=ts,
                exit_epoch=time.time(),
                entry_price=entry,
                exit_price=exit_price,
            )
        except Exception as exc:
            logger.debug("SCALP_POST_EXIT_PATH_SCHEDULE_SKIPPED %s", exc)

        try:
            from backend.services.scalp_market_memory import update_scalp_market_memory_on_close_sync

            setup = str(intel.get("scalp_setup") or intel.get("setup_name") or "")
            update_scalp_market_memory_on_close_sync(
                sym,
                setup=setup,
                net_pnl=net_usd,
                hold_seconds=hold_seconds,
                slippage=realized_slip,
                redis_client=self._redis,
            )
        except Exception as exc:
            logger.debug("SCALP_MARKET_MEMORY_CLOSE_SKIPPED %s", exc)

    def _try_exit(self, conn: sqlite3.Connection, row: sqlite3.Row, *, post_commit: list | None = None) -> None:
        sym = str(row["symbol"])
        snap = self.reader.read(sym)
        if snap is None:
            return
        self._write_market_cache(snap)
        self._record_momentum(snap)

        entry = float(row["entry_price"])
        qty = float(row["quantity"])
        ts, now_epoch = self._now()
        age = now_epoch - float(row["entry_time_epoch"])
        trade_id = str(row["trade_id"])

        entry_buy_impact = 0.0
        target_pct = self.econ.net_profit_target_pct
        pos_diag: dict = {}
        raw_diag = row["diagnostics_json"]
        if raw_diag:
            try:
                pos_diag = json.loads(raw_diag)
                setup_sig = pos_diag.get("setup_signal") or {}
                entry_buy_impact = float(setup_sig.get("impact_pct") or 0.0)
                target_pct = float(setup_sig.get("required_target_pct") or target_pct)
            except (json.JSONDecodeError, TypeError, ValueError):
                entry_buy_impact = 0.0

        pf = run_scalp_preflight(
            snap,
            self.econ,
            self.config,
            side="SELL",
            entry_price=entry,
            entry_buy_impact_pct=entry_buy_impact,
            quantity=qty,
            check_paper_enabled=True,
        )
        self._write_signal(
            sym,
            {
                "symbol": sym,
                "strategy_id": self._position_strategy_id(row),
                "side": "SELL",
                "preflight": pf.as_dict(),
                "paper_only": True,
            },
        )

        exit_price = pf.expected_avg_fill if pf.expected_avg_fill > 0 else pf.limit_sell_price
        net_pct = pf.expected_net_edge_pct
        net_usd = (exit_price - entry) * qty - (
            exit_price * qty * self.econ.exit_fee_pct() + exit_price * qty * self.econ.slippage_buffer_pct + entry * qty * self.econ.entry_fee_pct() + entry * qty * self.econ.slippage_buffer_pct
        )
        profit_hit = net_pct >= target_pct
        exit_spread_ok = pf.reject_reason != "SPREAD_TOO_WIDE"

        _, epoch = self._now()
        mom = self._momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
        track = track_from_row(row, pos_diag)
        review_interval = int(os.getenv("SCALP_REVIEW_INTERVAL_SEC", "30"))
        last_review_epoch = 0.0
        try:
            lrt = row["last_review_ts"]
            if lrt:
                last_review_epoch = datetime.fromisoformat(str(lrt).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            last_review_epoch = 0.0
        in_review_phase = age >= self.econ.stale_scalp_timeout_sec
        perform_review = in_review_phase and (track.stale_review_count == 0 or (epoch - last_review_epoch) >= review_interval)
        # Route exit decision through the signal engine when the flag is on.
        # v1 delegates to the same evaluate_exit (strong bounded logic already present:
        # profit target after costs, setup invalidation, momentum failed, hard max hold).
        # Future lab-validated exit rules (or adjusted thresholds) can live in the engine.
        review = None
        try:
            from backend.services.binance_scalp import scalp_signal_engine as _se

            if _se.scalp_signal_engine_enabled():
                ed = _se.exit_decision(
                    track_row=row,
                    snap=snap,
                    mom=mom,
                    hold_sec=age,
                    executable_net_pct=net_pct,
                    profit_hit=profit_hit,
                    exit_spread_ok=exit_spread_ok,
                    perform_review=perform_review,
                )
                if ed and ed.get("decision") == "SELL":
                    diag = ed.get("diagnostics") or {}
                    review = ExitReviewResult(
                        decision=DECISION_SELL,
                        state=str(diag.get("state") or "SIGNAL_ENGINE"),
                        reason=ed.get("reason") or "SIGNAL_ENGINE_SELL",
                        exit_reason=ed.get("exit_reason") or ed.get("reason") or "SIGNAL_ENGINE_SELL",
                        diagnostics=diag,
                        updated_track=ed.get("updated_track") or track,
                    )
                else:
                    review = None  # fall through to direct evaluate below if not sell
        except Exception:
            review = None

        if review is None:
            review = evaluate_exit(
                track=track,
                snap=snap,
                mom=mom,
                econ=self.econ,
                config=self.config,
                trade_id=trade_id,
                hold_sec=age,
                executable_net_pct=net_pct,
                profit_hit=profit_hit,
                exit_spread_ok=exit_spread_ok,
                perform_review=perform_review,
            )

        hard = _max_hold_hard_sec(self.econ)
        if age >= hard and (review.decision != DECISION_SELL or not review.exit_reason):
            forced_diag = dict(review.diagnostics or {})
            forced_diag["forced_max_hold"] = True
            forced_diag["hold_seconds"] = round(age, 1)
            review = ExitReviewResult(
                DECISION_SELL,
                STATE_MAX_HOLD_REVIEW,
                f"max_hold_forced_{hard}s",
                EXIT_MAX_HOLD_HARD_LIMIT,
                forced_diag,
                review.updated_track,
            )

        if perform_review or getattr(review, "decision", None) == DECISION_SELL:
            self._record_position_review(conn, trade_id=trade_id, sym=sym, review_diag=review.diagnostics)

        self._persist_position_track(
            conn,
            row,
            review.updated_track,
            ts=ts,
            reason=review.reason,
            review_lows=review.updated_track.review_lows,
            touch_review_ts=perform_review,
        )

        exit_gate = {
            "current_bid": snap.best_bid,
            "expected_sell_fill": exit_price,
            "net_pct": net_pct,
            "target_pct": self.econ.net_profit_target_pct,
            "sell_preflight_pass": profit_hit,
            "reject_reason": pf.reject_reason or None,
            "spread_pct": snap.spread_pct,
            "entry_buy_impact_pct": entry_buy_impact,
            "exit_sell_impact_pct": pf.sell_impact_pct,
            "age_sec": age,
            "exit_review": review.diagnostics,
            "exit_state": review.state,
        }

        if review.decision != DECISION_SELL or not review.exit_reason:
            return

        entry_notional = entry * qty
        persisted_entry_fee: float | None = None
        persisted_entry_slippage: float | None = None
        buy_cost_row = conn.execute(
            """
            SELECT notional, fee_usd, slippage_usd
            FROM scalp_paper_trades
            WHERE trade_id = ? AND upper(side) = 'BUY'
            LIMIT 1
            """,
            (trade_id,),
        ).fetchone()
        if buy_cost_row is not None:
            entry_notional = float(buy_cost_row["notional"])
            persisted_entry_fee = float(buy_cost_row["fee_usd"])
            persisted_entry_slippage = float(buy_cost_row["slippage_usd"])
        exit_notional = exit_price * qty
        round_trip_fees, round_trip_slippage = _round_trip_execution_costs(
            entry_notional=entry_notional,
            exit_notional=exit_notional,
            econ=self.econ,
            persisted_entry_fee=persisted_entry_fee,
            persisted_entry_slippage=persisted_entry_slippage,
        )

        self._execute_sell(
            conn,
            row,
            sym=sym,
            entry=entry,
            qty=qty,
            exit_price=exit_price,
            net_pct=net_pct,
            net_usd=net_usd,
            reason=review.exit_reason,
            pf_dict=pf.as_dict(),
            exit_gate=exit_gate,
            hold_seconds=age,
            spread_at_exit=float(snap.spread_pct),
        )

        close_payload = {
            "trade_id": trade_id,
            "sym": sym,
            "entry_diag": pos_diag,
            "exit_diag": dict(review.diagnostics or {}),
            "entry": entry,
            "qty": qty,
            "exit_price": exit_price,
            "net_usd": net_usd,
            "net_pct": net_pct,
            "hold_seconds": age,
            "spread_at_exit": float(snap.spread_pct),
            "reason": str(review.exit_reason or ""),
            "ts": ts,
            "row": dict(row),
            "now_epoch": now_epoch,
            "review_exit_reason": str(review.exit_reason or "SCALP_EXIT"),
            "entry_notional": entry_notional,
            "exit_notional": exit_notional,
            "fees_paid": round_trip_fees,
            "slippage_cost": round_trip_slippage,
        }

        def _after_commit() -> None:
            self._record_scalp_close_intelligence(
                trade_id=close_payload["trade_id"],
                sym=close_payload["sym"],
                entry_diag=close_payload["entry_diag"],
                exit_diag=close_payload["exit_diag"],
                entry=close_payload["entry"],
                qty=close_payload["qty"],
                exit_price=close_payload["exit_price"],
                net_usd=close_payload["net_usd"],
                net_pct=close_payload["net_pct"],
                hold_seconds=close_payload["hold_seconds"],
                spread_at_exit=close_payload["spread_at_exit"],
                reason=close_payload["reason"],
                ts=close_payload["ts"],
            )
            try:
                from backend.services.trade_learning_writer import TradeLearningRecord, record_trade_outcome

                _entry_diag = close_payload.get("entry_diag") or {}
                if not isinstance(_entry_diag, dict):
                    _entry_diag = {}
                _exit_diag = close_payload.get("exit_diag") or {}
                if not isinstance(_exit_diag, dict):
                    _exit_diag = {}
                _ctx_raw = _entry_diag.get("context_snapshot_json") or _entry_diag.get("context_snapshot")
                _ctx_obj = None
                if isinstance(_ctx_raw, dict):
                    _ctx_obj = _ctx_raw
                elif isinstance(_ctx_raw, str) and _ctx_raw.startswith("{"):
                    with contextlib.suppress(Exception):
                        _ctx_obj = json.loads(_ctx_raw)
                _sr = _entry_diag.get("symbol_ranking") if isinstance(_entry_diag.get("symbol_ranking"), dict) else {}
                _entry_score = _entry_diag.get("score") or _entry_diag.get("rank_score") or _entry_diag.get("selected_score") or _sr.get("rank_score") or _sr.get("best_rank_score")
                _rank = {
                    "strategy_id": str(close_payload["row"].get("strategy_id", "")),
                    "setup": str(close_payload["row"].get("strategy_id", "")),
                    "entry_score": _entry_score,
                    "rank_score": _entry_score,
                    "sig_passed": _entry_diag.get("sig_passed") if "sig_passed" in _entry_diag else _entry_diag.get("passed"),
                    "regime": (_ctx_obj or {}).get("market_regime") if isinstance(_ctx_obj, dict) else None,
                }
                rec = TradeLearningRecord(
                    symbol=close_payload["sym"].replace("/", ""),
                    entry_timestamp=float(close_payload["row"].get("entry_time_epoch") or 0),
                    exit_timestamp=close_payload["now_epoch"],
                    entry_price=close_payload["entry"],
                    exit_price=close_payload["exit_price"],
                    quantity=close_payload["qty"],
                    fees_paid=close_payload["fees_paid"],
                    slippage_cost=close_payload["slippage_cost"],
                    net_profit_usd=close_payload["net_usd"],
                    net_profit_pct=close_payload["net_pct"],
                    hold_seconds=close_payload["hold_seconds"],
                    decision_reason=close_payload["review_exit_reason"],
                    close_reason=close_payload["review_exit_reason"],
                    confidence=float(_entry_diag.get("confidence") or 0.0) or None,
                    rank_data=_rank,
                    indicators_at_entry={
                        "entry_diag": {k: _entry_diag.get(k) for k in list(_entry_diag)[:40]},
                        "context_snapshot": _ctx_obj,
                    },
                    indicators_while_holding={
                        "max_favorable_pct": _exit_diag.get("max_favorable_pct") or close_payload["row"].get("max_favorable_pct"),
                        "max_adverse_pct": _exit_diag.get("max_adverse_pct") or close_payload["row"].get("max_adverse_pct"),
                    },
                    indicators_at_sell={
                        "exit_diag": {k: _exit_diag.get(k) for k in list(_exit_diag)[:40]},
                        "reason": close_payload.get("reason"),
                        "spread_at_exit": close_payload.get("spread_at_exit"),
                    },
                    timeframes_used=["1m", "5m"],
                    extra={
                        "engine": "binance_scalp_paper",
                        "setup": str(close_payload["row"].get("strategy_id", "")),
                        "entry_notional": close_payload["entry_notional"],
                        "exit_notional": close_payload["exit_notional"],
                        "context_snapshot": _ctx_obj,
                    },
                )
                from backend.database_schema import DATABASE_PATH as _DAY_LEARNING_DB

                record_trade_outcome(rec, db_path=_DAY_LEARNING_DB)
            except Exception as _lw_exc:
                logger.debug("SCALP_LEARNING_WRITE_SKIPPED %s", _lw_exc)

            # Market-role outcome learner (strategy=scalp) — uses BUY entry snapshot
            try:
                from backend.services.market_role_outcome_learner import record_trade_outcome as _role_learn

                _entry_diag = close_payload.get("entry_diag") or {}
                if not isinstance(_entry_diag, dict):
                    _entry_diag = {}
                _ctx_snap = _entry_diag.get("context_snapshot_json") or "{}"
                if isinstance(_ctx_snap, dict):
                    _ctx_snap = json.dumps(_ctx_snap, separators=(",", ":"), default=str)
                _mfe = None
                _mae = None
                _exit_diag = close_payload.get("exit_diag") or {}
                _row = close_payload.get("row") or {}
                with contextlib.suppress(Exception):
                    _mfe = float(
                        (_exit_diag.get("max_favorable_pct") if isinstance(_exit_diag, dict) else None)
                        or (_row.get("max_favorable_pct") if isinstance(_row, dict) else None)
                        or _entry_diag.get("max_favorable_pct")
                        or 0
                    )
                with contextlib.suppress(Exception):
                    _mae = float(
                        (_exit_diag.get("max_adverse_pct") if isinstance(_exit_diag, dict) else None)
                        or (_row.get("max_adverse_pct") if isinstance(_row, dict) else None)
                        or _entry_diag.get("max_adverse_pct")
                        or 0
                    )
                _regime = "unknown"
                with contextlib.suppress(Exception):
                    _intel = json.loads(_ctx_snap) if isinstance(_ctx_snap, str) and _ctx_snap.startswith("{") else {}
                    if isinstance(_intel, dict):
                        _regime = str(_intel.get("market_regime") or "unknown")
                # Prefer BUY trade_id from position row when available
                _buy_tid = str((_row.get("trade_id") if isinstance(_row, dict) else None) or close_payload.get("trade_id") or "")
                _role_learn(
                    self.config.database_path,
                    trade_id=str(close_payload.get("trade_id") or "") + "_sell",
                    buy_trade_id=_buy_tid,
                    symbol=str(close_payload["sym"]).replace("/", ""),
                    strategy="scalp",
                    realized_pnl_pct=float(close_payload.get("net_pct") or 0.0),
                    hold_seconds=int(close_payload.get("hold_seconds") or 0),
                    exit_reason=str(close_payload.get("review_exit_reason") or close_payload.get("reason") or ""),
                    mfe_pct=_mfe,
                    mae_pct=_mae,
                    market_regime=_regime,
                    context_snapshot_json=str(_ctx_snap or "{}"),
                )
            except Exception as _role_exc:
                logger.debug("SCALP_ROLE_LEARNING_SKIPPED %s", _role_exc)

        if post_commit is not None:
            post_commit.append(_after_commit)
        else:
            _after_commit()

    def _exit_open_positions_now(self) -> None:
        """Evaluate existing path-aware exits before ranking/klines.

        Does not change PATH_MAX_ADVERSE_STOP or any other exit threshold.
        Removes only the evaluate_all / kline delay in front of _try_exit.
        """
        post_commit: list = []
        pending_sell = None
        try:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for row in self._open_positions(conn):
                    self._try_exit(conn, row, post_commit=post_commit)
                pending_sell = getattr(self, "_pending_sell_log", None)
                self._pending_sell_log = None
                conn.commit()
        except Exception:
            logger.exception("SCALP_EARLY_EXIT_PASS_FAILED")
            return
        if pending_sell:
            sym, net_usd, reason = pending_sell
            logger.info("SCALP_PAPER_SELL %s pnl=%.4f reason=%s", sym, net_usd, reason)
        for fn in post_commit:
            try:
                fn()
            except Exception as exc:
                logger.warning("SCALP_POST_COMMIT_HOOK_FAILED %s", exc)

    def _observe_markouts(self, snaps: dict) -> None:
        with contextlib.suppress(Exception):
            from backend.services.binance_scalp.scalp_markout import flush_completed, observe_book

            for sym, snap in snaps.items():
                observe_book(
                    sym,
                    bid=float(getattr(snap, "best_bid", 0.0) or 0.0),
                    ask=float(getattr(snap, "best_ask", 0.0) or 0.0),
                    bids=getattr(snap, "bids", None),
                    asks=getattr(snap, "asks", None),
                )
            flush_completed(self.config.database_path)

    def tick(self, *, rank: bool = True) -> None:
        self.config.assert_no_live_trading()
        if self.config.scalp_paper_enabled and self.config.scalp_paper_auto_arm:
            set_entry_armed(
                self._redis,
                prefix=self.config.redis_key_prefix,
                armed=True,
                persistent=True,
            )
        # Publish before heavy REST/klines so snapshot cannot go missing mid-tick.
        # Exit-only ticks skip opportunity labeling / kline-adjacent work.
        if rank:
            with contextlib.suppress(Exception):
                from backend.services.binance_scalp.scalp_post_exit_path import fill_due_post_exit_paths

                fill_due_post_exit_paths(self.config.database_path, self.reader, now_epoch=time.time())
            with contextlib.suppress(Exception):
                from backend.services.binance_scalp.scalp_opportunity_dataset import label_due_opportunities

                label_due_opportunities(self.config.database_path, self.reader, now_epoch=time.time())
            with contextlib.suppress(Exception):
                pre_open: list[sqlite3.Row] = []
                try:
                    with self._conn() as _c:
                        pre_open = self._open_positions(_c)
                except Exception:
                    pre_open = []
                blocked = "MAX_OPEN_POSITIONS" if pre_open and len(pre_open) >= int(self.config.max_open_positions) else None
                self._publish_api_status_snapshot(
                    open_rows=pre_open,
                    epoch=time.time(),
                    entry_blocked_reason=blocked,
                    snapshot_source="runner_tick_start",
                )
        if not self.config.scalp_paper_enabled:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for sym in self.config.products:
                    self._write_signal(
                        sym,
                        {
                            "symbol": sym,
                            "strategy_id": self.config.strategy_id,
                            "side": "SCAN",
                            "paper_enabled": False,
                            "reject_reason": SCALP_PAPER_DISABLED,
                        },
                    )
                self._publish_last_decision(decision="BLOCKED", reason=SCALP_PAPER_DISABLED)
                conn.commit()
            return

        if not self.econ.is_fee_model_verified():
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for sym in self.config.products:
                    self._record_reject(conn, sym, "BUY", FEE_MODEL_UNVERIFIED, "fee model not verified")
                self._publish_last_decision(decision="BLOCKED", reason=FEE_MODEL_UNVERIFIED)
                conn.commit()
            return

        _, epoch = self._now()
        snaps = {}
        for sym in self.config.products:
            snap = self.reader.read(sym)
            if snap is None:
                continue
            snaps[sym] = snap
            self._record_momentum(snap)
            self._write_market_cache(snap)
            with contextlib.suppress(Exception):
                from backend.services.microstructure_engine import record_snapshot

                if getattr(snap, "bids", None) and getattr(snap, "asks", None):
                    record_snapshot(sym, snap.bids, snap.asks)

        # Existing -15 bp authority must see the book before ranking work.
        from backend.services.binance_scalp.scalp_micro_latency import timed

        _exit_done = timed("event_to_exit_review")
        self._exit_open_positions_now()
        _exit_done()
        self._observe_markouts(snaps)
        if not rank:
            return

        for sym, snap in snaps.items():
            bars = self._klines.get(sym)
            regime = self._router._current_regime(sym, epoch, bars)
            self._seed_scalp_market_memory(sym, snap, micro_regime=regime)
            self._publish_scan_snapshot(sym, snap, micro_regime=regime, epoch=epoch)

        notional = float(self.config.max_notional_paper)
        with contextlib.suppress(Exception):
            with self._conn() as _cash:
                notional = min(notional, float(self._ledger(_cash)["cash_balance"]))
        pre_ranked: list[dict] = []
        try:
            pre_ranked = self._router.evaluate_all(epoch=epoch, notional_usd=notional) or []
        except Exception:
            logger.exception("SCALP_EVALUATE_ALL_FAILED")
            pre_ranked = []
        self._pending_opportunity_rows = pre_ranked
        self._pending_opportunity_epoch = epoch
        with contextlib.suppress(Exception):
            from backend.services.binance_scalp.scalp_markout import schedule_markout

            for row in pre_ranked:
                snap = snaps.get(str(row.get("symbol") or ""))
                if snap is None:
                    continue
                schedule_markout(
                    kind="candidate",
                    symbol=str(row.get("symbol") or ""),
                    side="BUY",
                    mid=float(getattr(snap, "mid", 0.0) or 0.0),
                    entry_px=float(getattr(snap, "best_ask", 0.0) or getattr(snap, "mid", 0.0) or 0.0),
                    qty=0.0,
                    notional=float(notional),
                    fee_pct=float(self.econ.entry_fee_pct() + self.econ.exit_fee_pct()),
                    slip_pct=float(self.econ.slippage_buffer_pct),
                    extra={"rank_score": row.get("rank_score")},
                )

        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            post_commit: list = []
            open_rows = self._open_positions(conn)
            entry_blocked: str | None = None
            for row in open_rows:
                self._try_exit(conn, row, post_commit=post_commit)
            open_count = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
            if open_count < self.config.max_open_positions:
                before_rejects = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]
                self._try_entry(conn, pre_ranked=pre_ranked)
                after_rejects = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]
                if after_rejects > before_rejects:
                    last = conn.execute("SELECT reason FROM scalp_rejects ORDER BY id DESC LIMIT 1").fetchone()
                    if last:
                        entry_blocked = str(last[0] or "")
            else:
                entry_blocked = "MAX_OPEN_POSITIONS"
                self._publish_last_decision(decision="BLOCKED", reason="MAX_OPEN_POSITIONS")
            if getattr(self, "_last_circuit_breaker_open", False):
                entry_blocked = "SCALP_CIRCUIT_BREAKER_OPEN"
            open_after = self._open_positions(conn)
            self._publish_runner_state(conn, open_rows=open_after, epoch=epoch, entry_blocked_reason=entry_blocked)
            pending_sell = getattr(self, "_pending_sell_log", None)
            self._pending_sell_log = None
            conn.commit()
            pending_opp = getattr(self, "_pending_opportunity_rows", None)
            pending_epoch = getattr(self, "_pending_opportunity_epoch", None)
            self._pending_opportunity_rows = None
            self._pending_opportunity_epoch = None
            if pending_opp:
                try:
                    from backend.services.binance_scalp.scalp_opportunity_dataset import record_opportunity_cycle

                    written = record_opportunity_cycle(
                        self.config.database_path,
                        rows=pending_opp,
                        epoch=pending_epoch,
                    )
                    logger.info(
                        "SCALP_OPPORTUNITY_CYCLE written=%s symbols=%s",
                        written,
                        [r.get("symbol") for r in pending_opp],
                    )
                except Exception:
                    logger.exception("SCALP_OPPORTUNITY_CYCLE_FAILED n=%s", len(pending_opp))
            if pending_sell:
                sym, net_usd, reason = pending_sell
                logger.info("SCALP_PAPER_SELL %s pnl=%.4f reason=%s", sym, net_usd, reason)
            for fn in post_commit:
                try:
                    fn()
                except Exception as exc:
                    logger.warning("SCALP_POST_COMMIT_HOOK_FAILED %s", exc)
            # Fast /api/scalp/status reads this Redis snapshot — never rebuilds on GET.
            self._publish_api_status_snapshot(
                open_rows=open_after,
                epoch=epoch,
                entry_blocked_reason=entry_blocked,
                snapshot_source="runner_tick",
                include_pnl=True,
            )

    def run_loop(self, interval_sec: float = 5.0, exit_interval_sec: float | None = None) -> None:
        if not self.config.scalp_paper_enabled:
            logger.error("Scalp paper loop idle: SCALP_PAPER_ENABLED=false (set true in .env and restart)")
            while True:
                time.sleep(max(interval_sec, 30.0))
            return
        if not self.econ.is_fee_model_verified():
            logger.error("Scalp paper loop idle: SCALP_FEE_MODEL_VERIFIED=false (set true in .env)")
            while True:
                time.sleep(max(interval_sec, 30.0))
            return
        self.config.assert_no_live_trading()
        armed = is_entry_armed(self._redis, prefix=self.config.redis_key_prefix)
        try:
            from backend.services.binance_scalp import scalp_signal_engine as _se

            engine_on = _se.scalp_signal_engine_enabled()
        except Exception:
            engine_on = False
        logger.info(
            "Binance scalp paper loop products=%s max_open=%s interval=%ss paper_only=True live_blocked=True calibration=%s profile=%s entry_armed=%s signal_engine=%s",
            self.config.products,
            self.config.max_open_positions,
            interval_sec,
            self.config.calibration_mode,
            self.config.calibration_profile if self.config.calibration_mode else "strict",
            armed if not self.config.calibration_mode else "auto",
            engine_on,
        )

        # Warm the in-memory MomentumTracker so the first real ticks are not
        # immediately rejected with MOMENTUM_DATA_INSUFFICIENT / history < 30s.
        # Mirrors the explicit warm used by /api/scalp/status diagnostics.
        # After this the engine can evaluate breakout_momentum (and other enabled
        # strategies) with real 15/30/60s rising checks and the full entry gate.
        warm_rounds = 12
        warm_interval = 5.0
        logger.info(
            "SCALP_WARMING momentum for %s (~%ss history for confirmed checks)",
            self.config.products,
            int(warm_rounds * warm_interval),
        )
        for _ in range(warm_rounds):
            for sym in self.config.products:
                snap = self.reader.read(sym)
                if snap:
                    self._record_momentum(snap)
            time.sleep(warm_interval)
        logger.info("SCALP_WARM complete — now evaluating entries with bounded exits (net-profit / momentum-fail / setup-invalid / hard max-hold)")
        # Seed open-symbol cache so heartbeat reports held slots honestly during first tick.
        with contextlib.suppress(Exception):
            with self._conn() as _c:
                rows = self._open_positions(_c)
                self._cached_open_symbols = [str(r["symbol"]) for r in rows]
                self._publish_api_status_snapshot(
                    open_rows=rows,
                    epoch=time.time(),
                    entry_blocked_reason=("MAX_OPEN_POSITIONS" if rows and len(rows) >= int(self.config.max_open_positions) else "WARM_COMPLETE"),
                    snapshot_source="runner_warm",
                )
        self._start_status_heartbeat(interval_sec=30.0)

        maybe_run_scalp_reject_retention(self.config.database_path)
        maybe_run_scalp_position_housekeeping(self.config.database_path)

        exit_sec = float(exit_interval_sec) if exit_interval_sec is not None else float(os.getenv("SCALP_EXIT_INTERVAL_SEC", "0.25"))
        rank_sec = float(interval_sec)
        last_rank = 0.0
        try:
            while True:
                try:
                    now = time.time()
                    do_rank = (now - last_rank) >= rank_sec
                    self.tick(rank=do_rank)
                    if do_rank:
                        last_rank = now
                        maybe_run_scalp_reject_retention(self.config.database_path)
                        maybe_run_scalp_position_housekeeping(self.config.database_path)
                except Exception as exc:
                    logger.exception("scalp paper tick error: %s", exc)
                    with contextlib.suppress(Exception):
                        self._publish_api_status_snapshot(
                            open_rows=None,
                            epoch=time.time(),
                            entry_blocked_reason=f"TICK_ERROR:{type(exc).__name__}",
                            snapshot_source="runner_error",
                        )
                try:
                    from backend.services.task_health_monitor import beat_sync

                    beat_sync("scalp_runner:tick", self._redis, extra={"products": ",".join(self.config.products)})
                except Exception:
                    pass
                time.sleep(max(0.05, min(exit_sec, rank_sec)))
        finally:
            self._heartbeat_stop = True
            self.shutdown()
