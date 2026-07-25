"""Binance.US scalp paper engine — paper fills only, no live orders."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

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
    entry_fee = (
        float(persisted_entry_fee)
        if persisted_entry_fee is not None
        else entry_notional * econ.taker_fee_pct
    )
    entry_slippage = (
        float(persisted_entry_slippage)
        if persisted_entry_slippage is not None
        else entry_notional * econ.slippage_buffer_pct
    )
    fees = entry_fee + exit_notional * econ.taker_fee_pct
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
        init_scalp_schema(self.config.database_path)
        if self.config.scalp_paper_enabled and self.config.scalp_paper_auto_arm:
            set_entry_armed(
                self._redis,
                prefix=self.config.redis_key_prefix,
                armed=True,
                persistent=True,
            )
        else:
            set_entry_armed(self._redis, prefix=self.config.redis_key_prefix, armed=False)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.database_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _now(self) -> tuple[str, float]:
        dt = datetime.now(timezone.utc)
        return dt.isoformat(), dt.timestamp()

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

    def _record_reject(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        side: str,
        reason: str,
        detail: str,
    ) -> None:
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

        Two conditions trigger the breaker:
        1. Today's closed-trade PnL is worse than -SCALP_DAILY_LOSS_LIMIT_PCT * principal.
        2. The last SCALP_MAX_CONSECUTIVE_LOSSES closed trades are all losses.
        """
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with self._conn() as conn:
                ledger = self._ledger(conn)
                principal = float(ledger["principal"])

                # Daily loss check
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(pnl_usd), 0) AS today_pnl
                    FROM scalp_paper_trades
                    WHERE upper(side) = 'SELL'
                      AND pnl_usd IS NOT NULL
                      AND date(created_at) = ?
                    """,
                    (today,),
                ).fetchone()
                today_pnl = float(row[0]) if row else 0.0
                daily_limit = self.config.daily_loss_limit_pct * principal
                if today_pnl <= -daily_limit:
                    logger.warning(
                        "[SCALP_CIRCUIT_BREAKER] Daily loss limit hit: today_pnl=%.4f limit=-%.4f principal=%.2f halt=True",
                        today_pnl,
                        daily_limit,
                        principal,
                    )
                    return True

                # Consecutive loss check
                max_consec = self.config.max_consecutive_losses
                recent_rows = conn.execute(
                    """
                    SELECT pnl_usd FROM scalp_paper_trades
                    WHERE upper(side) = 'SELL' AND pnl_usd IS NOT NULL
                    ORDER BY id DESC LIMIT ?
                    """,
                    (max_consec,),
                ).fetchall()
                if len(recent_rows) >= max_consec and all(float(r[0]) <= 0.0 for r in recent_rows):
                    logger.warning(
                        "[SCALP_CIRCUIT_BREAKER] %d consecutive losses, halt=True",
                        max_consec,
                    )
                    return True
        except Exception as e:
            logger.warning("[SCALP_CIRCUIT_BREAKER] Check failed (non-blocking): %s", e)
        return False

    def _entry_candidates(self, conn: sqlite3.Connection) -> list[tuple[str, MarketSnapshot, object]]:
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
        try:
            from backend.services.binance_scalp import scalp_signal_engine as _se

            if _se.scalp_signal_engine_enabled():
                _se.bind_paper_engine(router=self._router, momentum=self._momentum, klines=self._klines)
                ranked = self._router.evaluate_all(epoch=epoch, notional_usd=notional) or []
        except Exception:
            ranked = []

        if not ranked:
            ranked = self._router.evaluate_all(epoch=epoch, notional_usd=notional) or []

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

        from backend.services.binance_scalp.scalp_candidate_ranking import pick_best_global_candidate

        best = pick_best_global_candidate(ranked)
        if best is None:
            eligible_rows = [r for r in ranked if r.get("entry_eligible")]
            # Prefer an eligible row for reject reason; fall back to rank leader.
            top = eligible_rows[0] if eligible_rows else (ranked[0] if ranked else {})
            meta = top.get("rank_meta") or {}
            top_score = float(top.get("rank_score") or 0)
            second_score = float(eligible_rows[1].get("rank_score") or 0) if len(eligible_rows) > 1 else 0.0
            reason = (
                "RANK_LOW_CONFIDENCE_TIE"
                if eligible_rows and top_score < float(os.getenv("SCALP_MIN_CONFIDENT_RANK", "1.55"))
                else top.get("hard_block") or meta.get("hard_block") or top.get("soft_reason") or meta.get("soft_reason") or f"RANK_BELOW_MIN:{top.get('rank_score')}"
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
                        "all_symbols": [
                            {
                                "symbol": r["symbol"],
                                "rank_score": r.get("rank_score"),
                                "entry_eligible": r.get("entry_eligible"),
                                "best_setup": r.get("best_setup"),
                                "hard_block": r.get("hard_block"),
                                "soft_reason": r.get("soft_reason"),
                            }
                            for r in ranked
                        ],
                    }
                ),
            )
            ranked_summary = [{"symbol": r["symbol"], "rank_score": r.get("rank_score"), "entry_eligible": r.get("entry_eligible"), "hard_block": r.get("hard_block")} for r in ranked]
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
        ranked_summary = [{"symbol": r["symbol"], "rank_score": r.get("rank_score"), "entry_eligible": r.get("entry_eligible"), "hard_block": r.get("hard_block")} for r in ranked]
        if not getattr(sig, "passed", False):
            self._record_reject(
                conn,
                sym,
                "BUY",
                "RANKED_NOT_EXECUTABLE",
                json.dumps({"setup": sig.as_dict(), "rank_score": best.get("rank_score")}),
            )
            # The top-ranked candidate itself failed its preflight (spread/impact/
            # net-edge/depth) — a genuine operational failure for this attempt,
            # not an ordinary "nothing ranked" outcome.
            self._publish_last_decision(
                decision="BLOCKED",
                reason="RANKED_NOT_EXECUTABLE",
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
                }
                for r in ranked
            ],
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

    def _try_entry(self, conn: sqlite3.Connection) -> None:
        open_count = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
        if open_count >= self.config.max_open_positions:
            return

        if self._check_scalp_circuit_breaker():
            self._publish_last_decision(decision="BLOCKED", reason="SCALP_CIRCUIT_BREAKER_OPEN")
            return

        candidates = self._entry_candidates(conn)
        if not candidates:
            return

        sym, snap, sig = candidates[0]
        if conn.execute(
            "SELECT 1 FROM scalp_paper_positions WHERE symbol = ? AND status = 'OPEN' LIMIT 1",
            (sym,),
        ).fetchone():
            self._record_reject(conn, sym, "BUY", "SYMBOL_ALREADY_OPEN", json.dumps({"symbol": sym}))
            return

        ledger = self._ledger(conn)
        notional = min(self.config.max_notional_paper, float(ledger["cash_balance"]))
        if notional < 1.0:
            self._record_reject(conn, sym, "BUY", "INSUFFICIENT_CASH", f"cash={ledger['cash_balance']}")
            return

        limit_buy = sig.limit_buy_price
        qty = notional / limit_buy
        fee = notional * self.econ.taker_fee_pct
        slip = notional * self.econ.slippage_buffer_pct
        trade_id = f"scalp_paper_{sym}_{int(time.time() * 1000)}"
        ts, epoch = self._now()
        pre = dict(ledger)

        ranking_meta = getattr(self, "_last_ranking_meta", {}) or {}
        entry_intel = dict(ranking_meta.get("scalp_intelligence") or {})
        from backend.services.day_trade_thesis import scalp_strategy_to_thesis

        thesis_fields = scalp_strategy_to_thesis(sig.setup_name, sig.setup_context or {})
        soft_rank_entry = bool((sig.setup_context or {}).get("soft_rank_entry", False))
        entry_diag = {
            "setup_name": sig.setup_name,
            "setup_context": sig.setup_context,
            "setup_signal": sig.as_dict(),
            "passed": bool(sig.passed),
            "soft_rank_entry": soft_rank_entry,
            "entry_eligible": bool(sig.passed) and not soft_rank_entry,
            "selected_symbol": sym,
            "symbol_ranking": ranking_meta,
            "review_lows": [],
            "session_low_bid": limit_buy,
            "entry_time": ts,
            "spread_at_entry": float(snap.spread_pct),
            **thesis_fields,
            **entry_intel,
        }
        if entry_intel.get("feature_health_json") and "entry_scalp_vector" not in entry_diag:
            entry_diag["entry_scalp_vector"] = entry_intel.get("entry_scalp_vector") or []
        if entry_intel.get("scalp_setup"):
            entry_diag["scalp_setup"] = entry_intel.get("scalp_setup")
        if entry_intel.get("micro_regime"):
            entry_diag["micro_regime"] = entry_intel.get("micro_regime")

        # Entry-time market-role context (Redis — cross-process; for learning attribution)
        _role_ctx_snap = "{}"
        try:
            from backend.services.market_role_intelligence import get_role_context_snapshot_json

            _role_ctx_snap = get_role_context_snapshot_json(sym) or "{}"
        except Exception:
            _role_ctx_snap = "{}"
        entry_diag["context_snapshot_json"] = _role_ctx_snap

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
                        # Persist pass/soft-rank tags on the BUY row so closed-trade
                        # analytics do not depend on open-position diagnostics alone.
                        "passed": bool(sig.passed),
                        "soft_rank_entry": soft_rank_entry,
                        "entry_eligible": bool(sig.passed) and not soft_rank_entry,
                        "setup_name": sig.setup_name,
                        "rank_score": ranking_meta.get("rank_score"),
                        "selection_confidence": ranking_meta.get("selection_confidence"),
                        "context_snapshot_json": _role_ctx_snap,
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
        set_entry_armed(self._redis, prefix=self.config.redis_key_prefix, armed=False)

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
        fee = notional * self.econ.taker_fee_pct
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
        logger.info("SCALP_PAPER_SELL %s pnl=%.4f reason=%s", sym, net_usd, reason)

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
        sell_fee = notional * self.econ.taker_fee_pct
        buy_fee = entry * qty * self.econ.taker_fee_pct
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
            exit_price * qty * self.econ.taker_fee_pct
            + exit_price * qty * self.econ.slippage_buffer_pct
            + entry * qty * self.econ.taker_fee_pct
            + entry * qty * self.econ.slippage_buffer_pct
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
                    extra={
                        "engine": "binance_scalp_paper",
                        "setup": str(close_payload["row"].get("strategy_id", "")),
                        "entry_notional": close_payload["entry_notional"],
                        "exit_notional": close_payload["exit_notional"],
                    },
                )
                record_trade_outcome(rec, db_path=self.config.database_path)
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
                _buy_tid = str(
                    (_row.get("trade_id") if isinstance(_row, dict) else None)
                    or close_payload.get("trade_id")
                    or ""
                )
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

    def tick(self) -> None:
        self.config.assert_no_live_trading()
        if self.config.scalp_paper_enabled and self.config.scalp_paper_auto_arm:
            set_entry_armed(
                self._redis,
                prefix=self.config.redis_key_prefix,
                armed=True,
                persistent=True,
            )
        with self._conn() as conn:
            post_commit: list = []
            if not self.config.scalp_paper_enabled:
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
                for sym in self.config.products:
                    self._record_reject(conn, sym, "BUY", FEE_MODEL_UNVERIFIED, "fee model not verified")
                self._publish_last_decision(decision="BLOCKED", reason=FEE_MODEL_UNVERIFIED)
                conn.commit()
                return

            _, epoch = self._now()
            for sym in self.config.products:
                snap = self.reader.read(sym)
                if snap is None:
                    continue
                self._record_momentum(snap)
                self._write_market_cache(snap)
                bars = self._klines.get(sym)
                regime = self._router._current_regime(sym, epoch, bars)
                self._seed_scalp_market_memory(sym, snap, micro_regime=regime)
                self._publish_scan_snapshot(sym, snap, micro_regime=regime, epoch=epoch)

            open_rows = self._open_positions(conn)
            entry_blocked: str | None = None
            for row in open_rows:
                self._try_exit(conn, row, post_commit=post_commit)
            open_count = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]
            if open_count < self.config.max_open_positions:
                before_rejects = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]
                self._try_entry(conn)
                after_rejects = conn.execute("SELECT COUNT(*) FROM scalp_rejects").fetchone()[0]
                if after_rejects > before_rejects:
                    last = conn.execute("SELECT reason FROM scalp_rejects ORDER BY id DESC LIMIT 1").fetchone()
                    if last:
                        entry_blocked = str(last[0] or "")
            else:
                entry_blocked = "MAX_OPEN_POSITIONS"
                self._publish_last_decision(decision="BLOCKED", reason="MAX_OPEN_POSITIONS")
            self._publish_runner_state(conn, open_rows=self._open_positions(conn), epoch=epoch, entry_blocked_reason=entry_blocked)
            conn.commit()
            for fn in post_commit:
                try:
                    fn()
                except Exception as exc:
                    logger.warning("SCALP_POST_COMMIT_HOOK_FAILED %s", exc)

    def run_loop(self, interval_sec: float = 5.0) -> None:
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

        maybe_run_scalp_reject_retention(self.config.database_path)
        maybe_run_scalp_position_housekeeping(self.config.database_path)

        try:
            while True:
                try:
                    self.tick()
                    maybe_run_scalp_reject_retention(self.config.database_path)
                    maybe_run_scalp_position_housekeeping(self.config.database_path)
                except Exception as exc:
                    logger.exception("scalp paper tick error: %s", exc)
                try:
                    from backend.services.task_health_monitor import beat_sync

                    beat_sync("scalp_runner:tick", self._redis, extra={"products": ",".join(self.config.products)})
                except Exception:
                    pass
                time.sleep(interval_sec)
        finally:
            self.shutdown()
