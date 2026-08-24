"""Route paper scalp entries to highest-scoring valid strategy setup."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot, ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics, MomentumTracker
from backend.services.binance_scalp.redis_keys import ranking_meta_key
from backend.services.binance_scalp.scalp_candidate_ranking import (
    RankedCandidate,
    pick_best_ranked,
    prepare_entry_signal,
    rank_setup_signal,
)
from backend.services.binance_scalp.scalp_regime_classifier import (
    REGIME_DATA_MISSING,
    REGIME_RANGE,
    classify_scalp_regime,
)
from backend.services.binance_scalp.scalp_setup_measurements import evidence_rank_delta, measure_all_setups
from backend.services.binance_scalp.strategies import ALL_STRATEGIES, STRATEGY_NAMES, enabled_strategies
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.kline_cache import KlineCache, MIN_REGIME_1H_BARS
from backend.services.multi_horizon_ev import cached_multi_horizon_ev

logger = logging.getLogger(__name__)

# Last per-symbol ranking meta from evaluate_all() (item p22 unified EV
# contract wiring) — read-only diagnostic cache, never a decision input.
# In-process only: fast path for same-process callers (tests, and the scalp
# runner itself). The live scalp runner and the API/uvicorn process are
# separate OS processes, so this dict is invisible across processes —
# get_last_ranking_meta() falls back to the cross-process Redis snapshot
# (_publish_ranking_meta_to_redis / ranking_meta_key) for that case.
_LAST_RANKING_META_BY_SYMBOL: dict[str, dict[str, Any]] = {}


def _json_safe_ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip/convert the non-JSON-safe pieces of an evaluate_all() row
    (raw MarketSnapshot/ScalpSetupSignal objects) before publishing to
    Redis for cross-process diagnostic reads."""
    safe = {k: v for k, v in row.items() if k not in ("snap", "signal")}
    signal = row.get("signal")
    if signal is not None:
        with contextlib.suppress(Exception):
            safe["signal"] = signal.as_dict()
    return safe


def _publish_ranking_meta_to_redis(sym: str, row: dict[str, Any], *, redis_url: str, prefix: str) -> None:
    try:
        import json

        import redis as redis_sync

        client = redis_sync.from_url(redis_url, decode_responses=True)
        key = ranking_meta_key(prefix, sym)
        client.setex(key, 30, json.dumps(_json_safe_ranking_row(row), default=str))
    except Exception as exc:
        logger.debug("SCALP_RANKING_META_REDIS_PUBLISH_FAILED symbol=%s: %s", sym, exc)


def get_last_ranking_meta(symbol: str, *, redis_url: str | None = None, prefix: str | None = None) -> dict[str, Any] | None:
    """Most recent evaluate_all() ranking row for `symbol`. Checks the
    in-process cache first (same-process callers, e.g. tests or the scalp
    runner itself), then falls back to the cross-process Redis snapshot
    (needed for the API/uvicorn process, which runs as a separate OS
    process from the scalp runner and never shares its memory).
    Diagnostic-only (used by the unified EV contract API endpoint); never
    consulted by any entry/exit/sizing decision path."""
    row = _LAST_RANKING_META_BY_SYMBOL.get(symbol)
    if row is not None:
        return dict(row)
    try:
        import json

        import redis as redis_sync

        from backend.services.binance_scalp.config import ScalpConfig

        cfg_defaults = ScalpConfig.from_env()
        url = redis_url or cfg_defaults.redis_url
        pfx = prefix or cfg_defaults.redis_key_prefix
        client = redis_sync.from_url(url, decode_responses=True)
        raw = client.get(ranking_meta_key(pfx, symbol))
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.debug("SCALP_RANKING_META_REDIS_READ_FAILED symbol=%s: %s", symbol, exc)
    return None


def _mtf_confirmation_gate_enabled() -> bool:
    """When true (default), long SCALP entries require non-down 5m (and 15m) trend."""
    return str(os.getenv("SCALP_MTF_CONFIRMATION_GATE_ENABLED", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _mtf_require_15m() -> bool:
    return str(os.getenv("SCALP_MTF_REQUIRE_15M", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class ScalpStrategyRouter:
    def __init__(
        self,
        *,
        config: ScalpConfig,
        econ: ScalpEconomics,
        reader: ScalpMarketReader,
        momentum: MomentumTracker,
        klines: KlineCache | None = None,
    ) -> None:
        self.config = config
        self.econ = econ
        self.reader = reader
        self.momentum = momentum
        self.klines = klines or KlineCache()

    def evaluate_symbol(
        self,
        symbol: str,
        *,
        epoch: float,
        notional_usd: float,
        snap: MarketSnapshot | None = None,
        mom: MomentumDiagnostics | None = None,
        bars: list[dict] | None = None,
    ) -> tuple[ScalpSetupSignal | None, list[ScalpSetupSignal], dict[str, Any]]:
        sym = symbol.strip().upper()
        meta: dict[str, Any] = {
            "symbol": sym,
            "regime": REGIME_RANGE,
            "ranked": [],
            "best_rank_score": 0.0,
            "entry_eligible": False,
            "hard_block": None,
            "best_setup": None,
        }
        snap = snap or self.reader.read(sym)
        if snap is None:
            meta["hard_block"] = "STALE_DATA"
            return None, [], meta
        if mom is None:
            self.momentum.record(sym, epoch, snap.best_bid, snap.mid)
            mom = self.momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
        bars = bars if bars is not None else self.klines.get(sym)

        ctx = StrategyMarketContext(
            symbol=sym,
            snap=snap,
            mom=mom,
            bars_1m=bars,
            econ=self.econ,
            config=self.config,
            notional_usd=notional_usd,
        )

        regime = self._current_regime(sym, epoch, bars)
        meta["regime"] = regime
        if regime == REGIME_DATA_MISSING:
            meta["hard_block"] = "REGIME_DATA_MISSING"
            return None, [], meta
        # Multi-timeframe confirmation between 1m setups and 1h regime.
        # When SCALP_MTF_CONFIRMATION_GATE_ENABLED (default true), a down 5m/15m
        # trend blocks entry_eligible. Missing history (None) does not block.
        mtf_5m_trend_pct, mtf_5m_aligned = self._mtf_trend_confirmation(sym, "5m")
        mtf_15m_trend_pct, mtf_15m_aligned = self._mtf_trend_confirmation(sym, "15m")
        meta["mtf_5m_trend_pct"] = mtf_5m_trend_pct
        meta["mtf_5m_aligned"] = mtf_5m_aligned
        meta["mtf_15m_trend_pct"] = mtf_15m_trend_pct
        meta["mtf_15m_aligned"] = mtf_15m_aligned

        signals: list[ScalpSetupSignal] = []
        ranked_list: list[RankedCandidate] = []
        measurements: dict[str, dict[str, float]] = {}
        stage_ms: dict[str, float] = {}
        t_meas = time.perf_counter()
        with contextlib.suppress(Exception):
            measurements = measure_all_setups(ctx)
        stage_ms["measure_all_setups"] = round((time.perf_counter() - t_meas) * 1000.0, 1)
        meta["setup_measurements"] = measurements
        meta["measured_strategies"] = list(measurements.keys())
        compact_bars = []
        for bar in list(bars or [])[-30:]:
            if not isinstance(bar, dict):
                continue
            compact_bars.append(
                {
                    "open": float(bar.get("open") or 0),
                    "high": float(bar.get("high") or 0),
                    "low": float(bar.get("low") or 0),
                    "close": float(bar.get("close") or 0),
                    "volume": float(bar.get("volume") or 0),
                    "ts": bar.get("ts"),
                }
            )
        meta["bars_1m"] = compact_bars

        enabled_names = {s.name for s in enabled_strategies(self.config)}
        # Measure every module every cycle. Disabled modules stay out of the
        # executable pick but their features inform rank/learning.
        for strategy in ALL_STRATEGIES:
            t_strat = time.perf_counter()
            try:
                sig = strategy.evaluate(ctx)
                stage_ms[f"eval:{sig.setup_name}"] = round((time.perf_counter() - t_strat) * 1000.0, 1)
            except Exception as exc:
                stage_ms[f"eval:{getattr(strategy, 'name', '?')}"] = round((time.perf_counter() - t_strat) * 1000.0, 1)
                logger.warning("strategy %s failed %s: %s", getattr(strategy, "name", "?"), sym, exc)
                sig = ScalpSetupSignal(
                    symbol=sym,
                    side="BUY",
                    score=0.0,
                    setup_name=getattr(strategy, "name", "unknown"),
                    confidence=0.0,
                    entry_reason="",
                    invalidation_reason=None,
                    required_target_pct=0.0,
                    expected_move_pct=0.0,
                    spread_pct=getattr(ctx.snap, "spread_pct", 0.0),
                    impact_pct=0.0,
                    depth_sufficient=False,
                    limit_buy_price=getattr(ctx.snap, "best_ask", 0.0),
                    passed=False,
                    reject_reason=f"STRATEGY_ERROR:{exc}",
                    setup_context={"error": str(exc)[:200]},
                )
            feats = measurements.get(sig.setup_name) or {}
            ctx_map = dict(sig.setup_context or {})
            ctx_map["features"] = feats
            ctx_map["runtime_enabled"] = sig.setup_name in enabled_names
            from dataclasses import replace as _sig_replace

            sig = _sig_replace(sig, setup_context=ctx_map)
            signals.append(sig)
            t_rank = time.perf_counter()
            ranked_list.append(rank_setup_signal(sig, regime=regime, ctx=ctx))
            stage_ms[f"rank:{sig.setup_name}"] = round((time.perf_counter() - t_rank) * 1000.0, 1)
        meta["stage_ms"] = stage_ms

        meta["ranked"] = [
            {
                "setup_name": r.signal.setup_name,
                "rank_score": r.rank_score,
                "entry_eligible": r.entry_eligible,
                "hard_block": r.hard_block,
                "soft_reason": r.soft_reason,
                "passed": r.signal.passed,
                "regime_native": r.regime_native,
                "reachability_surplus": r.reachability_surplus,
                "selection_confidence": r.selection_confidence,
                # Diagnostics (existing values computed in rank_setup_signal)
                "base_score": getattr(r, "base_score", None),
                "momentum_boost": getattr(r, "momentum_boost", None),
                "reachability_multiplier": getattr(r, "reachability_multiplier", None),
                "expected_move_pct": getattr(r.signal, "expected_move_pct", None),
                "required_target_pct": getattr(r.signal, "required_target_pct", None),
                "target_gap_pct": getattr(r, "target_gap_pct", None),
                "regime": regime,
                "memory_delta": None,  # filled by enrichment in evaluate_all for global pick
                "recent_win_rate": None,
                "m15": None,
                "m30": None,
                "m60": None,
            }
            for r in ranked_list
        ]

        executable = [r for r in ranked_list if r.signal.setup_name in enabled_names]
        best_ranked = pick_best_ranked(executable) or pick_best_ranked(ranked_list)
        if best_ranked is None:
            meta["hard_block"] = "NO_CANDIDATES"
            return None, signals, meta
        meta["setup_evidence_delta"] = evidence_rank_delta(measurements)

        meta["reachability_surplus"] = best_ranked.reachability_surplus
        meta["selection_confidence"] = best_ranked.selection_confidence
        meta["best_rank_score"] = best_ranked.rank_score
        meta["best_setup"] = best_ranked.signal.setup_name
        entry_eligible = bool(best_ranked.entry_eligible)
        soft_reason = best_ranked.soft_reason
        hard_block = best_ranked.hard_block
        selection_confidence = best_ranked.selection_confidence

        # Architecture v2 (2026-08-11): "all-timeframes-must-agree" is exactly
        # the opinion-gate pattern the ranking architecture forbids. MTF
        # (dis)agreement is real evidence, but it now moves rank_score/size
        # down instead of setting entry_eligible=False. A down 5m/15m trend
        # no longer excludes BTC/ETH/SOL/XRP from ranking — it makes that
        # candidate compete with a real handicap and, via
        # scalp_dynamic_sizing.py, trade smaller if it still wins the pick.
        mtf_penalty_mult = 1.0
        mtf_conflict_reason: str | None = None
        if _mtf_confirmation_gate_enabled():
            if mtf_5m_aligned is False:
                mtf_conflict_reason = "MTF_5M_NOT_ALIGNED_RANKED"
                mtf_penalty_mult *= float(os.getenv("SCALP_MTF_5M_CONFLICT_RANK_MULT", "0.40"))
            if _mtf_require_15m() and mtf_15m_aligned is False:
                mtf_conflict_reason = mtf_conflict_reason or "MTF_15M_NOT_ALIGNED_RANKED"
                mtf_penalty_mult *= float(os.getenv("SCALP_MTF_15M_CONFLICT_RANK_MULT", "0.55"))
            if mtf_penalty_mult < 1.0:
                # MTF is a residual handicap only. Multiplying a typically
                # negative EV_10s primary would invert the penalty.
                old_score = float(best_ranked.rank_score)
                primary = float((getattr(best_ranked, "rank_components", None) or {}).get("EV_10s") or old_score)
                tie = old_score - primary
                rank_score_penalized = round(primary + tie * mtf_penalty_mult, 8)
                from dataclasses import replace as _replace

                best_ranked = _replace(best_ranked, rank_score=rank_score_penalized)
                soft_reason = soft_reason or mtf_conflict_reason
                selection_confidence = f"{selection_confidence}_mtf_conflict_ranked"
                with contextlib.suppress(Exception):
                    from backend.services.binance_scalp.config import get_scalp_config
                    from backend.services.scalp_gate_telemetry import record_gate_event

                    _db = get_scalp_config().database_path
                    record_gate_event(
                        _db,
                        gate_id="MTF_CONFLICT_RANKED",
                        symbol=sym,
                        outcome="ranked",
                        setup=best_ranked.signal.setup_name,
                        detail=f"{mtf_conflict_reason} mult={mtf_penalty_mult}",
                    )

        meta["best_rank_score"] = best_ranked.rank_score
        meta["entry_eligible"] = entry_eligible
        meta["hard_block"] = hard_block
        meta["soft_reason"] = soft_reason
        meta["selection_confidence"] = selection_confidence
        meta["mtf_penalty_mult"] = mtf_penalty_mult
        meta["arm_penalty_mult"] = float(best_ranked.arm_penalty_mult)
        meta["regime_mismatch"] = bool(best_ranked.regime_mismatch)
        meta["symbol_stall_risk"] = bool(best_ranked.symbol_stall_risk)
        meta["microstructure_adjustment"] = float(best_ranked.microstructure_adjustment)
        meta["static_rank_score"] = float(getattr(best_ranked, "raw_rank_score", 0.0) or 0.0)
        meta["learned_adjustment"] = float(getattr(best_ranked, "learned_adjustment", 0.0) or 0.0)
        meta["rank_components"] = dict(getattr(best_ranked, "rank_components", None) or {})
        meta["selection_version"] = (getattr(best_ranked, "rank_components", None) or {}).get("selection_version")
        _ev = getattr(best_ranked, "micro_ev", None) or {}
        for _ek in ("EV_1s", "EV_5s", "EV_10s", "EV_30s", "EV_60s"):
            if _ev.get(_ek) is not None:
                meta[_ek] = _ev.get(_ek)
        micro_q = 1.0
        with contextlib.suppress(Exception):
            ctx_micro = (best_ranked.signal.setup_context or {}) if best_ranked.signal else {}
            adverse = float(ctx_micro.get("p_adverse_move") or ctx_micro.get("adverse_selection_score") or 0.0)
            ev10 = float(ctx_micro.get("EV_10s") or 0.0)
            from backend.services.binance_scalp.config import get_scalp_config
            from backend.services.binance_scalp.scalp_micro_learning import micro_learning_adjustments
            from backend.services.microstructure_engine import compute_features

            mf = compute_features(sym) or {}
            learned = micro_learning_adjustments(
                get_scalp_config().database_path,
                symbol=sym,
                ofi_5s=float(mf.get("ofi_5s") or 0.0),
                obi_l5=float(mf.get("obi_l5") or 0.0),
                adverse_selection_score=float(mf.get("adverse_selection_score") or 0.0),
            )
            micro_q = float(learned.get("size_mult") or 1.0)
            micro_q *= max(0.70, min(1.15, 1.0 - 0.35 * adverse + 200.0 * ev10))
        meta["micro_quality_mult"] = round(max(0.70, min(1.15, micro_q)), 4)
        meta["strategy_passed"] = bool(best_ranked.signal.passed)
        meta["entry_owner"] = "strategy" if best_ranked.signal.passed else "ranking_ev"
        meta["ml_role"] = "rank_size"
        meta["decision_policy_version"] = "scalp_path_aware_v1"

        meta["stage_ms"] = stage_ms
        if entry_eligible:
            entry_sig = prepare_entry_signal(best_ranked, ctx)
            return entry_sig, signals, meta

        # Mechanical safety hard_block only — return best scored signal for
        # status/diagnostics; not executable.
        display_sig = best_ranked.signal
        return display_sig, signals, meta

    def _mtf_trend_confirmation(self, symbol: str, interval: str) -> tuple[float | None, bool | None]:
        """Best-effort higher-TF trend confirmation for long-only SCALP.

        Compares mean close of the last 3 bars vs the 3 before them.
        "Aligned" means trend_pct > 0. Returns (None, None) when history is
        insufficient — callers must not treat that as flat/down.
        """
        try:
            sym = symbol.strip().upper()
            if interval == "5m":
                bars = self.klines.get_5m(sym) or []
            elif interval == "15m":
                bars = self.klines.get_15m(sym) or []
            else:
                return None, None
            if len(bars) < 6:
                return None, None
            recent = bars[-3:]
            prior = bars[-6:-3]
            recent_mean = sum(float(b["close"]) for b in recent) / 3.0
            prior_mean = sum(float(b["close"]) for b in prior) / 3.0
            if prior_mean <= 0:
                return None, None
            trend_pct = (recent_mean - prior_mean) / prior_mean
            return trend_pct, trend_pct > 0.0
        except Exception:
            return None, None

    def _current_regime(self, symbol: str, epoch: float, bars: list[dict] | None) -> str:
        """1h regime from kline cache. Returns REGIME_DATA_MISSING on fetch failure (hard-blocks entry)."""
        sym = symbol.strip().upper()
        try:
            bars_1h = self.klines.get_1h(sym) or []
            if len(bars_1h) >= MIN_REGIME_1H_BARS:
                st = classify_scalp_regime(bars_1h, len(bars_1h) - 1)
                if st and st.regime:
                    return st.regime
        except Exception as e:
            logger.warning("[SCALP_REGIME] Failed to determine regime for %s: %s", sym, e)
            return REGIME_DATA_MISSING
        return REGIME_RANGE

    def evaluate_all(
        self,
        *,
        epoch: float,
        notional_usd: float,
    ) -> list[dict[str, Any]]:
        """Evaluate all products; return ranked entry candidates (eligible first)."""
        rows: list[dict[str, Any]] = []
        t_all = time.perf_counter()
        self._prefetch_history(list(self.config.products))
        for sym in self.config.products:
            # Fetch snapshot once and pass it through — evaluate_symbol() would
            # otherwise independently re-fetch the same symbol's depth via
            # self.reader.read(sym) internally (snap = snap or self.reader.read(sym)),
            # doubling Binance /api/v3/depth calls per symbol per cycle for no
            # benefit (same 5s cycle, no ranking/scoring logic depends on which
            # fetch is used). Pure duplicate-call elimination; no logic change.
            t_sym = time.perf_counter()
            snap = self.reader.read(sym)
            if snap is None:
                logger.info("SCALP_EVAL_TIMING symbol=%s snap=None elapsed_ms=%.0f", sym, (time.perf_counter() - t_sym) * 1000.0)
                continue
            best, all_sigs, meta = self.evaluate_symbol(sym, epoch=epoch, notional_usd=notional_usd, snap=snap)
            if best is None and not meta.get("ranked"):
                logger.info("SCALP_EVAL_TIMING symbol=%s no_ranked elapsed_ms=%.0f", sym, (time.perf_counter() - t_sym) * 1000.0)
                continue
            t_ev = time.perf_counter()
            mh_ev = cached_multi_horizon_ev(sym, "scalp").to_dict()
            ev_ms = round((time.perf_counter() - t_ev) * 1000.0, 1)
            row = {
                "symbol": sym,
                "snap": snap,
                "signal": best,
                "all_signals": [s.as_dict() for s in all_sigs],
                "rank_meta": meta,
                "rank_score": float(meta.get("best_rank_score") or 0.0),
                "entry_eligible": bool(meta.get("entry_eligible")),
                "hard_block": meta.get("hard_block"),
                "best_setup": meta.get("best_setup"),
                "soft_reason": meta.get("soft_reason"),
                "reachability_surplus": meta.get("reachability_surplus"),
                "selection_confidence": meta.get("selection_confidence"),
                # EV-sizing inputs (scalp_dynamic_sizing.py) — never gates.
                "strategy_passed": meta.get("strategy_passed"),
                "arm_penalty_mult": meta.get("arm_penalty_mult", 1.0),
                "mtf_penalty_mult": meta.get("mtf_penalty_mult", 1.0),
                "regime_mismatch": meta.get("regime_mismatch", False),
                "symbol_stall_risk": meta.get("symbol_stall_risk", False),
                "microstructure_adjustment": meta.get("microstructure_adjustment", 0.0),
                "learned_adjustment": meta.get("learned_adjustment", 0.0),
                "static_rank_score": meta.get("static_rank_score", 0.0),
                "EV_1s": meta.get("EV_1s"),
                "EV_5s": meta.get("EV_5s"),
                "EV_10s": meta.get("EV_10s"),
                "EV_30s": meta.get("EV_30s"),
                "EV_60s": meta.get("EV_60s"),
                "rank_components": meta.get("rank_components") or {},
                "selection_version": meta.get("selection_version"),
                # Item p11: composite EV across SCALP's realistic 30s-20m
                # holding horizons — diagnostic/ranking evidence only, never
                # a gate. TTL-cached (~5min) so this cheap-but-not-free
                # sqlite lookup doesn't run on every ~5s evaluate_all() tick.
                "multi_horizon_ev": mh_ev,
            }
            rows.append(row)
            _LAST_RANKING_META_BY_SYMBOL[sym] = dict(row)
            _publish_ranking_meta_to_redis(sym, row, redis_url=self.config.redis_url, prefix=self.config.redis_key_prefix)
            logger.info(
                "SCALP_EVAL_TIMING symbol=%s elapsed_ms=%.0f ev_ms=%s passed=%s reject=%s setups=%s",
                sym,
                (time.perf_counter() - t_sym) * 1000.0,
                ev_ms,
                bool(meta.get("strategy_passed")),
                meta.get("soft_reason") or meta.get("hard_block") or "",
                ",".join(f"{k}={v}" for k, v in (meta.get("stage_ms") or {}).items()),
            )

        rows.sort(
            key=lambda r: (
                -int(bool(r.get("entry_eligible"))),
                -float(r.get("rank_score") or 0.0),
                -float((r.get("rank_meta") or {}).get("reachability_surplus") or 0.0),
            )
        )
        logger.info(
            "SCALP_EVALUATE_ALL_DONE symbols=%s elapsed_ms=%.0f",
            [r.get("symbol") for r in rows],
            (time.perf_counter() - t_all) * 1000.0,
        )
        return rows

    def _prefetch_history(self, symbols: list[str]) -> None:
        """Independent kline I/O for all four coins. Cache writes are locked."""
        if not symbols:
            return

        def _one(sym: str) -> None:
            try:
                self.klines.get(sym)
                self.klines.get_5m(sym)
                self.klines.get_15m(sym)
                self.klines.get_1h(sym)
            except Exception as exc:
                logger.warning("SCALP_PREFETCH_FAILED symbol=%s err=%s", sym, exc)

        t0 = time.perf_counter()
        workers = min(4, max(1, len(symbols)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, symbols))
        logger.info("SCALP_PREFETCH_DONE symbols=%s elapsed_ms=%.0f", symbols, (time.perf_counter() - t0) * 1000.0)

    def strategy_inventory(self) -> dict[str, Any]:
        disabled = self.config.disabled_strategies
        return {
            "all": list(STRATEGY_NAMES),
            "enabled": [s.name for s in enabled_strategies(self.config)],
            "disabled": sorted(disabled),
        }
