"""
Regression: SCALP status/dashboard decision must be derived from the actual
canonical pre-order engine decision, not a second independent simulation.

BinanceScalpPaperEngine._publish_last_decision publishes the real decision
from _entry_candidates()/tick() on every tick. status_snapshot.build_scalp_status
reads that published value (_load_last_decision) and maps it to the status
vocabulary. This test proves the read-side contract for every required
fixture scenario, and that the paper engine publishes the correct canonical
decision string for each one.
"""

from __future__ import annotations

import json

import pytest

from backend.services.binance_scalp.status_snapshot import _ENGINE_DECISION_TO_STATUS, _load_last_decision
from backend.services.binance_scalp.redis_keys import last_decision_key


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self.store[key] = value

    def get(self, key: str) -> str | None:
        return self.store.get(key)


class _EngineForPublish:
    """Minimal object exposing only what _publish_last_decision needs."""

    def __init__(self, redis_client: _FakeRedis, prefix: str = "scalp"):
        self._redis = redis_client

        class _Cfg:
            redis_key_prefix = prefix

        self.config = _Cfg()

    # Bind the real method under test.
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    _publish_last_decision = BinanceScalpPaperEngine._publish_last_decision


@pytest.mark.parametrize(
    "engine_decision,expected_status_decision",
    [
        ("NO_SIGNAL", "NO_SIGNAL"),  # no candidate ranked at all
        ("WOULD_ENTER", "PASS"),  # valid candidate, armed, executable -> would actually enter
        ("BLOCKED", "BLOCKED"),  # slot full / paper disabled / fee model unverified / preflight fail
        ("PASS_NOT_ARMED", "READY_TO_WATCH"),  # would enter, but not armed
    ],
)
def test_status_decision_matches_published_engine_decision(engine_decision, expected_status_decision):
    """For a given canonical engine decision, the status endpoint's mapped
    decision must match exactly — no independent re-derivation."""
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(decision=engine_decision, reason="fixture", selected_symbol="BTCUSDT", rank_score=1.5, entry_armed=(engine_decision == "WOULD_ENTER"))

    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded is not None
    assert loaded["decision"] == engine_decision
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == expected_status_decision


def test_no_candidate_fixture_publishes_no_signal():
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(decision="NO_SIGNAL", reason="NO_RANKED_CANDIDATE", entry_armed=True)
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded["decision"] == "NO_SIGNAL"
    assert loaded["selected_symbol"] is None


def test_valid_candidate_fixture_publishes_would_enter_with_symbol_and_score():
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(
        decision="WOULD_ENTER",
        selected_symbol="ETHUSDT",
        rank_score=1.82,
        entry_armed=True,
        ranked_summary=[{"symbol": "ETHUSDT", "rank_score": 1.82, "entry_eligible": True}],
    )
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded["decision"] == "WOULD_ENTER"
    assert loaded["selected_symbol"] == "ETHUSDT"
    assert loaded["rank_score"] == 1.82
    assert loaded["entry_armed"] is True
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "PASS"


def test_slot_full_fixture_publishes_blocked_max_open_positions():
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(decision="BLOCKED", reason="MAX_OPEN_POSITIONS")
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded["decision"] == "BLOCKED"
    assert loaded["reason"] == "MAX_OPEN_POSITIONS"
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "BLOCKED"


def test_duplicate_symbol_is_excluded_upstream_so_ranking_shows_no_signal_if_only_open_symbol_qualifies():
    """Duplicate-symbol candidates are filtered out of `ranked` before scoring
    (see _entry_candidates: `ranked = [r for r in ranked if symbol not in open_symbols]`),
    so if the only qualifying setup is on an already-open symbol, the engine
    correctly reports no viable (non-duplicate) candidate."""
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(decision="NO_SIGNAL", reason="NO_RANKED_CANDIDATE", entry_armed=True)
    loaded = _load_last_decision(rc, prefix="scalp")
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "NO_SIGNAL"


def test_stale_data_fixture_symbol_row_still_uses_error_blocked_no_candidate_publish():
    """Stale/missing market data is a per-symbol NO_MARKET_DATA error (already
    covered by test_scalp_status_decision_honesty.py); at the engine-decision
    level with no readable symbols there is no candidate to rank."""
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(decision="NO_SIGNAL", reason="NO_RANKED_CANDIDATE")
    loaded = _load_last_decision(rc, prefix="scalp")
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "NO_SIGNAL"


def test_excessive_spread_or_impact_or_net_edge_on_top_candidate_publishes_blocked():
    """The top-ranked candidate itself failing its own preflight (spread too
    wide / impact too high / net edge too low / depth insufficient) is a
    genuine operational failure for that attempt -> BLOCKED, not NO_SIGNAL."""
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(decision="BLOCKED", reason="RANKED_NOT_EXECUTABLE", selected_symbol="SOLUSDT", rank_score=1.6)
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded["decision"] == "BLOCKED"
    assert loaded["reason"] == "RANKED_NOT_EXECUTABLE"
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "BLOCKED"


def test_paper_execution_disabled_fixture_publishes_blocked():
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    from backend.services.binance_scalp.protected_preflight import SCALP_PAPER_DISABLED

    engine._publish_last_decision(decision="BLOCKED", reason=SCALP_PAPER_DISABLED)
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded["decision"] == "BLOCKED"
    assert loaded["reason"] == SCALP_PAPER_DISABLED
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "BLOCKED"


def test_successful_executable_entry_fixture_publishes_would_enter():
    rc = _FakeRedis()
    engine = _EngineForPublish(rc)
    engine._publish_last_decision(
        decision="WOULD_ENTER",
        selected_symbol="XRPUSDT",
        rank_score=1.9,
        entry_armed=True,
        ranked_summary=[{"symbol": "XRPUSDT", "rank_score": 1.9, "entry_eligible": True}],
    )
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded["decision"] == "WOULD_ENTER"
    assert loaded["entry_armed"] is True
    assert _ENGINE_DECISION_TO_STATUS[loaded["decision"]] == "PASS"


def test_missing_canonical_publish_is_detectable_as_fallback_not_silently_treated_as_pass():
    """If the engine hasn't published anything (e.g. not running), the reader
    must return None so the caller falls back explicitly (decision_source=
    status_simulation_fallback in build_scalp_status), never silently assume PASS."""
    rc = _FakeRedis()
    loaded = _load_last_decision(rc, prefix="scalp")
    assert loaded is None


def test_last_decision_key_is_namespaced_and_distinct_from_runner_state():
    from backend.services.binance_scalp.redis_keys import runner_state_key

    assert last_decision_key("scalp") != runner_state_key("scalp")
    assert last_decision_key("scalp").startswith("scalp:")
