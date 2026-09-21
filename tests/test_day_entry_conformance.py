from backend.services.day_entry_conformance import ActualEntry, match_entries
from backend.services.day_production_lifecycle_replay import ReplayCandidate


def _cand(epoch: int, symbol: str, *, accepted: bool = True, reason: str = "ACCEPTED") -> ReplayCandidate:
    return ReplayCandidate(
        epoch=epoch,
        symbol=symbol,
        decision_id=f"d_{symbol}_{epoch}",
        accepted=accepted,
        first_reason=reason,
        cash=200.0,
        slot_occupancy=1,
        intact_4h_open=1,
        position_open=False,
        pending_order="",
        cooldown_until=0.0,
        fourh_intact=True,
        veto_ev=0.001,
        p_buy=0.4,
        final_selection_score=0.1,
    )


def test_match_exact_and_bar_cadence_and_miss():
    t0 = 1_788_000_000
    actual = [
        ActualEntry(t0, "", "BTCUSDT", "a1", 0.001, 70_000.0, 70.0),
        ActualEntry(t0 + 900, "", "ETHUSDT", "a2", 0.02, 2_400.0, 48.0),
        ActualEntry(t0 + 3600, "", "SOLUSDT", "a3", 0.5, 100.0, 50.0),
    ]
    replay = [
        _cand(t0, "BTCUSDT"),
        _cand(t0 + 780, "ETHUSDT"),
        _cand(t0 + 1800, "XRPUSDT"),
    ]
    report = match_entries(actual, replay, window_days=1.0)
    assert report.exact_matches == 1
    assert report.loop_cadence_matches == 0
    assert report.bar_cadence_matches == 1
    assert report.missed == 1
    assert report.false_positives == 1
    kinds = {m.kind for m in report.mismatches}
    assert "actual_entry_missed_by_replay" in kinds
    assert "replay_only_extra_entry" in kinds


def test_first_divergence_uses_stored_reject():
    t0 = 1_788_000_000
    actual = [ActualEntry(t0, "", "BTCUSDT", "a1", 0.001, 70_000.0, 70.0)]
    replay = [_cand(t0 + 10_000, "ETHUSDT")]
    rejects = [(t0, "BTCUSDT", "late_4h_rise_no_buy", "LATE_4H_RISE_1H_WEAK")]
    report = match_entries(actual, replay, rejects=rejects, window_days=1.0)
    miss = next(m for m in report.mismatches if m.kind == "actual_entry_missed_by_replay")
    extra = next(m for m in report.mismatches if m.kind == "replay_only_extra_entry")
    assert miss.first_divergence
    assert extra.first_divergence == "NO_STORED_REJECT_NEAR_REPLAY_ENTRY"
