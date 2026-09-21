from backend.services.day_experiment_registry import EMBARGO_SEC, LOCKED_TEST_FRAC, registry
from backend.services.day_net_challengers import (
    acceptance_test,
    audit_stored_145,
    expanding_folds,
    lock_slice,
    multiple_testing_note,
    purge_mask,
)
from backend.services.day_profit_attribution import CandidateLabel, GroupAttribution


def test_folds_lock_and_purge():
    n = 100
    a, b = lock_slice(n)
    assert b == 100
    assert a == int(n * (1 - LOCKED_TEST_FRAC)) or a == n - max(1, int(n * LOCKED_TEST_FRAC))
    folds = expanding_folds(n)
    assert folds
    assert all(train_end <= val_start < val_end for train_end, val_start, val_end in folds)
    assert EMBARGO_SEC >= 4 * 3600


def test_purge_drops_overlapping_outcome_intervals():
    def _attr(start, end):
        lab = CandidateLabel(
            decision_group_id="g",
            symbol="ETHUSDT",
            outcome_class="ranking_loser",
            labeled=True,
            label_kind="counterfactual_isolated",
            entry_epoch=start,
            exit_epoch=end,
            entry_price=1.0,
            exit_price=1.0,
            fill_status="counterfactual",
            filled_qty=1.0,
            latency_sec=0.0,
            gross_markout_bps={},
            mfe_bps=1.0,
            mae_bps=-1.0,
            time_to_mfe_sec=1.0,
            time_to_mae_sec=1.0,
            production_exit_gross_bps=0.0,
            commission_bps=1.0,
            spread_bps=1.0,
            slippage_bps=1.0,
            net_bps=-1.0,
            net_usd=-0.1,
            hold_sec=end - start,
            capital_hours=1.0,
            capture_ratio=None,
            exit_reason="X",
            favorable_first=None,
            interval_start=start,
            interval_end=end,
        )
        return GroupAttribution(
            decision_group_id="g",
            selected_symbol="ETHUSDT",
            selected_action="BUY_ETHUSDT",
            labels={"ETHUSDT": lab},
            best_eligible_symbol="HOLD",
            best_eligible_net_bps=0.0,
            regret_vs_best_bps=1.0,
            regret_vs_hold_bps=1.0,
            opportunity=False,
            selected_positive=False,
            root_cause="other",
        )

    rows = [_attr(0, 5000), _attr(0, 100)]
    kept = purge_mask(rows, [0, 1], val_start_epoch=1000)
    assert kept == [1]


def test_acceptance_rejects_negative_challenger():
    locked = {
        "champion": {"net_usd": -1.0, "net_bps": -10.0, "profit_factor": 0.5, "max_drawdown_bps": 20.0},
        "calibrated_score": {"net_usd": -2.0, "net_bps": -20.0, "profit_factor": 0.4, "max_drawdown_bps": 30.0},
        "pooled_ridge": {"net_usd": -3.0, "net_bps": -30.0, "profit_factor": 0.2, "max_drawdown_bps": 40.0},
    }
    folds = [
        {
            "champion": {"net_usd": -1.0},
            "calibrated_score": {"net_usd": -2.0},
            "pooled_ridge": {"net_usd": -3.0},
        }
    ]
    out = acceptance_test(locked, folds)
    assert out["calibrated_score"]["passed"] is False
    assert out["pooled_ridge"]["passed"] is False
    assert out["calibrated_score"]["promote"] is False


def test_145_audit_does_not_require_fitting():
    note = multiple_testing_note(10, 10)
    assert note["status"] == "sample_size_insufficient"
    reg = registry()
    assert "A" in {a["arm_id"] for a in reg["arms"]}
    assert audit_stored_145([])["n"] == 0
