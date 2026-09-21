"""SCALP V2 engine package.

Evidence-based exit calibration and opportunity identity for the SCALP V2
candidate engine. This package never places, cancels, or mutates live orders,
positions, cash balances, or accounting. All order authority remains with
LEGACY_DAY_LIVE until an explicit qualification and promotion commit.

Key modules:
    opportunity     — ScalpOpportunityId: deterministic ID, prevents same-move recycling
    exit_calibration — Evidence-based parameter overrides from candle recovery analysis
"""
