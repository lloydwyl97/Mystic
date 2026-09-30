"""SCALP V2 engine package.

Exit contract and opportunity identity for the live SCALP V2 engine. SCALP V2
entries are placed by the portfolio engine (execute_scalp_v2_buy_live) and
exited only by scalp_v2.exit_evaluator.

Key modules:
    opportunity     — ScalpOpportunityId: deterministic ID, prevents same-move recycling
    exit_calibration — Evidence-based parameter overrides from candle recovery analysis
"""
