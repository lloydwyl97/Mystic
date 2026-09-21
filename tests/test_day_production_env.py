import os

from backend.services.day_production_env import apply_env_map, runtime_float, snapshot_day_env


def test_apply_env_map_overrides_module_default_path():
    apply_env_map({"DAY_TARGET_NOTIONAL_PER_SLOT_USD": "4000"}, override=True)
    assert runtime_float("DAY_TARGET_NOTIONAL_PER_SLOT_USD", 2500.0) == 4000.0
    snap = snapshot_day_env()
    assert snap["DAY_TARGET_NOTIONAL_PER_SLOT_USD"] == "4000"
    os.environ.pop("DAY_TARGET_NOTIONAL_PER_SLOT_USD", None)


def test_runtime_float_uses_env_not_default():
    os.environ["DAY_MAX_DEPLOYED_USD"] = "9000"
    assert runtime_float("DAY_MAX_DEPLOYED_USD", 1.0) == 9000.0
    os.environ.pop("DAY_MAX_DEPLOYED_USD", None)
    assert runtime_float("DAY_MAX_DEPLOYED_USD", 1.0) == 1.0
