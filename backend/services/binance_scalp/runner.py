"""Entry point for mystic-scalp-paper.service — disabled by default."""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("binance_scalp_runner")


def main() -> int:
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine
    from backend.utils.process_singleton import SCALP_RUNNER_PIDFILE, ProcessAlreadyRunning, acquire_process_singleton

    try:
        acquire_process_singleton(SCALP_RUNNER_PIDFILE, label="SCALP runner")
    except ProcessAlreadyRunning:
        return 1

    cfg = get_scalp_config()
    cfg.assert_no_live_trading()
    mode = cfg.assert_structural_startup()
    logger.info("SCALP_STRUCTURAL_START mode=%s fill_model=structural_event_queue_v1 live_impossible=true", mode)
    if cfg.calibration_mode:
        logger.info(
            "SCALP_CALIBRATION_MODE active profile=%s products=%s",
            cfg.calibration_profile,
            cfg.products,
        )
    loop_sec = float(os.getenv("SCALP_RANK_INTERVAL_SEC", os.getenv("SCALP_LOOP_SEC", "1")))
    exit_sec = float(os.getenv("SCALP_EXIT_INTERVAL_SEC", "0.25"))
    engine = BinanceScalpPaperEngine()
    try:
        engine.run_loop(interval_sec=loop_sec, exit_interval_sec=exit_sec)
    except KeyboardInterrupt:
        logger.info("scalp paper interrupted")
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
