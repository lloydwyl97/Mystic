import logging

logger = logging.getLogger(__name__)


class SimulationLogger:
    def __init__(self) -> None:
        self._summary = {
            "avg_profit": 0.0,
            "win_rate": 0.5,
            "num_trades": 0,
        }

    def get_summary(self):
        return dict(self._summary)

    # Optional helpers to update during tests
    def update(self, avg_profit=None, win_rate=None, num_trades=None):
        if avg_profit is not None:
            self._summary["avg_profit"] = float(avg_profit)
        if win_rate is not None:
            self._summary["win_rate"] = float(win_rate)
        if num_trades is not None:
            self._summary["num_trades"] = int(num_trades)

    def log(self, message):
        logger.info(f"[SIMULATION] {message}")
