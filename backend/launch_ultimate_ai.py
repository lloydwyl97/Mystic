#!/usr/bin/env python3
"""
Ultimate AI Trading System Launcher
Launches the complete AI crypto trading machine with core modules.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "launcher.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ultimate_launcher")

# Import external modules (may not be available)
try:
    from capital_allocator import allocate_capital  # type: ignore[import-not-found]
except ImportError:
    allocate_capital = None  # type: ignore[assignment]

try:
    from db_logger import init_db  # type: ignore[import-not-found]
except ImportError:
    init_db = None  # type: ignore[assignment]

try:
    from hyper_tuner import optimize_rsi_ema_breakout  # type: ignore[import-not-found]
except ImportError:
    optimize_rsi_ema_breakout = None  # type: ignore[assignment]

try:
    from trade_logger import get_recent_trades  # type: ignore[import-not-found]
except ImportError:
    get_recent_trades = None  # type: ignore[assignment]

try:
    from watchdog import TradingWatchdog  # type: ignore[import-not-found]
except ImportError:
    TradingWatchdog = None  # type: ignore[assignment]


class UltimateAILauncher:
    def __init__(self) -> None:
        self.processes: dict[str, dict[str, object]] = {}
        self._stop_event = threading.Event()
        base_dir = Path(__file__).resolve().parent

        self.services: dict[str, dict[str, object]] = {
            "main_api": {
                "script": str(base_dir / "dashboard_api.py"),
                "port": 8000,
                "description": "Main FastAPI Dashboard",
            },
            "trade_logger": {
                "script": str(base_dir / "db_logger.py"),
                "port": None,
                "description": "Trade Logging System",
            },
            "strategy_mutator": {
                "script": str(base_dir / "mutator.py"),
                "port": None,
                "description": "Strategy Evolution Engine",
            },
            "hyper_optimizer": {
                "script": str(base_dir / "hyper_tuner.py"),
                "port": None,
                "description": "Hyperparameter Optimization",
            },
            "watchdog": {
                "script": str(base_dir / "watchdog.py"),
                "port": None,
                "description": "Health Monitoring",
            },
        }

    def _ascii_banner(self) -> str:
        line = "=" * 60
        return f"\n{line}\nMYSTIC AI TRADING SYSTEM LAUNCHER\nProduction-safe, Windows-friendly process manager\n{line}\n"

    def print_banner(self) -> None:
        logger.info(self._ascii_banner().strip("\n"))

    def check_dependencies(self) -> bool:
        logger.info("Checking system dependencies")
        required_files = [
            "models.py",
            "db_logger.py",
            "strategy_leaderboard.py",
            "mutator.py",
            "position_sizer.py",
            "capital_allocator.py",
            "yield_rotator.py",
            "watchdog.py",
            "dashboard_api.py",
            "hyper_tuner.py",
        ]
        base_dir = Path(__file__).resolve().parent
        missing = [f for f in required_files if not (base_dir / f).exists()]
        if missing:
            logger.error("Missing required files: %s", missing)
            return False
        logger.info("All required files present")
        return True

    def initialize_database(self) -> bool:
        logger.info("Initializing trade database")
        try:
            # Direct imports for production
            init_db()
            logger.info("Database initialized successfully")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Database initialization failed: %s", e)
            return False
        else:
            return True

    def _service_log_paths(self, service_name: str) -> tuple[Path, Path]:
        stdout_path = LOG_DIR / f"{service_name}.out.log"
        stderr_path = LOG_DIR / f"{service_name}.err.log"
        return stdout_path, stderr_path

    def _rotate_log_if_needed(self, log_path: Path, max_size_mb: int = 10) -> Path:
        """Rotate log file if it exceeds max size. Returns path to use."""
        if not log_path.exists():
            return log_path

        file_size_mb = log_path.stat().st_size / (1024 * 1024)
        if file_size_mb < max_size_mb:
            return log_path

        # Rotate log: .log -> .log.1, .log.1 -> .log.2, etc.
        backup_count = 3
        for i in range(backup_count - 1, 0, -1):
            old_backup = log_path.with_suffix(f".log.{i}")
            new_backup = log_path.with_suffix(f".log.{i + 1}")
            if old_backup.exists():
                if new_backup.exists():
                    new_backup.unlink()
                old_backup.rename(new_backup)

        # Move current log to .log.1
        backup_path = log_path.with_suffix(".log.1")
        if backup_path.exists():
            backup_path.unlink()
        log_path.rename(backup_path)

        # Return original path (now empty) for new writes
        return log_path

    def start_service(self, service_name: str, service_config: dict[str, object]) -> bool:
        stdout_f = None
        stderr_f = None
        added_to_processes = False
        try:
            script = str(service_config["script"])
            if not Path(script).exists():
                logger.error("Service %s script not found: %s", service_name, script)
                return False

            logger.info("Starting %s: %s", service_name, service_config.get("description", ""))

            stdout_path, stderr_path = self._service_log_paths(service_name)

            # Rotate logs if they're too large (10MB limit for startup logs)
            stdout_path = self._rotate_log_if_needed(stdout_path, max_size_mb=10)
            stderr_path = self._rotate_log_if_needed(stderr_path, max_size_mb=10)

            # Note: Files must stay open for subprocess.Popen lifetime, so we don't use context manager
            stdout_f = stdout_path.open("a", encoding="utf-8")
            stderr_f = stderr_path.open("a", encoding="utf-8")
            cmd = [sys.executable, script]

            # Start the process detached from launcher stdio; log to files
            process = subprocess.Popen(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=str(Path(script).parent),
                text=True,
            )

            # Give the process a moment to fail early if it will
            time.sleep(1.5)
            if process.poll() is None:
                self.processes[service_name] = {
                    "process": process,
                    "config": service_config,
                    "start_time": datetime.now(timezone.utc),
                    "stdout": stdout_f,
                    "stderr": stderr_f,
                }
                added_to_processes = True
                logger.info("%s started successfully (PID: %s)", service_name, process.pid)
            else:
                # process exited immediately; close logs and report failure
                logger.error("%s failed to start (exit code: %s)", service_name, process.returncode)
                try:
                    if stdout_f:
                        stdout_f.close()
                    if stderr_f:
                        stderr_f.close()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error starting %s: %s", service_name, e)
            # ensure files are closed if we opened them but didn't store them
            if not added_to_processes:
                try:
                    if stdout_f:
                        stdout_f.close()
                    if stderr_f:
                        stderr_f.close()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
            return False

    def start_all_services(self) -> bool:
        logger.info("Starting AI Trading services")
        started, failed = [], []
        for svc, cfg in self.services.items():
            if self.start_service(svc, cfg):
                started.append(svc)
            else:
                failed.append(svc)
        logger.info("Service startup summary: started=%d, failed=%d", len(started), len(failed))
        if started:
            logger.info("Running services: %s", ", ".join(started))
        if failed:
            logger.warning("Failed services: %s", ", ".join(failed))
        return len(failed) == 0

    def check_service_health(self) -> None:
        logger.info("Checking service health")
        for svc, info in self.processes.items():
            proc: subprocess.Popen = info["process"]  # type: ignore[index]
            if proc.poll() is None:
                logger.info("%s: Running (PID: %s)", svc, proc.pid)
            else:
                logger.warning("%s: Stopped (code: %s)", svc, proc.returncode)

    def monitor_services(self) -> None:
        logger.info("Service monitor started")
        while not self._stop_event.is_set():
            try:
                for svc, info in list(self.processes.items()):
                    proc: subprocess.Popen = info["process"]  # type: ignore[index]
                    if proc.poll() is not None:
                        logger.warning("%s has stopped, attempting restart", svc)
                        self._close_proc_logs(info)
                        if self.start_service(svc, info["config"]):  # type: ignore[call-arg]
                            logger.info("%s restarted successfully", svc)
                        else:
                            logger.error("Failed to restart %s", svc)
                self._stop_event.wait(timeout=30.0)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception("Monitoring loop error: %s", e)
                self._stop_event.wait(timeout=30.0)
        logger.info("Service monitor stopped")

    def stop_all_services(self) -> None:
        logger.info("Stopping all services")
        for svc, info in list(self.processes.items()):
            proc: subprocess.Popen = info["process"]  # type: ignore[index]
            if proc.poll() is None:
                logger.info("Stopping %s (PID: %s)", svc, proc.pid)
                try:
                    self._terminate_process(proc)
                    logger.info("%s stopped", svc)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception("Error stopping %s: %s", svc, e)
            self._close_proc_logs(info)
            self.processes.pop(svc, None)
        logger.info("All services stopped")

    def _terminate_process(self, proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing")
            proc.kill()
            proc.wait(timeout=5)

    def _close_proc_logs(self, info: dict[str, object]) -> None:
        for key in ("stdout", "stderr"):
            f = info.get(key)
            if hasattr(f, "close"):
                with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    f.close()  # type: ignore[attr-defined]

    def show_dashboard_info(self) -> None:
        logger.info("DASHBOARD ACCESS")
        logger.info("Main Dashboard:    http://localhost:8000/")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("Strategy API:      http://localhost:8000/api/leaderboard")
        logger.info("Trade History:     http://localhost:8000/api/trades")
        logger.info("System Health:     http://localhost:8000/health")

    def run_optimization(self) -> None:
        logger.info("Running strategy optimization")
        try:
            result = optimize_rsi_ema_breakout(method="genetic", rounds=20)
            if result:
                logger.info(
                    "Optimization completed | best_profit=%.2f | win_rate=%.2f%%",
                    result.get("total_profit", 0.0),
                    100.0 * result.get("win_rate", 0.0),
                )
            else:
                logger.warning("Optimization returned no result")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Optimization error: %s", e)

    def allocate_capital(self) -> None:
        logger.info("Allocating capital")
        try:
            allocations = allocate_capital(10000, method="performance")
            if allocations:
                for strategy, amount in allocations.items():
                    logger.info("Allocation | %s : $%s", strategy, amount)
            else:
                logger.warning("No allocations produced")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Capital allocation error: %s", e)

    def system_health_check(self) -> None:
        logger.info("System health check")
        try:
            watchdog = TradingWatchdog()
            summary = watchdog.get_system_summary()
            logger.info("Overall Health: %s", summary.get("overall_health"))
            logger.info(
                "Healthy Services: %s/%s",
                summary.get("healthy_services"),
                summary.get("total_services"),
            )
            logger.info("Health Percentage: %s%%", summary.get("health_percentage"))
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Health check error: %s", e)

    def view_recent_trades(self) -> None:
        logger.info("Recent trades")
        try:
            trades = get_recent_trades(10)
            if trades:
                for t in trades:
                    logger.info(
                        "%s | %s | %s | $%.2f",
                        t.get("timestamp"),
                        t.get("symbol"),
                        t.get("strategy"),
                        float(t.get("profit_usd", 0)),
                    )
            else:
                logger.info("No recent trades found")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception("Error viewing trades: %s", e)

    def restart_failed_services(self) -> None:
        logger.info("Restarting failed services")
        failed = []
        for svc, info in self.processes.items():
            proc: subprocess.Popen = info["process"]  # type: ignore[index]
            if proc.poll() is not None:
                failed.append(svc)
        if not failed:
            logger.info("No failed services to restart")
            return
        for svc in failed:
            logger.info("Restarting %s", svc)
            # close previous logs if any
            self._close_proc_logs(self.processes[svc])
            self.start_service(svc, self.services[svc])

    def launch_full_system(self) -> None:
        self.print_banner()

        if not self.check_dependencies():
            logger.error("Cannot start: missing dependencies")
            return

        if not self.initialize_database():
            logger.error("Cannot start: database initialization failure")
            return

        if not self.start_all_services():
            logger.warning("Some services failed to start; continuing")

        self.show_dashboard_info()

        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()

        def shutdown_handler(signum: int, _frame: object) -> None:
            logger.info("Shutdown signal received (%s); stopping services", signum)
            self._stop_event.set()
            self.stop_all_services()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                signal.signal(sig, shutdown_handler)

        try:
            while not self._stop_event.is_set():
                time.sleep(1.0)
        finally:
            self._stop_event.set()
            self.stop_all_services()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mystic AI Trading System Launcher")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode (CLI menu)",
    )
    args = parser.parse_args()

    launcher = UltimateAILauncher()

    if args.interactive:
        # Minimal interactive loop with logging output
        MENU = (
            "\n"
            "MYSTIC AI TRADING SYSTEM - INTERACTIVE MODE\n"
            "1) Start All Services\n"
            "2) Stop All Services\n"
            "3) Check Service Health\n"
            "4) Show Dashboard Info\n"
            "5) Run Strategy Optimization\n"
            "6) Allocate Capital\n"
            "7) System Health Check\n"
            "8) View Recent Trades\n"
            "9) Restart Failed Services\n"
            "0) Exit\n"
        )
        while True:
            try:
                print(MENU, end="", flush=True)
                choice = input("Select option (0-9): ").strip()
                if choice == "1":
                    launcher.start_all_services()
                elif choice == "2":
                    launcher.stop_all_services()
                elif choice == "3":
                    launcher.check_service_health()
                elif choice == "4":
                    launcher.show_dashboard_info()
                elif choice == "5":
                    launcher.run_optimization()
                elif choice == "6":
                    launcher.allocate_capital()
                elif choice == "7":
                    launcher.system_health_check()
                elif choice == "8":
                    launcher.view_recent_trades()
                elif choice == "9":
                    launcher.restart_failed_services()
                elif choice == "0":
                    logger.info("Exiting interactive mode")
                    launcher.stop_all_services()
                    break
                else:
                    logger.warning("Invalid option: %s", choice)
            except KeyboardInterrupt:
                logger.info("Interactive mode interrupted by user")
                launcher.stop_all_services()
                break
    else:
        launcher.launch_full_system()


if __name__ == "__main__":
    main()
