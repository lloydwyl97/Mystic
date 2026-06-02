import hashlib
import importlib.util
import inspect
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Initialize logger
logger = logging.getLogger(__name__)

# Configuration
MAX_EXECUTION_TIME = 30  # seconds
MAX_HISTORY_RECORDS = 1000
MAX_FILE_SIZE = 1024 * 1024  # 1MB


class AutoExecutor:
    """Autonomous module execution engine with safety and isolation"""

    def __init__(self, base_dir: str | None = None) -> None:
        self.executed_modules: list[str] = []
        self.execution_history: list[dict[str, Any]] = []
        self.base_dir = base_dir or str(Path(__file__).resolve().parent)
        self.generated_modules_dir = str(Path(self.base_dir) / "generated_modules")
        self.health_issues: list[str] = []

        logger.info(f"AutoExecutor initialized with base_dir: {self.base_dir}")
        logger.info(f"Generated modules directory: {self.generated_modules_dir}")

    def run_generated_module(self, file_path: str, timeout: int = MAX_EXECUTION_TIME) -> Any | None:
        """Execute a generated module dynamically with safety checks and timeouts"""
        start_time = time.time()

        try:
            # Validate file exists and is reasonable size
            if not self._validate_module_file(file_path):
                return None

            logger.info(f"Loading module: {file_path}")

            # Create unique module name to avoid collisions
            module_name = self._generate_unique_module_name(file_path)

            # Load the module dynamically with safety checks
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                error_msg = f"Failed to create spec for {file_path}"
                logger.error(error_msg)
                self._record_execution(file_path, None, "failed", error_msg)
                return None

            if spec.loader is None:
                error_msg = f"No loader available for {file_path}"
                logger.error(error_msg)
                self._record_execution(file_path, None, "failed", error_msg)
                return None

            module = importlib.util.module_from_spec(spec)

            # Execute module with timeout protection
            result = self._execute_with_timeout(spec, module, timeout)

            if result is not None and result.get("status") == "timeout":
                error_msg = f"Module execution timed out after {timeout}s"
                logger.warning(error_msg)
                self._record_execution(file_path, None, "timeout", error_msg)
                return None

            execution_result = result.get("result") if result else None
            execution_status = result.get("status", "success") if result else "success"

            # Record execution regardless of truthiness
            self._record_execution(file_path, execution_result, execution_status)
            self.executed_modules.append(file_path)

            logger.info(f"Module executed successfully: {file_path}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            error_msg = f"Error executing {file_path}: {e}"
            logger.error(error_msg, exc_info=True)  # Full traceback in logs
            self._record_execution(file_path, None, "failed", str(e))
            return None
        else:
            return execution_result
        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Module execution took {execution_time:.2f}s")

    def _validate_module_file(self, file_path: str) -> bool:
        """Validate that the module file is safe to execute"""
        try:
            if not Path(file_path).exists():
                logger.error(f"Module file does not exist: {file_path}")
                return False

            if not file_path.endswith(".py"):
                logger.error(f"File is not a Python module: {file_path}")
                return False

            # Check file size
            file_size = Path(file_path).stat().st_size
            if file_size > MAX_FILE_SIZE:
                logger.error(f"Module file too large ({file_size} bytes): {file_path}")
                return False

            # Basic syntax check
            file_path_obj = Path(file_path)
            with file_path_obj.open(encoding="utf-8") as f:
                try:
                    compile(f.read(), str(file_path_obj), "exec")
                except SyntaxError as e:
                    logger.exception(f"Syntax error in module {file_path}: {e}")
                    return False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error validating module file {file_path}: {e}")
            return False
        else:
            return True

    def _generate_unique_module_name(self, file_path: str) -> str:
        """Generate a unique module name to avoid sys.modules collisions"""
        try:
            # Use file path hash to create unique name
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            base_name = Path(file_path).name.replace(".py", "")
            result = f"autogen_{base_name}_{file_hash}"
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error generating unique module name: {e}")
            return f"autogen_{int(time.time())}"
        else:
            return result

    def _execute_with_timeout(self, spec: Any, module: Any, timeout: int) -> dict[str, Any] | None:
        """Execute module with timeout protection"""
        result = {"status": "unknown", "result": None}

        use_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
        old_handler = None

        def timeout_handler(_signum, _frame):
            msg = f"Module execution timed out after {timeout}s"
            raise TimeoutError(msg)

        try:
            # Set timeout alarm if available (POSIX)
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

            # Ensure module is importable under its spec name
            try:
                module_name = getattr(spec, "name", None)
                if module_name:
                    sys.modules[module_name] = module
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # If registering fails, proceed without failing execution
                logger.debug("Could not register module in sys.modules; continuing without registration")

            # Execute the module
            spec.loader.exec_module(module)

            # Find and execute the appropriate entry point
            execution_result = self._find_and_execute_entrypoint(module)
            result["status"] = "success"
            result["result"] = execution_result
        except TimeoutError:
            result["status"] = "timeout"
            return result
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            result["status"] = "failed"
            result["error"] = str(e)
            return result
        else:
            return result
        finally:
            # Cancel timeout and restore old handler if we set one
            if use_alarm:
                try:
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # Ignore errors while restoring signal handlers
                    pass

    def _find_and_execute_entrypoint(self, module: Any) -> Any | None:
        """Find and execute the appropriate entry point in the module"""
        try:
            # First, try to find a run() function
            if hasattr(module, "run") and callable(module.run):
                logger.debug("Executing run() function")
                return module.run()

            # Look for a class that implements execute_strategy
            for name in dir(module):
                if name.startswith("_"):
                    continue

                obj = getattr(module, name)

                # Check if it's a class with execute_strategy method
                if inspect.isclass(obj) and hasattr(obj, "execute_strategy") and callable(obj.execute_strategy):
                    logger.debug(f"Found class {name} with execute_strategy method")
                    try:
                        instance = obj()
                        return instance.execute_strategy()
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        logger.exception(f"Error instantiating class {name}: {e}")
                        continue

            logger.warning("No valid entry point found in module")
            result = None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error finding entry point: {e}")
            return None
        else:
            return result

    def _record_execution(self, file_path: str, result: Any, status: str, error: str | None = None) -> None:
        """Record execution result with proper timestamp and history management"""
        try:
            execution_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
                "result": result,
                "status": status,
                "error": error,
            }

            self.execution_history.append(execution_record)

            # Cap history to prevent unbounded growth
            if len(self.execution_history) > MAX_HISTORY_RECORDS:
                self.execution_history = self.execution_history[-MAX_HISTORY_RECORDS:]
                logger.debug(f"Trimmed execution history to {MAX_HISTORY_RECORDS} records")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error recording execution: {e}")

    def execute_all_generated_modules(self) -> list[tuple[str, Any]]:
        """Execute all modules in the generated_modules directory with deterministic ordering"""
        try:
            if not Path(self.generated_modules_dir).exists():
                logger.warning(f"No {self.generated_modules_dir} directory found")
                self.health_issues.append(f"Generated modules directory not found: {self.generated_modules_dir}")
                return []

            # Get all Python files and sort for deterministic execution
            try:
                all_files = [f.name for f in Path(self.generated_modules_dir).iterdir()]
                py_files = [f for f in all_files if f.endswith(".py")]
                py_files.sort()  # Deterministic ordering

                if not py_files:
                    logger.info("No Python modules found in generated_modules directory")
                    return []

                logger.info(f"Found {len(py_files)} Python modules to execute")

            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                error_msg = f"Error listing generated modules directory: {e}"
                logger.exception(error_msg)
                self.health_issues.append(error_msg)
                return []

            results = []
            for file_name in py_files:
                file_path = str(Path(self.generated_modules_dir) / file_name)

                # Execute module and record result regardless of truthiness
                result = self.run_generated_module(file_path)

                # Record execution outcome regardless of truthiness
                results.append((file_name, result))

                # Log execution outcome
                if result is not None:
                    logger.debug(f"Module {file_name} executed with result: {result}")
                else:
                    logger.debug(f"Module {file_name} executed with no result")

            logger.info(f"Executed {len(results)} modules")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            error_msg = f"Error executing generated modules: {e}"
            logger.error(error_msg, exc_info=True)
            self.health_issues.append(error_msg)
            return []
        else:
            return results

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics with standardized success rate as percentage"""
        try:
            successful = len([r for r in self.execution_history if r["status"] == "success"])
            failed = len([r for r in self.execution_history if r["status"] == "failed"])
            timeout = len([r for r in self.execution_history if r["status"] == "timeout"])
            total = len(self.execution_history)

            # Standardize success rate as percentage (0-100)
            success_rate_percent = (successful / total * 100) if total > 0 else 0

            return {
                "total_executions": total,
                "successful": successful,
                "failed": failed,
                "timeout": timeout,
                "success_rate_percent": round(success_rate_percent, 2),  # 0-100 percentage
                "success_rate_decimal": round(successful / total, 4) if total > 0 else 0,  # 0-1 decimal
                "latest_executions": self.execution_history[-5:] if self.execution_history else [],
                "health_issues": self.health_issues.copy(),
                "base_dir": self.base_dir,
                "generated_modules_dir": self.generated_modules_dir,
                "max_execution_time": MAX_EXECUTION_TIME,
                "max_history_records": MAX_HISTORY_RECORDS,
                "current_history_size": len(self.execution_history),
            }

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting execution stats: {e}")
            return {
                "error": str(e),
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "timeout": 0,
                "success_rate_percent": 0,
                "success_rate_decimal": 0,
                "latest_executions": [],
                "health_issues": [*self.health_issues.copy(), f"Stats error: {e}"],
            }


def run_generated_module(file_path: str, base_dir: str | None = None) -> Any | None:
    """Simple function interface for external calls with proper path handling"""
    try:
        executor = AutoExecutor(base_dir=base_dir)
        return executor.run_generated_module(file_path)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error in run_generated_module: {e}")
        return None


def execute_all_modules(base_dir: str | None = None) -> list[tuple[str, Any]]:
    """Execute all generated modules with proper path handling"""
    try:
        executor = AutoExecutor(base_dir=base_dir)
        return executor.execute_all_generated_modules()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error in execute_all_modules: {e}")
        return []


def get_execution_stats(base_dir: str | None = None) -> dict[str, Any]:
    """Get execution statistics"""
    try:
        executor = AutoExecutor(base_dir=base_dir)
        return executor.get_execution_stats()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Error getting execution stats: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test execution with proper logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing auto-execution...")

    # Execute all modules in generated_modules directory
    results = execute_all_modules()

    if results:
        logger.info(f"Executed {len(results)} modules:")
        for file_name, result in results:
            logger.info(f"  - {file_name}: {result}")
    else:
        logger.info("No modules to execute")

    # Show execution stats
    stats = get_execution_stats()
    logger.info(f"Execution stats: {stats}")
