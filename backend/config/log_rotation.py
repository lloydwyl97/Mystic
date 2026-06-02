"""
Log Rotation Configuration - Prevents log files from growing unbounded.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_log_rotation(
    logger_name: str,
    log_file: str | Path,
    max_bytes: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 5,  # Keep last 5 rotated files
    formatter: logging.Formatter | None = None,
) -> logging.Handler:
    """
    Configure rotating file handler for a logger.

    Args:
        logger_name: Name of the logger to configure
        log_file: Path to the log file
        max_bytes: Maximum file size before rotation (default 100 MB)
        backup_count: Number of backup files to keep (default 5)
        formatter: Custom formatter (default uses standard format)

    Returns:
        The configured RotatingFileHandler
    """
    logger = logging.getLogger(logger_name)

    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    # Set formatter
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    logging.info(f"Log rotation configured for {logger_name}: {log_file} (max {max_bytes / (1024 * 1024):.0f} MB, keep {backup_count} backups)")

    return handler


def configure_all_rotating_handlers(
    log_dir: str | Path = "logs",
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 5,
) -> dict[str, logging.Handler]:
    """
    Configure rotating handlers for all main loggers in the application.

    Args:
        log_dir: Directory to store log files (default "logs")
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Dictionary mapping logger names to their handlers
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    loggers_config = {
        "backend": "backend.log",
        "backend.agents.strategy_agent": "ai_strategy_generator_core.log",
        "backend.agents.agent_orchestrator": "agent_orchestrator.log",
        "backend.memory_monitor": "memory_monitor.log",
        "backend.process_memory_profiler": "process_memory.log",
        "uvicorn": "uvicorn.log",
    }

    handlers = {}
    for logger_name, log_file in loggers_config.items():
        try:
            handler = setup_log_rotation(
                logger_name,
                log_dir / log_file,
                max_bytes=max_bytes,
                backup_count=backup_count,
            )
            handlers[logger_name] = handler
        except Exception as e:
            logging.warning(f"Failed to configure rotation for {logger_name}: {e}")

    return handlers
