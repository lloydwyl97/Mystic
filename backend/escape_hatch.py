#!/usr/bin/env python3
"""
Escape Hatch Migration Utility for Mystic Trading Platform

Safe, reliable file migration utility with Windows/Python 3.12+ compatibility.
Provides structured logging, progress hooks, safety checks, and resumable operations.
No hardcoded data, placeholders, or Docker assumptions.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MigrationError:
    """Structured error record"""

    path: str
    operation: str
    errno: int | None
    message: str
    timestamp: str


@dataclass
class MigrationSummary:
    """Structured migration summary"""

    source: str
    destination: str
    dry_run: bool
    created_dirs: int
    copied_files: int
    copied_bytes: int
    skipped_files: int
    skipped_dirs: int
    errors: list[MigrationError]
    errors_by_code: dict[str, int]
    skipped_by_rule: dict[str, int]
    locked_files_count: int
    preflight_required_bytes: int
    free_space_bytes_before: int
    free_space_bytes_after: int
    duration_sec: float
    success: bool


class CancellationToken:
    """Thread-safe cancellation token"""

    def __init__(self) -> None:
        self._cancelled = threading.Event()

    def cancel(self):
        """Cancel the operation"""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """Check if operation is cancelled"""
        return self._cancelled.is_set()


def _get_disk_usage(path: Path) -> int:
    """Get free disk space in bytes"""
    try:
        if sys.platform == "win32":
            # Windows - use parent directory if path doesn't exist
            check_path = path
            if not check_path.exists():
                check_path = check_path.parent
                if not check_path.exists():
                    check_path = Path.cwd()

            _total, _used, free = shutil.disk_usage(check_path)
            return free
        # POSIX
        statvfs = os.statvfs(path)
        return statvfs.f_frsize * statvfs.f_bavail
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.warning(f"Could not get disk usage for {path}: {e}")
        return 0


def _normalize_path(path: str | Path) -> Path:
    """Normalize path for Windows compatibility"""
    path = Path(path).resolve()

    # Handle Windows long path support
    if sys.platform == "win32":
        path_str = str(path)
        if len(path_str) > 260 and not path_str.startswith("\\\\?\\"):
            # Use extended-length path prefix
            path = Path(f"\\\\?\\{path_str}")

    return path


def _is_safe_path(path: Path) -> bool:
    """Check if path is safe for Windows"""
    if sys.platform != "win32":
        return True

    # path_str = str(path).lower()  # Unused

    # Check for reserved device names
    reserved_names = {
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
    }

    for part in path.parts:
        name = part.lower().split(".")[0]  # Remove extension
        if name in reserved_names:
            return False

    return True


def _matches_pattern(path: Path, patterns: set[str]) -> bool:
    """Check if path matches any exclusion pattern"""
    path_str = str(path).replace("\\", "/")  # Normalize separators

    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str.lower(), pattern.lower()):
            return True
        # Check individual components
        for part in path.parts:
            if fnmatch.fnmatch(part, pattern) or fnmatch.fnmatch(part.lower(), pattern.lower()):
                return True

    return False


def _is_special_file(path: Path) -> bool:
    """Check if file is special (device, socket, etc.)"""
    try:
        stat_info = path.stat()
        mode = stat_info.st_mode

        # Check for special file types
        return bool(stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return False


def _preflight_check(src: Path, dst: Path, exclude_patterns: set[str]) -> dict[str, Any]:
    """Perform pre-flight safety checks"""
    logger.info("Performing pre-flight safety checks...")

    # Check source exists and is readable
    if not src.exists():
        msg = f"Source path does not exist: {src}"
        raise FileNotFoundError(msg)

    if not src.is_dir():
        msg = f"Source path is not a directory: {src}"
        raise ValueError(msg)

    if not os.access(src, os.R_OK):
        msg = f"Source path is not readable: {src}"
        raise PermissionError(msg)

    # Check destination safety
    if dst == src:
        msg = "Destination cannot be the same as source"
        raise ValueError(msg)

    # Check if destination is inside source
    try:
        src_rel = dst.relative_to(src)
        is_dst_in_src = bool(src_rel)
    except ValueError:
        is_dst_in_src = False  # Not relative; OK

    # Check if source is inside destination
    try:
        dst_rel = src.relative_to(dst)
        is_src_in_dst = bool(dst_rel)
    except ValueError:
        is_src_in_dst = False  # Not relative; OK

    # Validate paths outside try blocks to avoid TRY301
    if is_dst_in_src:
        msg = "Destination is inside source directory"
        raise ValueError(msg)

    if is_src_in_dst:
        msg = "Source is inside destination directory"
        raise ValueError(msg)

    # Check path safety
    if not _is_safe_path(src):
        msg = f"Source path contains unsafe characters: {src}"
        raise ValueError(msg)

    if not _is_safe_path(dst):
        msg = f"Destination path contains unsafe characters: {dst}"
        raise ValueError(msg)

    # Calculate required space
    total_bytes = 0
    total_files = 0
    total_dirs = 0

    logger.info("Calculating required space...")
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        try:
            rel_root = root_path.relative_to(src)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            rel_root = Path()

        if _matches_pattern(rel_root, exclude_patterns):
            dirs[:] = []  # Don't descend
            continue

        total_dirs += 1

        for fname in files:
            rel_file = rel_root / fname
            if _matches_pattern(rel_file, exclude_patterns):
                continue

            src_file = root_path / fname
            try:
                total_bytes += src_file.stat().st_size
                total_files += 1
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pass

    # Check free space
    free_space = _get_disk_usage(dst.parent if dst.exists() else dst)

    logger.info(f"Pre-flight check complete: {total_files} files, {total_dirs} dirs, {total_bytes} bytes required")

    return {
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_bytes": total_bytes,
        "free_space": free_space,
        "sufficient_space": free_space >= total_bytes * 1.1,  # 10% buffer
    }


def migrate_to_new_node(
    data_path: str = "./data",
    backup_path: str = "./escape",
    exclude: Iterable[str] | None = None,
    dry_run: bool = False,
    atomic: bool = True,
    resumable: bool = False,
    skip_locked_files: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_token: CancellationToken | None = None,
    fail_on_first_error: bool = False,
    max_errors: int = 1000,
) -> MigrationSummary:
    """
    Copy a live data directory to a backup location with comprehensive safety checks.

    Args:
        data_path: Source directory path
        backup_path: Destination directory path
        exclude: List of glob patterns to exclude
        dry_run: If True, only calculate what would be copied
        atomic: If True, copy to temp location then rename
        resumable: If True, skip already-copied files
        skip_locked_files: If True, skip files that can't be opened
        max_retries: Maximum retries for locked files
        retry_delay: Delay between retries in seconds
        progress_callback: Optional callback for progress updates
        cancellation_token: Optional cancellation token
        fail_on_first_error: If True, stop on first error
        max_errors: Maximum errors before stopping

    Returns:
        MigrationSummary with detailed results
    """
    started = time.time()

    # Normalize paths
    src = _normalize_path(data_path)
    dst = _normalize_path(backup_path)

    # Process exclude patterns
    exclude_patterns = set()
    if exclude:
        for pattern in exclude:
            pattern_clean = pattern.strip("/\\")
            if pattern_clean:
                exclude_patterns.add(pattern_clean)

    # Add default safe exclusions
    safe_excludes = {
        "*.log",
        "*.tmp",
        "*.temp",
        "*.swp",
        "*.lock",
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/.DS_Store",
    }

    exclude_patterns |= safe_excludes

    # Prepare final and working destination paths
    dst_final = dst
    dst_work = dst_final
    if atomic and not dry_run:
        # create a unique temp directory beside final destination
        suffix = f".tmp_migration_{os.getpid()}_{int(time.time())}"
        dst_work = dst_final.parent / (dst_final.name + suffix)

    # Preflight check
    preflight = _preflight_check(src, dst_final, exclude_patterns)

    # Initialize counters and state
    created_dirs = 0
    copied_files = 0
    copied_bytes = 0
    skipped_files = 0
    skipped_dirs = 0
    errors: list[MigrationError] = []
    errors_by_code: dict[str, int] = {}
    skipped_by_rule: dict[str, int] = {}
    locked_files_count = 0
    processed_files = 0

    # Progress updater
    def update_progress():
        if progress_callback:
            progress_callback(
                {
                    "source": str(src),
                    "destination": str(dst_final if atomic and not dry_run else dst_work),
                    "created_dirs": created_dirs,
                    "copied_files": copied_files,
                    "copied_bytes": copied_bytes,
                    "skipped_files": skipped_files,
                    "skipped_dirs": skipped_dirs,
                    "errors": len(errors),
                    "locked_files": locked_files_count,
                    "duration_sec": time.time() - started,
                }
            )

    # Create working directory if needed
    try:
        if not dry_run:
            Path(dst_work).mkdir(parents=True, exist_ok=True)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Could not create destination directory {dst_work}: {e}"
        raise RuntimeError(msg) from e

    try:
        # Walk source directory
        for root, dirs, files in os.walk(src):
            if cancellation_token and cancellation_token.is_cancelled():
                logger.info("Cancellation requested, stopping migration")
                break

            root_path = Path(root)
            try:
                rel_root = root_path.relative_to(src)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                rel_root = Path()

            # Check if this directory is excluded
            if _matches_pattern(rel_root, exclude_patterns):
                skipped_dirs += 1
                skipped_by_rule["excluded_dir"] = skipped_by_rule.get("excluded_dir", 0) + 1
                dirs[:] = []
                continue

            # Create corresponding destination directory
            dst_root = dst_work / rel_root
            if not dry_run:
                try:
                    Path(dst_root).mkdir(parents=True, exist_ok=True)
                    created_dirs += 1
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    error = MigrationError(
                        path=str(dst_root),
                        operation="mkdir",
                        errno=getattr(e, "errno", None),
                        message=str(e),
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    errors.append(error)
                    errors_by_code[error.operation] = errors_by_code.get(error.operation, 0) + 1
                    if fail_on_first_error:
                        raise

            # Process files in this directory
            for fname in files:
                if cancellation_token and cancellation_token.is_cancelled():
                    logger.info("Cancellation requested, stopping migration")
                    break

                processed_files += 1

                rel_file = rel_root / fname
                if _matches_pattern(rel_file, exclude_patterns):
                    skipped_files += 1
                    skipped_by_rule["excluded_file"] = skipped_by_rule.get("excluded_file", 0) + 1
                    continue

                src_file = root_path / fname
                dst_file = dst_root / fname

                # Skip special files
                if _is_special_file(src_file):
                    skipped_files += 1
                    skipped_by_rule["special_file"] = skipped_by_rule.get("special_file", 0) + 1
                    logger.debug(f"Skipping special file: {src_file}")
                    continue

                # Check if file should be resumed (resumable mode)
                if resumable and not dry_run and dst_file.exists():
                    try:
                        src_stat = src_file.stat()
                        dst_stat = dst_file.stat()
                        if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
                            skipped_files += 1
                            skipped_by_rule["already_copied"] = skipped_by_rule.get("already_copied", 0) + 1
                            continue
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass  # Continue with copy

                if dry_run:
                    try:
                        file_size = src_file.stat().st_size
                        copied_files += 1
                        copied_bytes += file_size
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass
                    continue

                # Copy file with retry logic
                success = False
                for attempt in range(max_retries + 1):
                    try:
                        shutil.copy2(src_file, dst_file)
                        success = True
                        break
                    except (PermissionError, OSError) as e:
                        if attempt < max_retries:
                            logger.debug(f"Retry {attempt + 1} for {src_file}: {e}")
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        if skip_locked_files:
                            locked_files_count += 1
                            skipped_files += 1
                            skipped_by_rule["locked_file"] = skipped_by_rule.get("locked_file", 0) + 1
                            logger.warning(f"Skipping locked file: {src_file}")
                        else:
                            error = MigrationError(
                                path=str(src_file),
                                operation="copy",
                                errno=getattr(e, "errno", None),
                                message=str(e),
                                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            )
                            errors.append(error)
                            errors_by_code[error.operation] = errors_by_code.get(error.operation, 0) + 1

                            if fail_on_first_error:
                                raise
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                        error = MigrationError(
                            path=str(src_file),
                            operation="copy",
                            errno=getattr(e, "errno", None),
                            message=str(e),
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                        errors.append(error)
                        errors_by_code[error.operation] = errors_by_code.get(error.operation, 0) + 1

                        if fail_on_first_error:
                            raise
                        break

                if success:
                    try:
                        file_size = src_file.stat().st_size
                        copied_files += 1
                        copied_bytes += file_size
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                        pass

                # Check error limit
                if len(errors) >= max_errors:
                    logger.error(f"Maximum errors reached ({max_errors}), stopping migration")
                    break

                # Update progress periodically
                if processed_files % 100 == 0:
                    update_progress()

            if len(errors) >= max_errors:
                break

        # Atomic rename / move
        if atomic and not dry_run and dst_work != dst_final:
            try:
                # Remove existing final destination if present
                if dst_final.exists():
                    if dst_final.is_dir():
                        shutil.rmtree(dst_final)
                    else:
                        dst_final.unlink()
                # Move working to final location
                shutil.move(str(dst_work), str(dst_final))
                logger.info(f"Atomic move completed: {dst_work} -> {dst_final}")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                error = MigrationError(
                    path=str(dst_work),
                    operation="atomic_move",
                    errno=getattr(e, "errno", None),
                    message=str(e),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                errors.append(error)
                errors_by_code[error.operation] = errors_by_code.get(error.operation, 0) + 1
                # Attempt cleanup of work dir
                try:
                    if dst_work.exists():
                        if dst_work.is_dir():
                            shutil.rmtree(dst_work)
                        else:
                            dst_work.unlink()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    pass
                msg = f"Atomic move failed: {e}"
                raise RuntimeError(msg) from e

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Migration failed: {e}")
        # Cleanup working directory if atomic and not final
        if atomic and not dry_run and dst_work.exists():
            try:
                if dst_work.is_dir():
                    shutil.rmtree(dst_work)
                else:
                    dst_work.unlink()
                logger.info(f"Cleaned up temp directory: {dst_work}")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                pass
        # Create summary even on failure
        duration = time.time() - started
        success = False
        update_progress()
        summary = MigrationSummary(
            source=str(src),
            destination=str(dst_final if atomic and not dry_run else dst_work),
            dry_run=dry_run,
            created_dirs=created_dirs,
            copied_files=copied_files,
            copied_bytes=copied_bytes,
            skipped_files=skipped_files,
            skipped_dirs=skipped_dirs,
            errors=errors,
            errors_by_code=errors_by_code,
            skipped_by_rule=skipped_by_rule,
            locked_files_count=locked_files_count,
            preflight_required_bytes=preflight["total_bytes"],
            free_space_bytes_before=preflight["free_space"],
            free_space_bytes_after=_get_disk_usage(Path(str(dst_final if atomic and not dry_run else dst_work))),
            duration_sec=round(duration, 3),
            success=success,
        )
        return summary
    else:
        duration = time.time() - started
        success = len(errors) == 0 and not (cancellation_token and cancellation_token.is_cancelled())

        # Final progress update
        update_progress()

        summary = MigrationSummary(
            source=str(src),
            destination=str(dst_final if atomic and not dry_run else dst_work),
            dry_run=dry_run,
            created_dirs=created_dirs,
            copied_files=copied_files,
            copied_bytes=copied_bytes,
            skipped_files=skipped_files,
            skipped_dirs=skipped_dirs,
            errors=errors,
            errors_by_code=errors_by_code,
            skipped_by_rule=skipped_by_rule,
            locked_files_count=locked_files_count,
            preflight_required_bytes=preflight["total_bytes"],
            free_space_bytes_before=preflight["free_space"],
            free_space_bytes_after=_get_disk_usage(Path(str(dst_final if atomic and not dry_run else dst_work))),
            duration_sec=round(duration, 3),
            success=success,
        )

        logger.info("Migration completed")
        logger.info(f"Dirs created: {created_dirs}")
        logger.info(f"Files copied: {copied_files}")
        logger.info(f"Bytes copied: {copied_bytes}")
        logger.info(f"Skipped files: {skipped_files}")
        logger.info(f"Skipped dirs: {skipped_dirs}")
        logger.info(f"Errors: {len(errors)}")
        logger.info(f"Duration: {summary.duration_sec}s")
        logger.info(f"Success: {success}")

        if errors:
            logger.warning(f"Some errors occurred: {len(errors)} total")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"Error: {error.operation} {error.path}: {error.message}")

        return summary


if __name__ == "__main__":
    """CLI interface for migration utility"""
    import argparse

    parser = argparse.ArgumentParser(description="Escape Hatch Migration Utility")
    parser.add_argument("source", nargs="?", default="./data", help="Source directory path")
    parser.add_argument("destination", nargs="?", default="./escape", help="Destination directory path")
    parser.add_argument(
        "--exclude",
        action="append",
        help="Exclusion patterns (can be used multiple times)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without copying files")
    parser.add_argument(
        "--atomic",
        action="store_true",
        default=True,
        help="Use atomic copy (default: True)",
    )
    parser.add_argument("--resumable", action="store_true", help="Enable resumable mode")
    parser.add_argument(
        "--skip-locked",
        action="store_true",
        default=True,
        help="Skip locked files (default: True)",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for locked files")
    parser.add_argument("--fail-fast", action="store_true", help="Fail on first error")
    parser.add_argument("--max-errors", type=int, default=1000, help="Maximum errors before stopping")

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.source).is_absolute() and not Path(args.destination).is_absolute():
        logger.warning("Using relative paths - ensure working directory is correct")

    try:
        summary = migrate_to_new_node(
            data_path=args.source,
            backup_path=args.destination,
            exclude=args.exclude,
            dry_run=args.dry_run,
            atomic=args.atomic,
            resumable=args.resumable,
            skip_locked_files=args.skip_locked,
            max_retries=args.max_retries,
            fail_on_first_error=args.fail_fast,
            max_errors=args.max_errors,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Migration failed: {e}")
        sys.exit(1)
    else:
        if summary.success:
            logger.info("Migration completed successfully")
            sys.exit(0)
        else:
            logger.error("Migration completed with errors")
            sys.exit(1)
