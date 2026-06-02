"""
Smart Training Data Manager — bounded retention for training JSON on disk.

Actual behavior (see SmartTrainingDataManager.smart_cleanup):
- Never deletes ``*_latest.json`` files (they are excluded from grouping).
- Timestamped ``SYMBOL_*.json`` chunks: keep recent files (retention window),
  per-symbol count cap (``MAX_TRAINING_FILES_PER_SYMBOL``), and optional
  ``_analyze_file_quality`` edge-case keep (high feature-vector std).
- Deletes mainly ``excess_files`` beyond the per-symbol cap / age rules.

Emergency path (``emergency_cleanup``): keeps newest 30 per symbol plus files
scoring ``quality >= 0.9`` from the same analyzer.

This is rule-based housekeeping, not an LLM judging "good" vs "bad" rows.
"""

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Configuration - BALANCED: Keep enough data for learning, but prevent runaway growth
MAX_TRAINING_JSON_AGE_DAYS = int(os.getenv("TRAINING_JSON_RETENTION_DAYS", "30"))  # Keep 30 days
MAX_TRAINING_FILES_PER_SYMBOL = int(os.getenv("MAX_TRAINING_FILES_PER_SYMBOL", "500"))  # 500 files per symbol
MIN_CONFIDENCE_TO_KEEP = float(os.getenv("MIN_CONFIDENCE_TO_KEEP", "0.1"))  # Keep low-confidence examples too
MAX_DISK_USAGE_GB = float(os.getenv("MAX_TRAINING_DISK_GB", "50"))  # Max 50 GB for training data


class SmartTrainingDataManager:
    """
    On-disk training JSON retention: age + per-symbol file limits + emergency shrink.

    Does not score trades by PnL or "AI quality"; ``_analyze_file_quality`` mostly
    returns keep, with a volatility heuristic to flag edge-case files worth retaining.
    """

    def __init__(self, training_data_dir: str = "models/training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        # Track what we've cleaned
        self.last_cleanup_time = 0
        self.cleanup_stats = {
            "total_cleaned": 0,
            "low_quality_removed": 0,
            "redundant_removed": 0,
            "stale_removed": 0,
            "kept_winners": 0,
            "kept_edge_cases": 0,
        }

    def should_run_cleanup(self) -> bool:
        """Run cleanup once per day for balanced maintenance"""
        return time.time() - self.last_cleanup_time > 86400  # 1 day

    async def smart_cleanup(self) -> dict[str, Any]:
        """
        Apply retention rules to timestamped training JSON files (not ``*_latest``).

        Returns stats about deleted vs kept files.
        """
        try:
            logger.info(" Starting SMART training data cleanup for maximum AI learning...")
            start_time = time.time()

            # Get all training files
            all_files = list(self.training_data_dir.glob("*_*.json"))
            logger.info(f"Found {len(all_files)} training files")

            if not all_files:
                return {"status": "no_files", "message": "No training files to clean"}

            # Group files by symbol
            files_by_symbol = self._group_files_by_symbol(all_files)

            # Analyze each symbol's training data
            cleanup_decisions = {}
            for symbol, files in files_by_symbol.items():
                cleanup_decisions[symbol] = self._analyze_symbol_data(symbol, files)

            # Execute cleanup based on decisions
            total_deleted = 0
            total_kept = 0
            disk_freed_mb = 0

            for _symbol, decisions in cleanup_decisions.items():
                for file_path, decision, reason in decisions:
                    if decision == "DELETE":
                        try:
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            file_path.unlink()
                            total_deleted += 1
                            disk_freed_mb += file_size_mb
                            self._update_stats(reason)
                        except Exception as e:
                            logger.debug(f"Error deleting {file_path}: {e}")
                    else:
                        total_kept += 1

            # Update cleanup time
            self.last_cleanup_time = time.time()
            elapsed = time.time() - start_time

            result = {
                "status": "success",
                "files_deleted": total_deleted,
                "files_kept": total_kept,
                "disk_freed_mb": round(disk_freed_mb, 2),
                "elapsed_seconds": round(elapsed, 2),
                "stats": self.cleanup_stats.copy(),
            }

            logger.info(f" Cleanup complete: Deleted {total_deleted} files, kept {total_kept} files, freed {disk_freed_mb:.1f} MB")
            logger.info(f" Quality stats: {self.cleanup_stats}")
        except Exception as e:
            logger.exception(f"Error in smart cleanup: {e}")
            return {"status": "error", "message": str(e)}
        else:
            return result

    def _group_files_by_symbol(self, files: list[Path]) -> dict[str, list[Path]]:
        """Group training files by symbol"""
        by_symbol = defaultdict(list)

        for file_path in files:
            # Skip _latest.json files (always keep)
            if file_path.stem.endswith("_latest"):
                continue

            # Extract symbol from filename (e.g., BTCUSDT_20251204_153045.json)
            try:
                symbol = file_path.stem.split("_")[0]
                by_symbol[symbol].append(file_path)
            except (IndexError, ValueError):
                logger.debug(f"Could not parse symbol from {file_path.name}")
                continue

        return dict(by_symbol)

    def _analyze_symbol_data(self, symbol: str, files: list[Path]) -> list[tuple[Path, str, str]]:
        """
        Analyze training files for a symbol and decide what to keep/delete.

        Returns: List of (file_path, "KEEP" | "DELETE", reason)
        """
        decisions = []

        # Sort files by modification time (newest first)
        files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

        cutoff_time = time.time() - (MAX_TRAINING_JSON_AGE_DAYS * 86400)

        high_quality_count = 0
        edge_case_count = 0
        recent_count = 0

        for idx, file_path in enumerate(files_sorted):
            file_mtime = file_path.stat().st_mtime
            is_recent = file_mtime > cutoff_time

            # RULE 1: Keep the most recent MAX_TRAINING_FILES_PER_SYMBOL files
            if idx < MAX_TRAINING_FILES_PER_SYMBOL and is_recent:
                decisions.append((file_path, "KEEP", "recent_data"))
                recent_count += 1
                continue

            # RULE 2: Keep ALL training data for AI learning (don't judge "quality")
            quality_score = self._analyze_file_quality(file_path)

            # Keep edge cases (rare patterns worth learning from)
            if quality_score == -1.0:  # Edge case marker
                decisions.append((file_path, "KEEP", "edge_case"))
                edge_case_count += 1
                continue

            # RULE 3: Keep recent data (within retention period)
            if is_recent:
                decisions.append((file_path, "KEEP", "recent_data"))
                recent_count += 1
                continue

            # RULE 4: For older data, keep up to the limit (FIFO)
            if idx < MAX_TRAINING_FILES_PER_SYMBOL:
                decisions.append((file_path, "KEEP", "historical_data"))
                continue

            # RULE 5: Delete excess files beyond the limit (one decision per file)
            decisions.append((file_path, "DELETE", "excess_files"))
            continue

        logger.debug(f"{symbol}: Keeping {recent_count} recent, {high_quality_count} high-quality, {edge_case_count} edge cases")

        return decisions

    def _analyze_file_quality(self, file_path: Path) -> float:
        """
        FIXED: Keep ALL training data for AI learning.

        AI needs to see failures, mistakes, and diverse examples to learn properly.
        Don't remove "low quality" data - it's essential for learning!

        Returns:
        - 1.0: Keep (any valid data is valuable for learning)
        - -1.0: Edge case (extreme volatility, rare pattern)
        """
        try:
            with file_path.open() as f:
                data = json.load(f)

            if not isinstance(data, list) or not data:
                return 0.5  # Empty or invalid

            # Check for edge cases (extreme values) - these are valuable for learning
            for sample in data[:10]:  # Check first 10 samples
                if not isinstance(sample, dict):
                    continue

                features = sample.get("features", [])
                if not features or len(features) < 10:
                    continue

                try:
                    feature_array = np.array(features, dtype=float)

                    # Detect extreme volatility (valuable edge case for learning)
                    volatility = np.std(feature_array)
                    if volatility > 10.0:  # High volatility = rare pattern worth learning
                        return -1.0  # Edge case marker - keep!

                except (ValueError, TypeError):
                    continue

            # Keep ALL valid training data - AI needs diversity to learn
            # Don't judge "quality" - let the AI learn from all examples

        except Exception as e:
            logger.exception(f"Error analyzing {file_path}: {e}")
            return 0.5  # Default to average quality
        else:
            return 1.0  # Keep all data

    def _update_stats(self, reason: str) -> None:
        """Update cleanup statistics"""
        self.cleanup_stats["total_cleaned"] += 1

        if reason == "low_quality":
            self.cleanup_stats["low_quality_removed"] += 1
        elif reason == "redundant":
            self.cleanup_stats["redundant_removed"] += 1
        elif reason == "stale":
            self.cleanup_stats["stale_removed"] += 1
        elif reason == "high_quality":
            self.cleanup_stats["kept_winners"] += 1
        elif reason == "edge_case":
            self.cleanup_stats["kept_edge_cases"] += 1

    def get_disk_usage_stats(self) -> dict[str, Any]:
        """Get current disk usage statistics"""
        try:
            total_size_mb = 0
            file_count = 0

            for file_path in self.training_data_dir.glob("*_*.json"):
                total_size_mb += file_path.stat().st_size / (1024 * 1024)
                file_count += 1

            return {
                "total_size_mb": round(total_size_mb, 2),
                "total_size_gb": round(total_size_mb / 1024, 2),
                "file_count": file_count,
                "limit_gb": MAX_DISK_USAGE_GB,
                "usage_percent": round((total_size_mb / 1024 / MAX_DISK_USAGE_GB) * 100, 1),
            }
        except Exception as e:
            logger.exception(f"Error getting disk usage stats: {e}")
            return {"error": str(e)}

    def emergency_cleanup(self) -> dict[str, Any]:
        """
        Emergency cleanup when disk is nearly full.
        More aggressive - only keeps the absolute best data.
        """
        try:
            logger.warning(" EMERGENCY CLEANUP: Disk usage critical!")

            # Delete everything except:
            # 1. _latest.json files (most recent)
            # 2. Last 30 files per symbol
            # 3. High-quality files (quality >= 0.9)

            all_files = list(self.training_data_dir.glob("*_*.json"))
            files_by_symbol = self._group_files_by_symbol(all_files)

            deleted = 0
            kept = 0

            for _symbol, files in files_by_symbol.items():
                # Sort by modification time
                files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

                for idx, file_path in enumerate(files_sorted):
                    # Keep latest 30 files
                    if idx < 30:
                        kept += 1
                        continue

                    # Check quality
                    quality = self._analyze_file_quality(file_path)
                    if quality >= 0.9:  # Excellent quality
                        kept += 1
                        continue

                    # Delete
                    try:
                        file_path.unlink()
                        deleted += 1
                    except Exception as e:
                        logger.debug(f"Error deleting {file_path}: {e}")

            logger.warning(f" Emergency cleanup: Deleted {deleted}, kept {kept}")
        except Exception as e:
            logger.exception(f"Error in emergency cleanup: {e}")
            return {"status": "error", "message": str(e)}
        else:
            return {
                "status": "emergency_cleanup",
                "deleted": deleted,
                "kept": kept,
            }


# Global instance
smart_training_data_manager = SmartTrainingDataManager()
