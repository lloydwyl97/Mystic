"""
Unicode-safe logging configuration for Windows console compatibility.
This module provides Unicode-safe logging handlers that replace problematic
Unicode characters with ASCII-safe alternatives.
"""

import logging
import sys
from typing import ClassVar


class UnicodeSafeFormatter(logging.Formatter):
    """A formatter that safely handles Unicode characters"""

    # Character replacements for Windows console compatibility
    UNICODE_REPLACEMENTS: ClassVar[dict[str, str]] = {
        "[OK]": "[OK]",
        "[ERROR]": "[ERROR]",
        "[TARGET]": "[TARGET]",
        "[WARNING]": "[WARNING]",
        "[TOOL]": "[TOOL]",
        "[DATA]": "[DATA]",
        "[LAUNCH]": "[LAUNCH]",
        "[IDEA]": "[IDEA]",
        "[HOT]": "[HOT]",
        "[STAR]": "[STAR]",
        "[CELEBRATE]": "[CELEBRATE]",
        "[UP]": "[UP]",
        "[DOWN]": "[DOWN]",
        "[LOCK]": "[LOCK]",
        "[UNLOCK]": "[UNLOCK]",
        "[FAST]": "[FAST]",
        "[SHIELD]": "[SHIELD]",
        "[CIRCUS]": "[CIRCUS]",
        "[TROPHY]": "[TROPHY]",
        "[ART]": "[ART]",
        "[SEARCH]": "[SEARCH]",
        "[NOTE]": "[NOTE]",
        "[MUSIC]": "[MUSIC]",
        "[MOVIE]": "[MOVIE]",
        "[GAME]": "[GAME]",
        "[HOME]": "[HOME]",
        "[WORLD]": "[WORLD]",
        "[SUN]": "[SUN]",
        "[MOON]": "[MOON]",
        "[DIAMOND]": "[DIAMOND]",
        "[MEDAL]": "[MEDAL]",
        "[BADGE]": "[BADGE]",
        "[ROSETTE]": "[ROSETTE]",
        "[RIBBON]": "[RIBBON]",
        "[GOLD]": "[GOLD]",
        "[SILVER]": "[SILVER]",
        "[BRONZE]": "[BRONZE]",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with Unicode character replacement"""
        try:
            # Get the formatted message
            msg = super().format(record)

            # Replace Unicode characters with ASCII-safe alternatives
            for unicode_char, replacement in self.UNICODE_REPLACEMENTS.items():
                msg = msg.replace(unicode_char, replacement)
        except UnicodeEncodeError:
            # If Unicode encoding still fails, use ASCII-safe version
            try:
                msg = super().format(record)
                return msg.encode("ascii", "replace").decode("ascii")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # Last resort: return a simple error message
                return f"Logging error: {record.levelname} - {record.name}"
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Handle any other formatting errors
            return f"Logging error: {record.levelname} - {record.name}"
        else:
            return msg


class UnicodeSafeStreamHandler(logging.StreamHandler):
    """A StreamHandler that safely handles Unicode characters"""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with Unicode safety"""
        try:
            msg = self.format(record)
            # Replace Unicode characters with ASCII-safe alternatives
            msg = self._replace_unicode_chars(msg)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # If Unicode encoding still fails, use ASCII-safe version
            try:
                msg = self.format(record)
                msg = self._replace_unicode_chars(msg)
                msg = msg.encode("ascii", "replace").decode("ascii")
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # Last resort: write a simple error message
                stream = self.stream
                stream.write(f"Logging error: {record.levelname} - {record.name}\n")
                self.flush()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # Handle any other logging errors silently
            pass

    def _replace_unicode_chars(self, msg: str) -> str:
        """Replace Unicode characters with ASCII-safe alternatives"""
        replacements = {
            "[OK]": "[OK]",
            "[ERROR]": "[ERROR]",
            "[TARGET]": "[TARGET]",
            "[WARNING]": "[WARNING]",
            "[TOOL]": "[TOOL]",
            "[DATA]": "[DATA]",
            "[LAUNCH]": "[LAUNCH]",
            "[IDEA]": "[IDEA]",
            "[HOT]": "[HOT]",
            "[STAR]": "[STAR]",
            "[CELEBRATE]": "[CELEBRATE]",
            "[UP]": "[UP]",
            "[DOWN]": "[DOWN]",
            "[LOCK]": "[LOCK]",
            "[UNLOCK]": "[UNLOCK]",
            "[FAST]": "[FAST]",
            "[SHIELD]": "[SHIELD]",
            "[CIRCUS]": "[CIRCUS]",
            "[TROPHY]": "[TROPHY]",
            "[ART]": "[ART]",
            "[SEARCH]": "[SEARCH]",
            "[NOTE]": "[NOTE]",
            "[MUSIC]": "[MUSIC]",
            "[MOVIE]": "[MOVIE]",
            "[GAME]": "[GAME]",
            "[HOME]": "[HOME]",
            "[WORLD]": "[WORLD]",
            "[SUN]": "[SUN]",
            "[MOON]": "[MOON]",
            "[DIAMOND]": "[DIAMOND]",
            "[MEDAL]": "[MEDAL]",
            "[BADGE]": "[BADGE]",
            "[ROSETTE]": "[ROSETTE]",
            "[RIBBON]": "[RIBBON]",
            "[GOLD]": "[GOLD]",
            "[SILVER]": "[SILVER]",
            "[BRONZE]": "[BRONZE]",
        }

        for unicode_char, replacement in replacements.items():
            msg = msg.replace(unicode_char, replacement)

        return msg


def configure_unicode_safe_logging() -> None:
    """Configure logging with Unicode-safe handlers"""
    try:
        # Get the root logger
        root_logger = logging.getLogger()

        # Remove all existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create Unicode-safe handler
        safe_handler = UnicodeSafeStreamHandler(sys.stdout)
        safe_handler.setFormatter(UnicodeSafeFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        safe_handler.setLevel(logging.INFO)

        # Add the safe handler to root logger
        root_logger.addHandler(safe_handler)
        root_logger.setLevel(logging.INFO)

        # Disable propagation to prevent duplicate logs
        root_logger.propagate = False

        # Patch the logging module to ensure all loggers use Unicode-safe handlers
        _patch_logging_module()

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # If configuration fails, continue without it
        pass


def _patch_logging_module() -> None:
    """Configure logging with Unicode-safe handlers without patching the module"""
    try:
        # Create a comprehensive Unicode-safe handler
        safe_handler = UnicodeSafeStreamHandler(sys.stdout)
        safe_handler.setFormatter(UnicodeSafeFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        safe_handler.setLevel(logging.INFO)

        # Apply to root logger only
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(safe_handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        # If configuration fails, continue without it
        pass


def get_unicode_safe_logger(name: str) -> logging.Logger:
    """Get a logger with Unicode-safe configuration"""
    logger = logging.getLogger(name)

    # Ensure the logger uses our Unicode-safe configuration
    if not logger.handlers:
        logger.addHandler(UnicodeSafeStreamHandler(sys.stdout))
        logger.setFormatter(UnicodeSafeFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


# Configure Unicode-safe logging when this module is imported
configure_unicode_safe_logging()
