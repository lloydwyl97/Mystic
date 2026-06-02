"""
Script to update imports in migrated files - Live Configuration Only

All configuration values come from live config - no hardcoded values.
"""

import json
import os
import re
from pathlib import Path

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None


def update_imports_in_file(file_path: str) -> bool:
    """Update imports in a file."""
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return False

    # Get encoding from live config
    encoding = _get_file_encoding()

    # Read file content
    file_path_obj = Path(file_path)
    with file_path_obj.open(encoding=encoding) as f:
        content = f.read()

    # Update imports
    updated_content = content

    # Mapping of old module paths to new module paths - from live config
    mappings = _get_import_mappings()

    # Replace occurrences in import/from lines while preserving any trailing submodules
    for src, dst in mappings.items():
        # Match lines starting with "from" or "import" followed by the source module path.
        # Keep any trailing ".submodule" intact by only matching the base src path.
        pattern = rf"(?m)^(from|import)\s+{re.escape(src)}\b"
        replacement = r"\1 " + dst
        updated_content = re.sub(pattern, replacement, updated_content)

    # Write updated content
    if content != updated_content:
        with file_path_obj.open("w", encoding=encoding) as f:
            f.write(updated_content)
        print(f"Updated imports in {file_path}")
        return True
    print(f"No imports to update in {file_path}")
    return False


def update_imports_in_directory(directory_path: str) -> bool:
    """Update imports in all Python files in a directory."""
    if not Path(directory_path).exists():
        print(f"Directory not found: {directory_path}")
        return False

    # Get file pattern from live config
    file_pattern = _get_file_pattern()

    # Get all Python files in directory
    python_files = list(Path(directory_path).glob(file_pattern))

    # Update imports in each file
    updated_count = 0
    for file_path in python_files:
        if update_imports_in_file(str(file_path)):
            updated_count += 1

    print(f"Updated imports in {updated_count} files in {directory_path}")
    return True


def update_all_imports() -> None:
    """Update imports in all relevant files."""
    print("Starting import updates...")

    # Get target directory from live config
    target_directory = _get_target_directory()

    # Update imports in new structure
    update_imports_in_directory(target_directory)

    print("\nImport updates complete!")


# ------------------------------------------------------------------------------
# Configuration helpers (live config)
# ------------------------------------------------------------------------------
def _get_import_mappings() -> dict[str, str]:
    """Get import mappings from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "update_imports", None)
            if value and hasattr(value, "mappings"):
                mappings = value.mappings
                if isinstance(mappings, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in mappings.items()):
                    return mappings
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable or default mappings
    mappings_json = os.getenv("UPDATE_IMPORTS_MAPPINGS", "").strip()
    if mappings_json:
        try:
            mappings = json.loads(mappings_json)
            if isinstance(mappings, dict):
                return mappings
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            pass
    # Default mappings
    return {
        "backend.endpoints": "backend.app.api.v1.routers",
        "backend.services": "backend.app.services",
        "backend.models": "backend.app.domain.models",
        "backend.config": "backend.app.core",
    }


def _get_target_directory() -> str:
    """Get target directory from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "update_imports", None)
            if value and hasattr(value, "target_directory"):
                directory = value.target_directory
                if isinstance(directory, str) and directory:
                    return directory.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    directory = os.getenv("UPDATE_IMPORTS_TARGET_DIRECTORY", "backend/app").strip()
    return directory if directory else "backend/app"


def _get_file_pattern() -> str:
    """Get file pattern from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "update_imports", None)
            if value and hasattr(value, "file_pattern"):
                pattern = value.file_pattern
                if isinstance(pattern, str) and pattern:
                    return pattern.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    pattern = os.getenv("UPDATE_IMPORTS_FILE_PATTERN", "**/*.py").strip()
    return pattern if pattern else "**/*.py"


def _get_file_encoding() -> str:
    """Get file encoding from live configuration."""
    if _mystic_config is not None:
        try:
            value = getattr(_mystic_config, "update_imports", None)
            if value and hasattr(value, "file_encoding"):
                encoding = value.file_encoding
                if isinstance(encoding, str) and encoding:
                    return encoding.strip()
        except (AttributeError, ValueError, TypeError):
            pass
    # Fallback to environment variable
    encoding = os.getenv("UPDATE_IMPORTS_FILE_ENCODING", "utf-8").strip()
    return encoding if encoding else "utf-8"


if __name__ == "__main__":
    update_all_imports()
