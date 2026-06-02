"""
Script to run linter on new structure.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a command (list or string) and return its success and output."""
    try:
        # Accept either a list (preferred) or a string command
        if isinstance(command, str):
            # Run string under the shell for compatibility
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            # Run list without shell (safer)
            result = subprocess.run(command, shell=False, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    else:
        return True, result.stdout


def lint_directory(directory_path):
    """Run linter on a directory."""
    if not Path(directory_path).is_dir():
        print(f"Directory not found: {directory_path}")
        return False

    print(f"Running linter on {directory_path}...")

    # Use the current Python executable to run ruff for consistency
    cmd_fix = [sys.executable, "-m", "ruff", "check", directory_path, "--select=E,W", "--fix", "--target-version", "py310"]
    success, output = run_command(cmd_fix)
    print(output)

    # Run ruff check again to verify (no fix)
    cmd_check = [sys.executable, "-m", "ruff", "check", directory_path]
    success, output = run_command(cmd_check)
    print(output)

    return success


def lint_all():
    """Run linter on all relevant directories."""
    print("Starting linting...")

    # Lint new structure
    success = lint_directory("backend/app")

    if success:
        print("\nLinting successful!")
    else:
        print("\nLinting failed!")
        sys.exit(1)


if __name__ == "__main__":
    lint_all()
