"""
Script to run all migration steps.
"""

import subprocess
import sys


def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n=== {description} ===\n")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    else:
        return True


def run_migration():
    """Run all migration steps."""
    print("Starting migration to new structure...")

    # Step 1: Migrate endpoints
    if not run_command("python backend/migrate_endpoints.py", "Migrating Endpoints"):
        print("Failed to migrate endpoints!")
        sys.exit(1)

    # Step 2: Update imports
    if not run_command("python backend/update_imports.py", "Updating Imports"):
        print("Failed to update imports!")
        sys.exit(1)

    # Step 3: Run linter
    if not run_command("python backend/lint_new_structure.py", "Running Linter"):
        print("Linting failed!")
        sys.exit(1)

    print("\nMigration complete!")


if __name__ == "__main__":
    run_migration()
