"""
Migration script to move endpoints from old structure to new structure.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def create_directory_if_not_exists(directory_path):
    """Create directory if it doesn't exist."""
    path = Path(directory_path)
    if path.exists():
        if not path.is_dir():
            msg = f"Path exists and is not a directory: {directory_path}"
            raise OSError(msg)
        return
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {directory_path}")


def _migrate_file(source_file, target_dir, target_file_name, label=None):
    """Helper to migrate a single file to a target directory with error handling."""
    if not Path(source_file).exists():
        print(f"Source file not found: {source_file}")
        return False

    try:
        create_directory_if_not_exists(target_dir)
    except OSError as e:
        print(f"Failed to ensure target directory {target_dir}: {e}")
        return False

    target_file = str(Path(target_dir) / target_file_name)
    try:
        shutil.copy2(source_file, target_file)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        print(f"Failed to copy {source_file} to {target_file}: {e}")
        return False

    if label:
        print(f"Migrated {label} endpoints from {source_file} to {target_file}")
    else:
        print(f"Migrated endpoints from {source_file} to {target_file}")
    return True


def migrate_health_endpoints():
    """Migrate health endpoints to new structure."""
    source_file = "backend/endpoints/health_endpoints.py"
    target_dir = "backend/app/api/v1/routers"
    return _migrate_file(source_file, target_dir, "health.py", label="health")


def migrate_auth_endpoints():
    """Migrate auth endpoints to new structure."""
    source_file = "backend/endpoints/auth.py"
    target_dir = "backend/app/api/v1/routers"
    return _migrate_file(source_file, target_dir, "auth.py", label="auth")


def migrate_trading_endpoints():
    """Migrate trading endpoints to new structure."""
    source_file = "backend/endpoints/trading/trading_endpoints.py"
    target_dir = "backend/app/api/v1/routers"
    return _migrate_file(source_file, target_dir, "trades.py", label="trading")


def migrate_portfolio_endpoints():
    """Migrate portfolio endpoints to new structure."""
    source_file = "backend/endpoints/portfolio/portfolio_endpoints.py"
    target_dir = "backend/app/api/v1/routers"
    return _migrate_file(source_file, target_dir, "accounts.py", label="portfolio")


def migrate_market_endpoints():
    """Migrate market endpoints to new structure."""
    source_file = "backend/endpoints/market/market_data_endpoints.py"
    target_dir = "backend/app/api/v1/routers"
    return _migrate_file(source_file, target_dir, "markets.py", label="market")


def migrate_all_endpoints():
    """Migrate all endpoints to new structure."""
    print("Starting endpoint migration...")

    # Migrate endpoints
    health_migrated = migrate_health_endpoints()
    auth_migrated = migrate_auth_endpoints()
    trading_migrated = migrate_trading_endpoints()
    portfolio_migrated = migrate_portfolio_endpoints()
    market_migrated = migrate_market_endpoints()

    # Print summary
    print("\nMigration Summary:")
    print(f"Health endpoints: {'Migrated' if health_migrated else 'Not found'}")
    print(f"Auth endpoints: {'Migrated' if auth_migrated else 'Not found'}")
    print(f"Trading endpoints: {'Migrated' if trading_migrated else 'Not found'}")
    print(f"Portfolio endpoints: {'Migrated' if portfolio_migrated else 'Not found'}")
    print(f"Market endpoints: {'Migrated' if market_migrated else 'Not found'}")

    logger = logging.getLogger(__name__)
    logger.info("\nMigration complete!")


if __name__ == "__main__":
    migrate_all_endpoints()
