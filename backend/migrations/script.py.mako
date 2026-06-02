"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

Live Database Migration - All Live Data, No Fallback/Hardcoded Data

This migration operates on live database connections (backend port 8000).
All operations:
- Apply live schema changes to production database (backend port 8000)
- Modify live database tables, columns, indexes, and constraints
- No fallback/hardcoded data - all migrations operate on live database
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- Database schema: Live schema changes applied to production database
- Database tables: Live tables modified by migration operations
- All migrations operate on live database - no mock/test data

Endpoint References:
- Backend API: Port 8000 (migrations applied to live database used by backend)
- Database: Live database connection (from DATABASE_URL or AIDBManager)
- All migrations use live connections - no fallback/hardcoded data

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic for live database migrations
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """
    Apply live database migration to production database (backend port 8000).
    
    All operations modify live database schema, tables, columns, indexes, etc.
    No fallback/hardcoded data - all operations on live database.
    """
    ${upgrades if upgrades else "    # No live schema changes in this migration"}


def downgrade() -> None:
    """
    Revert live database migration on production database (backend port 8000).
    
    All operations revert live database schema changes.
    No fallback/hardcoded data - all operations on live database.
    """
    ${downgrades if downgrades else "    # No live schema revert in this migration"}
