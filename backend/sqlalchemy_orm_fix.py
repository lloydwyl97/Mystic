"""
Fix for SQLAlchemy ORM imports in SQLAlchemy 1.0.15
"""

import sqlalchemy
import sqlalchemy.ext.declarative

# Add declarative_base to sqlalchemy.orm if it doesn't exist
if not hasattr(sqlalchemy.orm, "declarative_base"):
    sqlalchemy.orm.declarative_base = sqlalchemy.ext.declarative.declarative_base

# Create a dummy Mapped class if it doesn't exist (for SQLAlchemy 2.0 compatibility)
if not hasattr(sqlalchemy.orm, "Mapped"):

    class Mapped:
        pass

    sqlalchemy.orm.Mapped = Mapped

# Create a dummy mapped_column function if it doesn't exist (for SQLAlchemy 2.0 compatibility)
if not hasattr(sqlalchemy.orm, "mapped_column"):

    def mapped_column(*args, **kwargs):
        return sqlalchemy.Column(*args, **kwargs)

    sqlalchemy.orm.mapped_column = mapped_column
