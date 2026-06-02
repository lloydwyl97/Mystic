"""Initial AI models

Revision ID: 0cf39eaa7b98
Revises:
Create Date: 2025-08-29 20:36:20.477813+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0cf39eaa7b98"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -- ai_live_predictions ---------------------------------------------------
    op.create_table(
        "ai_live_predictions",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("symbol", sa.String(length=50), index=True),
        sa.Column("timeframe", sa.String(length=32), index=True),
        sa.Column("prob_up", sa.Float(), nullable=False, server_default="0"),
        sa.Column("prob_down", sa.Float(), nullable=False, server_default="0"),
        sa.Column("features", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        op.f("ix_ai_live_predictions_symbol"),
        "ai_live_predictions",
        ["symbol"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ai_live_predictions_timeframe"),
        "ai_live_predictions",
        ["timeframe"],
        unique=False,
    )

    # -- ai_live_signals -------------------------------------------------------
    op.create_table(
        "ai_live_signals",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("symbol", sa.String(length=50), index=True),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("reason", sa.String(length=512)),
        sa.Column("price", sa.Float()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("consumed", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.CheckConstraint("side in ('buy','sell','hold')", name="ck_ai_live_signals_side"),
    )
    op.create_index(
        op.f("ix_ai_live_signals_symbol"),
        "ai_live_signals",
        ["symbol"],
        unique=False,
    )
    op.create_index(
        "ix_ai_live_signals_created_at",
        "ai_live_signals",
        ["created_at"],
        unique=False,
    )

    # -- ai_live_strategies ----------------------------------------------------
    op.create_table(
        "ai_live_strategies",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("symbol", sa.String(length=50)),
        sa.Column("desc", sa.String(length=1000)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("win_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.UniqueConstraint("name", "symbol", name="uq_ai_live_strategies_name_symbol"),
    )
    op.create_index(
        op.f("ix_ai_live_strategies_name"),
        "ai_live_strategies",
        ["name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ai_live_strategies_symbol"),
        "ai_live_strategies",
        ["symbol"],
        unique=False,
    )

    # -- ai_live_strategy_kpi --------------------------------------------------
    op.create_table(
        "ai_live_strategy_kpi",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("strategy_name", sa.String(length=100), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("closed_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("winning_trades", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "last_updated",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("strategy_name", name="uq_ai_live_strategy_kpi_strategy_name"),
    )
    op.create_index(
        op.f("ix_ai_live_strategy_kpi_strategy_name"),
        "ai_live_strategy_kpi",
        ["strategy_name"],
        unique=True,
    )

    # -- ai_live_trade_fills ---------------------------------------------------
    op.create_table(
        "ai_live_trade_fills",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("exchange", sa.String(length=50), nullable=False, index=True),
        sa.Column("symbol", sa.String(length=50), nullable=False, index=True),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False, server_default="0"),
        sa.Column("price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost", sa.Float()),
        sa.Column("fee", sa.Float()),
        sa.Column("fee_currency", sa.String(length=20)),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("trade_id", sa.String(length=128)),
        sa.Column("order_id", sa.String(length=128)),
        sa.Column("strategy_name", sa.String(length=100)),
        sa.UniqueConstraint("exchange", "trade_id", name="uq_ai_live_exchange_trade_id"),
        sa.CheckConstraint("side in ('buy','sell')", name="ck_ai_live_trade_fills_side"),
    )
    op.create_index(
        op.f("ix_ai_live_trade_fills_exchange"),
        "ai_live_trade_fills",
        ["exchange"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ai_live_trade_fills_symbol"),
        "ai_live_trade_fills",
        ["symbol"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ai_live_trade_fills_trade_id"),
        "ai_live_trade_fills",
        ["trade_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_ai_live_trade_fills_ts"),
        "ai_live_trade_fills",
        ["ts"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_ai_live_trade_fills_ts"), table_name="ai_live_trade_fills")
    op.drop_index(op.f("ix_ai_live_trade_fills_trade_id"), table_name="ai_live_trade_fills")
    op.drop_index(op.f("ix_ai_live_trade_fills_symbol"), table_name="ai_live_trade_fills")
    op.drop_index(op.f("ix_ai_live_trade_fills_exchange"), table_name="ai_live_trade_fills")
    op.drop_table("ai_live_trade_fills")

    op.drop_index(op.f("ix_ai_live_strategy_kpi_strategy_name"), table_name="ai_live_strategy_kpi")
    op.drop_table("ai_live_strategy_kpi")

    op.drop_index(op.f("ix_ai_live_strategies_symbol"), table_name="ai_live_strategies")
    op.drop_index(op.f("ix_ai_live_strategies_name"), table_name="ai_live_strategies")
    op.drop_table("ai_live_strategies")

    op.drop_index("ix_ai_live_signals_created_at", table_name="ai_live_signals")
    op.drop_index(op.f("ix_ai_live_signals_symbol"), table_name="ai_live_signals")
    op.drop_table("ai_live_signals")

    op.drop_index(op.f("ix_ai_live_predictions_timeframe"), table_name="ai_live_predictions")
    op.drop_index(op.f("ix_ai_live_predictions_symbol"), table_name="ai_live_predictions")
    op.drop_table("ai_live_predictions")
