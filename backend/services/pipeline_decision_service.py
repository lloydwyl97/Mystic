"""
LOW #5 FIX: Unified Pipeline Decision Service

Consolidates duplicate _update_pipeline_decision logic from:
- portfolio_engine_integration.py
- paper_trading_endpoints.py

This ensures consistent database handling and avoids logic drift.
"""

import asyncio
import logging
import sqlite3
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)


async def update_pipeline_decision(decision_id: str, updates: dict[str, Any]) -> None:
    """
    Update existing pipeline decision with new stage data.

    Unified implementation replacing duplicates in:
    - portfolio_engine_integration._update_pipeline_decision
    - paper_trading_endpoints._update_pipeline_decision

    Args:
        decision_id: Unique decision identifier
        updates: Dictionary of fields to update (e.g., {"stage": "GATES", "gate_result": "APPROVED"})
    """
    try:

        def _db_operation():
            with connect_rw(DATABASE_PATH) as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.cursor()

                update_fields = []
                update_values = []

                for key, value in updates.items():
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)

                update_values.append(decision_id)

                cursor.execute(
                    f"""
                    UPDATE pipeline_decisions
                    SET {", ".join(update_fields)}
                    WHERE decision_id = ?
                    """,
                    update_values,
                )

                conn.commit()

        await asyncio.to_thread(run_locked_retry, _db_operation)

    except Exception as e:
        logger.warning(f"Failed to update pipeline decision {decision_id}: {e}")
