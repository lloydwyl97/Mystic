"""The DAY label read must not scan every historical markout."""

from __future__ import annotations

from backend.services.adaptive_learning import _connect


def test_unresolved_markout_query_uses_the_due_index(tmp_path) -> None:
    db = str(tmp_path / "m.db")
    conn = _connect(db)
    try:
        plan = " ".join(
            str(row[3])
            for row in conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM adaptive_candidate_markouts
                WHERE resolved=0 AND engine_id=?
                ORDER BY learned ASC, id ASC
                LIMIT 200
                """,
                ("DAY_V2",),
            )
        )
    finally:
        conn.close()
    assert "idx_adaptive_markouts_due" in plan
    assert "SCAN" not in plan
