"""Read-only: which of the 12 structured HOLD categories actually get emitted,
and does every emitted record carry the nine required fields?

Section 5 requires each HOLD record to carry: symbol, decision/intent id, timestamp,
controlling authority, exact reason, observed values, required values, whether live
execution is blocked, and the next reevaluation / expiry.
"""

from __future__ import annotations

import json
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"

REQUIRED_CATEGORIES = [
    "MODEL_HOLD_TELEMETRY",
    "NO_RANKED_CANDIDATE",
    "WAITING_FOR_DIP",
    "TRAILING_LOW",
    "WAITING_FOR_REBOUND",
    "OPEN_POSITION_HOLD",
    "HARD_SAFETY_BLOCK",
    "CAPITAL_OR_SLOT_BLOCK",
    "ORDER_PENDING",
    "DATA_REPAIR_REQUIRED",
    "OPERATOR_CONTROL_BLOCK",
    "COOLDOWN_ACTIVE",
    "TRAILING_BUY_TERMINAL",
]

REQUIRED_FIELDS = [
    "symbol",
    "category",
    "exact_reason",
    "controlling_authority",
    "blocks_live_execution",
    "first_seen_ts",
    "last_seen_ts",
    "next_reevaluation",
]


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== category constants defined in code ===")
    try:
        import backend.services.day_decision_state as dds

        defined = {n for n in dir(dds) if n.isupper()}
        for cat in REQUIRED_CATEGORIES:
            val = getattr(dds, cat, None)
            print(f"  {cat:<24} defined={cat in defined} value={val!r}")
    except Exception as e:  # pragma: no cover - probe only
        print(f"  could not import day_decision_state: {e}")

    print("\n=== episode table: emitted categories ===")
    try:
        rows = list(
            c.execute(
                "SELECT category, COUNT(*) n, COUNT(DISTINCT symbol) syms, "
                "MIN(first_seen_ts) first_seen, MAX(last_seen_ts) last_seen, "
                "SUM(observation_count) obs "
                "FROM day_decision_hold_episodes GROUP BY category ORDER BY n DESC"
            )
        )
    except sqlite3.OperationalError as e:
        print(f"  table missing: {e}")
        return
    emitted = {r["category"] for r in rows}
    for r in rows:
        print(f"  {r['category']:<24} episodes={r['n']:<5} symbols={r['syms']} observations={r['obs']}")

    print("\n=== coverage vs the 12 required categories ===")
    for cat in REQUIRED_CATEGORIES:
        print(f"  {cat:<24} {'EMITTED' if cat in emitted else 'not yet observed'}")
    extra = emitted - set(REQUIRED_CATEGORIES)
    if extra:
        print(f"\n  categories beyond the required 12: {sorted(extra)}")

    print("\n=== episode columns ===")
    cols = [x[1] for x in c.execute("PRAGMA table_info(day_decision_hold_episodes)")]
    print("  " + ", ".join(cols))

    print("\n=== field completeness on emitted episodes ===")
    for f in REQUIRED_FIELDS:
        if f not in cols:
            print(f"  {f:<24} COLUMN MISSING")
            continue
        miss = c.execute(f"SELECT COUNT(*) FROM day_decision_hold_episodes WHERE {f} IS NULL OR TRIM(CAST({f} AS TEXT))=''").fetchone()[0]
        print(f"  {f:<24} null_or_blank={miss}")

    for f in ("last_observed_json", "required_json"):
        if f in cols:
            n = c.execute(f"SELECT COUNT(*) FROM day_decision_hold_episodes WHERE {f} IS NULL OR TRIM({f}) IN ('', '{{}}')").fetchone()[0]
            print(f"  {f:<24} empty={n}")

    print("\n=== per-category field completeness (the 9 required fields) ===")
    print(f"  {'category':<24} {'n':>4} {'blocks=1':>9} {'reason':>7} {'auth':>5} {'req_json':>9} {'next_reeval':>12} {'dec_or_intent':>14}")
    nonblank = "{0} IS NOT NULL AND TRIM(CAST({0} AS TEXT)) <> ''"
    checks = [
        ("blocks_live_execution=1", 9),
        (nonblank.format("exact_reason"), 7),
        (nonblank.format("controlling_authority"), 5),
        ("required_json IS NOT NULL AND TRIM(required_json) NOT IN ('', '{}')", 9),
        (nonblank.format("next_reevaluation"), 12),
        (
            "(" + nonblank.format("decision_id") + ") OR (" + nonblank.format("intent_id") + ")",
            14,
        ),
    ]
    for cat in sorted(emitted):
        n = c.execute("SELECT COUNT(*) FROM day_decision_hold_episodes WHERE category=?", (cat,)).fetchone()[0]
        cells = []
        for where, width in checks:
            got = c.execute(
                f"SELECT COUNT(*) FROM day_decision_hold_episodes WHERE category=? AND ({where})",
                (cat,),
            ).fetchone()[0]
            cells.append(f"{got:>{width}}")
        print(f"  {cat:<24} {n:>4} " + " ".join(cells))

    print("\n=== newest 10 episodes (real column names) ===")
    for r in c.execute("SELECT * FROM day_decision_hold_episodes ORDER BY last_seen_ts DESC LIMIT 10"):
        d = dict(r)
        print(f"  {d.get('symbol')!s:<14} {d.get('category')!s:<22} blocks={d.get('blocks_live_execution')} obs_n={d.get('observation_count')}")
        print(f"      reason={str(d.get('exact_reason'))[:40]:<40} authority={str(d.get('controlling_authority'))[:44]}")
        print(f"      decision_id={str(d.get('decision_id'))[:26]:<26} intent_id={str(d.get('intent_id'))[:22]}")
        print(f"      next_reeval={str(d.get('next_reevaluation'))[:22]:<22} expires={str(d.get('expires_at'))[:22]}")
        print(f"      observed={str(d.get('last_observed_json') or '')[:80]}")
        print(f"      required={str(d.get('required_json') or '')[:80]}")

    print("\n=== episodes missing required_json, by category ===")
    for r in c.execute("SELECT category, COUNT(*) n FROM day_decision_hold_episodes WHERE required_json IS NULL OR TRIM(required_json) IN ('', '{}') GROUP BY category ORDER BY n DESC"):
        print(f"  {r['category']:<24} missing={r['n']}")

    print("\n=== operational_state schema + hold snapshot keys ===")
    try:
        ocols = [x[1] for x in c.execute("PRAGMA table_info(operational_state)")]
        print("  columns: " + ", ".join(ocols))
        vcol = next((x for x in ("value", "state_value", "json_value", "payload") if x in ocols), None)
        kcol = next((x for x in ("key", "state_key", "name") if x in ocols), None)
        if kcol:
            for r in c.execute(f"SELECT {kcol} k FROM operational_state WHERE {kcol} LIKE '%hold%' OR {kcol} LIKE '%decision%' ORDER BY {kcol} LIMIT 10"):
                print(f"  key={r['k']}")
                if vcol:
                    raw = c.execute(
                        f"SELECT {vcol} FROM operational_state WHERE {kcol}=?",
                        (r["k"],),
                    ).fetchone()[0]
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            print(f"      symbols_in_map={sorted(parsed.keys())}")
                    except Exception:
                        print(f"      raw={str(raw)[:70]}")
    except sqlite3.OperationalError as e:
        print(f"  {e}")


if __name__ == "__main__":
    main()
