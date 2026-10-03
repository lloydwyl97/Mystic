#!/usr/bin/env python3
"""Single maintenance entry point for Mystic production (see backend/services/mystic_maintenance.py).

  mystic_maintenance.py run [--dry-run] [--no-reboot]   hourly orchestrator (cron, as root)
  mystic_maintenance.py post-reboot-verify               @reboot contract check
  mystic_maintenance.py status                           print the last run summary

Owner-only subcommands (invoked by ``run`` via runuser so DB files stay mystic-owned):
  backup-create [--dry-run] | backup-verify [--dry-run] | backup-retention --mode M [--dry-run] | db-report [--dry-run] | db-retention [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import mystic_maintenance as m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["run", "post-reboot-verify", "status", "backup-create", "backup-verify", "backup-retention", "db-report", "db-retention"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-reboot", action="store_true")
    parser.add_argument("--mode", default="normal", choices=["normal", "aggressive", "critical"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
    cfg = m.MaintConfig()

    if args.task == "run":
        out = m.run_maintenance(cfg, dry_run=args.dry_run, allow_reboot=not args.no_reboot)
    elif args.task == "post-reboot-verify":
        out = m.post_reboot_verify(cfg)
        m.record_post_reboot(cfg, out)
    elif args.task == "status":
        out = m.maintenance_status_summary(cfg)
    elif args.task == "backup-create":
        out = m.create_backup(cfg, dry_run=args.dry_run)
    elif args.task == "backup-verify":
        out = m.verify_unverified_backups(cfg, dry_run=args.dry_run)
    elif args.task == "backup-retention":
        out = m.apply_backup_retention(cfg, mode=args.mode, dry_run=args.dry_run, opened=m.open_paths())
    elif args.task == "db-report":
        out = m.db_report(cfg.live_db, checkpoint_threshold_bytes=cfg.wal_checkpoint_bytes, dry_run=args.dry_run)
    else:
        out = m.db_table_retention(cfg.live_db, dry_run=args.dry_run)
    print(json.dumps(out, default=str))
    return 0 if not (isinstance(out, dict) and out.get("status") == "error") else 1


if __name__ == "__main__":
    sys.exit(main())
