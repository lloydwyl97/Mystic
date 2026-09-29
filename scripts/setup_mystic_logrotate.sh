#!/usr/bin/env bash
# setup_mystic_logrotate.sh — Install bounded log rotation for Mystic on Ocean.
#
# Policy
# ──────
#   • Installs deploy/logrotate-mystic (the canonical policy).
#   • Rotate at 20 MB (checked hourly), keep 60 compressed copies (>= 48 h), maxage 7 days.
#   • Never touch mystic_trading.db, *.db, *.db-wal, *.db-shm, *.json, *.bak*.
#   • Safe to run multiple times (idempotent).
#
# Run on Ocean as root:
#   bash scripts/setup_mystic_logrotate.sh

set -euo pipefail

CONF=/etc/logrotate.d/mystic
LOG_DIR=/home/mystic/mystic/logs

echo "Installing logrotate config → $CONF"

cp "$(dirname "$0")/../deploy/logrotate-mystic" "$CONF"

chmod 644 "$CONF"

echo "Verifying logrotate config..."
logrotate --debug "$CONF" 2>&1 | head -30

echo ""
echo "Disk usage before forced rotation:"
df -h /
ls -lh "$LOG_DIR"/*.log 2>/dev/null | sort -k5 -rh | head -10

echo ""
echo "Running forced rotation (--force)..."
logrotate --force "$CONF"

echo ""
echo "Disk usage after rotation:"
df -h /
ls -lh "$LOG_DIR"/ 2>/dev/null | sort -k5 -rh | head -20

echo ""
echo "Done. Logrotate installed at $CONF"
echo "Logs retain: 60 compressed copies (>= 48 h), 20MB trigger, maxage 7 days."
echo "Active log files are preserved via copytruncate (no restart required)."
