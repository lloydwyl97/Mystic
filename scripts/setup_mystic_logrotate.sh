#!/usr/bin/env bash
# setup_mystic_logrotate.sh — Install bounded log rotation for Mystic on Ocean.
#
# Policy
# ──────
#   • Rotate daily when > 20 MB or weekly unconditionally.
#   • Keep 7 rotated copies (7 days of history).
#   • Compress rotated logs with gzip (saves ~90% space).
#   • Never touch mystic_trading.db, *.db, *.db-wal, *.db-shm, *.json, *.bak*.
#   • Safe to run multiple times (idempotent).
#
# Run on Ocean as root:
#   bash scripts/setup_mystic_logrotate.sh

set -euo pipefail

CONF=/etc/logrotate.d/mystic
LOG_DIR=/home/mystic/mystic/logs

echo "Installing logrotate config → $CONF"

cat > "$CONF" << 'EOF'
/home/mystic/mystic/logs/*.log {
    daily
    size 20M
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    # Do NOT follow symlinks (no logs are symlinks).
    nolinks
    # Owner / permissions preserved.
    su mystic mystic
}
EOF

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
echo "Logs retain: 7 compressed copies, daily/20MB trigger."
echo "Active log files are preserved via copytruncate (no restart required)."
