#!/bin/bash
# Hourly local disk hygiene: rotate logs, purge pip/worktree/cursor leftovers.
# Safe while Mystic is running. Does not VACUUM SQLite (offline only).

set -euo pipefail

REPO="${MYSTIC_REPO:-/home/mystic/mystic}"
HOME_DIR="${HOME:-/home/mystic}"
LOGROTATE_CONF="${MYSTIC_LOGROTATE_CONF:-$REPO/deploy/mystic-logrotate.conf}"
LOGROTATE_STATE="${MYSTIC_LOGROTATE_STATE:-$REPO/.logrotate.state}"
PIP_BIN="${MYSTIC_PIP:-$REPO/venv/bin/pip}"
WORKTREE_DAYS="${MYSTIC_WORKTREE_MAX_DAYS:-14}"
CURSOR_BIN_KEEP="${MYSTIC_CURSOR_BIN_KEEP:-1}"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

if [ -f "$LOGROTATE_CONF" ] && [ -x /usr/sbin/logrotate ]; then
  /usr/sbin/logrotate -s "$LOGROTATE_STATE" "$LOGROTATE_CONF" || log "WARN logrotate failed"
fi

if [ -x "$PIP_BIN" ]; then
  "$PIP_BIN" cache purge >/dev/null 2>&1 || true
fi
rm -rf "${HOME_DIR}/.cache/pip"

WORKTREES="${HOME_DIR}/.cursor/worktrees"
if [ -d "$WORKTREES" ]; then
  find "$WORKTREES" -mindepth 1 -maxdepth 1 -type d -mtime "+${WORKTREE_DAYS}" -exec rm -rf {} +
fi

CURSOR_BIN="${HOME_DIR}/.cursor-server/bin/linux-x64"
if [ -d "$CURSOR_BIN" ]; then
  running=""
  for pid in $(pgrep -f 'cursor-server/.*/out/server-main.js' 2>/dev/null || true); do
    exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
    case "$exe" in
      "${CURSOR_BIN}/"*)
        running="${running} $(echo "$exe" | awk -F/ '{print $(NF-1)}')"
        ;;
    esac
  done
  mapfile -t versions < <(ls -1t "$CURSOR_BIN" 2>/dev/null || true)
  keep=0
  for ver in "${versions[@]}"; do
    [ -z "$ver" ] && continue
    [ ! -d "${CURSOR_BIN}/${ver}" ] && continue
    case " $running " in
      *" $ver "*) continue ;;
    esac
    keep=$((keep + 1))
    if [ "$keep" -gt "$CURSOR_BIN_KEEP" ]; then
      rm -rf "${CURSOR_BIN}/${ver}"
      log "removed stale cursor-server ${ver}"
    fi
  done
fi

DEBUG_LOG="${HOME_DIR}/.cursor/debug.log"
if [ -f "$DEBUG_LOG" ]; then
  size=$(stat -c%s "$DEBUG_LOG" 2>/dev/null || echo 0)
  if [ "${size:-0}" -gt 104857600 ]; then
    : > "$DEBUG_LOG"
    log "truncated ${DEBUG_LOG} (${size} bytes)"
  fi
fi
