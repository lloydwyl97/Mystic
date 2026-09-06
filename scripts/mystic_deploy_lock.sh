#!/bin/bash
# Operations-only deployment lock for Mystic.
#
# Canonical path: /run/mystic/deploy.lock
# Legacy alias:   /tmp/mystic_maintenance.lock  (honored by the watchdog so an
#                 in-flight operator using the old path is still protected)
#
# This is NOT a trading, execution, or strategy gate. start_mystic.sh still
# starts while the lock is held — that is required so an approved start can
# happen before the lock is released. Only watchdog_mystic.sh is suppressed.
#
# Usage:
#   mystic_deploy_lock.sh acquire [--sha HASH] [--reason TEXT]
#   mystic_deploy_lock.sh status
#   mystic_deploy_lock.sh release
#   mystic_deploy_lock.sh check     # exit 0 = watchdog must stay down
#
# There is no auto-expiry. A stale lock needs an explicit release.

set -u

CANONICAL_LOCK="${MYSTIC_DEPLOY_LOCK:-/run/mystic/deploy.lock}"
LEGACY_LOCK="${MYSTIC_MAINTENANCE_LOCK:-/tmp/mystic_maintenance.lock}"

_utc_now() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

_lock_dir() {
    dirname -- "$CANONICAL_LOCK"
}

_write_lock() {
    local path="$1"
    local sha="${2:-}"
    local reason="${3:-deploy}"
    local dir
    dir=$(_lock_dir)
    if ! mkdir -p "$dir" 2>/dev/null; then
        echo "ERROR: cannot create lock directory $dir" >&2
        return 1
    fi
    umask 022
    local tmp
    tmp="${path}.tmp.$$"
    cat >"$tmp" <<EOF
{
  "kind": "mystic_deploy_lock",
  "created_at": "$(_utc_now)",
  "operator": "$(id -un 2>/dev/null || echo unknown)",
  "hostname": "$(hostname 2>/dev/null || echo unknown)",
  "pid": "$$",
  "intended_sha": "${sha}",
  "reason": "${reason}"
}
EOF
    # ln without -f is atomic: succeeds only when $path does not already exist.
    if ! ln "$tmp" "$path" 2>/dev/null; then
        rm -f -- "$tmp"
        echo "ERROR: deploy lock already exists at $path" >&2
        echo "       run: $0 status" >&2
        return 1
    fi
    rm -f -- "$tmp"
}

_print_status() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "unlocked  path=$path"
        return 1
    fi
    if [ ! -f "$path" ]; then
        echo "LOCKED_MALFORMED  path=$path  reason=not_a_regular_file"
        return 0
    fi
    if [ ! -r "$path" ]; then
        echo "LOCKED_MALFORMED  path=$path  reason=unreadable"
        return 0
    fi
    echo "LOCKED  path=$path"
    cat "$path"
    return 0
}

cmd="${1:-}"
shift || true

case "$cmd" in
    acquire)
        sha=""
        reason="deploy"
        while [ $# -gt 0 ]; do
            case "$1" in
                --sha) sha="${2:-}"; shift 2 ;;
                --reason) reason="${2:-}"; shift 2 ;;
                *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
            esac
        done
        if ! _write_lock "$CANONICAL_LOCK" "$sha" "$reason"; then
            exit 1
        fi
        # Mirror a marker at the legacy path so an old watchdog still suppresses.
        if [ "$LEGACY_LOCK" != "$CANONICAL_LOCK" ]; then
            printf '%s\n' "legacy_alias_of=$CANONICAL_LOCK" >"$LEGACY_LOCK" 2>/dev/null || true
        fi
        echo "acquired $CANONICAL_LOCK"
        ;;
    status)
        if _print_status "$CANONICAL_LOCK"; then
            exit 0
        fi
        if [ "$LEGACY_LOCK" != "$CANONICAL_LOCK" ] && [ -e "$LEGACY_LOCK" ]; then
            _print_status "$LEGACY_LOCK"
            exit 0
        fi
        echo "unlocked"
        exit 1
        ;;
    release)
        removed=0
        if [ -e "$CANONICAL_LOCK" ]; then
            if [ ! -f "$CANONICAL_LOCK" ]; then
                echo "ERROR: $CANONICAL_LOCK is not a regular file; resolve manually" >&2
                exit 1
            fi
            rm -f -- "$CANONICAL_LOCK"
            removed=1
        fi
        if [ "$LEGACY_LOCK" != "$CANONICAL_LOCK" ] && [ -e "$LEGACY_LOCK" ]; then
            rm -f -- "$LEGACY_LOCK"
            removed=1
        fi
        if [ "$removed" -eq 0 ]; then
            echo "already unlocked"
        else
            echo "released"
        fi
        ;;
    check)
        # Exit 0 => watchdog must not start/restart Mystic.
        for path in "$CANONICAL_LOCK" "$LEGACY_LOCK"; do
            [ -z "$path" ] && continue
            if [ -e "$path" ] || [ -L "$path" ]; then
                if [ ! -f "$path" ]; then
                    echo "WATCHDOG_SUPPRESSED_MALFORMED_LOCK path=$path"
                    exit 0
                fi
                echo "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK path=$path"
                exit 0
            fi
        done
        exit 1
        ;;
    *)
        echo "Usage: $0 acquire [--sha HASH] [--reason TEXT] | status | release | check" >&2
        exit 2
        ;;
esac
