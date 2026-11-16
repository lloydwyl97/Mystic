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
# There is no time limit. The lock is valid only while its owner process is
# still the same process (pid + /proc start time). A dead or reused pid is
# recovered by `check` so supervision is not suppressed forever.

set -u

CANONICAL_LOCK="${MYSTIC_DEPLOY_LOCK:-/run/mystic/deploy.lock}"
LEGACY_LOCK="${MYSTIC_MAINTENANCE_LOCK:-/tmp/mystic_maintenance.lock}"

_utc_now() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

_lock_dir() {
    dirname -- "$CANONICAL_LOCK"
}

_proc_starttime() {
    local pid="$1"
    local stat rest
    [ -r "/proc/${pid}/stat" ] || return 1
    stat=$(cat "/proc/${pid}/stat") || return 1
    # comm can contain spaces. starttime is field 22, so field 20 after "pid (comm)".
    rest="${stat#*) }"
    awk '{print $20}' <<<"$rest"
}

_json_field() {
    local path="$1"
    local key="$2"
    sed -n "s/.*\"${key}\": *\"\\([^\"]*\\)\".*/\\1/p" "$path" | head -1
}

_owner_alive() {
    local path="$1"
    local pid starttime now host
    pid=$(_json_field "$path" pid)
    starttime=$(_json_field "$path" starttime_ticks)
    [ -n "$pid" ] && [ -n "$starttime" ] || return 1
    now=$(_proc_starttime "$pid") || return 1
    [ "$now" = "$starttime" ] || return 1
    host=$(_json_field "$path" hostname)
    if [ -n "$host" ] && [ "$host" != "$(hostname 2>/dev/null || echo unknown)" ]; then
        return 1
    fi
    return 0
}

_quarantine() {
    local path="$1"
    local dest="${path}.recovered.$(_utc_now | tr -d ':').$$"
    if mv -- "$path" "$dest" 2>/dev/null; then
        echo "DEPLOY_LOCK_RECOVERED path=$path dest=$dest"
        return 0
    fi
    rm -rf -- "$path" 2>/dev/null || true
    echo "DEPLOY_LOCK_RECOVERED path=$path dest=removed"
}

_write_lock() {
    local path="$1"
    local sha="${2:-}"
    local reason="${3:-deploy}"
    local dir owner_pid starttime
    owner_pid="${MYSTIC_DEPLOY_OWNER_PID:-$PPID}"
    starttime=$(_proc_starttime "$owner_pid") || {
        echo "ERROR: deploy owner pid=$owner_pid is not alive" >&2
        return 1
    }
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
  "pid": "${owner_pid}",
  "starttime_ticks": "${starttime}",
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
        echo "DEPLOY_LOCK_ACQUIRED path=$CANONICAL_LOCK pid=${MYSTIC_DEPLOY_OWNER_PID:-$PPID}"
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
            echo "DEPLOY_LOCK_RELEASED"
            echo "released"
        fi
        ;;
    check)
        # Exit 0 => a live owner holds the lock; supervision must wait.
        # Exit 1 => no lock.
        # Exit 3 => stale or malformed lock was recovered; supervision may run.
        stale=0
        for path in "$CANONICAL_LOCK" "$LEGACY_LOCK"; do
            [ -z "$path" ] && continue
            [ "$CANONICAL_LOCK" = "$LEGACY_LOCK" ] && [ "$path" = "$LEGACY_LOCK" ] && [ "$stale" -eq 1 ] && continue
            if [ ! -e "$path" ] && [ ! -L "$path" ]; then
                continue
            fi
            if [ ! -f "$path" ] || [ ! -r "$path" ]; then
                echo "DEPLOY_LOCK_STALE reason=malformed path=$path"
                _quarantine "$path"
                stale=1
                continue
            fi
            if _owner_alive "$path"; then
                echo "WATCHDOG_SUPPRESSED_DEPLOYMENT_LOCK path=$path"
                exit 0
            fi
            echo "DEPLOY_LOCK_OWNER_DEAD path=$path"
            _quarantine "$path"
            stale=1
        done
        if [ "$stale" -eq 1 ]; then
            exit 3
        fi
        exit 1
        ;;
    *)
        echo "Usage: $0 acquire [--sha HASH] [--reason TEXT] | status | release | check" >&2
        exit 2
        ;;
esac
