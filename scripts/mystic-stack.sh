#!/usr/bin/env bash
# Mystic local stack control (systemd user units)
set -euo pipefail
cmd="${1:-status}"
case "$cmd" in
  start)   systemctl --user start mystic.target ;;
  stop)    systemctl --user stop mystic.target ;;
  restart) systemctl --user restart mystic.target ;;
  status)  systemctl --user status mystic.target --no-pager; echo; systemctl --user list-units 'mystic-*' --no-pager ;;
  logs)
    unit="${2:-mystic-uvicorn.service}"
    journalctl --user -u "$unit" -f
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [unit]}"
    echo "Units: mystic-uvicorn mystic-market-data mystic-ai-context mystic-ai-signals mystic-portfolio mystic-ai-learning"
    exit 1
    ;;
esac
