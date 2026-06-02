#!/usr/bin/env bash
# One-time local setup: venv, env file, systemd user units, Redis check.
set -euo pipefail

MYSTIC_ROOT="${MYSTIC_ROOT:-/home/mystic/mystic}"
cd "$MYSTIC_ROOT"

echo "=== Mystic local setup ==="
echo "Root: $MYSTIC_ROOT"

if ! command -v redis-cli >/dev/null 2>&1; then
  echo "WARN: redis-cli not found. Install Redis and ensure it runs on 127.0.0.1:6379"
else
  if redis-cli ping >/dev/null 2>&1; then
    echo "OK: Redis PONG"
  else
    echo "WARN: Redis not responding. Start with: sudo systemctl start redis-server"
  fi
fi

if [ ! -d venv ]; then
  echo "Creating venv..."
  python3 -m venv venv
fi

echo "Installing dependencies (may take several minutes)..."
./venv/bin/pip install -U pip wheel
./venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit API keys before trading."
fi

mkdir -p logs models/active models/training_data models/versions

SYSTEMD_USER="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$SYSTEMD_USER"
for unit in deploy/systemd/user/*; do
  dest="$SYSTEMD_USER/$(basename "$unit")"
  if [ "$MYSTIC_ROOT" != "/home/mystic/mystic" ]; then
    sed "s|/home/mystic/mystic|$MYSTIC_ROOT|g" "$unit" > "$dest"
  else
    cp "$unit" "$dest"
  fi
done

systemctl --user daemon-reload
systemctl --user enable mystic.target
echo ""
echo "Setup complete."
echo "  Edit: $MYSTIC_ROOT/.env"
echo "  Start: cd $MYSTIC_ROOT && ./start_mystic.sh core"
echo "  Or:    systemctl --user start mystic.target"
echo "  Dashboard: http://127.0.0.1:8000/dashboard/"
