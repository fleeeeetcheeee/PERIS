#!/bin/zsh
# launchd entrypoint for the TradingAgents desk server (UI + background worker).
# KeepAlive in the plist restarts it if it dies; exec hands the process to launchd.
set -euo pipefail

DESK="/Users/fletcherlee/Documents/PERIS/PERIS/overnight-desk"
UV="/Users/fletcherlee/anaconda3/bin/uv"
LOG="$DESK/logs/tradingagents_desk.log"

cd "$DESK"
mkdir -p "$DESK/logs"
# rotate at ~5 MB, keep one previous
if [[ -f "$LOG" && $(stat -f%z "$LOG") -gt 5242880 ]]; then
  mv "$LOG" "$LOG.1"
fi
echo "=== desk server starting $(date) ===" >> "$LOG"

exec "$UV" run python -m tradingagents.app >> "$LOG" 2>&1
