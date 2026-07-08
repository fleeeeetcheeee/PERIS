#!/bin/zsh
# launchd entrypoint for the nightly pipeline. Kept as a script so PATH, cwd, and
# log rotation live in one debuggable place instead of inside a plist.
set -euo pipefail

DESK="/Users/fletcherlee/Documents/PERIS/PERIS/overnight-desk"
UV="/Users/fletcherlee/anaconda3/bin/uv"
LOG="$DESK/logs/nightly_$(date +%Y-%m-%d).log"

cd "$DESK"
echo "=== nightly run started $(date) ===" >> "$LOG"

# Skip weekends/holidays cheaply: nightly.py is idempotent anyway (signals table
# is UNIQUE per date), so a redundant run just re-emits the last session's ticket.
rc=0
"$UV" run python -m jobs.nightly --stage all >> "$LOG" 2>&1 || rc=$?

echo "=== nightly run finished $(date) exit=$rc ===" >> "$LOG"

# Keep the last 30 logs
ls -t "$DESK"/logs/nightly_*.log 2>/dev/null | tail -n +31 | xargs rm -f --

exit $rc
