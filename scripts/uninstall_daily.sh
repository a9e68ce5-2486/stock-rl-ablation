#!/usr/bin/env bash
# Uninstall the Scout daily launchd job.

set -e
LABEL="${1:-com.victor.scout-daily}"
PLIST_DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [ ! -f "$PLIST_DEST" ]; then
    echo "❌ Not installed: $PLIST_DEST does not exist."
    exit 1
fi

launchctl unload "$PLIST_DEST" 2>/dev/null || true
rm "$PLIST_DEST"
echo "🗑️  Uninstalled $LABEL"
echo "   Logs and reports under results/daily{,_logs}/ are kept."
echo "   Delete them manually if you want."
