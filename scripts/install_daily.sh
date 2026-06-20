#!/usr/bin/env bash
# Install / reinstall the launchd agent that runs Scout every morning.
# Default schedule: 08:00 local time. Override with --hour and --minute.

set -e

HOUR=8
MINUTE=0
LABEL="com.victor.scout-daily"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --hour)   HOUR="$2";   shift 2;;
        --minute) MINUTE="$2"; shift 2;;
        --label)  LABEL="$2";  shift 2;;
        -h|--help)
            cat <<EOF
Install macOS launchd job for Scout daily run.

Usage:
    ./scripts/install_daily.sh                  # default 08:00
    ./scripts/install_daily.sh --hour 7 --minute 30
    ./scripts/install_daily.sh --label com.victor.scout-evening --hour 20

To uninstall:
    ./scripts/uninstall_daily.sh

To trigger now (test):
    launchctl start $LABEL
EOF
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

PROJECT_ROOT="/Users/victor/Documents/GitHub/AI agent/stock-rl"
RUN_SCRIPT="$PROJECT_ROOT/scripts/run_daily.sh"
PLIST_DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="$PROJECT_ROOT/results/daily_logs"

# Make sure the run script is executable
chmod +x "$RUN_SCRIPT"
mkdir -p "$LOG_DIR"

# Generate plist with absolute paths
cat > "$PLIST_DEST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${RUN_SCRIPT}</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>${HOUR}</integer>
        <key>Minute</key>
        <integer>${MINUTE}</integer>
    </dict>

    <key>RunAtLoad</key>
    <false/>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd.out</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd.err</string>

    <key>WorkingDirectory</key>
    <string>${PROJECT_ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>
    </dict>

    <!-- Run when missed (laptop closed at scheduled time) -->
    <key>StartCalendarIntervalCatchup</key>
    <true/>
</dict>
</plist>
EOF

echo "📄 Plist written: $PLIST_DEST"

# Unload any old version (silently)
launchctl unload "$PLIST_DEST" 2>/dev/null || true

# Load it
launchctl load "$PLIST_DEST"

echo "✅ Loaded launchd job: $LABEL"
echo "   Schedule       : every day at $(printf '%02d:%02d' "$HOUR" "$MINUTE")"
echo "   Logs           : $LOG_DIR/<date>.log"
echo "   Stdout / stderr: $LOG_DIR/launchd.{out,err}"
echo ""

# Show status
echo "Current launchd state:"
launchctl list | grep "$LABEL" || echo "  (not visible yet — should appear within seconds)"

echo ""
echo "Test it right now:  launchctl start $LABEL"
echo "Uninstall:          ./scripts/uninstall_daily.sh"
