#!/usr/bin/env bash
# Scout daily runner. Triggered by launchd every morning.
# - Activates venv
# - Runs discovery mode (top 5 picks) and watchlist (NVDA/AVGO/TSM)
# - Writes dated reports to results/daily/
# - Sends a macOS notification when done
# - Cleans up daily reports older than 60 days

set -u

# === Config ===
PROJECT_ROOT="/Users/victor/Documents/GitHub/AI agent/stock-rl"
VENV_DIR="/Users/victor/Documents/GitHub/AI agent/stock-rl-env"
WATCHLIST="NVDA AVGO TSM"   # edit to add/remove tickers
TOP_K=5

# === Paths ===
DATE="$(date +%Y-%m-%d)"
TIME="$(date +%H:%M:%S)"
LOG_DIR="$PROJECT_ROOT/results/daily_logs"
REPORT_DIR="$PROJECT_ROOT/results/daily"
mkdir -p "$LOG_DIR" "$REPORT_DIR"
LOG_FILE="$LOG_DIR/${DATE}.log"

# === Activate venv ===
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

# === Run ===
{
    echo ""
    echo "════════════════════════════════════════"
    echo "🦞 Scout daily run — $DATE $TIME"
    echo "════════════════════════════════════════"

    echo ""
    echo "🔍 Discovery (top $TOP_K undiscovered candidates)..."
    python3 scout/picks_with_thesis.py \
        --top_k "$TOP_K" \
        --out "results/daily/agent_${DATE}.md" \
        && AGENT_OK=1 || AGENT_OK=0

    echo ""
    echo "📡 Watchlist ($WATCHLIST)..."
    # shellcheck disable=SC2086
    python3 scout/watchlist.py $WATCHLIST \
        --out "results/daily/watchlist_${DATE}.md" \
        && WATCH_OK=1 || WATCH_OK=0

    echo ""
    echo "── Cleanup: removing reports older than 60 days ──"
    find "$REPORT_DIR" -type f -mtime +60 -delete
    find "$LOG_DIR" -type f -mtime +60 -delete

    echo ""
    echo "✅ Done at $(date +%H:%M:%S)"
} >> "$LOG_FILE" 2>&1

# === Notify ===
if command -v osascript >/dev/null 2>&1; then
    if [ "${AGENT_OK:-0}" = "1" ] && [ "${WATCH_OK:-0}" = "1" ]; then
        TITLE="🦞 Scout — Daily ready"
        MSG="$TOP_K picks + watchlist updated. Open results/daily/"
    else
        TITLE="🦞 Scout — Errors"
        MSG="See $LOG_FILE"
    fi
    osascript -e "display notification \"$MSG\" with title \"$TITLE\"" 2>/dev/null || true
fi
