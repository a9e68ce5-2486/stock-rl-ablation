#!/usr/bin/env bash
# Stock RL pipeline wrapper.
# Activates the venv and runs a subcommand. Usage:
#   ./run.sh train          # Train PPO
#   ./run.sh test-data      # Smoke test data fetch
#   ./run.sh test-env       # Smoke test gym env
#   ./run.sh tensorboard    # Open TensorBoard
#   ./run.sh shell          # Drop into venv Python REPL
#   ./run.sh install <pkg>  # Install a package into the venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$(dirname "$PROJECT_ROOT")/stock-rl-env"

# Sanity check venv exists
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "❌ Virtual env not found at: $VENV_DIR"
    echo "   Create it with:"
    echo "   cd $(dirname "$PROJECT_ROOT") && python3 -m venv stock-rl-env"
    exit 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Make sure we're using the venv Python
PYTHON_PATH=$(which python3)
if [[ "$PYTHON_PATH" != "$VENV_DIR"* ]]; then
    echo "⚠️  Warning: Python is $PYTHON_PATH, expected venv path"
fi

cd "$PROJECT_ROOT"

CMD="${1:-}"

case "$CMD" in
    train)
        echo "🚀 Training PPO..."
        python3 policy/train_ppo.py
        ;;
    eval)
        shift
        echo "📊 Evaluating saved model..."
        python3 policy/eval.py "$@"
        ;;
    train-eval)
        echo "🚀 Training PPO + auto-eval after..."
        CONFIG="${CONFIG:-phase2_relaxed}"
        RUN_ID="${RUN_ID:-$CONFIG}"
        export CONFIG RUN_ID
        python3 policy/train_ppo.py && \
        echo "" && \
        echo "════════════════════════════════════════" && \
        echo "🦞 Training done ($CONFIG). Running evaluation..." && \
        echo "════════════════════════════════════════" && \
        python3 policy/eval.py "policy/checkpoints/ppo_${RUN_ID}.zip" "$RUN_ID" "$CONFIG"
        ;;
    ablation)
        echo "🧪 Running ablation study..."
        python3 policy/ablation.py
        ;;
    multi-seed)
        echo "🌱 Running multi-seed validation..."
        python3 policy/multi_seed.py
        ;;
    results)
        echo "📋 Showing accumulated results..."
        python3 policy/results_io.py
        ;;
    scout-universe)
        echo "📋 Showing scout universe..."
        python3 scout/fetch_universe.py
        ;;
    scout-label)
        shift
        echo "🔬 Labeling rallies across universe..."
        python3 scout/label_rallies.py "$@"
        ;;
    scout-analyze)
        echo "📊 Analyzing rally labels..."
        python3 scout/analyze_rallies.py
        ;;
    scout-features)
        shift
        echo "🧮 Computing features..."
        python3 scout/compute_features.py "$@"
        ;;
    scout-train)
        echo "🏋️  Training rally classifier..."
        python3 scout/train_classifier.py
        ;;
    scout-pipeline)
        echo "🦞 Full scout pipeline..."
        python3 scout/label_rallies.py && \
        python3 scout/compute_features.py && \
        python3 scout/train_classifier.py
        ;;
    scout-validate)
        echo "🔬 Multi-seed + top-K validation..."
        python3 scout/validate.py
        ;;
    scout-benchmarks)
        echo "🏆 Running benchmark cases (MU + similar rallies)..."
        python3 scout/benchmark_cases.py
        ;;
    scout-walkforward)
        echo "🔬 Walk-forward benchmark (truly out-of-sample)..."
        python3 scout/benchmark_walkforward.py
        ;;
    scout-agent)
        shift
        echo "🤖 Scout agent: picks + LLM thesis..."
        python3 scout/picks_with_thesis.py "$@"
        ;;
    scout-thesis-test)
        shift
        echo "🦞 Smoke test thesis writer..."
        python3 scout/thesis_writer.py "$@"
        ;;
    scout-macro)
        shift
        echo "🌍 Macro context briefing..."
        python3 scout/macro_context.py "$@"
        ;;
    scout-sector)
        shift
        echo "🏭 Sector news lookup..."
        python3 scout/sector_news.py "$@"
        ;;
    scout-earnings)
        shift
        echo "📅 Next earnings lookup..."
        python3 scout/earnings_calendar.py "$@"
        ;;
    scout-chart)
        shift
        echo "📊 Chart analysis (MACD/Bollinger/Trend/Candles)..."
        python3 scout/chart_analysis.py "$@"
        ;;
    scout-watch)
        shift
        echo "📡 Watchlist monitor: chart + buy/sell/hold..."
        python3 scout/watchlist.py "$@"
        ;;
    portfolio)
        shift
        echo "📊 Portfolio monitor (your real positions)..."
        python3 scout/portfolio.py "$@"
        ;;
    portfolio-init)
        echo "📝 Creating positions.json template..."
        python3 scout/portfolio.py --init
        ;;
    portfolio-edit)
        FILE="results/positions.json"
        if [ ! -f "$FILE" ]; then
            python3 scout/portfolio.py --init
        fi
        echo "📝 Opening $FILE..."
        open "$FILE"
        ;;
    daily-install)
        shift
        echo "⏰ Installing daily launchd job..."
        bash scripts/install_daily.sh "$@"
        ;;
    daily-uninstall)
        echo "🗑️  Uninstalling daily launchd job..."
        bash scripts/uninstall_daily.sh
        ;;
    daily-test)
        echo "🧪 Triggering daily run now..."
        launchctl start com.victor.scout-daily 2>/dev/null && \
            echo "✅ Triggered. Watch results/daily_logs/$(date +%Y-%m-%d).log for output."
        ;;
    daily-status)
        echo "📋 Daily launchd status..."
        launchctl list | grep scout-daily || echo "(not installed)"
        echo ""
        echo "Recent log files:"
        ls -lh results/daily_logs/*.log 2>/dev/null | tail -5 || echo "  (none yet)"
        echo ""
        echo "Recent reports:"
        ls -lh results/daily/*.md 2>/dev/null | tail -10 || echo "  (none yet)"
        ;;
    daily-today)
        TODAY="$(date +%Y-%m-%d)"
        BRIEFING="results/daily/briefing_${TODAY}.html"
        if [ -f "$BRIEFING" ]; then
            echo "📂 Opening today's briefing in browser..."
            open "$BRIEFING"
        else
            echo "  ⚠️  $BRIEFING not yet generated"
            echo "  Try: ./run.sh daily-test (run pipeline now) or wait for 08:00."
        fi
        ;;
    daily-briefing)
        TODAY="$(date +%Y-%m-%d)"
        echo "🎨 (Re)building HTML briefing for $TODAY..."
        python3 scout/briefing.py --date "$TODAY" && \
            open "results/daily/briefing_${TODAY}.html"
        ;;
    daily-markdown)
        TODAY="$(date +%Y-%m-%d)"
        echo "📂 Opening raw markdown reports..."
        [ -f "results/daily/agent_${TODAY}.md" ] && open "results/daily/agent_${TODAY}.md"
        [ -f "results/daily/watchlist_${TODAY}.md" ] && open "results/daily/watchlist_${TODAY}.md"
        ;;
    daily-yesterday)
        YDAY="$(date -v-1d +%Y-%m-%d)"
        echo "📂 Opening yesterday's reports ($YDAY)..."
        [ -f "results/daily/agent_${YDAY}.md" ] && open "results/daily/agent_${YDAY}.md"
        [ -f "results/daily/watchlist_${YDAY}.md" ] && open "results/daily/watchlist_${YDAY}.md"
        ;;
    daily-list)
        echo "📅 All daily reports:"
        ls -lt results/daily/ 2>/dev/null | grep '\.md$' | head -30 || echo "  (none yet)"
        ;;
    daily-tail)
        TODAY="$(date +%Y-%m-%d)"
        LOGFILE="results/daily_logs/${TODAY}.log"
        if [ -f "$LOGFILE" ]; then
            echo "📜 Tailing $LOGFILE..."
            tail -50 "$LOGFILE"
        else
            echo "  ⚠️ No log for today yet."
            LATEST="$(ls -t results/daily_logs/*.log 2>/dev/null | head -1)"
            [ -n "$LATEST" ] && echo "Latest log: $LATEST" && tail -30 "$LATEST"
        fi
        ;;
    daily-peek)
        # Quick text preview of today's picks without opening anything
        TODAY="$(date +%Y-%m-%d)"
        AGENT_FILE="results/daily/agent_${TODAY}.md"
        WATCH_FILE="results/daily/watchlist_${TODAY}.md"
        echo ""
        if [ -f "$AGENT_FILE" ]; then
            echo "═══════════════════════════════════════"
            echo "🔍 DISCOVERY ($AGENT_FILE)"
            echo "═══════════════════════════════════════"
            grep -E "^## |^### 一句話結論" "$AGENT_FILE" | head -20
        fi
        echo ""
        if [ -f "$WATCH_FILE" ]; then
            echo "═══════════════════════════════════════"
            echo "📡 WATCHLIST ($WATCH_FILE)"
            echo "═══════════════════════════════════════"
            grep -E "^## |^### 建議|Net signal score" "$WATCH_FILE" | head -30
        fi
        ;;
    scout-picks)
        shift
        echo "🦞 Generating today's picks..."
        python3 scout/picks_today.py "$@"
        ;;
    test-data)
        echo "📦 Testing data pipeline..."
        python3 data/fetch_prices.py
        ;;
    test-env)
        echo "🎮 Testing gym env..."
        python3 env/portfolio_env.py
        ;;
    tensorboard)
        echo "📊 Starting TensorBoard at http://localhost:6006"
        # Auto-install if missing
        python3 -c "import tensorboard" 2>/dev/null || pip install tensorboard
        tensorboard --logdir tb_logs --port 6006
        ;;
    shell)
        echo "🐍 Dropping into venv Python REPL (Ctrl+D to exit)"
        python3
        ;;
    install)
        shift
        if [ -z "$*" ]; then
            echo "Usage: ./run.sh install <package> [more packages...]"
            exit 1
        fi
        pip install "$@"
        ;;
    "")
        cat <<EOF
🦞 Stock RL pipeline runner

Usage: ./run.sh <command>

== 🤖 Scout Agent (Day 2 — stock screening) ==
    scout-agent           ⭐ Top picks + LLM thesis (the agent)
                            args: --top_k N --as_of YYYY-MM-DD
    scout-picks           Top picks with feature bullets only (no LLM)
    scout-walkforward     Walk-forward benchmark (truly out-of-sample)
    scout-benchmarks      Benchmark cases (with leaky training)
    scout-validate        Multi-seed AUC + top-K precision
    scout-train           Train rally classifier
    scout-features        Compute features for full universe
    scout-label           Re-label rallies in universe
    scout-analyze         Statistics on rally labels
    scout-universe        Show universe (129 tickers)
    scout-pipeline        Full pipeline: label → features → train
    scout-thesis-test     Smoke test LLM thesis writer with mock features

== 📊 RL Portfolio (Day 1 — exploration) ==
    train                 Train PPO model (~10-20 min on M-series MPS)
    train-eval            Train then auto-eval
    eval                  Evaluate a saved model
    ablation              Run ablation across configs (single-seed)
    multi-seed            Multi-seed validation for a config
    results               Print accumulated CSV results table
    test-data             Smoke test: fetch prices + compute features
    test-env              Smoke test: gym env reset/step
    tensorboard           Start TensorBoard at http://localhost:6006

== ⏰ Daily Automation ==
    daily-install         Install macOS launchd job (default 08:00)
    daily-uninstall       Remove the launchd job
    daily-test            Trigger today's run NOW (for testing)
    daily-status          Show launchd status + recent files
    daily-today           ⭐ Open today's reports in default app
    daily-yesterday       Open yesterday's reports
    daily-list            List all daily reports (newest first)
    daily-tail            tail last 50 lines of today's log
    daily-peek            Quick text preview without opening files

== 🛠️ Misc ==
    shell                 Drop into venv Python REPL
    install <pkg>         Install a package into the venv

Active venv: $VENV_DIR
EOF
        ;;
    *)
        echo "❌ Unknown command: $CMD"
        echo "Run ./run.sh with no arguments to see usage."
        exit 1
        ;;
esac
