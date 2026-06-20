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

Commands:
    train         Train PPO model (~10-20 min on M-series MPS)
    test-data     Smoke test: fetch prices + compute features
    test-env      Smoke test: gym env reset/step
    tensorboard   Start TensorBoard at http://localhost:6006
    shell         Drop into venv Python REPL
    install <pkg> Install a package into the venv

Active venv: $VENV_DIR
EOF
        ;;
    *)
        echo "❌ Unknown command: $CMD"
        echo "Run ./run.sh with no arguments to see usage."
        exit 1
        ;;
esac
