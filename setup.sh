#!/usr/bin/env bash
# Build mirpy and optionally run the test suite.
#
# Usage:
#   ./setup.sh                  # create/reuse ./venv, rebuild, no tests
#   ./setup.sh --test           # rebuild + fast tests
#   ./setup.sh --test-all       # rebuild + fast + benchmark + integration tests
#   ./setup.sh my-venv --test   # custom venv dir + fast tests
#
# First non-flag argument is taken as the venv directory (default: venv).
set -euo pipefail

VENV="venv"
RUN_TESTS=0
RUN_HEAVY=0

for arg in "$@"; do
    case "$arg" in
        --test)     RUN_TESTS=1 ;;
        --test-all) RUN_TESTS=1; RUN_HEAVY=1 ;;
        -*)         echo "Unknown option: $arg"; exit 1 ;;
        *)          VENV="$arg" ;;
    esac
done

# ── Virtualenv ────────────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ] && [ ! -f "$VENV/Scripts/activate" ]; then
    echo "Creating virtualenv: $VENV"
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# ── Build ─────────────────────────────────────────────────────────────────────
# Remove stale cmake build artifacts; scikit-build-core writes into ./build
# and a partial previous build can cause confusing errors on the next run.
if [ -d "build" ]; then
    echo "Removing stale build/ directory..."
    rm -rf build
fi

echo "Building and installing mirpy (editable)..."
# CMAKE_POLICY_VERSION_MINIMUM silences CMake 3.27+ policy warnings from
# scikit-build-core's internal configuration step.
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e . --no-build-isolation

# ── Tests ─────────────────────────────────────────────────────────────────────
if [ "$RUN_TESTS" -eq 1 ]; then
    echo ""
    echo "Running fast test suite..."
    python -m pytest tests -m "not benchmark and not integration" -q

    if [ "$RUN_HEAVY" -eq 1 ]; then
        echo ""
        echo "Running benchmark and integration tests..."
        RUN_BENCHMARKS=1 RUN_INTEGRATION=1 \
            python -m pytest tests -m "benchmark or integration" -q
    fi
fi

echo ""
echo "Done.  Activate the environment with:"
echo "  source $VENV/bin/activate"
