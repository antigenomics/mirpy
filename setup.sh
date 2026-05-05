#!/usr/bin/env bash
# Build mirpy, optionally install docs dependencies, and optionally run tests.
#
# Usage:
#   ./setup.sh                  # create/reuse ./venv, rebuild, no tests
#   ./setup.sh --docs           # rebuild and install docs requirements
#   ./setup.sh --test           # rebuild + fast tests
#   ./setup.sh --test-all       # rebuild + fast + benchmark + integration tests
#   ./setup.sh my-venv --test   # custom venv dir + fast tests
#
# First non-flag argument is taken as the venv directory (default: venv).
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

VENV="venv"
INSTALL_DOCS=0
RUN_TESTS=0
RUN_HEAVY=0

for arg in "$@"; do
    case "$arg" in
        --docs)     INSTALL_DOCS=1 ;;
        --test)     RUN_TESTS=1 ;;
        --test-all) RUN_TESTS=1; RUN_HEAVY=1 ;;
        -*)         echo "Unknown option: $arg"; exit 1 ;;
        *)          VENV="$arg" ;;
    esac
done

# ── Virtualenv ────────────────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtualenv: $VENV"
    python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
PYTHON_BIN="$VENV/bin/python"

# Verify that we're using the venv's pip, not the global one (safety check)
if ! "$PYTHON_BIN" -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix else 1)"; then
    echo "Error: Python is not running in a virtual environment."
    echo "This safety check prevents accidental installation to the global Python."
    exit 1
fi

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel --quiet

echo "Installing dependencies from requirements.txt..."
"$PYTHON_BIN" -m pip install -r requirements.txt --quiet

if [ "$INSTALL_DOCS" -eq 1 ]; then
    echo "Installing documentation dependencies..."
    "$PYTHON_BIN" -m pip install -r docs/requirements.txt --quiet
fi

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
CMAKE_POLICY_VERSION_MINIMUM=3.5 "$PYTHON_BIN" -m pip install -e . --no-build-isolation

# ── Tests ─────────────────────────────────────────────────────────────────────
if [ "$RUN_TESTS" -eq 1 ]; then
    echo ""
    echo "Running fast test suite..."
    "$PYTHON_BIN" -m pytest tests -m "not benchmark and not integration" -q

    if [ "$RUN_HEAVY" -eq 1 ]; then
        echo ""
        echo "Running benchmark and integration tests..."
        RUN_BENCHMARKS=1 RUN_INTEGRATION=1 \
            "$PYTHON_BIN" -m pytest tests -m "benchmark or integration" -q
    fi
fi

echo ""
echo "Done.  Activate the environment with:"
echo "  source $VENV/bin/activate"
