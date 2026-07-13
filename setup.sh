#!/usr/bin/env bash
# mirpy v3 bootstrap — reproducible install (conda-based, pure Python, no C build).
#
# Steps:
#   1. Create/update the `mirpy` conda environment from environment.yml.
#   2. pip install -e ../seqtree ../vdjtools (if present) then -e .
#   3. Optionally install docs deps / run tests.
#
# Flags:
#   --no-conda      Use the already-active environment instead of creating `mirpy`.
#   --docs          Install docs deps.
#   --test          Install [dev,bench] and run the test suite.
#   --test-all      Same as --test (no separate benchmark tier in v3 yet).
#
# Usage:
#   bash setup.sh [--no-conda] [--docs] [--test] [--test-all]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ENV_NAME="mirpy"
USE_CONDA=1
INSTALL_DOCS=0
RUN_TESTS=0
RUN_HEAVY=0

for arg in "$@"; do
  case "$arg" in
    --no-conda) USE_CONDA=0 ;;
    --docs)     INSTALL_DOCS=1 ;;
    --test)     RUN_TESTS=1 ;;
    --test-all) RUN_TESTS=1; RUN_HEAVY=1 ;;
    --help|-h)  sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[mirpy]\033[0m %s\n' "$*"; }

# --- 1. conda environment --------------------------------------------------
if [[ "$USE_CONDA" -eq 1 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH; install miniconda/anaconda or pass --no-conda." >&2
    exit 1
  fi
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "conda env '$ENV_NAME' exists — updating from environment.yml"
    conda env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
  else
    log "creating conda env '$ENV_NAME' from environment.yml"
    conda env create -f "$ROOT/environment.yml"
  fi
  RUN="conda run -n $ENV_NAME"
else
  RUN=""
fi

# --- 2. editable install (pure Python — no C build) ------------------------
# Co-developed siblings first if present, else PyPI resolves them.
for parent in seqtree vdjtools; do
  if [[ -d "$ROOT/../$parent" ]]; then
    log "pip install -e ../$parent"
    $RUN python -m pip install -e "$ROOT/../$parent"
  fi
done
log "pip install -e ."
$RUN python -m pip install -e "$ROOT"

# --- 3. optional docs ------------------------------------------------------
if [[ "$INSTALL_DOCS" -eq 1 ]]; then
  log "installing docs deps"
  $RUN python -m pip install -e ".[docs]"
fi

# --- 4. verification -------------------------------------------------------
log "verifying install"
$RUN python -c "import mir; from mir.embedding.tcremp import TCREmp; print('mir import OK')"

# --- 5. optional tests -----------------------------------------------------
if [[ "$RUN_TESTS" -eq 1 ]]; then
  log "installing test + bench tooling"
  $RUN python -m pip install -e ".[dev,bench]"
  log "running test suite"
  $RUN python -m pytest "$ROOT/tests" -q
fi

log "done."
if [[ "$USE_CONDA" -eq 1 ]]; then
  echo "  conda activate $ENV_NAME"
fi
