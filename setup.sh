#!/usr/bin/env bash
# mirpy v3 bootstrap — reproducible install into a repo-local .venv with uv.
#
# Portable: runs under bash OR zsh (bash setup.sh / zsh setup.sh / ./setup.sh). Not fish.
#
# mirpy itself is a pure-Python `py3-none-any` package (no C build). The heavy machinery
# (alignment, Pgen, sampling) is reused from the compiled `seqtree` and `vdjtools` wheels,
# which pip resolves from PyPI — unless you pass --dev-parents to editable-install the
# co-developed sibling checkouts from ../ instead.
#
# Steps:
#   1. Create/activate a repo-local .venv (uv if present, else python -m venv).
#   2. (optional) editable-install co-developed sibling parents from ../ if present.
#   3. pip install -e ".[dev,bench]".
#
# Flags:
#   --dev-parents  Editable-install ../seqtree ../vdjtools ../vdjmatch if they exist locally
#                  (they are co-developed; otherwise the PyPI releases are used). Building the
#                  siblings compiles their C++ _core extensions (needs a C++ toolchain).
#   --docs         Also install the [docs] extra.
#   --tests        Run the fast test suite after install.
#
# Requirements: a C++ toolchain (Xcode Command Line Tools on macOS, build-essential on Linux)
# is needed ONLY when --dev-parents rebuilds seqtree/vdjtools from source. The `[build]` extra
# (arda, BioPython) is for regenerating bundled resources and is not installed here.
#
# Usage: bash setup.sh [--dev-parents] [--docs] [--tests]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"   # $0, not ${BASH_SOURCE}: works in bash AND zsh
DEV_PARENTS=0
INSTALL_DOCS=0
DO_TESTS=0

for arg in "$@"; do
  case "$arg" in
    --dev-parents) DEV_PARENTS=1 ;;
    --docs)        INSTALL_DOCS=1 ;;
    --tests)       DO_TESTS=1 ;;
    --no-conda)    ;;  # accepted for backward-compat; conda is no longer used (no-op)
    --help|-h)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[mirpy]\033[0m %s\n' "$*"; }

# --- 1. repo-local .venv (uv preferred) ------------------------------------
VENV="$ROOT/.venv"
if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
  [ -d "$VENV" ] || { log "creating .venv with uv (Python 3.12)"; uv venv --python 3.12 "$VENV"; }
else
  log "uv not found — using python -m venv + pip (install uv for faster installs: https://docs.astral.sh/uv/)"
  PIP="python -m pip"
  [ -d "$VENV" ] || { log "creating .venv"; python -m venv "$VENV"; }
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"   # activate script is bash/zsh compatible

# --- 2. co-developed sibling parents (optional) ----------------------------
if [ "$DEV_PARENTS" -eq 1 ]; then
  for parent in seqtree vdjtools vdjmatch; do
    if [ -f "$ROOT/../$parent/pyproject.toml" ]; then
      log "editable-install ../$parent (compiles its _core extension)"
      $PIP install -e "$ROOT/../$parent"
    fi
  done
fi

# --- 3. editable install (pure Python — no C build for mir itself) ---------
EXTRAS="dev,bench"
[ "$INSTALL_DOCS" -eq 1 ] && EXTRAS="$EXTRAS,docs"
log "$PIP install -e .[$EXTRAS]"
$PIP install -e "$ROOT[$EXTRAS]"

# --- 4. verification -------------------------------------------------------
log "verifying install"
python -c "import mir; from mir.embedding.tcremp import TCREmp; print('mir', mir.__version__, 'import OK')"

# --- 5. optional tests -----------------------------------------------------
if [ "$DO_TESTS" -eq 1 ]; then
  log "running fast tests"
  python -m pytest "$ROOT/tests" -q -m "not integration and not benchmark"
fi

log "done."
echo "  source $VENV/bin/activate"
