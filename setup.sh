#!/usr/bin/env bash
# mirpy v3 bootstrap — reproducible install into a repo-local .venv with uv.
#
# Portable: runs under bash OR zsh (bash setup.sh / zsh setup.sh / ./setup.sh). Not fish.
#
# uv is required. mirpy is developed exclusively against uv-managed environments; there is
# deliberately no pip/virtualenv fallback, so every checkout resolves dependencies the same
# way. Install it once with:  curl -LsSf https://astral.sh/uv/install.sh | sh
#
# mirpy itself is a pure-Python `py3-none-any` package (no C build). The heavy machinery
# (alignment, Pgen, sampling) is reused from the compiled `seqtree` and `vdjtools` wheels,
# which uv resolves from PyPI — unless you pass --dev-parents to editable-install the
# co-developed sibling checkouts from ../ instead.
#
# Steps:
#   1. Create/activate a repo-local .venv (uv venv, Python 3.12).
#   2. (optional) editable-install co-developed sibling parents from ../ if present.
#   3. uv pip install -e ".[dev,bench]".
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
PYTHON_VERSION=3.12
DEV_PARENTS=0
INSTALL_DOCS=0
DO_TESTS=0

for arg in "$@"; do
  case "$arg" in
    --dev-parents) DEV_PARENTS=1 ;;
    --docs)        INSTALL_DOCS=1 ;;
    --tests)       DO_TESTS=1 ;;
    --help|-h)     sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[mirpy]\033[0m %s\n' "$*"; }

# --- 0. uv is a hard requirement ------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[mirpy] uv is required but was not found on PATH.

  Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh
               (or: brew install uv)

  Then re-run: bash setup.sh
MSG
  exit 1
fi
log "using $(uv --version)"

# --- 1. repo-local .venv ---------------------------------------------------
VENV="$ROOT/.venv"
[ -d "$VENV" ] || { log "creating .venv (Python $PYTHON_VERSION)"; uv venv --python "$PYTHON_VERSION" "$VENV"; }
# shellcheck disable=SC1091
. "$VENV/bin/activate"   # activate script is bash/zsh compatible

# A pre-existing .venv is reused as-is, so it can silently be the wrong interpreter
# (e.g. a stray `uv run` creates one on the default Python). Say so rather than build on it.
HAVE=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$HAVE" != "$PYTHON_VERSION" ]; then
  log "WARNING: .venv is Python $HAVE, expected $PYTHON_VERSION"
  log "         recreate it with: uv venv --clear --python $PYTHON_VERSION .venv"
fi

# --- 2. co-developed sibling parents (optional) ----------------------------
if [ "$DEV_PARENTS" -eq 1 ]; then
  for parent in seqtree vdjtools vdjmatch; do
    if [ -f "$ROOT/../$parent/pyproject.toml" ]; then
      log "editable-install ../$parent (compiles its _core extension)"
      uv pip install -e "$ROOT/../$parent"
    fi
  done
fi

# --- 3. editable install (pure Python — no C build for mir itself) ---------
EXTRAS="dev,bench"
[ "$INSTALL_DOCS" -eq 1 ] && EXTRAS="$EXTRAS,docs"
log "uv pip install -e .[$EXTRAS]"
uv pip install -e "$ROOT[$EXTRAS]"

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
