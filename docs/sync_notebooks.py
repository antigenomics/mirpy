#!/usr/bin/env python3
"""
Synchronise notebook symlinks in docs/notebooks/ with the canonical
notebooks/ directory at the repo root.

Run from the repo root:
    python3 docs/sync_notebooks.py

Or from the docs/ directory:
    python3 sync_notebooks.py

Safe to run repeatedly: skips symlinks that are already correct.
"""
import os
import pathlib
import sys

# Resolve paths relative to this script's location (docs/).
DOCS_DIR = pathlib.Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent
NB_SRC = REPO_ROOT / "notebooks"
NB_DST = DOCS_DIR / "notebooks"

# The symlink target is always written as a relative path so the repo
# remains portable (no hardcoded absolute paths).
SYMLINK_PREFIX = pathlib.Path("../../notebooks")


def sync():
    created = 0
    skipped = 0
    for nb in sorted(NB_SRC.glob("*.ipynb")):
        link = NB_DST / nb.name
        target = SYMLINK_PREFIX / nb.name
        if link.is_symlink() and os.readlink(link) == str(target):
            skipped += 1
        else:
            link.unlink(missing_ok=True)
            link.symlink_to(target)
            created += 1
            print(f"  linked: {nb.name}")
    print(f"sync-notebooks: {created} created/updated, {skipped} already up-to-date")


if __name__ == "__main__":
    sync()
    sys.exit(0)
