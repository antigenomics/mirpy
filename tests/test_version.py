"""The version literal and the changelog's top entry must name the same release.

`pyproject.toml` takes the version from ``src/mir/__init__.py::__version__``, so a release commit
that writes the changelog and forgets that literal publishes under the *previous* version and the
upload cannot be undone. Measured 2026-09-27: the 4.0.0 commit left ``__version__`` at ``3.20.2``
with a ``## 4.0.0`` changelog entry above it, and arda shipped the same slip one repo over, which is
why 2.30.0 never reached PyPI. Nothing else in either suite looks at the literal.
"""

import re
import tomllib
from pathlib import Path

import mir

ROOT = Path(__file__).resolve().parents[1]


def test_the_version_literal_matches_the_changelog_top_entry():
    head = re.search(r"^## (\d+\.\d+\.\d+)", Path(ROOT, "CHANGELOG.md").read_text(), re.M)
    assert head, "CHANGELOG.md has no `## <version>` entry"
    assert mir.__version__ == head.group(1)


def test_pyproject_reads_the_version_from_that_literal():
    """If this ever stops being true, the test above stops guarding the published version."""
    cfg = tomllib.loads(Path(ROOT, "pyproject.toml").read_text())
    assert "version" in cfg["project"].get("dynamic", [])
    assert cfg["tool"]["hatch"]["version"]["path"] == "src/mir/__init__.py"
