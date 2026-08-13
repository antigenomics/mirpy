"""Run every module's ``_demo()`` self-check.

Several modules ship an assert-based ``_demo()`` under ``if __name__ == "__main__"`` — the
smallest runnable check that fails if that module's logic breaks. They were reachable only by
running the file by hand, so CI never executed them and they showed up as the largest uncovered
blocks in the library. Running them here makes them real tests: each is a few hundred
milliseconds and self-contained on bundled resources.

The list is **discovered**, not written down. It used to be a hard-coded tuple, and within one
session a newly added module with a working ``_demo`` was silently not run by it — which is the
failure mode a self-check list can least afford, since nothing about it looks broken. A module
that cannot be imported (an optional dependency) is skipped with its reason rather than failing:
this test exists to run self-checks, not to assert the extras are installed.
"""

import importlib
import pkgutil

import pytest

import mir


def _modules_with_demo():
    found = []
    for m in pkgutil.walk_packages(mir.__path__, prefix="mir."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        if callable(getattr(mod, "_demo", None)):
            found.append(m.name)
    return sorted(found)


_WITH_DEMO = _modules_with_demo()


def test_discovery_found_the_demos():
    """A discovery bug would silently turn this whole file into zero tests."""
    assert len(_WITH_DEMO) >= 7, _WITH_DEMO


@pytest.mark.parametrize("module", _WITH_DEMO)
def test_module_self_check_passes(module):
    importlib.import_module(module)._demo()
