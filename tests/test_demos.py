"""Run every module's ``_demo()`` self-check.

Several modules ship an assert-based ``_demo()`` under ``if __name__ == "__main__"`` — the
smallest runnable check that fails if that module's logic breaks. They were reachable only by
running the file by hand, so CI never executed them and they showed up as the largest uncovered
blocks in the library. Running them here makes them real tests: each is a few hundred
milliseconds and self-contained on bundled resources.

``mir.aliases`` is absent from the list on purpose: its self-check is inlined in the ``__main__``
block rather than a ``_demo`` function, and the module is already fully covered.
"""

import importlib

import pytest

_WITH_DEMO = [
    "mir.bench.eval",
    "mir.cohort",
    "mir.explain",
    "mir.generate",
    "mir.repertoire",
    "mir.track",
    "mir.twin",
]


@pytest.mark.parametrize("module", _WITH_DEMO)
def test_module_self_check_passes(module):
    importlib.import_module(module)._demo()
