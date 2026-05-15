"""Benchmarks for PairedRepertoire <-> SampleRepertoire conversion."""

from __future__ import annotations

import os
import time

import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell import PairedRepertoire

RUN_BENCHMARK = os.getenv("RUN_BENCHMARK") == "1"
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN_BENCHMARK, reason="set RUN_BENCHMARK=1 to run benchmark tests"),
]


def _build_sample(scale: int = 3000) -> SampleRepertoire:
    tra = [Clonotype(sequence_id=f"tra{i}", locus="TRA", junction_aa="CAAAAF") for i in range(scale)]
    trb = [Clonotype(sequence_id=f"trb{i}", locus="TRB", junction_aa="CASSFF") for i in range(scale)]
    return SampleRepertoire(
        loci={
            "TRA": LocusRepertoire(tra, locus="TRA"),
            "TRB": LocusRepertoire(trb, locus="TRB"),
        },
        sample_id="bench_sample",
    )


def test_sample_to_paired_conversion_speed() -> None:
    sample = _build_sample(scale=2000)
    pairing_rows = [(f"pair{i}", "TRA", "TRB", f"tra{i}", f"trb{i}") for i in range(2000)]

    t0 = time.perf_counter()
    paired = PairedRepertoire.from_sample_repertoire(sample, pairing_rows=pairing_rows)
    dt = time.perf_counter() - t0

    assert paired.paired_locus_repertoires["TRA_TRB"].clonotype_count == 2000
    assert dt < 5.0


def test_paired_to_sample_conversion_speed() -> None:
    sample = _build_sample(scale=2000)
    pairing_rows = [(f"pair{i}", "TRA", "TRB", f"tra{i}", f"trb{i}") for i in range(2000)]
    paired = PairedRepertoire.from_sample_repertoire(sample, pairing_rows=pairing_rows)

    t0 = time.perf_counter()
    collapsed = paired.to_sample_repertoire()
    dt = time.perf_counter() - t0

    assert collapsed.loci["TRA"].clonotype_count == 2000
    assert collapsed.loci["TRB"].clonotype_count == 2000
    assert dt < 5.0
