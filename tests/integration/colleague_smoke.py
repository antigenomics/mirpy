#!/usr/bin/env python3
"""Smoke test for a fresh `mir` install (the colleague experience).

Verifies that a clean install — e.g. ``pip install
"git+https://github.com/antigenomics/mirpy.git@dev"`` in an isolated
environment — can embed clonotypes with **both** TCREmp feature modes using only
the bundled resources (no arda, no mmseqs2, no network).

Run::

    python tests/integration/colleague_smoke.py

Exits non-zero on any failure so it can gate CI / a manual install check.
"""

from __future__ import annotations

import sys

import numpy as np

from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneLibrary
from mir.embedding.tcremp import TCREmp

# A handful of human TRB clonotypes (no data files needed).
_CLONOTYPES = [
    Clonotype(sequence_id="1", locus="TRB", v_call="TRBV9*01",
              j_call="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF", _validate=False),
    Clonotype(sequence_id="2", locus="TRB", v_call="TRBV19*01",
              j_call="TRBJ2-1*01", junction_aa="CASSIRSTDTQYF", _validate=False),
    Clonotype(sequence_id="3", locus="TRB", v_call="TRBV28*01",
              j_call="TRBJ1-1*01", junction_aa="CASSLAPGATNEKLFF", _validate=False),
]


def check_region_annotations() -> None:
    """The companion region_annotations.txt must ship and load."""
    lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
    cdr1 = lib.entries["TRBV9*01"].region_aa.get("cdr1")
    assert cdr1, "region_annotations.txt missing or not packaged (no CDR1 for TRBV9*01)"
    print(f"  region annotations OK (TRBV9*01 CDR1 = {cdr1})")


def check_embedding(mode: str) -> None:
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=50, mode=mode)
    X = model.embed(_CLONOTYPES, n_jobs=1)
    assert X.shape == (len(_CLONOTYPES), 3 * 50), f"{mode}: unexpected shape {X.shape}"
    assert X.dtype == np.float32, f"{mode}: dtype {X.dtype}"
    assert not np.isnan(X).any(), f"{mode}: NaN in embedding"
    print(f"  mode={mode}: embed OK -> {X.shape}")


def main() -> int:
    print(f"mir import OK (python {sys.version.split()[0]})")
    check_region_annotations()
    for mode in ("vjcdr3", "cdr123"):
        check_embedding(mode)
    print("colleague smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
