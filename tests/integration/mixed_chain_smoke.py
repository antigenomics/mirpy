#!/usr/bin/env python3
"""Colleague workflow check: a mixed-chain (multi-locus) AIRR file -> TCREmp.

Verifies the documented per-locus pattern on a single TSV containing several loci
(here TRA + TRB): ``AIRRParser(locus=...).parse(file)`` must (1) read standard
AIRR ``v_call``/``j_call`` columns and (2) filter to the requested locus, so each
chain embeds against its own TCREmp model in both feature modes.

Run in a clean install (e.g. ``pip install
"git+https://github.com/antigenomics/mirpy.git@dev"``)::

    python tests/integration/mixed_chain_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mir.basic.pgen import OlgaModel
from mir.common.parser import AIRRParser
from mir.embedding.tcremp import TCREmp

OUT = Path("/tmp/mixed.airr.tsv")
N_PER_LOCUS = 200
LOCI = ("TRA", "TRB")


def build_mixed_file() -> int:
    rows = ["locus\tv_call\tj_call\tjunction_aa"]
    for locus in LOCI:
        recs = OlgaModel(locus=locus, species="human", seed=1).generate_sequences_with_meta(
            n=N_PER_LOCUS, pgens=False, seed=1)
        for r in recs:
            rows.append(f"{locus}\t{r['v_call']}\t{r['j_call']}\t{r['junction_aa']}")
    OUT.write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def main() -> int:
    n = build_mixed_file()
    print(f"wrote {OUT} with {n} mixed-locus rows ({' + '.join(LOCI)})")
    ok = True
    for locus in LOCI:
        clonos = AIRRParser(locus=locus).parse(str(OUT))  # filters to this locus
        filtered_ok = len(clonos) == N_PER_LOCUS
        ok &= filtered_ok
        print(f"\n[{locus}] parsed {len(clonos)} clonotypes "
              f"({'filtered OK' if filtered_ok else 'FILTER FAIL'})")
        for mode in ("vjcdr3", "cdr123"):
            X = TCREmp.from_defaults("human", locus, n_prototypes=100, mode=mode).embed(
                clonos, n_jobs=1)
            good = X.shape == (len(clonos), 300) and not np.isnan(X).any()
            ok &= good
            print(f"  mode={mode:7s} -> {X.shape}  {'OK' if good else 'FAIL'}")
    print("\nMIXED-CHAIN COLLEAGUE WORKFLOW:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
