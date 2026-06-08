#!/usr/bin/env python3
"""Build the companion region-annotation library from arda.

Annotates every germline V and J allele in the bundled gene libraries (OLGA +
IMGT, human + mouse, all seven loci) with arda and writes their germline-encoded
FR/CDR amino-acid subsequences to ``region_annotations.txt`` next to this script.

Output schema (tab-separated, with header)::

    species  locus  gene  allele  fwr1_aa  cdr1_aa  fwr2_aa  cdr2_aa  fwr3_aa  jcdr3_aa  fwr4_aa

V rows fill the V-side columns (``fwr1_aa``..``fwr3_aa``); J rows fill
``jcdr3_aa`` (the J contribution to CDR3) and ``fwr4_aa``.  Missing regions are
left blank.  This file is checked into the repository and loaded at runtime by
:meth:`mir.common.gene_library.GeneLibrary.load_default`; plain embedding and
similarity therefore never need arda.

Run once, from the repository root, in an environment with the ``arda`` extra
and the ``mmseqs2`` binary on ``PATH``::

    pip install -e ".[arda]"
    conda install -c bioconda mmseqs2
    python mir/resources/gene_library/build_region_annotations.py

Options::

    --organism {human,mouse}   restrict to one organism (default: both)
    --sensitivity FLOAT        mmseqs2 sensitivity (default: 7.0)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Editable install puts the repo root on sys.path; fallback for direct runs.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mir.common.gene_library import GeneLibrary, _REGION_COLUMNS
from mir.common.region_annotation import annotate_gene_library

RESOURCES_DIR = _HERE
OUTPUT_PATH = RESOURCES_DIR / "region_annotations.txt"
LOG_PATH = RESOURCES_DIR / "build_region_annotations.log"

ORGANISMS = ("human", "mouse")
ALL_LOCI = {"TRB", "TRA", "TRG", "TRD", "IGH", "IGK", "IGL"}
SOURCES = ("olga", "imgt")

_HEADER = ["species", "locus", "gene", "allele"] + _REGION_COLUMNS


def _merged_library(organism: str) -> GeneLibrary:
    """Union of OLGA and IMGT V/J alleles for *organism* (region cols disabled)."""
    entries: dict[str, object] = {}
    for source in SOURCES:
        try:
            lib = GeneLibrary.load_default(
                loci=ALL_LOCI, species={organism}, source=source, with_regions=False
            )
        except Exception as exc:  # noqa: BLE001 - a source may be missing
            print(f"  [skip] {organism} {source}: {exc}", file=sys.stderr)
            continue
        for allele, entry in lib.entries.items():
            entries.setdefault(allele, entry)
    return GeneLibrary(entries, complete=True)


def build(organisms: tuple[str, ...], sensitivity: float) -> list[list[str]]:
    """Annotate the requested organisms and return sorted output rows."""
    rows: list[list[str]] = []
    for organism in organisms:
        lib = _merged_library(organism)
        n_vj = len([e for e in lib.get_entries() if e.gene in ("V", "J")])
        print(f"=== {organism}: annotating {n_vj} V/J alleles via arda ===")
        annotations = annotate_gene_library(lib, organism, sensitivity=sensitivity)
        annotated = 0
        for allele, regions in annotations.items():
            entry = lib.entries[allele]
            row = [organism, entry.locus, entry.gene, allele]
            row += [regions.get(col[:-3], "") for col in _REGION_COLUMNS]
            rows.append(row)
            annotated += 1
        print(f"  {organism}: {annotated} alleles annotated")
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    return rows


def write_rows(rows: list[list[str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("\t".join(_HEADER) + "\n")
        for row in rows:
            out.write("\t".join(row) + "\n")
    print(f"Wrote {len(rows)} region rows -> {path}")


def _append_log(rows: list[list[str]]) -> None:
    from collections import Counter
    counts = Counter((r[0], r[1], r[2]) for r in rows)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n{'=' * 70}", f"region annotations  |  {ts}  |  {len(rows)} rows", "=" * 70]
    for key in sorted(counts):
        lines.append(f"  {key[0]:<7}{key[1]:<6}{key[2]:<3}{counts[key]:>6}")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", choices=ORGANISMS, default=None)
    ap.add_argument("--sensitivity", type=float, default=7.0)
    args = ap.parse_args()

    organisms = (args.organism,) if args.organism else ORGANISMS
    rows = build(organisms, args.sensitivity)
    write_rows(rows, OUTPUT_PATH)
    _append_log(rows)
