"""Bake germline FR/CDR region annotations from **arda** (build-time only).

Reads arda's per-scaffold AA region markup (``markup.aa.tsv``) and writes one row per V/J allele
with its germline-encoded FR/CDR amino-acid subsequences, keyed by arda's IMGT allele name. This
is the arda-native reference that :mod:`build_germline_dist` bakes into the germline distance
matrices, so mirpy's germline geometry uses exactly the germline arda assigns to real data.

Output schema (tab-separated, with header)::

    species  locus  gene  allele  fwr1_aa  cdr1_aa  fwr2_aa  cdr2_aa  fwr3_aa  jcdr3_aa  fwr4_aa

Per allele:

* **V** (``gene="V"``): ``fwr1..fwr3``, ``cdr1``, ``cdr2`` — V-determined, so identical across every
  scaffold carrying that V; take the first.
* **J** (``gene="J"``): ``fwr4`` — J-determined; and ``jcdr3`` (the J germline contribution to CDR3)
  = the longest common suffix of the scaffold CDR3s for that J (arda's own convention), taken after
  the last ambiguous ``X`` position.

Only alleles arda actually builds scaffolds for are emitted — exactly the alleles arda annotation
assigns to real data, so a prototype/query gene name always resolves (no silent max-distance fallback).

Run (needs the ``[build]`` extra: ``arda-mapper`` + ``ARDA_HOME`` pointing at an arda checkout/cache)::

    ARDA_HOME=/path/to/arda python mir/resources/gene_library/build_region_annotations.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "region_annotations.txt"
_ORGANISMS = ("human", "mouse")
_LOCI = ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
_COLS = ["species", "locus", "gene", "allele",
         "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "jcdr3_aa", "fwr4_aa"]


def _lcsuffix(strs: list[str]) -> str:
    """Longest common suffix of a list of strings."""
    strs = [s for s in strs if s]
    if not strs:
        return ""
    shortest = min(strs, key=len)
    best = ""
    for i in range(1, len(shortest) + 1):
        suf = shortest[-i:]
        if all(s.endswith(suf) for s in strs):
            best = suf
        else:
            break
    return best


def _jcdr3(cdr3s: list[str]) -> str:
    """J germline CDR3 contribution = common suffix of scaffold CDR3s, after the last ``X``."""
    return _lcsuffix(cdr3s).rsplit("X", 1)[-1]


def _markup(organism: str) -> pl.DataFrame:
    from arda.paths import vdj_dir  # arda ([build] extra); ARDA_HOME locates the database

    return pl.read_csv(vdj_dir(organism) / "markup.aa.tsv", separator="\t", infer_schema_length=50000)


def build() -> None:
    rows: list[dict] = []
    for organism in _ORGANISMS:
        mk = _markup(organism)
        for locus in _LOCI:
            sub = mk.filter(pl.col("locus") == locus)
            if sub.height == 0:
                continue
            for r in sub.group_by("v_call").first().iter_rows(named=True):
                rows.append({"species": organism, "locus": locus, "gene": "V", "allele": r["v_call"],
                             "fwr1_aa": r["fwr1"] or "", "cdr1_aa": r["cdr1"] or "",
                             "fwr2_aa": r["fwr2"] or "", "cdr2_aa": r["cdr2"] or "",
                             "fwr3_aa": r["fwr3"] or "", "jcdr3_aa": "", "fwr4_aa": ""})
            for j_call in sub["j_call"].unique().to_list():
                g = sub.filter(pl.col("j_call") == j_call)
                rows.append({"species": organism, "locus": locus, "gene": "J", "allele": j_call,
                             "fwr1_aa": "", "cdr1_aa": "", "fwr2_aa": "", "cdr2_aa": "", "fwr3_aa": "",
                             "jcdr3_aa": _jcdr3(g["cdr3"].to_list()), "fwr4_aa": g["fwr4"].to_list()[0] or ""})
    df = pl.DataFrame(rows).select(_COLS).sort(["species", "locus", "gene", "allele"])
    df.write_csv(_OUT, separator="\t")
    print(f"wrote {df.height} rows -> {_OUT}")
    print(df.group_by(["species", "locus", "gene"]).len().sort(["species", "locus", "gene"]))


if __name__ == "__main__":
    build()
