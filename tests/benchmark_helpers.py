"""Shared helpers for benchmark tests."""

from __future__ import annotations

import gzip
import os
import time
from pathlib import Path

import pytest

from mir.common.alleles import allele_to_major
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire

ASSETS = Path(__file__).parent / "assets"
GILG_FILE = ASSETS / "gilgfvftl_trb_cdr3.txt.gz"
BENCH_LOG = Path(__file__).parent / "benchmarks.log"


def benchmark_log_line(message: str) -> None:
    BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with BENCH_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")


def synthetic_control_size(default: int = 1_000_000) -> int:
    raw = os.getenv("MIRPY_BENCH_SYNTHETIC_N")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1_000, value)


def _mk_clonotype(sid: str, aa: str, *, v_gene: str = "TRBV7-9*01", j_gene: str = "TRBJ2-1*01") -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=aa,
        v_gene=allele_to_major(v_gene),
        j_gene=allele_to_major(j_gene),
        duplicate_count=1,
        _validate=False,
    )


def load_gilg_target_repertoire(*, max_sequences: int | None = None) -> LocusRepertoire:
    if not GILG_FILE.exists():
        pytest.skip("GIL target asset missing: tests/assets/gilgfvftl_trb_cdr3.txt.gz")

    with gzip.open(GILG_FILE, "rt", encoding="utf-8") as fh:
        seqs = [line.strip() for line in fh if line.strip()]
    if max_sequences is not None:
        seqs = seqs[:max_sequences]
    clonotypes = [_mk_clonotype(f"g{i}", seq) for i, seq in enumerate(seqs)]
    return LocusRepertoire(clonotypes=clonotypes, locus="TRB")


def synthetic_control_repertoire(
    *,
    manager: ControlManager,
    species: str = "human",
    locus: str = "TRB",
    n: int = 1_000_000,
    require_cached: bool = True,
) -> LocusRepertoire:
    if require_cached:
        control_path = manager.synthetic_control_path(species, locus, n)
        if not control_path.exists():
            pytest.skip(
                f"Synthetic control cache not found at {control_path}. "
                "Build it first with mirpy-control-setup or set MIRPY_BENCH_REQUIRE_CACHED_CONTROL=0."
            )

    df = manager.ensure_and_load_control_df(
        "synthetic",
        species,
        locus,
        n=n,
        seed=42,
        chunk_size=100_000,
        progress=True,
    )

    clonotypes = [
        _mk_clonotype(
            f"c{i}",
            str(rec.get("junction_aa", "")),
            v_gene=str(rec.get("v_gene", "TRBV7-9*01")),
            j_gene=str(rec.get("j_gene", "TRBJ2-1*01")),
        )
        for i, rec in enumerate(df.to_dict(orient="records"))
    ]
    return LocusRepertoire(clonotypes=clonotypes, locus=locus)
