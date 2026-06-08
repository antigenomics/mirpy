"""Shared helpers for single-cell parity/integration tests."""

from __future__ import annotations

from pathlib import Path


def discover_dcode_donor_pair() -> tuple[Path, Path] | None:
    """Return a matching (all_contig, consensus) 10x donor file pair, or None.

    Looks under ``airr_benchmark/dcode`` for a donor whose
    ``*_all_contig_annotations.csv.gz`` has a matching
    ``*_consensus_annotations.csv.gz``. Returns ``None`` when the benchmark
    assets are absent so callers can skip.
    """
    dcode_root = Path(__file__).resolve().parents[1] / "airr_benchmark" / "dcode"
    if not dcode_root.exists():
        return None

    all_contig_files = sorted(dcode_root.glob("*_all_contig_annotations.csv.gz"))
    consensus_files = sorted(dcode_root.glob("*_consensus_annotations.csv.gz"))
    if not all_contig_files or not consensus_files:
        return None

    all_contig = all_contig_files[0]
    consensus = next(
        (
            p
            for p in consensus_files
            if p.name.replace("_consensus_annotations", "")
            == all_contig.name.replace("_all_contig_annotations", "")
        ),
        None,
    )
    if consensus is None:
        return None
    return all_contig, consensus
