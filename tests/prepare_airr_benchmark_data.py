from __future__ import annotations

import csv
import gzip
import os
import re
import shutil
from pathlib import Path

csv.field_size_limit(10_000_000)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "isalgo/airr_benchmark"

DATASET_ROOT = REPO_ROOT / "airr_benchmark"
TESTS_DIR = REPO_ROOT / "tests"
ASSETS_DIR = TESTS_DIR / "assets"
REAL_REPS_DIR = ASSETS_DIR / "real_repertoires"
SRX_DIR = ASSETS_DIR / "srx_repertoires"

# Written after a successful bootstrap so we can skip on subsequent runs.
_SENTINEL = DATASET_ROOT / ".test_data_ready"

# Patterns always fetched — sufficient for unit tests.
_MINIMAL_PATTERNS = [
    "sra/meta.tsv",
    "sra/samples.tar.gz",
    "tcrnet/B35+.txt.gz",
    "alice/yf/Q1_d0.tsv.gz",
    "alice/yf/Q1_d15.tsv.gz",
    "gliph/gliph_trb.tsv.gz",
    "vdjdb/**",
    "vdjtools_lite/**",
]

# Extra patterns only fetched for integration / benchmark runs (large blobs).
_SC_PATTERNS = [
    "dcode/**",
]


def _has_10x_vdj_v1_assets() -> bool:
    dcode_root = DATASET_ROOT / "dcode"
    return (
        dcode_root.exists()
        and any(dcode_root.glob("*_all_contig_annotations.csv.gz"))
        and any(dcode_root.glob("*_consensus_annotations.csv.gz"))
    )

TOY_DATASET_METADATA_TSV = """sample_id\tfile_name\tbatch_id
s1\tvdjtools_trb_d_dot.tsv\tbatch_A
s2\tvdjtools_trb_d_dot_2.tsv\tbatch_B
"""

META_CSV = """,sample_id,file_name,status
0,id_1,repertoire_1.csv,healthy
0,id_2,repertoire_2.csv,healthy
0,id_3,repertoire_3.csv,ill
0,id_4,repertoire_4.csv,ill
"""

REPERTOIRE_FILES: dict[str, str] = {
    "repertoire_1.csv": """,count,freq,cdr3nt,cdr3aa,v,d,j,VEnd,DStart,DEnd,JStart
0,60,0.3,xx,CASTA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
0,40,0.2,xy,CASTA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
1,50,0.25,xx,CTALF,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
2,30,0.15,xx,CLAMF,TRBV28,.,TRBJ2-1,-1,-1,-1,-1
3,10,0.05,xx,CFRRA,TRBV11,.,TRBJ2-1,-1,-1,-1,-1
4,10,0.05,xx,CRFFA,TRBV11,.,TRBJ2-7,-1,-1,-1,-1
4,10,0.05,xx,CGGCF,TRBV11,.,TRBJ2-7,-1,-1,-1,-1
""",
    "repertoire_2.csv": """,count,freq,cdr3nt,cdr3aa,v,d,j,VEnd,DStart,DEnd,JStart
0,100,0.5,xx,CALTA,TRBV11,.,TRBJ1-1,-1,-1,-1,-1
1,50,0.25,xx,CRFRF,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
2,30,0.15,xx,CTAMF,TRBV28,.,TRBJ2-1,-1,-1,-1,-1
3,10,0.05,xx,CFRRA,TRBV2,.,TRBJ2-1,-1,-1,-1,-1
4,10,0.05,xx,CDDDA,TRBV11,.,TRBJ2-7,-1,-1,-1,-1
""",
    "repertoire_3.csv": """,count,freq,cdr3nt,cdr3aa,v,d,j,VEnd,DStart,DEnd,JStart
0,40,0.3,xx,CRFMA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
0,20,0.1,xy,CRFMA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
1,50,0.25,xx,CTFFA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
2,30,0.10,xx,CGGGF,TRBV11,.,TRBJ2-1,-1,-1,-1,-1
2,10,0.05,xy,CGGGF,TRBV11,.,TRBJ2-1,-1,-1,-1,-1
3,30,0.15,xx,CLKMA,TRBV2,.,TRBJ2-1,-1,-1,-1,-1
3,20,0.10,xy,CGGRF,TRBV2,.,TRBJ2-1,-1,-1,-1,-1
4,10,0.05,xx,CKMLA,TRBV28,.,TRBJ2-7,-1,-1,-1,-1
4,10,0.05,xx,CGGGG,TRBV28,.,TRBJ2-7,-1,-1,-1,-1
""",
    "repertoire_4.csv": """,count,freq,cdr3nt,cdr3aa,v,d,j,VEnd,DStart,DEnd,JStart
0,60,0.3,xx,CRKMA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
1,50,0.25,xx,CALMA,TRBV2,.,TRBJ1-1,-1,-1,-1,-1
2,10,0.05,xx,CGRGF,TRBV11,.,TRBJ2-1,-1,-1,-1,-1
2,20,0.10,xy,CGRGF,TRBV11,.,TRBJ2-1,-1,-1,-1,-1
2,20,0.10,xy,CGQGF,TRBV11,.,TRBJ2-1,-1,-1,-1,-1
3,50,0.25,xx,CQQQA,TRBV2,.,TRBJ2-1,-1,-1,-1,-1
4,10,0.05,xx,CPPPA,TRBV28,.,TRBJ2-7,-1,-1,-1,-1
""",
}


def _verbose(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg)


def _is_valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            fh.read(1024)
        return True
    except (OSError, EOFError):
        return False


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def _convert_yf_vdjtools_to_airr(src: Path, dst: Path) -> None:
    with gzip.open(src, "rt", encoding="utf-8") as in_fh:
        reader = csv.DictReader(in_fh, delimiter="\t")
        rows: list[dict[str, str]] = []
        for i, row in enumerate(reader):
            junction_aa = (row.get("CDR3.amino.acid.sequence") or "").strip()
            v_gene = (row.get("bestVGene") or "").strip()
            j_gene = (row.get("bestJGene") or "").strip()
            if not junction_aa or not v_gene or not j_gene:
                continue
            duplicate_count = str(row.get("Read.count") or "1")
            junction_nt = (row.get("CDR3.nucleotide.sequence") or "").strip()
            if not junction_nt:
                junction_nt = "N" * max(3, len(junction_aa) * 3)
            rows.append(
                {
                    "sequence_id": str(i),
                    "junction": junction_nt,
                    "junction_aa": junction_aa,
                    "v_gene": v_gene,
                    "j_gene": j_gene,
                    "duplicate_count": duplicate_count,
                    "locus": "TRB",
                }
            )

    with gzip.open(dst, "wt", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(
            out_fh,
            fieldnames=["sequence_id", "junction", "junction_aa", "v_gene", "j_gene", "duplicate_count", "locus"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_gilg_and_llw(vdjdb_slim: Path) -> tuple[int, int]:
    gilg: list[str] = []
    seen = set()
    llw_rows: list[dict[str, str]] = []

    with gzip.open(vdjdb_slim, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gene = (row.get("gene") or "").strip().upper()
            junction_aa = (row.get("cdr3") or "").strip()
            epitope = (row.get("antigen.epitope") or "").strip().upper()
            mhc_a = (row.get("mhc.a") or "").strip()
            v_gene = (row.get("v.segm") or "").strip()
            j_gene = (row.get("j.segm") or "").strip()

            if gene == "TRB" and epitope == "GILGFVFTL" and junction_aa and junction_aa not in seen:
                seen.add(junction_aa)
                gilg.append(junction_aa)

            if gene == "TRB" and epitope == "LLWNGPMAV" and "A*02" in mhc_a and junction_aa and v_gene and j_gene:
                llw_rows.append(
                    {
                        "junction_aa": junction_aa,
                        "v_gene": v_gene,
                        "j_gene": j_gene,
                        "duplicate_count": "1",
                        "locus": "TRB",
                    }
                )

    gilg_path = ASSETS_DIR / "gilgfvftl_trb_junctions.txt.gz"
    with gzip.open(gilg_path, "wt", encoding="utf-8") as out:
        for seq in gilg:
            out.write(seq + "\n")

    llw_path = ASSETS_DIR / "llwngpmav_trb_a02.tsv.gz"
    with gzip.open(llw_path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=["junction_aa", "v_gene", "j_gene", "duplicate_count", "locus"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(llw_rows)

    return len(gilg), len(llw_rows)


def _write_olga_1000(vdjdb_slim: Path) -> int:
    rows: list[tuple[str, str, str, str]] = []
    with gzip.open(vdjdb_slim, "rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if (row.get("gene") or "").strip().upper() != "TRB":
                continue
            junction_aa = (row.get("cdr3") or "").strip()
            v_gene = (row.get("v.segm") or "TRBV7-9*01").strip() or "TRBV7-9*01"
            j_gene = (row.get("j.segm") or "TRBJ2-1*01").strip() or "TRBJ2-1*01"
            if not junction_aa:
                continue
            nt = "NNN" * len(junction_aa)
            rows.append((nt, junction_aa, v_gene, j_gene))

    if not rows:
        raise RuntimeError("No TRB rows available in VDJdb slim to build OLGA test asset")

    out_rows = [rows[i % len(rows)] for i in range(1000)]
    out_path = ASSETS_DIR / "olga_humanTRB_1000.txt.gz"
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerows(out_rows)
    return len(out_rows)


def _derive_assets(*, verbose: bool = False) -> None:
    """Build derived test assets from downloaded raw files."""
    # Remove legacy pre-reorganization directories if they still exist.
    for legacy in (TESTS_DIR / "real_repertoires", TESTS_DIR / "srx_repertoires"):
        if legacy.exists():
            shutil.rmtree(legacy)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REAL_REPS_DIR.mkdir(parents=True, exist_ok=True)
    SRX_DIR.mkdir(parents=True, exist_ok=True)

    vdjtools_dir = DATASET_ROOT / "vdjtools_lite"
    for src in vdjtools_dir.glob("*.txt.gz"):
        _copy(src, REAL_REPS_DIR / src.name)
    for name in ("metadata_aging.txt", "metadata_hsct.txt"):
        src = vdjtools_dir / name
        if src.exists():
            _copy(src, REAL_REPS_DIR / name)

    _copy(vdjtools_dir / "old_mixcr.gz", ASSETS_DIR / "old_mixcr.gz")
    _copy(vdjtools_dir / "vdjtools_trb_d_dot.tsv", ASSETS_DIR / "vdjtools_trb_d_dot.tsv")
    _copy(vdjtools_dir / "vdjtools_trb_d_dot_2.tsv", ASSETS_DIR / "vdjtools_trb_d_dot_2.tsv")

    vdjdb_candidates = sorted(DATASET_ROOT.glob("vdjdb/vdjdb-*/vdjdb.slim.txt.gz"))
    vdjdb_full_candidates = sorted(DATASET_ROOT.glob("vdjdb/vdjdb-*/vdjdb_full.txt.gz"))
    if not vdjdb_candidates:
        raise RuntimeError("Could not locate vdjdb.slim.txt.gz in downloaded dataset")
    if not vdjdb_full_candidates:
        raise RuntimeError("Could not locate vdjdb_full.txt.gz in downloaded dataset")
    vdjdb_local = vdjdb_candidates[-1]
    vdjdb_full_local = vdjdb_full_candidates[-1]

    if not _is_valid_gzip(vdjdb_local):
        raise RuntimeError(f"Downloaded VDJdb slim is corrupt: {vdjdb_local}")
    if not _is_valid_gzip(vdjdb_full_local):
        raise RuntimeError(f"Downloaded VDJdb full is corrupt: {vdjdb_full_local}")

    _copy(vdjdb_local, ASSETS_DIR / "vdjdb.slim.txt.gz")
    _copy(vdjdb_full_local, ASSETS_DIR / "vdjdb_full.txt.gz")
    _convert_yf_vdjtools_to_airr(
        DATASET_ROOT / "alice/yf/Q1_d0.tsv.gz",
        REAL_REPS_DIR / "Q1_0_F1.airr.tsv.gz",
    )
    _convert_yf_vdjtools_to_airr(
        DATASET_ROOT / "alice/yf/Q1_d15.tsv.gz",
        REAL_REPS_DIR / "Q1_15_F1.airr.tsv.gz",
    )
    _copy(DATASET_ROOT / "tcrnet/B35+.txt.gz", REAL_REPS_DIR / "B35+.txt.gz")
    _copy(DATASET_ROOT / "sra/meta.tsv", SRX_DIR / "meta.tsv")
    _copy(DATASET_ROOT / "sra/samples.tar.gz", SRX_DIR / "samples.tar.gz")

    _write_text(ASSETS_DIR / "toy_dataset_metadata.tsv", TOY_DATASET_METADATA_TSV)
    _write_text(ASSETS_DIR / "meta.csv", META_CSV)
    for name, content in REPERTOIRE_FILES.items():
        _write_text(ASSETS_DIR / name, content)

    gilg_n, llw_n = _write_gilg_and_llw(ASSETS_DIR / "vdjdb.slim.txt.gz")
    olga_n = _write_olga_1000(ASSETS_DIR / "vdjdb.slim.txt.gz")

    if verbose:
        print(f"prepared test data under {TESTS_DIR}")
        print(f"vdjdb source: {vdjdb_local.relative_to(DATASET_ROOT)}")
        print(f"vdjdb full source: {vdjdb_full_local.relative_to(DATASET_ROOT)}")
        print(f"derived GILG junction count: {gilg_n}")
        print(f"derived LLW rows: {llw_n}")
        print(f"derived OLGA rows: {olga_n}")


def ensure_test_data(
    *, force: bool = False, verbose: bool = False, include_sc_assets: bool = False
) -> None:
    """Download and derive test assets from the HuggingFace benchmark dataset.

    Args:
        force: Re-download even if the sentinel already exists.
        verbose: Print a progress summary.
        include_sc_assets: Also fetch large single-cell (dcode/**) blobs.
            Enable only for integration / benchmark runs to avoid long downloads
            during ordinary unit-test collection.
    """
    sc_ready = not include_sc_assets or _has_10x_vdj_v1_assets()
    if (
        not force
        and _SENTINEL.exists()
        and (ASSETS_DIR / "vdjdb.slim.txt.gz").exists()
        and (ASSETS_DIR / "vdjdb_full.txt.gz").exists()
        and sc_ready
    ):
        return

    if force:
        for d in (ASSETS_DIR,):
            if d.exists():
                shutil.rmtree(d)

    from huggingface_hub import snapshot_download

    patterns = list(_MINIMAL_PATTERNS)
    if include_sc_assets:
        patterns.extend(_SC_PATTERNS)

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(DATASET_ROOT),
        allow_patterns=patterns,
    )

    _derive_assets(verbose=verbose)
    _SENTINEL.touch()


def main() -> None:
    force = os.getenv("MIRPY_TEST_DATA_FORCE", "0") in {"1", "true", "TRUE", "yes", "YES"}
    include_sc = os.getenv("MIRPY_TEST_DATA_SC", "0") in {"1", "true", "TRUE", "yes", "YES"}
    ensure_test_data(force=force, verbose=True, include_sc_assets=include_sc)


if __name__ == "__main__":
    main()
