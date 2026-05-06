from __future__ import annotations

import csv
import gzip
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path

csv.field_size_limit(10_000_000)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "isalgo/airr_benchmark"
HF_BASE = "https://huggingface.co/datasets/isalgo/airr_benchmark/resolve/main"
HF_TREE_API = "https://huggingface.co/api/datasets/isalgo/airr_benchmark/tree/main?recursive=true"

DATASET_ROOT = REPO_ROOT / "airr_benchmark"
TESTS_DIR = REPO_ROOT / "tests"
ASSETS_DIR = TESTS_DIR / "assets"
REAL_REPS_DIR = TESTS_DIR / "real_repertoires"
SRX_DIR = TESTS_DIR / "srx_repertoires"

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


def _load_tree_paths() -> list[str]:
    with urllib.request.urlopen(HF_TREE_API, timeout=60) as response:
        payload = json.load(response)
    return [item.get("path", "") for item in payload if item.get("path")]


def _download(path: str, dst: Path, *, verbose: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{path}"
    _verbose(f"download {path}", verbose)
    part = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as src, open(part, "wb") as out:
        shutil.copyfileobj(src, out)
    part.replace(dst)


def _is_valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            fh.read(1024)
        return True
    except OSError:
        return False
    except EOFError:
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
            cdr3 = (row.get("cdr3") or "").strip()
            epitope = (row.get("antigen.epitope") or "").strip().upper()
            mhc_a = (row.get("mhc.a") or "").strip()
            v_gene = (row.get("v.segm") or "").strip()
            j_gene = (row.get("j.segm") or "").strip()

            if gene == "TRB" and epitope == "GILGFVFTL" and cdr3 and cdr3 not in seen:
                seen.add(cdr3)
                gilg.append(cdr3)

            if gene == "TRB" and epitope == "LLWNGPMAV" and "A*02" in mhc_a and cdr3 and v_gene and j_gene:
                llw_rows.append(
                    {
                        "junction_aa": cdr3,
                        "v_gene": v_gene,
                        "j_gene": j_gene,
                        "duplicate_count": "1",
                        "locus": "TRB",
                    }
                )

    gilg_path = ASSETS_DIR / "gilgfvftl_trb_cdr3.txt.gz"
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
            cdr3 = (row.get("cdr3") or "").strip()
            v_gene = (row.get("v.segm") or "TRBV7-9*01").strip() or "TRBV7-9*01"
            j_gene = (row.get("j.segm") or "TRBJ2-1*01").strip() or "TRBJ2-1*01"
            if not cdr3:
                continue
            # OlgaParser only requires non-empty junction strings here.
            nt = "NNN" * len(cdr3)
            rows.append((nt, cdr3, v_gene, j_gene))

    if not rows:
        raise RuntimeError("No TRB rows available in VDJdb slim to build OLGA test asset")

    out_rows = [rows[i % len(rows)] for i in range(1000)]
    out_path = ASSETS_DIR / "olga_humanTRB_1000.txt.gz"
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerows(out_rows)
    return len(out_rows)


def ensure_test_data(*, force: bool = False, verbose: bool = False) -> None:
    tree_paths = _load_tree_paths()

    vdjdb_candidates = [
        p
        for p in tree_paths
        if re.match(r"^vdjdb/vdjdb-\d{4}-\d{2}-\d{2}/vdjdb\.slim\.txt\.gz$", p)
    ]
    if not vdjdb_candidates:
        raise RuntimeError("Could not locate vdjdb.slim.txt.gz in airr_benchmark")
    vdjdb_rel = sorted(vdjdb_candidates)[-1]

    required_remote = [
        "sra/meta.tsv",
        "sra/samples.tar.gz",
        "tcrnet/B35+.txt.gz",
        "alice/yf/Q1_d0.tsv.gz",
        "alice/yf/Q1_d15.tsv.gz",
        vdjdb_rel,
    ]
    required_remote.extend(
        p for p in tree_paths if p.startswith("vdjtools_lite/") and not p.endswith("/")
    )

    for rel in required_remote:
        _download(rel, DATASET_ROOT / rel, verbose=verbose)

    vdjdb_local = DATASET_ROOT / vdjdb_rel
    if not _is_valid_gzip(vdjdb_local):
        if vdjdb_local.exists():
            vdjdb_local.unlink()
        _download(vdjdb_rel, vdjdb_local, verbose=verbose)

    if force:
        for d in (ASSETS_DIR, REAL_REPS_DIR, SRX_DIR):
            if d.exists():
                shutil.rmtree(d)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REAL_REPS_DIR.mkdir(parents=True, exist_ok=True)
    SRX_DIR.mkdir(parents=True, exist_ok=True)

    vdjtools_dir = DATASET_ROOT / "vdjtools_lite"
    for src in vdjtools_dir.glob("*.txt.gz"):
        _copy(src, REAL_REPS_DIR / src.name)
    for name in ("metadata_aging.txt", "metadata_hsct.txt"):
        _copy(vdjtools_dir / name, REAL_REPS_DIR / name)

    _copy(vdjtools_dir / "old_mixcr.gz", ASSETS_DIR / "old_mixcr.gz")
    _copy(vdjtools_dir / "vdjtools_trb_d_dot.tsv", ASSETS_DIR / "vdjtools_trb_d_dot.tsv")
    _copy(vdjtools_dir / "vdjtools_trb_d_dot_2.tsv", ASSETS_DIR / "vdjtools_trb_d_dot_2.tsv")

    _copy(DATASET_ROOT / vdjdb_rel, ASSETS_DIR / "vdjdb.slim.txt.gz")
    _convert_yf_vdjtools_to_airr(
        DATASET_ROOT / "alice/yf/Q1_d0.tsv.gz",
        ASSETS_DIR / "Q1_0_F1.airr.tsv.gz",
    )
    _convert_yf_vdjtools_to_airr(
        DATASET_ROOT / "alice/yf/Q1_d15.tsv.gz",
        ASSETS_DIR / "Q1_15_F1.airr.tsv.gz",
    )
    _copy(DATASET_ROOT / "tcrnet/B35+.txt.gz", ASSETS_DIR / "B35+.txt.gz")
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
        print(f"vdjdb source: {vdjdb_rel}")
        print(f"derived GILG cdr3 count: {gilg_n}")
        print(f"derived LLW rows: {llw_n}")
        print(f"derived OLGA rows: {olga_n}")


def main() -> None:
    force = os.getenv("MIRPY_TEST_DATA_FORCE", "0") in {"1", "true", "TRUE", "yes", "YES"}
    ensure_test_data(force=force, verbose=True)


if __name__ == "__main__":
    main()
