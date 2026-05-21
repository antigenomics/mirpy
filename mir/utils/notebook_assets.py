"""Helpers for downloading and locating notebook datasets.

These utilities keep notebook asset handling consistent with the test
bootstrap flow: datasets are downloaded on first use into
``notebooks/assets/large`` and ignored by git.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root from a notebook or script working directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "mir").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the mirpy repository root starting from {current}"
    )


def notebook_assets_root(repo_root: Path | None = None) -> Path:
    """Return ``notebooks/assets`` and create it when missing."""
    root = find_repo_root(repo_root)
    assets_root = root / "notebooks" / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    return assets_root


def notebook_large_assets_root(repo_root: Path | None = None) -> Path:
    """Return ``notebooks/assets/large`` and create it when missing."""
    large_root = notebook_assets_root(repo_root) / "large"
    large_root.mkdir(parents=True, exist_ok=True)
    return large_root


def _ensure_dataset(
    repo_id: str,
    local_name: str,
    repo_root: Path | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    dataset_root = notebook_large_assets_root(repo_root) / local_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_root),
        allow_patterns=allow_patterns,
    )
    return dataset_root


def ensure_airr_benchmark(
    repo_root: Path | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download or refresh the AIRR benchmark dataset into notebook assets."""
    return _ensure_dataset(
        repo_id="isalgo/airr_benchmark",
        local_name="airr_benchmark",
        repo_root=repo_root,
        allow_patterns=allow_patterns,
    )


def ensure_airr_covid19(repo_root: Path | None = None) -> Path:
    """Download or refresh the AIRR COVID-19 dataset into notebook assets."""
    return _ensure_dataset(
        repo_id="isalgo/airr_covid19",
        local_name="airr_covid19",
        repo_root=repo_root,
    )


def ensure_airr_yfv19(repo_root: Path | None = None) -> Path:
    """Download or refresh the AIRR YFV19 dataset into notebook assets."""
    return _ensure_dataset(
        repo_id="isalgo/airr_yfv19",
        local_name="airr_yfv19",
        repo_root=repo_root,
    )


def find_airr_benchmark_vdjdb_slim(dataset_root: Path) -> Path:
    """Return the latest ``vdjdb.slim.txt.gz`` file inside AIRR benchmark."""
    candidates = sorted(dataset_root.glob("vdjdb/vdjdb-*/vdjdb.slim.txt.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find vdjdb.slim.txt.gz under {dataset_root / 'vdjdb'}"
        )
    return candidates[-1]


def find_airr_benchmark_vdjdb_full(dataset_root: Path) -> Path:
    """Return the latest ``vdjdb_full.txt.gz`` file inside AIRR benchmark."""
    candidates = sorted(dataset_root.glob("vdjdb/vdjdb-*/vdjdb_full.txt.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find vdjdb_full.txt.gz under {dataset_root / 'vdjdb'}"
        )
    return candidates[-1]


def find_airr_benchmark_sra_meta(dataset_root: Path) -> tuple[Path, Path]:
    """Return ``(samples.tar.gz, meta.tsv)`` for the AIRR benchmark SRA bundle."""
    tarball = dataset_root / "sra" / "samples.tar.gz"
    metadata = dataset_root / "sra" / "meta.tsv"
    if not tarball.exists() or not metadata.exists():
        raise FileNotFoundError(
            f"Could not find SRA files under {dataset_root / 'sra'}"
        )
    return tarball, metadata


def find_airr_benchmark_tcrnet_file(dataset_root: Path, filename: str) -> Path:
    """Return a file from ``airr_benchmark/tcrnet`` by name."""
    path = dataset_root / "tcrnet" / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find {filename} under {dataset_root / 'tcrnet'}")
    return path


def ensure_airr_benchmark_alice(
    repo_root: Path | None = None,
    subsets: list[str] | None = None,
) -> Path:
    """Download the ALICE subset of the AIRR benchmark dataset.

    Args:
        subsets: Sub-folders to fetch. Defaults to ``["yf", "as"]``.
            Pass ``["yf"]`` for YF-only or ``["as"]`` for AS-only downloads.

    Returns:
        ``notebooks/assets/large/airr_benchmark`` root.
    """
    if subsets is None:
        subsets = ["yf", "as"]
    patterns = [f"alice/{sub}/*" for sub in subsets]
    return ensure_airr_benchmark(repo_root=repo_root, allow_patterns=patterns)


def find_airr_benchmark_dcode_10x_vdj_v1_donor(
    dataset_root: Path,
    donor_id: str,
) -> tuple[Path, Path]:
    """Return (all_contig, consensus) files for one dcode 10x_vdj_v1 donor."""
    dcode_root = dataset_root / "dcode"
    all_contig = sorted(
        dcode_root.glob(f"*{donor_id}*_all_contig_annotations.csv.gz")
    )
    consensus = sorted(
        dcode_root.glob(f"*{donor_id}*_consensus_annotations.csv.gz")
    )
    if not all_contig or not consensus:
        raise FileNotFoundError(
            f"Could not find 10x_vdj_v1 donor {donor_id!r} under {dcode_root}"
        )
    return all_contig[0], consensus[0]


def find_airr_benchmark_motif_pwms(dataset_root: Path) -> Path:
    """Return the latest ``motif_pwms.txt.gz`` file inside AIRR benchmark."""
    candidates = sorted(dataset_root.glob("vdjdb/vdjdb-*/motif_pwms.txt.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find motif_pwms.txt.gz under {dataset_root / 'vdjdb'}.  "
            "Run ensure_airr_benchmark(allow_patterns=['vdjdb/**']) first."
        )
    return candidates[-1]


def find_airr_benchmark_dcode_10x_vdj_v1_donor_matrix(
    dataset_root: Path,
    donor_id: str,
) -> Path:
    """Return donor-specific dcode 10x_vdj_v1 binarized CITE-seq matrix path."""
    dcode_root = dataset_root / "dcode"
    matrices = sorted(dcode_root.glob(f"*{donor_id}*_binarized_matrix.csv.gz"))
    if not matrices:
        raise FileNotFoundError(
            f"Could not find 10x_vdj_v1 donor matrix {donor_id!r} under {dcode_root}"
        )
    return matrices[0]