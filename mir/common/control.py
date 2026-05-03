"""Control/background repertoire management.

This module manages synthetic (OLGA-generated) and real (HuggingFace AIRR)
control datasets used for neighborhood/enrichment workflows.

Design goals:
- explicit setup step for expensive control generation/download,
- reproducible cache/registry of what is available locally,
- easy use in local development and batch (e.g. Slurm) environments.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd

from mir import get_resource_path
from mir.common.alleles import allele_with_default
from mir.basic.aliases import (
    OLGA_SUFFIX_TO_LOCUS,
    locus_search_tokens,
    normalize_locus_alias,
    normalize_species_alias,
)
from mir.basic.pgen import OlgaModel

_CONTROL_ENV = "MIRPY_CONTROL_DIR"
_MANIFEST_FILE = "manifest.json"
_DEFAULT_HF_DATASET = "isalgo/airr_control"
_LOCKS_DIR = ".locks"
_DEFAULT_LOCK_TIMEOUT_S = 1200.0
_DEFAULT_LOCK_POLL_S = 0.25
_DEFAULT_STALE_LOCK_S = 12 * 1200.0

@dataclass
class ControlRecord:
    """Single control artifact record stored in manifest."""

    control_type: str
    species: str
    locus: str
    path: str
    format: str
    source: str
    n: int | None = None
    created_at_utc: str | None = None


class ControlManager:
    """Manage control repertoire setup, registry, and loading."""

    def __init__(self, control_dir: str | Path | None = None) -> None:
        self.control_dir = resolve_control_dir(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.control_dir / _MANIFEST_FILE
        self._locks_dir = self.control_dir / _LOCKS_DIR
        self._locks_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_lock_path = self._locks_dir / "manifest.lock"

    # ---------------------------
    # Canonicalization
    # ---------------------------
    @staticmethod
    def canonical_species(species: str) -> str:
        return normalize_species_alias(species)

    @staticmethod
    def canonical_locus(locus: str) -> str:
        return normalize_locus_alias(locus)

    # ---------------------------
    # Manifest
    # ---------------------------
    def load_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {"records": {}}
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("records", {})
        return data

    def save_manifest(self, manifest: dict[str, dict]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self._manifest_lock_path):
            _atomic_write_json(self.manifest_path, manifest)

    @staticmethod
    def _record_key(control_type: str, species: str, locus: str, n: int | None = None) -> str:
        if control_type == "synthetic" and n is not None:
            return f"{control_type}:{species}:{locus}:n={int(n)}"
        return f"{control_type}:{species}:{locus}"

    def register_record(self, record: ControlRecord) -> None:
        key = self._record_key(record.control_type, record.species, record.locus, record.n)
        with _file_lock(self._manifest_lock_path):
            manifest = self.load_manifest()
            manifest["records"][key] = asdict(record)
            _atomic_write_json(self.manifest_path, manifest)

    def get_record(
        self,
        control_type: str,
        species: str,
        locus: str,
        *,
        n: int | None = None,
    ) -> ControlRecord | None:
        manifest = self.load_manifest()
        records = manifest["records"]
        ctype = control_type.strip().lower()

        if ctype != "synthetic":
            rec = records.get(self._record_key(ctype, species, locus))
            if rec is None:
                return None
            return ControlRecord(**rec)

        if n is not None:
            rec = records.get(self._record_key(ctype, species, locus, n))
            if rec is None:
                legacy = records.get(self._record_key(ctype, species, locus))
                if legacy is None:
                    return None
                legacy_rec = ControlRecord(**legacy)
                return legacy_rec if legacy_rec.n == int(n) else None
            return ControlRecord(**rec)

        legacy = records.get(self._record_key(ctype, species, locus))
        if legacy is not None:
            return ControlRecord(**legacy)

        prefix = f"{ctype}:{species}:{locus}:n="
        matches = [ControlRecord(**rec) for key, rec in records.items() if key.startswith(prefix)]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"Multiple synthetic controls registered for {species}/{locus}; "
            "specify n explicitly when loading"
        )

    def list_available_controls(self) -> list[ControlRecord]:
        manifest = self.load_manifest()
        out: list[ControlRecord] = []
        for rec in manifest["records"].values():
            out.append(ControlRecord(**rec))
        return out

    # ---------------------------
    # Availability from OLGA resources
    # ---------------------------
    @staticmethod
    def list_available_olga_models(model_root: str | Path | None = None) -> list[tuple[str, str]]:
        """Return available (species, locus) pairs from local OLGA resources."""
        if model_root is None:
            model_root = Path(cast(str, get_resource_path("olga/default_models")))
        else:
            model_root = Path(model_root)

        if not model_root.exists():
            return []

        out: list[tuple[str, str]] = []
        for child in model_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if "_" not in name:
                continue
            species, suffix = name.split("_", 1)
            species = species.lower().strip()
            suffix_key = suffix.upper().strip()
            locus = OLGA_SUFFIX_TO_LOCUS.get(suffix_key)
            if not locus:
                continue
            out.append((species, locus))

        out = sorted(set(out))
        return out

    # ---------------------------
    # Synthetic control (OLGA)
    # ---------------------------
    def synthetic_control_path(self, species: str, locus: str, n: int) -> Path:
        return self.control_dir / "synthetic" / species / locus / f"olga_n{n}.pkl"

    def _control_lock_path(
        self,
        control_type: str,
        species: str,
        locus: str,
        *,
        n: int | None = None,
    ) -> Path:
        safe = f"{control_type}_{species}_{locus}".replace("/", "_")
        if control_type == "synthetic" and n is not None:
            safe = f"{safe}_n{int(n)}"
        return self._locks_dir / f"{safe}.lock"

    def ensure_synthetic_control(
        self,
        species: str,
        locus: str,
        *,
        n: int = 10_000_000,
        n_jobs: int = 1,
        overwrite: bool = False,
        seed: int = 42,
        chunk_size: int = 100_000,
        progress: bool = True,
    ) -> ControlRecord:
        species_c = self.canonical_species(species)
        locus_c = self.canonical_locus(locus)

        available = set(self.list_available_olga_models())
        if (species_c, locus_c) not in available:
            raise ValueError(
                f"OLGA model not available for species={species_c!r}, locus={locus_c!r}. "
                f"Available: {sorted(available)}"
            )

        path = self.synthetic_control_path(species_c, locus_c, n)
        lock_path = self._control_lock_path("synthetic", species_c, locus_c, n=n)
        with _file_lock(lock_path):
            if path.exists() and not overwrite:
                try:
                    _read_pickle(path)
                except Exception:
                    path.unlink(missing_ok=True)
                else:
                    rec = ControlRecord(
                        control_type="synthetic",
                        species=species_c,
                        locus=locus_c,
                        path=str(path),
                        format="pickle",
                        source="olga",
                        n=n,
                        created_at_utc=_utc_now(),
                    )
                    self.register_record(rec)
                    return rec

            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                df = generate_synthetic_olga_control(
                    species=species_c,
                    locus=locus_c,
                    n=n,
                    n_jobs=n_jobs,
                    seed=seed,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            except TypeError:
                # Backward-compat for monkeypatched helpers with legacy signature.
                df = generate_synthetic_olga_control(
                    species=species_c,
                    locus=locus_c,
                    n=n,
                    seed=seed,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            _write_pickle(df, path)

            rec = ControlRecord(
                control_type="synthetic",
                species=species_c,
                locus=locus_c,
                path=str(path),
                format="pickle",
                source="olga",
                n=n,
                created_at_utc=_utc_now(),
            )
            self.register_record(rec)
            return rec

    # ---------------------------
    # Real control (HuggingFace)
    # ---------------------------
    def real_control_path(self, species: str, locus: str) -> Path:
        return self.control_dir / "real" / species / locus / "airr_control.pkl"

    def ensure_real_control(
        self,
        species: str,
        locus: str,
        *,
        dataset_repo: str = _DEFAULT_HF_DATASET,
        overwrite: bool = False,
        hf_cache_dir: str | Path | None = None,
    ) -> ControlRecord:
        species_c = self.canonical_species(species)
        locus_c = self.canonical_locus(locus)

        path = self.real_control_path(species_c, locus_c)
        lock_path = self._control_lock_path("real", species_c, locus_c)
        with _file_lock(lock_path):
            if path.exists() and not overwrite:
                try:
                    _read_pickle(path)
                except Exception:
                    path.unlink(missing_ok=True)
                else:
                    rec = ControlRecord(
                        control_type="real",
                        species=species_c,
                        locus=locus_c,
                        path=str(path),
                        format="pickle",
                        source=f"huggingface:{dataset_repo}",
                        n=None,
                        created_at_utc=_utc_now(),
                    )
                    self.register_record(rec)
                    return rec

            path.parent.mkdir(parents=True, exist_ok=True)
            local_snapshot = _download_hf_snapshot(dataset_repo, cache_dir=hf_cache_dir)
            source_file = _find_real_control_file(local_snapshot, species_c, locus_c)
            if source_file is None:
                raise FileNotFoundError(
                    f"No .ntvj file found for species={species_c!r}, locus={locus_c!r} "
                    f"in dataset {dataset_repo!r}"
                )

            df = build_real_control_from_ntvj(source_file)
            _write_pickle(df, path)

            rec = ControlRecord(
                control_type="real",
                species=species_c,
                locus=locus_c,
                path=str(path),
                format="pickle",
                source=f"huggingface:{dataset_repo}",
                n=len(df),
                created_at_utc=_utc_now(),
            )
            self.register_record(rec)
            return rec

    # ---------------------------
    # Unified API
    # ---------------------------
    def ensure_control(
        self,
        control_type: str,
        species: str,
        locus: str,
        **kwargs,
    ) -> ControlRecord:
        ctype = (control_type or "").strip().lower()
        if ctype == "synthetic":
            return self.ensure_synthetic_control(species, locus, **kwargs)
        if ctype == "real":
            return self.ensure_real_control(species, locus, **kwargs)
        raise ValueError("control_type must be 'synthetic' or 'real'")

    def load_control_df(
        self,
        control_type: str,
        species: str,
        locus: str,
        *,
        n: int | None = None,
        wait_if_building: bool = True,
        wait_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
    ) -> pd.DataFrame:
        species_c = self.canonical_species(species)
        locus_c = self.canonical_locus(locus)
        ctype = control_type.strip().lower()
        if wait_if_building:
            lock_path = self._control_lock_path(ctype, species_c, locus_c, n=n)
            _wait_for_lock_release(lock_path, timeout_s=wait_timeout_s)
        rec = self.get_record(ctype, species_c, locus_c, n=n)
        if rec is None:
            raise FileNotFoundError(
                f"Control not registered: type={control_type!r}, species={species_c!r}, locus={locus_c!r}, n={n!r}"
            )
        return _read_pickle(Path(rec.path))

    def ensure_and_load_control_df(
        self,
        control_type: str,
        species: str,
        locus: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Ensure control exists on disk and return it as a DataFrame."""
        n = kwargs.get("n") if control_type.strip().lower() == "synthetic" else None
        self.ensure_control(control_type, species, locus, **kwargs)
        return self.load_control_df(control_type, species, locus, n=n)


# ---------------------------
# Public helper functions
# ---------------------------

def resolve_control_dir(control_dir: str | Path | None = None) -> Path:
    """Resolve control root directory.

    Priority:
    1. explicit function argument,
    2. MIRPY_CONTROL_DIR env var,
    3. ~/.cache/mirpy/controls
    """
    if control_dir is not None:
        return Path(control_dir).expanduser().resolve()

    env_val = os.getenv(_CONTROL_ENV)
    if env_val:
        return Path(env_val).expanduser().resolve()

    return (Path.home() / ".cache" / "mirpy" / "controls").resolve()


def generate_synthetic_olga_control(
    *,
    species: str,
    locus: str,
    n: int,
    seed: int,
    chunk_size: int,
    progress: bool,
    n_jobs: int = 1,
    zipf_alpha: float = 2.0,
    max_duplicate_count: int = 10_000,
) -> pd.DataFrame:
    """Generate synthetic OLGA control DataFrame with ntvj columns.

    Output columns:
    - duplicate_count
    - junction
    - junction_aa
    - v_gene
    - j_gene
    - log2_pgen
    """
    model = OlgaModel(species=species, locus=locus, seed=seed)
    records = model.generate_pool(n, n_jobs=max(1, int(n_jobs)), seed=seed)

    rng = np.random.default_rng(seed + 1_000_003)
    dup_counts = rng.zipf(a=zipf_alpha, size=len(records))
    dup_counts = np.minimum(dup_counts, max_duplicate_count)

    rows: list[dict[str, str | int]] = []
    for i, rec in enumerate(records):
        rows.append(
            {
                "duplicate_count": int(dup_counts[i]),
                "junction": str(rec["junction"]),
                "junction_aa": str(rec["junction_aa"]),
                "v_gene": _normalize_allele(str(rec["v_gene"])),
                "j_gene": _normalize_allele(str(rec["j_gene"])),
                "log2_pgen": float(rec["log2_pgen"]),
            }
        )
    if progress:
        print(f"Generated synthetic control {species}/{locus}: {len(rows)}/{n}")

    return pd.DataFrame.from_records(rows)


def compute_control_pgen_records(
    control_df: pd.DataFrame,
    *,
    locus: str,
    species: str = "human",
    seed: int = 42,
    n_jobs: int = 1,
    pgen_adjustment=None,
) -> list[dict[str, str | float]]:
    """Compute log2-Pgen records from control tables for VDJBet bin pooling.

    The returned records contain ``junction_aa``, normalized ``v_gene``/``j_gene``,
    and ``log2_pgen`` for direct use in Pgen-bin mock sampling.
    """
    required = ["junction_aa", "v_gene", "j_gene"]
    missing = [c for c in required if c not in control_df.columns]
    if missing:
        raise ValueError(f"control_df missing required columns: {missing}")

    df = control_df.dropna(subset=required).copy()
    if df.empty:
        return []

    if "log2_pgen" in df.columns:
        rows = [
            (
                str(jaa),
                _normalize_allele(str(vg)),
                _normalize_allele(str(jg)),
                float(l2p),
            )
            for jaa, vg, jg, l2p in df[
                ["junction_aa", "v_gene", "j_gene", "log2_pgen"]
            ].itertuples(index=False)
            if str(jaa)
        ]
    else:
        rows = [
            (
                str(jaa),
                _normalize_allele(str(vg)),
                _normalize_allele(str(jg)),
                None,
            )
            for jaa, vg, jg in df[
                ["junction_aa", "v_gene", "j_gene"]
            ].itertuples(index=False)
            if str(jaa)
        ]
    if not rows:
        return []

    def _worker(batch: list[tuple[str, str, str, float | None]], worker_seed: int) -> list[dict[str, str | float]]:
        model: OlgaModel | None = None
        out: list[dict[str, str | float]] = []
        for jaa, vg, jg, log2_pgen in batch:
            if log2_pgen is not None:
                if pgen_adjustment is None:
                    out.append(
                        {
                            "junction_aa": jaa,
                            "v_gene": vg,
                            "j_gene": jg,
                            "log2_pgen": float(log2_pgen),
                        }
                    )
                    continue
                factor = float(pgen_adjustment.factor(locus, vg, jg))
                if factor <= 0:
                    continue
                out.append(
                    {
                        "junction_aa": jaa,
                        "v_gene": vg,
                        "j_gene": jg,
                        "log2_pgen": float(log2_pgen) + math.log2(factor),
                    }
                )
                continue

            if model is None:
                model = OlgaModel(species=species, locus=locus, seed=worker_seed)
            pgen_val = model.compute_pgen_junction_aa(jaa)
            if pgen_val is None or pgen_val <= 0:
                continue
            if pgen_adjustment is not None:
                pgen_val = pgen_adjustment.adjust_pgen(locus, vg, jg, pgen_val)
                if pgen_val <= 0:
                    continue
            out.append(
                {
                    "junction_aa": jaa,
                    "v_gene": vg,
                    "j_gene": jg,
                    "log2_pgen": math.log2(float(pgen_val)),
                }
            )
        return out

    jobs = max(1, int(n_jobs))
    if jobs == 1 or len(rows) < 1024:
        return _worker(rows, seed)

    chunk_size = max(256, len(rows) // jobs)
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    records: list[dict[str, str | float]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_worker, chunk, seed + i + 1) for i, chunk in enumerate(chunks)]
        for future in futures:
            records.extend(future.result())
    return records


def build_real_control_from_ntvj(path: str | Path) -> pd.DataFrame:
    """Load .ntvj-like VDJtools file and normalize to ntvj schema.

    Alleles are appended as *01 when absent.
    """
    p = Path(path)
    df = pd.read_csv(p, sep="\t")
    renamed = {
        "cdr3nt": "junction",
        "cdr3aa": "junction_aa",
        "v": "v_gene",
        "j": "j_gene",
    }
    for src, dst in renamed.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    count_candidates = ["duplicate_count", "count", "#count", "clonotype_count"]
    count_col = next((c for c in count_candidates if c in df.columns), None)
    if count_col is None:
        df["duplicate_count"] = 1
    elif count_col != "duplicate_count":
        df["duplicate_count"] = df[count_col]

    required = ["duplicate_count", "junction", "junction_aa", "v_gene", "j_gene"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")

    out = df[required].copy()
    out["duplicate_count"] = pd.to_numeric(out["duplicate_count"], errors="coerce").fillna(1).astype(int)
    out.loc[out["duplicate_count"] < 1, "duplicate_count"] = 1
    out["v_gene"] = out["v_gene"].astype(str).map(_normalize_allele)
    out["j_gene"] = out["j_gene"].astype(str).map(_normalize_allele)
    return out


def _normalize_allele(gene_name: str) -> str:
    return allele_with_default(gene_name)


def _find_real_control_file(snapshot_root: str | Path, species: str, locus: str) -> Path | None:
    root = Path(snapshot_root)
    if not root.exists():
        return None

    species_tokens = {
        species,
        {"human": "hsa", "mouse": "mmu"}.get(species, species),
    }
    locus_tokens = locus_search_tokens(locus)

    candidates = sorted({
        *root.rglob("*.ntvj"),
        *root.rglob("*.ntvj.*"),
        *root.rglob("*ntvj*.tsv"),
        *root.rglob("*ntvj*.tsv.gz"),
    })
    for fp in candidates:
        name_l = fp.name.lower()
        if any(tok in name_l for tok in species_tokens) and any(tok in name_l for tok in locus_tokens):
            return fp

    # Fallback: first .ntvj containing locus token only.
    for fp in candidates:
        name_l = fp.name.lower()
        if any(tok in name_l for tok in locus_tokens):
            return fp
    return None


def _download_hf_snapshot(repo_id: str, cache_dir: str | Path | None = None) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - validated in unit tests via monkeypatch
        raise ImportError(
            "huggingface_hub is required to download real controls; install it first"
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=(str(cache_dir) if cache_dir is not None else None),
    )


def _write_pickle(df: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _read_pickle(path: Path) -> pd.DataFrame:
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Pickle at {path} is not a pandas DataFrame")
    return obj


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


@contextmanager
def _file_lock(
    lock_path: Path,
    *,
    timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
    poll_s: float = _DEFAULT_LOCK_POLL_S,
    stale_after_s: float = _DEFAULT_STALE_LOCK_S,
):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd: int | None = None

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            meta = f"pid={os.getpid()} started={time.time():.6f}\n"
            os.write(fd, meta.encode("utf-8", errors="ignore"))
            break
        except FileExistsError:
            # Prune stale locks to survive crashed workers on shared infra.
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_after_s:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue

            if (time.monotonic() - start) >= timeout_s:
                raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass


def _wait_for_lock_release(
    lock_path: Path,
    *,
    timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
    poll_s: float = _DEFAULT_LOCK_POLL_S,
) -> None:
    start = time.monotonic()
    while lock_path.exists():
        if (time.monotonic() - start) >= timeout_s:
            raise TimeoutError(f"Timeout waiting for build lock release: {lock_path}")
        time.sleep(poll_s)


def _parse_locus_arg(value: str) -> list[str]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("At least one locus must be provided")
    return parts


def _parse_species_arg(value: str) -> list[str]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("At least one species must be provided")
    return parts


def control_setup_cli(argv: list[str] | None = None) -> int:
    """CLI for prebuilding/downloading controls.

    Example:
      mirpy-control-setup --type synthetic --species human,mouse --loci TRA,TRB --n 1000000
    """
    parser = argparse.ArgumentParser(description="Setup mirpy control/background data")
    parser.add_argument("--type", choices=["synthetic", "real"], required=True)
    parser.add_argument("--species", required=True, help="Comma-separated species aliases")
    parser.add_argument("--loci", required=True, help="Comma-separated loci aliases")
    parser.add_argument("--control-dir", default=None)
    parser.add_argument("--n", type=int, default=10_000_000, help="Synthetic sample size")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-repo", default=_DEFAULT_HF_DATASET)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    mgr = ControlManager(args.control_dir)
    species_list = _parse_species_arg(args.species)
    locus_list = _parse_locus_arg(args.loci)

    for species in species_list:
        for locus in locus_list:
            if args.type == "synthetic":
                rec = mgr.ensure_synthetic_control(
                    species,
                    locus,
                    n=args.n,
                    overwrite=args.overwrite,
                    seed=args.seed,
                    chunk_size=args.chunk_size,
                )
            else:
                rec = mgr.ensure_real_control(
                    species,
                    locus,
                    dataset_repo=args.dataset_repo,
                    overwrite=args.overwrite,
                    hf_cache_dir=args.hf_cache_dir,
                )
            print(f"ready: {rec.control_type} {rec.species}/{rec.locus} -> {rec.path}")

    print(f"manifest: {mgr.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(control_setup_cli())
