"""Control/background repertoire management.

This module **orchestrates** everything needed to provide background controls for
neighborhood-enrichment workflows (ALICE, TCRNET).  It covers:

- **Synthetic controls** (OLGA-generated): generate once, cache to disk, reuse
  across all subsequent mirpy runs without regeneration.  Existing caches can be
  extended (append) or trimmed (head) to satisfy a new size request without
  rebuilding from scratch.
- **Real controls** (downloaded from HuggingFace AIRR dataset): snapshot-aware
  download that refreshes only when the upstream dataset changes.
- **Manifest registry**: a ``manifest.json`` file tracks every cached file
  (type, species, locus, size, source, timestamp) so any run can discover what
  is available locally without scanning the filesystem.
- **Thread and Slurm safety**: per-control file locks (``O_CREAT | O_EXCL``)
  prevent two concurrent processes from building the same control.  Stale locks
  left by crashed workers are pruned automatically after
  ``MIRPY_CONTROL_LOCK_STALE_S`` seconds (default 4 h).  A second process that
  arrives while a build is in progress simply waits (``load_control_df`` with
  ``wait_if_building=True``) and reads the finished file.
- **On-demand cleanup and regeneration**: :meth:`ControlManager.cleanup_cache`
  removes broken/orphan files and reconciles the manifest;
  ``ensure_*_control(..., overwrite=True)`` forces a full rebuild.

Typical usage
-------------
From Python::

    from mir.common.control import ControlManager
    mgr = ControlManager()
    mgr.ensure_synthetic_control("human", "TRB", n=10_000_000)
    df = mgr.load_control_df("synthetic", "human", "TRB", n=10_000_000)

From the CLI (pre-build before a Slurm job array)::

    mirpy-control-setup --type synthetic --species human --loci TRB --n 10000000

Environment variable ``MIRPY_CONTROL_DIR`` overrides the default cache root
(``~/.cache/mirpy/controls``).
"""

from __future__ import annotations

import argparse
import math
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

_MP_CTX = multiprocessing.get_context("spawn")
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from mir import get_resource_path
from mir.common.alleles import allele_with_default
from mir.basic.aliases import (
    OLGA_SUFFIX_TO_LOCUS,
    locus_search_tokens,
    normalize_locus_alias,
    normalize_species_alias,
)
from mir.basic.pgen import McPgenPool, OlgaModel

_CONTROL_ENV = "MIRPY_CONTROL_DIR"
_MANIFEST_FILE = "manifest.json"
_DEFAULT_HF_DATASET = "isalgo/airr_control"
_LOCKS_DIR = ".locks"
_DEFAULT_LOCK_TIMEOUT_S = 1200.0
_DEFAULT_LOCK_POLL_S = 0.25
_DEFAULT_STALE_LOCK_S = 12 * 1200.0


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve worker count, defaulting to all available CPUs."""
    if n_jobs is None:
        return max(1, int(os.cpu_count() or 1))
    return max(1, int(n_jobs))


def _compute_log2_pgen_chunk(
    args: tuple[list[str], str, str, int],
) -> dict[str, float]:
    """Compute log2(Pgen) for a chunk of unique junction_aa strings."""
    junction_aas, species, locus, seed = args
    model = OlgaModel(species=species, locus=locus, seed=seed)
    out: dict[str, float] = {}
    for jaa in junction_aas:
        pgen_val = model.compute_pgen_junction_aa(jaa)
        if pgen_val is None or pgen_val <= 0:
            continue
        out[jaa] = math.log2(float(pgen_val))
    return out

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
        n_jobs: int | None = None,
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

        resolved_n_jobs = _resolve_n_jobs(n_jobs)

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

            if not overwrite:
                # Reuse existing synthetic controls for same species/locus.
                # If an existing cache is larger, keep first n rows.
                # If smaller, append exactly (n - N) newly generated rows.
                candidates: list[tuple[ControlRecord, pl.DataFrame]] = []
                for rec in self.list_available_controls():
                    if rec.control_type != "synthetic":
                        continue
                    if rec.species != species_c or rec.locus != locus_c:
                        continue
                    if rec.n is None:
                        continue
                    rec_path = Path(rec.path)
                    if not rec_path.exists():
                        continue
                    try:
                        rec_df = _read_pickle(rec_path)
                    except Exception:
                        rec_path.unlink(missing_ok=True)
                        continue
                    candidates.append((rec, rec_df))

                # Exact-size cache already exists.
                for rec, rec_df in candidates:
                    if len(rec_df) != n:
                        continue
                    if Path(rec.path) != path:
                        _write_pickle(rec_df, path)
                    out_rec = ControlRecord(
                        control_type="synthetic",
                        species=species_c,
                        locus=locus_c,
                        path=str(path),
                        format="pickle",
                        source="olga",
                        n=n,
                        created_at_utc=_utc_now(),
                    )
                    self.register_record(out_rec)
                    return out_rec

                # Use the smallest available cache that is still >= n.
                larger = [(rec, df) for rec, df in candidates if len(df) > n]
                if larger:
                    rec, base_df = min(larger, key=lambda x: len(x[1]))
                    trimmed = base_df.head(n)
                    _write_pickle(trimmed, path)
                    out_rec = ControlRecord(
                        control_type="synthetic",
                        species=species_c,
                        locus=locus_c,
                        path=str(path),
                        format="pickle",
                        source=f"{rec.source}|derived:head({n})",
                        n=n,
                        created_at_utc=_utc_now(),
                    )
                    self.register_record(out_rec)
                    if progress:
                        print(
                            f"Reused synthetic control {species_c}/{locus_c}: "
                            f"trimmed {len(base_df)} -> {n}"
                        )
                    return out_rec

                # Extend the largest available cache that is < n.
                smaller = [(rec, df) for rec, df in candidates if len(df) < n]
                if smaller:
                    rec, base_df = max(smaller, key=lambda x: len(x[1]))
                    need = n - len(base_df)
                    extra = generate_synthetic_olga_control(
                        species=species_c,
                        locus=locus_c,
                        n=need,
                        n_jobs=resolved_n_jobs,
                        seed=seed + len(base_df),
                        chunk_size=chunk_size,
                        progress=progress,
                    )
                    combined = pl.concat([base_df, extra])
                    _write_pickle(combined, path)
                    out_rec = ControlRecord(
                        control_type="synthetic",
                        species=species_c,
                        locus=locus_c,
                        path=str(path),
                        format="pickle",
                        source=f"{rec.source}|extended:+{need}",
                        n=n,
                        created_at_utc=_utc_now(),
                    )
                    self.register_record(out_rec)
                    if progress:
                        print(
                            f"Extended synthetic control {species_c}/{locus_c}: "
                            f"{len(base_df)} + {need} -> {n}"
                        )
                    return out_rec

            path.parent.mkdir(parents=True, exist_ok=True)
            df = generate_synthetic_olga_control(
                species=species_c,
                locus=locus_c,
                n=n,
                n_jobs=resolved_n_jobs,
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

    def cleanup_cache(
        self,
        *,
        cleanup_synthetic: bool = True,
        cleanup_real: bool = True,
        remove_orphans: bool = True,
    ) -> dict[str, int]:
        """Clean stale/broken cache files and reconcile manifest entries.

        Returns counters for removed/kept entries and files.
        """
        summary = {
            "manifest_entries_removed": 0,
            "invalid_files_removed": 0,
            "orphan_files_removed": 0,
            "manifest_entries_kept": 0,
        }
        allow_types = {
            t
            for t, enabled in (("synthetic", cleanup_synthetic), ("real", cleanup_real))
            if enabled
        }

        with _file_lock(self._manifest_lock_path):
            manifest = self.load_manifest()
            records = manifest.get("records", {})
            kept: dict[str, dict] = {}
            referenced_paths: set[Path] = set()

            for key, rec_raw in records.items():
                rec = ControlRecord(**rec_raw)
                if rec.control_type not in allow_types:
                    kept[key] = rec_raw
                    continue

                p = Path(rec.path)
                if not p.exists():
                    summary["manifest_entries_removed"] += 1
                    continue
                try:
                    _read_pickle(p)
                except Exception:
                    p.unlink(missing_ok=True)
                    summary["invalid_files_removed"] += 1
                    summary["manifest_entries_removed"] += 1
                    continue

                kept[key] = rec_raw
                referenced_paths.add(p.resolve())
                summary["manifest_entries_kept"] += 1

            manifest["records"] = kept
            _atomic_write_json(self.manifest_path, manifest)

        if remove_orphans:
            roots: list[Path] = []
            if cleanup_synthetic:
                roots.append(self.control_dir / "synthetic")
            if cleanup_real:
                roots.append(self.control_dir / "real")
            for root in roots:
                if not root.exists():
                    continue
                for p in root.rglob("*.pkl"):
                    rp = p.resolve()
                    if rp in referenced_paths:
                        continue
                    p.unlink(missing_ok=True)
                    summary["orphan_files_removed"] += 1

        return summary

    def refresh_real_controls(
        self,
        *,
        dataset_repo: str = _DEFAULT_HF_DATASET,
        hf_cache_dir: str | Path | None = None,
        species: list[str] | None = None,
        loci: list[str] | None = None,
    ) -> dict[str, int]:
        """Refresh real controls when upstream HF dataset snapshot changes."""
        snapshot = _download_hf_snapshot(dataset_repo, cache_dir=hf_cache_dir)
        snapshot_id = Path(snapshot).name
        species_filter = {self.canonical_species(s) for s in species} if species else None
        locus_filter = {self.canonical_locus(l) for l in loci} if loci else None

        total = 0
        updated = 0
        skipped = 0
        for rec in self.list_available_controls():
            if rec.control_type != "real":
                continue
            if species_filter is not None and rec.species not in species_filter:
                continue
            if locus_filter is not None and rec.locus not in locus_filter:
                continue
            total += 1
            src = rec.source or ""
            current_snapshot = src.split("@", 1)[1] if "@" in src else ""
            if current_snapshot == snapshot_id and Path(rec.path).exists():
                skipped += 1
                continue
            self.ensure_real_control(
                rec.species,
                rec.locus,
                dataset_repo=dataset_repo,
                overwrite=True,
                hf_cache_dir=hf_cache_dir,
            )
            updated += 1

        return {
            "checked": total,
            "updated": updated,
            "up_to_date": skipped,
        }

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
                source=f"huggingface:{dataset_repo}@{Path(local_snapshot).name}",
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
    ) -> pl.DataFrame:
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
    ) -> pl.DataFrame:
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
    n_jobs: int | None = None,
    zipf_alpha: float = 2.0,
    max_duplicate_count: int = 10_000,
) -> pl.DataFrame:
    """Generate synthetic OLGA control DataFrame with ntvj columns.

    Output columns:
    - duplicate_count
    - junction
    - junction_aa
    - v_call
    - j_call
    - log2_pgen
    """
    jobs = _resolve_n_jobs(n_jobs)
    model = OlgaModel(species=species, locus=locus, seed=seed)
    records = model.generate_pool(n, n_jobs=jobs, seed=seed)

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
                "v_call": _normalize_allele(str(rec["v_call"])),
                "j_call": _normalize_allele(str(rec["j_call"])),
                "log2_pgen": float(rec["log2_pgen"]),
            }
        )
    if progress:
        print(f"Generated synthetic control {species}/{locus}: {len(rows)}/{n}")

    return pl.from_dicts(rows)


def compute_control_pgen_records(
    control_df: pl.DataFrame,
    *,
    locus: str,
    species: str = "human",
    seed: int = 42,
    n_jobs: int | None = None,
    pgen_adjustment=None,
) -> list[dict[str, str | float]]:
    """Compute log2-Pgen records from control tables for VDJBet bin pooling.

    The returned records contain ``junction_aa``, normalized ``v_call``/``j_call``,
    and ``log2_pgen`` for direct use in Pgen-bin mock sampling.

    Args:
        control_df: DataFrame with ``junction_aa``, ``v_call``, ``j_call`` columns.
            An optional ``log2_pgen`` column enables the fast path (no OLGA calls).
        locus: Receptor locus (e.g. ``"TRB"``).
        species: Species name (default ``"human"``).
        seed: Random seed for OLGA model initialization.
        n_jobs: Worker processes for Pgen computation (default: all CPUs).
        pgen_adjustment: Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`
            for V/J-specific Pgen scaling.  When provided, each record's
            ``log2_pgen`` is adjusted by ``log2(factor(locus, v, j))``.
    """
    required = ["junction_aa", "v_call", "j_call"]
    missing = [c for c in required if c not in control_df.columns]
    if missing:
        raise ValueError(f"control_df missing required columns: {missing}")

    df = control_df.drop_nulls(subset=required)
    if df.is_empty():
        return []

    if "log2_pgen" in df.columns:
        rows = [
            (
                str(jaa),
                _normalize_allele(str(vg)),
                _normalize_allele(str(jg)),
                float(l2p),
            )
            for jaa, vg, jg, l2p in df.select(
                ["junction_aa", "v_call", "j_call", "log2_pgen"]
            ).iter_rows()
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
            for jaa, vg, jg in df.select(
                ["junction_aa", "v_call", "j_call"]
            ).iter_rows()
            if str(jaa)
        ]
    if not rows:
        return []

    jobs = _resolve_n_jobs(n_jobs)

    factor_cache: dict[tuple[str, str], float] = {}
    if pgen_adjustment is not None:
        unique_pairs = {(vg, jg) for _, vg, jg, _ in rows}
        factor_cache = {
            pair: float(pgen_adjustment.factor(locus, pair[0], pair[1]))
            for pair in unique_pairs
        }

    # Fast path when control table already has log2_pgen values.
    if all(log2_pgen is not None for _, _, _, log2_pgen in rows):
        records: list[dict[str, str | float]] = []
        for jaa, vg, jg, log2_pgen in rows:
            l2p = float(log2_pgen)
            if pgen_adjustment is not None:
                factor = factor_cache.get((vg, jg), 1.0)
                if factor <= 0:
                    continue
                l2p += math.log2(factor)
            records.append(
                {
                    "junction_aa": jaa,
                    "v_call": vg,
                    "j_call": jg,
                    "log2_pgen": l2p,
                }
            )
        return records

    # Real-control path: compute base log2(Pgen) once per unique junction_aa.
    ordered_unique_jaa = list(dict.fromkeys(jaa for jaa, _, _, _ in rows))
    log2_by_jaa: dict[str, float] = {}

    if jobs == 1 or len(ordered_unique_jaa) < 2048:
        model = OlgaModel(species=species, locus=locus, seed=seed)
        for jaa in ordered_unique_jaa:
            pgen_val = model.compute_pgen_junction_aa(jaa)
            if pgen_val is None or pgen_val <= 0:
                continue
            log2_by_jaa[jaa] = math.log2(float(pgen_val))
    else:
        chunk_size = max(1024, len(ordered_unique_jaa) // jobs)
        chunks = [
            ordered_unique_jaa[i : i + chunk_size]
            for i in range(0, len(ordered_unique_jaa), chunk_size)
        ]
        args = [
            (chunk, species, locus, seed + i + 1)
            for i, chunk in enumerate(chunks)
        ]
        with ProcessPoolExecutor(max_workers=jobs, mp_context=_MP_CTX) as executor:
            for partial in executor.map(_compute_log2_pgen_chunk, args):
                log2_by_jaa.update(partial)

    records: list[dict[str, str | float]] = []
    for jaa, vg, jg, _ in rows:
        base = log2_by_jaa.get(jaa)
        if base is None:
            continue
        l2p = base
        if pgen_adjustment is not None:
            factor = factor_cache.get((vg, jg), 1.0)
            if factor <= 0:
                continue
            l2p += math.log2(factor)
        records.append(
            {
                "junction_aa": jaa,
                "v_call": vg,
                "j_call": jg,
                "log2_pgen": l2p,
            }
        )
    return records


def build_real_control_from_ntvj(path: str | Path) -> pl.DataFrame:
    """Load .ntvj-like VDJtools file and normalize to ntvj schema.

    Alleles are appended as ``*01`` when absent.
    """
    p = Path(path)
    df = pl.read_csv(p, separator="\t", infer_schema_length=10000, ignore_errors=True)

    # Rename legacy column names to AIRR equivalents
    rename_map = {}
    for src, dst in {"cdr3nt": "junction", "cdr3aa": "junction_aa", "v": "v_call", "j": "j_call"}.items():
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(rename_map)

    count_candidates = ["duplicate_count", "count", "#count", "clonotype_count"]
    count_col = next((c for c in count_candidates if c in df.columns), None)
    if count_col is None:
        df = df.with_columns(pl.lit(1).alias("duplicate_count"))
    elif count_col != "duplicate_count":
        df = df.rename({count_col: "duplicate_count"})

    required = ["duplicate_count", "junction", "junction_aa", "v_call", "j_call"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")

    df = df.select(required).with_columns([
        pl.col("duplicate_count").cast(pl.Int64, strict=False).fill_null(1).clip(lower_bound=1),
        pl.col("v_call").cast(pl.Utf8).map_elements(_normalize_allele, return_dtype=pl.Utf8),
        pl.col("j_call").cast(pl.Utf8).map_elements(_normalize_allele, return_dtype=pl.Utf8),
    ])
    return df


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


def _write_pickle(df: pl.DataFrame, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)


# Legacy internal gene-column names found in caches written before the
# v_gene→v_call AIRR-naming unification.  Renamed to internal names on load.
_LEGACY_GENE_COLUMNS: dict[str, str] = {
    "v_gene": "v_call", "d_gene": "d_call", "j_gene": "j_call", "c_gene": "c_call",
}


def _normalize_cached_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename legacy ``*_gene`` columns to internal ``*_call`` names."""
    rename = {c: _LEGACY_GENE_COLUMNS[c] for c in df.columns if c in _LEGACY_GENE_COLUMNS}
    return df.rename(rename) if rename else df


def _read_pickle(path: Path) -> pl.DataFrame:
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, pl.DataFrame):
        return _normalize_cached_columns(obj)
    # Backward compatibility: old cache files stored pandas DataFrames.
    try:
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            return _normalize_cached_columns(pl.from_pandas(obj))
    except ImportError:
        pass
    raise TypeError(f"Pickle at {path} is not a DataFrame (got {type(obj).__name__})")


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


_MC_POOL_SESSION_CACHE: dict[tuple, Any] = {}


def get_mc_pool_from_control(
    *,
    locus: str = "TRB",
    species: str = "human",
    n: int = 10_000_000,
    seed: int = 42,
    skip_ends: int = 2,
    n_jobs: int | None = None,
) -> McPgenPool:
    """Get or build a synthetic McPgenPool, using the disk cache when available.

    Loads sequences from the ControlManager disk cache (if the synthetic control
    exists for the given ``(species, locus, n)``) and builds an
    :class:`~mir.basic.pgen.McPgenPool` from them.  If no disk cache exists,
    the control is generated via OLGA, saved to disk for future runs, and the
    pool is built from the result.  A session-level in-memory cache avoids
    rebuilding the pool within the same process.

    Supports any pool size including 100M (matching the original ALICE paper),
    provided the machine has sufficient RAM (~20 GB for 100M sequences on top
    of analysis workloads; 10M is the practical default for 32 GB machines).

    Args:
        locus: Receptor locus (e.g. ``"TRB"``).
        species: ``"human"`` or ``"mouse"``.
        n: Pool size (productive sequences).
        seed: OLGA generation seed.
        skip_ends: Terminal positions to skip for 1mm Pgen (default 2).
        n_jobs: Worker processes for pool generation (default: all CPUs).

    Returns:
        :class:`~mir.basic.pgen.McPgenPool` ready for ``pgen_1mm_bulk`` queries.
    """
    from mir.basic.pgen import get_p_productive

    species_c = ControlManager.canonical_species(species)
    locus_c = ControlManager.canonical_locus(locus)
    jobs = _resolve_n_jobs(n_jobs)
    key = (species_c, locus_c, n, seed, skip_ends)

    cached = _MC_POOL_SESSION_CACHE.get(key)
    if cached is not None:
        return cached

    mgr = ControlManager()
    df: pl.DataFrame | None = None
    try:
        df = mgr.load_control_df("synthetic", species_c, locus_c, n=n, wait_if_building=False)
    except Exception:
        pass

    if df is None:
        rec = mgr.ensure_synthetic_control(
            species_c, locus_c, n=n, n_jobs=jobs, seed=seed, progress=True,
        )
        df = _read_pickle(Path(rec.path))

    sequences = df["junction_aa"].to_list()
    p_prod = get_p_productive(locus_c, species_c)
    n_total = max(len(sequences), int(round(len(sequences) / p_prod)))
    pool = McPgenPool(sequences, n_total, skip_ends=skip_ends, locus=locus_c, species=species_c)
    _MC_POOL_SESSION_CACHE[key] = pool
    return pool


def control_setup_cli(argv: list[str] | None = None) -> int:
    """CLI for prebuilding/downloading controls.

    Example:
      mirpy-control-setup --type synthetic --species human,mouse --loci TRA,TRB --n 1000000
    """
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

    parser = argparse.ArgumentParser(description="Setup mirpy control/background data")
    parser.add_argument("--type", choices=["synthetic", "real"], required=False)
    parser.add_argument("--species", required=True, help="Comma-separated species aliases")
    parser.add_argument("--loci", required=True, help="Comma-separated loci aliases")
    parser.add_argument("--control-dir", default=None)
    parser.add_argument("--n", type=int, default=10_000_000, help="Synthetic sample size")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-repo", default=_DEFAULT_HF_DATASET)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None, help="Workers for synthetic generation (default: all CPUs)")
    parser.add_argument("--cleanup", action="store_true", help="Clean cache/manifest before setup")
    parser.add_argument("--refresh-real", action="store_true", help="Refresh existing real controls when HF snapshot changes")
    args = parser.parse_args(argv)

    mgr = ControlManager(args.control_dir)

    if args.cleanup:
        summary = mgr.cleanup_cache(cleanup_synthetic=True, cleanup_real=True, remove_orphans=True)
        print(f"cleanup: {summary}")

    if args.refresh_real:
        species_list = _parse_species_arg(args.species) if args.species else None
        locus_list = _parse_locus_arg(args.loci) if args.loci else None
        ref = mgr.refresh_real_controls(
            dataset_repo=args.dataset_repo,
            hf_cache_dir=args.hf_cache_dir,
            species=species_list,
            loci=locus_list,
        )
        print(f"refresh_real: {ref}")

    if args.type is None:
        print(f"manifest: {mgr.manifest_path}")
        return 0

    species_list = _parse_species_arg(args.species)
    locus_list = _parse_locus_arg(args.loci)

    for species in species_list:
        for locus in locus_list:
            if args.type == "synthetic":
                rec = mgr.ensure_synthetic_control(
                    species,
                    locus,
                    n=args.n,
                    n_jobs=args.n_jobs,
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
