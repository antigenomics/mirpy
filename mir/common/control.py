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
import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from mir import get_resource_path
from mir.basic.pgen import OlgaModel

_CONTROL_ENV = "MIRPY_CONTROL_DIR"
_MANIFEST_FILE = "manifest.json"
_DEFAULT_HF_DATASET = "isalgo/airr_control"

_SPECIES_ALIASES: dict[str, str] = {
    "human": "human",
    "hsa": "human",
    "homosapiens": "human",
    "homo_sapiens": "human",
    "homo sapiens": "human",
    "mouse": "mouse",
    "mmu": "mouse",
    "musmusculus": "mouse",
    "mus_musculus": "mouse",
    "mus musculus": "mouse",
}

# Canonical IMGT locus aliases used by users/notebooks/tools.
_LOCUS_ALIASES: dict[str, str] = {
    "TRA": "TRA",
    "TALPHA": "TRA",
    "T_ALPHA": "TRA",
    "T-ALPHA": "TRA",
    "TRA": "TRA",
    "ALPHA": "TRA",
    "TRB": "TRB",
    "TBETA": "TRB",
    "T_BETA": "TRB",
    "T-BETA": "TRB",
    "BETA": "TRB",
    "TRG": "TRG",
    "TGAMMA": "TRG",
    "T_GAMMA": "TRG",
    "T-GAMMA": "TRG",
    "GAMMA": "TRG",
    "TRD": "TRD",
    "TDELTA": "TRD",
    "T_DELTA": "TRD",
    "T-DELTA": "TRD",
    "DELTA": "TRD",
    "IGH": "IGH",
    "BHEAVY": "IGH",
    "B_HEAVY": "IGH",
    "B-HEAVY": "IGH",
    "HEAVY": "IGH",
    "IGK": "IGK",
    "BKAPPA": "IGK",
    "B_KAPPA": "IGK",
    "B-KAPPA": "IGK",
    "KAPPA": "IGK",
    "IGL": "IGL",
    "BLAMBDA": "IGL",
    "B_LAMBDA": "IGL",
    "B-LAMBDA": "IGL",
    "LAMBDA": "IGL",
}

_OLGA_SUFFIX_TO_LOCUS: dict[str, str] = {
    "T_ALPHA": "TRA",
    "T_BETA": "TRB",
    "T_GAMMA": "TRG",
    "T_DELTA": "TRD",
    "B_HEAVY": "IGH",
    "B_KAPPA": "IGK",
    "B_LAMBDA": "IGL",
}


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

    # ---------------------------
    # Canonicalization
    # ---------------------------
    @staticmethod
    def canonical_species(species: str) -> str:
        key = (species or "").strip().lower().replace("-", "_")
        if key not in _SPECIES_ALIASES:
            raise ValueError(f"Unsupported species alias: {species!r}")
        return _SPECIES_ALIASES[key]

    @staticmethod
    def canonical_locus(locus: str) -> str:
        key = (locus or "").strip().upper().replace("-", "_")
        key = key.replace(" ", "")
        if key not in _LOCUS_ALIASES:
            raise ValueError(f"Unsupported locus alias: {locus!r}")
        return _LOCUS_ALIASES[key]

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
        with self.manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    @staticmethod
    def _record_key(control_type: str, species: str, locus: str) -> str:
        return f"{control_type}:{species}:{locus}"

    def register_record(self, record: ControlRecord) -> None:
        manifest = self.load_manifest()
        key = self._record_key(record.control_type, record.species, record.locus)
        manifest["records"][key] = asdict(record)
        self.save_manifest(manifest)

    def get_record(self, control_type: str, species: str, locus: str) -> ControlRecord | None:
        manifest = self.load_manifest()
        key = self._record_key(control_type, species, locus)
        rec = manifest["records"].get(key)
        if rec is None:
            return None
        return ControlRecord(**rec)

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
            model_root = Path(get_resource_path("olga/default_models"))
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
            locus = _OLGA_SUFFIX_TO_LOCUS.get(suffix_key)
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

    def ensure_synthetic_control(
        self,
        species: str,
        locus: str,
        *,
        n: int = 10_000_000,
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
        if path.exists() and not overwrite:
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
        if path.exists() and not overwrite:
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

    def load_control_df(self, control_type: str, species: str, locus: str) -> pd.DataFrame:
        species_c = self.canonical_species(species)
        locus_c = self.canonical_locus(locus)
        rec = self.get_record(control_type, species_c, locus_c)
        if rec is None:
            raise FileNotFoundError(
                f"Control not registered: type={control_type!r}, species={species_c!r}, locus={locus_c!r}"
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
        self.ensure_control(control_type, species, locus, **kwargs)
        return self.load_control_df(control_type, species, locus)


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
) -> pd.DataFrame:
    """Generate synthetic OLGA control DataFrame with ntvj columns.

    Output columns:
    - junction
    - junction_aa
    - v_gene
    - j_gene
    """
    model = OlgaModel(species=species, locus=locus, seed=seed)
    generator: Callable[[], dict] = (
        model._gen_one_vdj_with_meta if model.is_d_present else model._gen_one_vj_with_meta
    )

    rows: list[dict[str, str]] = []
    generated = 0
    while generated < n:
        batch = min(chunk_size, n - generated)
        for _ in range(batch):
            rec = generator()
            rows.append(
                {
                    "junction": str(rec["junction"]),
                    "junction_aa": str(rec["junction_aa"]),
                    "v_gene": _normalize_allele(str(rec["v_gene"])),
                    "j_gene": _normalize_allele(str(rec["j_gene"])),
                }
            )
        generated += batch
        if progress and (generated % (chunk_size * 10) == 0 or generated == n):
            print(f"Generated synthetic control {species}/{locus}: {generated}/{n}")

    return pd.DataFrame.from_records(rows)


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

    required = ["junction", "junction_aa", "v_gene", "j_gene"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")

    out = df[required].copy()
    out["v_gene"] = out["v_gene"].astype(str).map(_normalize_allele)
    out["j_gene"] = out["j_gene"].astype(str).map(_normalize_allele)
    return out


def _normalize_allele(gene_name: str) -> str:
    s = (gene_name or "").strip()
    if not s:
        return s
    if "*" in s:
        return s
    return f"{s}*01"


def _find_real_control_file(snapshot_root: str | Path, species: str, locus: str) -> Path | None:
    root = Path(snapshot_root)
    if not root.exists():
        return None

    species_tokens = {
        species,
        {"human": "hsa", "mouse": "mmu"}.get(species, species),
    }
    locus_tokens = {
        locus.lower(),
        {
            "TRA": "talpha",
            "TRB": "tbeta",
            "TRG": "tgamma",
            "TRD": "tdelta",
            "IGH": "bheavy",
            "IGK": "bkappa",
            "IGL": "blambda",
        }.get(locus, locus.lower()),
    }

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

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return snapshot_download(**kwargs)


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
