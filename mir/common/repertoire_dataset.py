"""Dataset container for multiple sample repertoires with metadata.

This module introduces :class:`RepertoireDataset`, a lightweight holder for
:class:`~mir.common.repertoire.SampleRepertoire` objects keyed by ``sample_id``
and accompanied by per-sample metadata.
"""

from __future__ import annotations

import gc
import gzip
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import polars as pl

from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import SampleRepertoire, infer_locus

logger = logging.getLogger(__name__)


def _parse_sample_task(
    args: tuple,
) -> tuple | None:
    """Parse one sample file in a worker thread.

    Returns (sid, SampleRepertoire, rec) or None when below min_duplicate_count.
    Threads share the parser object and return objects directly (no serialization).
    Polars I/O and column extraction release the GIL, so threads parallelize well.
    """
    sid, path_str, rec, parser, min_dup = args
    # Fast path: polars-based parser returns col_groups (no Clonotype construction).
    # Clonotype objects are materialised lazily in the main thread on first access.
    if hasattr(parser, '_polars_to_col_groups'):
        df = parser._read_polars(path_str)
        if min_dup > 0 and 'duplicate_count' in df.columns:
            import polars as _pl
            total = df['duplicate_count'].cast(_pl.Int64).fill_null(1).sum() or 0
            if total < min_dup:
                return None
        col_groups = parser._polars_to_col_groups(df)
        return sid, col_groups, rec
    # Legacy path for non-polars parsers.
    from mir.common.repertoire import SampleRepertoire
    clonotypes = parser.parse(path_str)
    if min_dup > 0 and sum(c.duplicate_count for c in clonotypes) < min_dup:
        return None
    srep = SampleRepertoire.from_clonotypes(clonotypes, sample_id=sid, sample_metadata=rec)
    srep.sample_id = sid
    return sid, srep, rec


class RepertoireDataset:
    """Collection of :class:`SampleRepertoire` objects and sample metadata.

    Parameters
    ----------
    samples
        Mapping ``sample_id -> SampleRepertoire``.
    metadata
        Mapping ``sample_id -> metadata dict``. Each metadata record must
        contain ``sample_id``.
    """

    def __init__(
        self,
        samples: dict[str, SampleRepertoire],
        metadata: dict[str, dict] | None = None,
    ) -> None:
        self.samples: dict[str, SampleRepertoire] = dict(samples)
        self.metadata: dict[str, dict] = {}

        metadata = metadata or {}
        for sample_id, sample in self.samples.items():
            # Start from whatever the sample already carries, then layer on
            # any caller-supplied overrides (caller takes precedence).
            record: dict = dict(sample.sample_metadata) if sample.sample_metadata else {}
            record.update(metadata.get(sample_id, {}))
            record["sample_id"] = sample_id
            self.metadata[sample_id] = record
            # Sync both fields back so sample and dataset stay consistent.
            sample.sample_id = sample_id
            sample.sample_metadata = record

    @property
    def metadata_df(self) -> pd.DataFrame:
        """Return metadata as a DataFrame indexed by ``sample_id``."""
        if not self.metadata:
            return pd.DataFrame(columns=["sample_id"]).set_index("sample_id")
        return pd.DataFrame(self.metadata.values()).set_index("sample_id", drop=False)

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        parser: ClonotypeTableParser | None = None,
        metadata_file: str = "metadata.tsv",
        file_name_to_sample_id: Callable[[str], str] | None = None,
        metadata_sep: str = "\t",
    ) -> "RepertoireDataset":
        """Load a dataset from a folder containing sample files and metadata.

        The metadata table must contain a ``sample_id`` column and either:
        - a ``file_name`` column with relative paths to sample files, or
        - a ``file_name_to_sample_id`` callback that maps ``sample_id`` to a
          relative file path.
        """
        base = Path(folder)
        meta_path = base / metadata_file
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata file not found: {meta_path}")

        metadata_df = pd.read_csv(meta_path, sep=metadata_sep)
        if "sample_id" not in metadata_df.columns:
            raise ValueError("metadata must contain a 'sample_id' column")

        has_file_name = "file_name" in metadata_df.columns
        if not has_file_name and file_name_to_sample_id is None:
            raise ValueError(
                "metadata must contain 'file_name' column or file_name_to_sample_id must be provided"
            )

        parser = parser or ClonotypeTableParser()
        samples: dict[str, SampleRepertoire] = {}
        metadata: dict[str, dict] = {}

        for _, row in metadata_df.iterrows():
            sample_id = str(row["sample_id"])
            rel_path = str(row["file_name"]) if has_file_name else str(file_name_to_sample_id(sample_id))
            sample_path = base / rel_path
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"sample file for sample_id={sample_id!r} not found: {sample_path}"
                )

            clonotypes = parser.parse(str(sample_path))
            for c in clonotypes:
                if not c.locus:
                    c.locus = infer_locus(c.j_gene or c.v_gene or "")

            sample_rep = SampleRepertoire.from_clonotypes(
                clonotypes,
                sample_id=sample_id,
                sample_metadata=row.to_dict(),
            )
            sample_rep.sample_id = sample_id

            samples[sample_id] = sample_rep
            rec = row.to_dict()
            rec["sample_id"] = sample_id
            if not has_file_name:
                rec["file_name"] = rel_path
            metadata[sample_id] = rec

        return cls(samples=samples, metadata=metadata)

    @classmethod
    def from_folder_polars(
        cls,
        folder: str | Path,
        *,
        parser: ClonotypeTableParser | None = None,
        metadata_file: str = "metadata.tsv",
        file_name_column: str = "file_name",
        sample_id_column: str = "sample_id",
        file_name_to_sample_id: Callable[[str], str] | None = None,
        metadata_sep: str = "\t",
        skip_missing_files: bool = True,
        min_duplicate_count: int = 0,
        n_workers: int = 6,
        progress: bool = True,
        progress_every: int = 100,
    ) -> "RepertoireDataset":
        """Load dataset from metadata with polars and build SampleRepertoire objects.

        Parameters
        ----------
        folder
            Root directory containing sample files and the metadata table.
        parser
            Parser to use; defaults to :class:`~mir.common.parser.ClonotypeTableParser`.
        metadata_file
            Name of the metadata TSV file inside *folder*.
        file_name_column, sample_id_column
            Column names in the metadata table.
        file_name_to_sample_id
            Optional callable mapping ``sample_id`` → relative file path when
            ``file_name_column`` is absent.
        metadata_sep
            Separator for the metadata file.
        skip_missing_files
            Skip rows whose sample file does not exist on disk.
        min_duplicate_count
            Samples whose total ``duplicate_count`` summed over all clonotypes
            is strictly below this threshold are silently skipped.
        n_workers
            Number of parallel threads for reading sample files (default: 6).
        progress
            Print periodic progress messages to stdout (with timestamps).
        progress_every
            Report progress every this many completed samples (default: 100).
        """
        base = Path(folder)
        meta_path = base / metadata_file
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata file not found: {meta_path}")

        metadata_df = pl.read_csv(meta_path, separator=metadata_sep, infer_schema_length=10_000)
        cols = set(metadata_df.columns)
        if sample_id_column not in cols:
            raise ValueError(f"metadata must contain a {sample_id_column!r} column")

        has_file_name = file_name_column in cols
        if not has_file_name and file_name_to_sample_id is None:
            raise ValueError(
                f"metadata must contain {file_name_column!r} column or "
                "file_name_to_sample_id must be provided"
            )

        _parser = parser or ClonotypeTableParser()

        # Build the work list: (sample_id, file_path, metadata_dict)
        tasks: list[tuple[str, Path, dict]] = []
        for row in metadata_df.iter_rows(named=True):
            sample_id = str(row[sample_id_column])
            rel_path = (
                str(row[file_name_column]) if has_file_name
                else str(file_name_to_sample_id(sample_id))  # type: ignore[misc]
            )
            sample_path = base / rel_path
            if not sample_path.exists():
                if skip_missing_files:
                    logger.debug("skipping missing file for %s: %s", sample_id, sample_path)
                    continue
                raise FileNotFoundError(
                    f"sample file for sample_id={sample_id!r} not found: {sample_path}"
                )
            rec = dict(row)
            rec["sample_id"] = sample_id
            if not has_file_name:
                rec[file_name_column] = rel_path
            tasks.append((sample_id, sample_path, rec))

        total = len(tasks)
        _t0 = time.perf_counter()
        if progress:
            print(f"[{time.strftime('%H:%M:%S')}] Loading {total} samples using {n_workers} worker(s)…",
                  flush=True)

        samples: dict[str, SampleRepertoire] = {}
        metadata: dict[str, dict] = {}

        # Build task args: parser is shared across threads (read-only use, thread-safe)
        task_args = [(sid, str(path), rec, _parser, min_duplicate_count)
                     for sid, path, rec in tasks]

        # ThreadPoolExecutor: polars read/parse releases the GIL so threads
        # run truly in parallel for I/O.  Objects are returned by reference
        # (shared heap) with zero serialization cost.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            done = 0
            skipped = 0
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_parse_sample_task, t): t[0] for t in task_args}
                for fut in as_completed(futures):
                    result = fut.result()
                    done += 1
                    if result is None:
                        skipped += 1
                    else:
                        sid, payload, rec = result
                        if isinstance(payload, dict):
                            # col_groups from fast polars path — build lazily
                            from mir.common.repertoire import SampleRepertoire as _SR, LocusRepertoire as _LR
                            loci = {
                                loc: _LR._from_lazy_cols(loc, cols)
                                for loc, cols in payload.items()
                            }
                            srep = _SR(loci=loci, sample_id=sid, sample_metadata=rec)
                        else:
                            srep = payload  # legacy SampleRepertoire
                        srep.sample_id = sid
                        samples[sid] = srep
                        metadata[sid] = rec
                    if progress and done % progress_every == 0:
                        elapsed = time.perf_counter() - _t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta_s = (total - done) / rate if rate > 0 else float('inf')
                        eta_str = f"{eta_s:.0f}s" if eta_s < 3600 else "--"
                        print(
                            f"[{time.strftime('%H:%M:%S')}]  {done}/{total}"
                            + (f"  ({skipped} below threshold)" if skipped else "")
                            + f"  {rate:.1f} samples/s  ETA {eta_str}",
                            flush=True,
                        )
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()

        if progress:
            elapsed = time.perf_counter() - _t0
            n_loaded = len(samples)
            rate = n_loaded / elapsed if elapsed > 0 else 0
            print(
                f"[{time.strftime('%H:%M:%S')}] Done: {n_loaded} samples in {elapsed:.1f}s"
                + f"  ({rate:.1f} samples/s)"
                + (f"  {skipped} skipped (< {min_duplicate_count} duplicates)" if skipped else "")
                + ".",
                flush=True,
            )

        return cls(samples=samples, metadata=metadata)

    def __iter__(self) -> Iterator[SampleRepertoire]:
        return iter(self.samples.values())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_id: str) -> SampleRepertoire:
        return self.samples[sample_id]

    def to_pickle(self, path: str | Path) -> Path:
        """Serialize dataset to pickle and return output path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    @classmethod
    def from_pickle(cls, path: str | Path) -> "RepertoireDataset":
        """Load dataset from pickle.

        Warning:
            ``pickle`` is not secure against untrusted input.
        """
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected pickled {cls.__name__}, got {type(obj).__name__}")
        return obj

    def write_folder(
        self,
        folder: str | Path,
        *,
        format: str = "tsv",
        gzip_output: bool = False,
        metadata_file: str = "metadata.tsv",
    ) -> Path:
        """Write dataset as per-sample repertoire files plus aggregated metadata.

        A ``file_name`` column is added/overwritten in metadata and saved as TSV.
        """
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: list[dict] = []
        for sample_id, sample in self.samples.items():
            for locus, lr in sample.loci.items():
                base_name = f"{sample_id}_{locus}.{format}"
                rel_name = base_name + (".gz" if gzip_output and format in {"tsv", "csv"} else "")
                out_path = out_dir / rel_name

                if format == "parquet":
                    lr.write_polars(out_path, format="parquet")
                elif format in {"ipc", "feather"}:
                    lr.write_polars(out_path, format="ipc")
                elif format in {"tsv", "csv"}:
                    df = lr.to_polars().to_pandas()
                    sep = "\t" if format == "tsv" else ","
                    if gzip_output:
                        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
                            df.to_csv(fh, sep=sep, index=False)
                    else:
                        df.to_csv(out_path, sep=sep, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format!r}")

                row = dict(self.metadata.get(sample_id, sample.sample_metadata or {}))
                row["sample_id"] = sample_id
                row["locus"] = locus
                row["file_name"] = rel_name
                metadata_rows.append(row)

        meta_df = pd.DataFrame(metadata_rows)
        meta_path = out_dir / metadata_file
        meta_df.to_csv(meta_path, sep="\t", index=False)
        return meta_path
