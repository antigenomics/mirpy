"""Dataset container for multiple sample repertoires with metadata.

This module introduces :class:`RepertoireDataset`, a lightweight holder for
:class:`~mir.common.repertoire.SampleRepertoire` objects keyed by ``sample_id``
and accompanied by per-sample metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import polars as pl

from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import SampleRepertoire, infer_locus


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
    ) -> "RepertoireDataset":
        """Load dataset from metadata with polars and build SampleRepertoire objects.

        This variant is intended for large cohorts where metadata parsing with
        polars is preferable. Missing sample files can be skipped.
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
                f"metadata must contain {file_name_column!r} column or file_name_to_sample_id must be provided"
            )

        parser = parser or ClonotypeTableParser()
        samples: dict[str, SampleRepertoire] = {}
        metadata: dict[str, dict] = {}

        for row in metadata_df.iter_rows(named=True):
            sample_id = str(row[sample_id_column])
            rel_path = str(row[file_name_column]) if has_file_name else str(file_name_to_sample_id(sample_id))
            sample_path = base / rel_path

            if not sample_path.exists():
                if skip_missing_files:
                    continue
                raise FileNotFoundError(
                    f"sample file for sample_id={sample_id!r} not found: {sample_path}"
                )

            clonotypes = parser.parse(str(sample_path))
            for c in clonotypes:
                if not c.locus:
                    c.locus = infer_locus(c.j_gene or c.v_gene or "")

            rec = dict(row)
            rec["sample_id"] = sample_id
            if not has_file_name:
                rec[file_name_column] = rel_path

            sample_rep = SampleRepertoire.from_clonotypes(
                clonotypes,
                sample_id=sample_id,
                sample_metadata=rec,
            )
            sample_rep.sample_id = sample_id
            samples[sample_id] = sample_rep
            metadata[sample_id] = rec

        return cls(samples=samples, metadata=metadata)

    def __iter__(self) -> Iterator[SampleRepertoire]:
        return iter(self.samples.values())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_id: str) -> SampleRepertoire:
        return self.samples[sample_id]
