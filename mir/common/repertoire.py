"""Repertoire data structures.

:class:`LocusRepertoire` — single-locus immune repertoire.
:class:`SampleRepertoire` — multi-locus collection for one biological sample.
``Repertoire`` is a backward-compatible alias for :class:`LocusRepertoire`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import polars as pl

from mir.common.clonotype import Clonotype


_GENE_PREFIX_TO_LOCUS: dict[str, str] = {
    "TRA": "TRA", "TRB": "TRB", "TRG": "TRG", "TRD": "TRD",
    "IGH": "IGH", "IGK": "IGK", "IGL": "IGL",
}


def infer_locus(gene_name: str) -> str:
    """Return the locus code for a gene name (e.g. ``'TRBV6-9'`` → ``'TRB'``)."""
    prefix = gene_name[:3].upper()
    return _GENE_PREFIX_TO_LOCUS.get(prefix, "")


class LocusRepertoire:
    """Single-locus immune repertoire.

    Parameters
    ----------
    clonotypes:
        List of :class:`~mir.common.clonotype.Clonotype` objects.  All
        clonotypes that carry a non-empty ``locus`` field must match
        *locus* (if provided).
    locus:
        IMGT locus code (``"TRB"``, ``"TRA"``, …).  Empty string means
        the locus is unspecified.
    repertoire_id:
        Free-form identifier for this repertoire (e.g. a sample accession).
    repertoire_metadata:
        Arbitrary key/value metadata dict.
    """

    def __init__(
        self,
        clonotypes: list[Clonotype],
        locus: str = "",
        repertoire_id: str = "",
        repertoire_metadata: dict | None = None,
        # ---- legacy params (still accepted for backward compat) ----
        is_sorted: bool = False,   # ignored — is_sorted is now a computed property
        metadata=None,             # mapped to repertoire_metadata
        gene: str | None = None,   # mapped to locus
    ) -> None:
        # Handle legacy constructor params
        if metadata is not None and repertoire_metadata is None:
            repertoire_metadata = dict(metadata) if metadata is not None else {}
        if gene is not None and not locus:
            locus = gene

        if locus:
            bad = {c.locus for c in clonotypes if c.locus and c.locus != locus}
            if bad:
                raise ValueError(
                    f"Clonotypes have loci {bad!r} inconsistent with "
                    f"repertoire locus {locus!r}"
                )
        self.clonotypes = list(clonotypes)
        self.locus = locus
        self.repertoire_id = repertoire_id
        self.repertoire_metadata: dict = repertoire_metadata if repertoire_metadata is not None else {}

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    @property
    def clonotype_count(self) -> int:
        """Number of distinct clonotypes."""
        return len(self.clonotypes)

    @property
    def duplicate_count(self) -> int:
        """Total read/cell count across all clonotypes."""
        return sum(c.duplicate_count for c in self.clonotypes)

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @property
    def is_sorted(self) -> bool:
        """True iff clonotypes are ordered by decreasing duplicate_count."""
        return all(
            a.duplicate_count >= b.duplicate_count
            for a, b in zip(self.clonotypes, self.clonotypes[1:])
        )

    def sort(self) -> None:
        """Sort clonotypes by duplicate_count descending (in-place)."""
        self.clonotypes.sort(key=lambda c: c.duplicate_count, reverse=True)

    def top(self, n: int = 100) -> list[Clonotype]:
        """Return the *n* most abundant clonotypes, sorting first if needed."""
        if not self.is_sorted:
            self.sort()
        return self.clonotypes[:n]

    # ------------------------------------------------------------------
    # Polars I/O
    # ------------------------------------------------------------------

    def to_polars(self) -> pl.DataFrame:
        """Serialise clonotypes to a Polars DataFrame (AIRR schema)."""
        return Clonotype.to_polars(self.clonotypes)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        locus: str = "",
        repertoire_id: str = "",
        repertoire_metadata: dict | None = None,
    ) -> LocusRepertoire:
        """Deserialise a Polars DataFrame (AIRR schema) into a :class:`LocusRepertoire`."""
        return cls(
            clonotypes=Clonotype.from_polars(df),
            locus=locus,
            repertoire_id=repertoire_id,
            repertoire_metadata=repertoire_metadata,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def subsample_functional(self) -> LocusRepertoire:
        """Keep only coding clonotypes (no stop codons / non-standard AA)."""
        return LocusRepertoire(
            [x for x in self.clonotypes if x.is_coding()],
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def subsample_nonfunctional(self) -> LocusRepertoire:
        """Keep only non-coding clonotypes."""
        return LocusRepertoire(
            [x for x in self.clonotypes if not x.is_coding()],
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def subsample_by_lambda(self, function) -> LocusRepertoire:
        return LocusRepertoire(
            [x for x in self.clonotypes if function(x)],
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def sample_n(self, n: int = 100, sample_random: bool = False, random_seed: int = 42) -> LocusRepertoire:
        if n is None:
            return self
        random.seed(random_seed)
        selected = random.sample(self.clonotypes, n) if sample_random else self.clonotypes[:n]
        return LocusRepertoire(selected, locus=self.locus, repertoire_metadata=dict(self.repertoire_metadata))

    def subtract_background(
        self,
        other: LocusRepertoire,
        odds_ratio_threshold: float = 2,
        compare_by=lambda x: (x.junction_aa, str(x.v_gene)),
    ) -> LocusRepertoire:
        pre = {compare_by(x) for x in other.clonotypes}
        post = {compare_by(x) for x in self.clonotypes}
        shared = pre & post
        old_freq = {
            compare_by(c): c.duplicate_count / other.duplicate_count
            for c in other if compare_by(c) in shared
        }
        result = []
        for c in self:
            key = compare_by(c)
            if key in shared:
                if c.duplicate_count / self.duplicate_count / old_freq[key] > odds_ratio_threshold:
                    result.append(c)
            else:
                result.append(c)
        return LocusRepertoire(result, locus=self.locus, repertoire_metadata=dict(self.repertoire_metadata))

    @property
    def evaluate_segment_usage(self) -> dict:
        usage: dict[str, int] = defaultdict(int)
        for c in self.clonotypes:
            usage[c.v_gene] += 1
            usage[c.j_gene] += 1
            if c.d_gene:
                usage[c.d_gene] += 1
        return dict(usage)

    def make_chunks(self, number_of_chunks: int, save_path=None) -> list:
        """Split into *number_of_chunks* sub-repertoires (or pickle files)."""
        import os, pickle
        if number_of_chunks < 1:
            raise ValueError("number_of_chunks must be at least 1")
        chunk_size = (len(self.clonotypes) + number_of_chunks - 1) // number_of_chunks
        chunks = []
        for i in range(number_of_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(self.clonotypes))
            chunk = LocusRepertoire(self.clonotypes[start:end], locus=self.locus)
            if save_path:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                fpath = os.path.join(save_path, f"chunk_{i}.pkl")
                with open(fpath, "wb") as f:
                    pickle.dump(chunk, f)
                chunks.append(fpath)
            else:
                chunks.append(chunk)
        return chunks

    def serialize(self):
        """Serialise to a pandas DataFrame (legacy helper)."""
        import pandas as pd
        from collections import defaultdict as _dd
        d: dict[str, list] = _dd(list)
        for c in self.clonotypes:
            for k, v in c.serialize().items():
                d[k].append(v)
        return pd.DataFrame(d, index=[c.id for c in self.clonotypes])

    # ------------------------------------------------------------------
    # Backward-compat properties / attributes
    # ------------------------------------------------------------------

    @property
    def number_of_clones(self) -> int:
        return self.clonotype_count

    @property
    def number_of_reads(self) -> int:
        return self.duplicate_count

    @property
    def total(self) -> int:
        return self.duplicate_count

    @property
    def metadata(self) -> dict:
        return self.repertoire_metadata

    @metadata.setter
    def metadata(self, value) -> None:
        self.repertoire_metadata = dict(value) if value is not None else {}

    @property
    def gene(self) -> str:
        return self.locus

    @property
    def sorted(self) -> bool:
        return self.is_sorted

    @sorted.setter
    def sorted(self, value: bool) -> None:
        pass  # is_sorted is computed; setter is a no-op for backward compat

    @property
    def segment_usage(self):
        return None  # legacy attribute; use evaluate_segment_usage property

    @segment_usage.setter
    def segment_usage(self, value) -> None:
        pass

    # ------------------------------------------------------------------
    # Trie property (kept for downstream code)
    # ------------------------------------------------------------------

    @property
    def trie(self):
        from tcrtrie import Trie
        if not hasattr(self, '_trie'):
            self._trie = Trie(
                sequences=[str(c.junction_aa) for c in self.clonotypes],
                vGenes=[str(c.v_gene) for c in self.clonotypes],
                jGenes=[str(c.j_gene) for c in self.clonotypes],
            )
        return self._trie

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.clonotype_count

    def __iter__(self) -> Iterator[Clonotype]:
        return iter(self.clonotypes)

    def __getitem__(self, idx):
        return self.clonotypes[idx]

    def __add__(self, other: LocusRepertoire) -> LocusRepertoire:
        if not isinstance(other, LocusRepertoire):
            raise TypeError(f"Cannot add LocusRepertoire and {type(other).__name__}")
        if self.locus and other.locus and self.locus != other.locus:
            raise ValueError(
                f"Cannot merge repertoires with different loci: "
                f"{self.locus!r} and {other.locus!r}"
            )
        merged_meta = {**self.repertoire_metadata, **other.repertoire_metadata}
        return LocusRepertoire(
            self.clonotypes + other.clonotypes,
            locus=self.locus or other.locus,
            repertoire_metadata=merged_meta,
        )

    def __str__(self) -> str:
        return (
            f"LocusRepertoire(locus={self.locus!r}, "
            f"clonotypes={self.clonotype_count}, "
            f"duplicate_count={self.duplicate_count})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __copy__(self) -> LocusRepertoire:
        return LocusRepertoire(
            list(self.clonotypes),
            locus=self.locus,
            repertoire_id=self.repertoire_id,
            repertoire_metadata=dict(self.repertoire_metadata),
        )


# ---------------------------------------------------------------------------
# SampleRepertoire
# ---------------------------------------------------------------------------

class SampleRepertoire:
    """Multi-locus immune repertoire for a single biological sample.

    Parameters
    ----------
    loci:
        Dict mapping locus code → :class:`LocusRepertoire`.
    sample_id:
        Sample identifier (e.g. SRX accession).
    sample_metadata:
        Arbitrary key/value metadata dict.
    """

    def __init__(
        self,
        loci: dict[str, LocusRepertoire],
        sample_id: str = "",
        sample_metadata: dict | None = None,
    ) -> None:
        self.loci = dict(loci)
        self.sample_id = sample_id
        self.sample_metadata: dict = sample_metadata if sample_metadata is not None else {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_clonotypes(
        cls,
        clonotypes: list[Clonotype],
        sample_id: str = "",
        sample_metadata: dict | None = None,
    ) -> SampleRepertoire:
        """Build a :class:`SampleRepertoire` by grouping *clonotypes* by locus."""
        groups: dict[str, list[Clonotype]] = defaultdict(list)
        for c in clonotypes:
            groups[c.locus].append(c)
        loci = {
            locus: LocusRepertoire(clones, locus=locus)
            for locus, clones in groups.items()
        }
        return cls(loci=loci, sample_id=sample_id, sample_metadata=sample_metadata)

    # ------------------------------------------------------------------
    # Aggregated clonotype access
    # ------------------------------------------------------------------

    @property
    def clonotypes(self) -> list[Clonotype]:
        """All clonotypes across all loci (concatenated)."""
        result: list[Clonotype] = []
        for lr in self.loci.values():
            result.extend(lr.clonotypes)
        return result

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @property
    def is_sorted(self) -> bool:
        """True iff every per-locus repertoire is sorted."""
        return all(lr.is_sorted for lr in self.loci.values())

    def sort(self) -> None:
        """Sort every per-locus repertoire by duplicate_count descending."""
        for lr in self.loci.values():
            lr.sort()

    # ------------------------------------------------------------------
    # Polars I/O
    # ------------------------------------------------------------------

    def to_polars(self) -> pl.DataFrame:
        """Concatenate all per-locus DataFrames into one."""
        frames = [lr.to_polars() for lr in self.loci.values() if lr.clonotype_count]
        if not frames:
            return Clonotype.to_polars([])
        return pl.concat(frames)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        locus_column: str = "locus",
        sample_id: str = "",
        sample_metadata: dict | None = None,
    ) -> SampleRepertoire:
        """Deserialise a Polars DataFrame, grouping rows by *locus_column*."""
        loci: dict[str, LocusRepertoire] = {}
        if locus_column in df.columns:
            for locus_val in df[locus_column].unique().to_list():
                locus_str = str(locus_val) if locus_val is not None else ""
                group = df.filter(pl.col(locus_column) == locus_val)
                loci[locus_str] = LocusRepertoire.from_polars(group, locus=locus_str)
        else:
            loci[""] = LocusRepertoire.from_polars(df)
        return cls(loci=loci, sample_id=sample_id, sample_metadata=sample_metadata)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __getitem__(self, locus: str) -> LocusRepertoire:
        return self.loci[locus]

    def __contains__(self, locus: str) -> bool:
        return locus in self.loci

    def __iter__(self) -> Iterator[LocusRepertoire]:
        return iter(self.loci.values())

    def __len__(self) -> int:
        return len(self.loci)

    def __str__(self) -> str:
        parts = ", ".join(
            f"{k}: {v.clonotype_count}" for k, v in self.loci.items()
        )
        return f"SampleRepertoire(id={self.sample_id!r}, loci={{{parts}}})"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

Repertoire = LocusRepertoire
