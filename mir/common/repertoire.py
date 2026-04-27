"""Repertoire data structures.

:class:`LocusRepertoire` — single-locus immune repertoire.
:class:`SampleRepertoire` — multi-locus collection for one biological sample.
``Repertoire`` is a backward-compatible alias for :class:`LocusRepertoire`.
"""

from __future__ import annotations

import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import polars as pl

from mir.common.clonotype import Clonotype


_GENE_PREFIX_TO_LOCUS: dict[str, str] = {
    "TRA": "TRA", "TRB": "TRB", "TRG": "TRG", "TRD": "TRD",
    "IGH": "IGH", "IGK": "IGK", "IGL": "IGL",
}


def infer_locus(gene_name: str) -> str:
    """Return the locus code inferred from a gene name.

    Uses the three-letter IMGT prefix (``TRBV…`` → ``"TRB"``,
    ``TRAV…`` → ``"TRA"``, ``IGHV…`` → ``"IGH"``, …).
    Returns an empty string when the prefix is unrecognised.

    Examples
    --------
    >>> infer_locus("TRBV6-9")
    'TRB'
    >>> infer_locus("unknown")
    ''
    """
    return _GENE_PREFIX_TO_LOCUS.get(gene_name[:3].upper(), "")


class LocusRepertoire:
    """Single-locus immune repertoire.

    Stores a list of :class:`~mir.common.clonotype.Clonotype` objects that all
    belong to the same IMGT locus (TRA, TRB, IGH, …).  Counts are computed on
    demand; the list is kept in the order supplied unless :meth:`sort` is called.

    Parameters
    ----------
    clonotypes:
        Clonotype list.  Any clonotype whose ``locus`` field is non-empty must
        match *locus*; otherwise a :exc:`ValueError` is raised.
    locus:
        IMGT locus code (``"TRB"``, ``"TRA"``, …).  An empty string means
        the locus is unspecified and no locus consistency check is performed.
    repertoire_id:
        Free-form identifier for this repertoire (e.g. a sample accession).
    repertoire_metadata:
        Arbitrary key/value metadata dict.

    Notes
    -----
    ``metadata``, ``gene``, and ``is_sorted`` are accepted as legacy keyword
    arguments and silently mapped to ``repertoire_metadata``, ``locus``, and
    ignored, respectively, to preserve backward compatibility.
    """

    def __init__(
        self,
        clonotypes: list[Clonotype],
        locus: str = "",
        repertoire_id: str = "",
        repertoire_metadata: dict | None = None,
        # ---- backward-compat params ----
        is_sorted: bool = False,    # ignored: is_sorted is now a computed property
        metadata=None,              # mapped to repertoire_metadata
        gene: str | None = None,    # mapped to locus
    ) -> None:
        if metadata is not None and repertoire_metadata is None:
            repertoire_metadata = dict(metadata)
        if gene is not None and not locus:
            locus = gene

        if locus:
            bad = {c.locus for c in clonotypes if c.locus and c.locus != locus}
            if bad:
                raise ValueError(
                    f"Clonotypes have loci {bad!r} inconsistent with "
                    f"repertoire locus {locus!r}"
                )

        self.clonotypes: list[Clonotype] = list(clonotypes)
        self.locus: str = locus
        self.repertoire_id: str = repertoire_id
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
        """Total read/cell count (sum of :attr:`Clonotype.duplicate_count`)."""
        return sum(c.duplicate_count for c in self.clonotypes)

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @property
    def is_sorted(self) -> bool:
        """True iff clonotypes are in non-increasing ``duplicate_count`` order."""
        return all(
            a.duplicate_count >= b.duplicate_count
            for a, b in zip(self.clonotypes, self.clonotypes[1:])
        )

    def sort(self) -> None:
        """Sort clonotypes by ``duplicate_count`` descending, in-place."""
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
        """Serialise clonotypes to a Polars DataFrame using the AIRR schema."""
        return Clonotype.to_polars(self.clonotypes)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        locus: str = "",
        repertoire_id: str = "",
        repertoire_metadata: dict | None = None,
    ) -> LocusRepertoire:
        """Deserialise a Polars DataFrame (AIRR schema) into a :class:`LocusRepertoire`.

        Parameters
        ----------
        df:
            DataFrame with AIRR-schema column names (see
            :attr:`Clonotype._POLARS_SCHEMA`).
        locus:
            Locus code to assign to the resulting repertoire.
        repertoire_id:
            Identifier for the resulting repertoire.
        repertoire_metadata:
            Metadata dict for the resulting repertoire.
        """
        return cls(
            clonotypes=Clonotype.from_polars(df),
            locus=locus,
            repertoire_id=repertoire_id,
            repertoire_metadata=repertoire_metadata,
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def subsample_functional(self) -> LocusRepertoire:
        """Return a new repertoire keeping only coding clonotypes."""
        return LocusRepertoire(
            [x for x in self.clonotypes if x.is_coding()],
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def subsample_nonfunctional(self) -> LocusRepertoire:
        """Return a new repertoire keeping only non-coding clonotypes."""
        return LocusRepertoire(
            [x for x in self.clonotypes if not x.is_coding()],
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def subsample_by_lambda(
        self,
        function: Callable[[Clonotype], bool],
    ) -> LocusRepertoire:
        """Return a new repertoire keeping clonotypes that satisfy *function*.

        Parameters
        ----------
        function:
            Predicate called on each :class:`Clonotype`; clonotypes for which
            it returns ``True`` are retained.
        """
        return LocusRepertoire(
            [x for x in self.clonotypes if function(x)],
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def sample_n(
        self,
        n: int | None = None,
        sample_random: bool = False,
        random_seed: int = 42,
    ) -> LocusRepertoire:
        """Return a sub-repertoire of *n* clonotypes.

        Parameters
        ----------
        n:
            Number of clonotypes to keep.  ``None`` returns ``self`` unchanged.
        sample_random:
            When ``True``, draw *n* clonotypes at random instead of taking the
            first *n*.
        random_seed:
            Seed for the random sampler (ignored when ``sample_random=False``).
        """
        if n is None:
            return self
        random.seed(random_seed)
        selected = (
            random.sample(self.clonotypes, n)
            if sample_random
            else self.clonotypes[:n]
        )
        return LocusRepertoire(
            selected,
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def subtract_background(
        self,
        other: LocusRepertoire,
        odds_ratio_threshold: float = 2.0,
        compare_by: Callable[[Clonotype], object] = lambda x: (x.junction_aa, x.v_gene),
    ) -> LocusRepertoire:
        """Remove clonotypes whose fold-enrichment over *other* is below threshold.

        Clonotypes shared with *other* are kept only if their frequency in
        ``self`` is at least *odds_ratio_threshold* times their frequency in
        *other*.  Clonotypes absent from *other* are always kept.

        Parameters
        ----------
        other:
            Background (control) repertoire.
        odds_ratio_threshold:
            Minimum fold-change to retain a shared clonotype.
        compare_by:
            Key function used to match clonotypes between repertoires.
        """
        pre    = {compare_by(x) for x in other.clonotypes}
        shared = {compare_by(x) for x in self.clonotypes} & pre
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
        return LocusRepertoire(
            result,
            locus=self.locus,
            repertoire_metadata=dict(self.repertoire_metadata),
        )

    def segment_usage(self) -> dict[str, int]:
        """Return a dict counting occurrences of each V/D/J gene segment."""
        usage: dict[str, int] = defaultdict(int)
        for c in self.clonotypes:
            usage[c.v_gene] += 1
            usage[c.j_gene] += 1
            if c.d_gene:
                usage[c.d_gene] += 1
        return dict(usage)

    def make_chunks(
        self,
        number_of_chunks: int,
        save_path: str | Path | None = None,
    ) -> list[LocusRepertoire | str]:
        """Split into *number_of_chunks* contiguous sub-repertoires.

        Parameters
        ----------
        number_of_chunks:
            How many chunks to produce.  Must be ≥ 1 and ≤ ``clonotype_count``.
        save_path:
            When provided, each chunk is pickled under ``save_path/chunk_<i>.pkl``
            and the list of file paths is returned instead of
            :class:`LocusRepertoire` objects.

        Returns
        -------
        list
            Either a list of :class:`LocusRepertoire` objects (``save_path=None``)
            or a list of path strings when ``save_path`` is given.
        """
        if number_of_chunks < 1:
            raise ValueError("number_of_chunks must be at least 1")
        chunk_size = (len(self.clonotypes) + number_of_chunks - 1) // number_of_chunks
        chunks: list[LocusRepertoire | str] = []
        for i in range(number_of_chunks):
            start = i * chunk_size
            end   = min((i + 1) * chunk_size, len(self.clonotypes))
            chunk = LocusRepertoire(self.clonotypes[start:end], locus=self.locus)
            if save_path is not None:
                out_dir = Path(save_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                fpath = out_dir / f"chunk_{i}.pkl"
                with fpath.open("wb") as fh:
                    pickle.dump(chunk, fh)
                chunks.append(str(fpath))
            else:
                chunks.append(chunk)
        return chunks

    def serialize(self) -> pd.DataFrame:
        """Serialise to a pandas DataFrame (legacy helper).

        Returns a DataFrame indexed by ``sequence_id`` with one row per
        clonotype and one column per :meth:`Clonotype.serialize` key.
        """
        data: dict[str, list] = defaultdict(list)
        for c in self.clonotypes:
            for k, v in c.serialize().items():
                data[k].append(v)
        return pd.DataFrame(data, index=[c.id for c in self.clonotypes])

    def to_pickle(self, path: str | Path) -> Path:
        """Serialize this repertoire to a pickle file and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    @classmethod
    def from_pickle(cls, path: str | Path) -> LocusRepertoire:
        """Load a pickled :class:`LocusRepertoire` from disk."""
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected pickled {cls.__name__}, got {type(obj).__name__}"
            )
        return obj

    # ------------------------------------------------------------------
    # Backward-compat properties
    # ------------------------------------------------------------------

    @property
    def number_of_clones(self) -> int:
        """Backward-compat alias for :attr:`clonotype_count`."""
        return self.clonotype_count

    @property
    def number_of_reads(self) -> int:
        """Backward-compat alias for :attr:`duplicate_count`."""
        return self.duplicate_count

    @property
    def total(self) -> int:
        """Backward-compat alias for :attr:`duplicate_count`."""
        return self.duplicate_count

    @property
    def metadata(self) -> dict:
        """Backward-compat alias for :attr:`repertoire_metadata`."""
        return self.repertoire_metadata

    @metadata.setter
    def metadata(self, value) -> None:
        self.repertoire_metadata = dict(value) if value is not None else {}

    @property
    def gene(self) -> str:
        """Backward-compat alias for :attr:`locus`."""
        return self.locus

    @property
    def sorted(self) -> bool:
        """Backward-compat alias for :attr:`is_sorted`."""
        return self.is_sorted

    @sorted.setter
    def sorted(self, value: bool) -> None:
        pass  # no-op: is_sorted is computed, not stored

    # evaluate_segment_usage was a @property in the old API; keep it working.
    @property
    def evaluate_segment_usage(self) -> dict[str, int]:
        """Backward-compat property alias for :meth:`segment_usage`."""
        return self.segment_usage()

    # ------------------------------------------------------------------
    # Trie (kept for downstream code)
    # ------------------------------------------------------------------

    @property
    def trie(self):
        """Lazy :class:`tcrtrie.Trie` built from junction_aa / V / J strings."""
        from tcrtrie import Trie
        if not hasattr(self, "_trie"):
            self._trie = Trie(
                sequences=[c.junction_aa for c in self.clonotypes],
                vGenes=[c.v_gene for c in self.clonotypes],
                jGenes=[c.j_gene for c in self.clonotypes],
            )
        return self._trie

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.clonotype_count

    def __iter__(self) -> Iterator[Clonotype]:
        return iter(self.clonotypes)

    def __getitem__(self, idx: int | slice) -> Clonotype | list[Clonotype]:
        return self.clonotypes[idx]

    def __add__(self, other: LocusRepertoire) -> LocusRepertoire:
        if not isinstance(other, LocusRepertoire):
            raise TypeError(
                f"unsupported operand type(s) for +: "
                f"'LocusRepertoire' and {type(other).__name__!r}"
            )
        if self.locus and other.locus and self.locus != other.locus:
            raise ValueError(
                f"Cannot merge repertoires with different loci: "
                f"{self.locus!r} and {other.locus!r}"
            )
        return LocusRepertoire(
            self.clonotypes + other.clonotypes,
            locus=self.locus or other.locus,
            repertoire_metadata={**self.repertoire_metadata, **other.repertoire_metadata},
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

    Groups :class:`LocusRepertoire` objects by locus code so that
    per-chain analysis (diversity, V-gene usage, etc.) can be performed
    independently while the full sample remains a single object.

    Parameters
    ----------
    loci:
        Mapping of locus code (e.g. ``"TRB"``) to its
        :class:`LocusRepertoire`.
    sample_id:
        Sample identifier (e.g. an SRA accession such as ``"SRR8363891"``).
    sample_metadata:
        Arbitrary key/value metadata dict (e.g. donor age, disease status).

    Examples
    --------
    Build from a flat list of clonotypes whose locus has been pre-assigned::

        from mir.common.clonotype import Clonotype
        from mir.common.repertoire import SampleRepertoire, infer_locus

        clonotypes = [
            Clonotype(junction_aa="CASSEGF", v_gene="TRBV3-1", locus="TRB"),
            Clonotype(junction_aa="CATSEGF", v_gene="TRAV21",  locus="TRA"),
        ]
        sr = SampleRepertoire.from_clonotypes(clonotypes, sample_id="donor_1")
        trb = sr["TRB"]
    """

    def __init__(
        self,
        loci: dict[str, LocusRepertoire],
        sample_id: str = "",
        sample_metadata: dict | None = None,
    ) -> None:
        self.loci: dict[str, LocusRepertoire] = dict(loci)
        self.sample_id: str = sample_id
        self.sample_metadata: dict = sample_metadata if sample_metadata is not None else {}

    def to_pickle(self, path: str | Path) -> Path:
        """Serialize this sample repertoire to a pickle file and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    @classmethod
    def from_pickle(cls, path: str | Path) -> SampleRepertoire:
        """Load a pickled :class:`SampleRepertoire` from disk."""
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected pickled {cls.__name__}, got {type(obj).__name__}"
            )
        return obj

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
        """Build a :class:`SampleRepertoire` by grouping *clonotypes* by locus.

        Parameters
        ----------
        clonotypes:
            Flat list of :class:`Clonotype` objects.  The ``locus`` field on
            each clonotype determines which :class:`LocusRepertoire` it is
            placed into.
        sample_id:
            Identifier for the resulting sample.
        sample_metadata:
            Metadata dict for the resulting sample.
        """
        groups: dict[str, list[Clonotype]] = defaultdict(list)
        for c in clonotypes:
            groups[c.locus].append(c)
        loci = {
            locus: LocusRepertoire(clones, locus=locus)
            for locus, clones in groups.items()
        }
        return cls(loci=loci, sample_id=sample_id, sample_metadata=sample_metadata)

    # ------------------------------------------------------------------
    # Aggregated access
    # ------------------------------------------------------------------

    @property
    def clonotypes(self) -> list[Clonotype]:
        """All clonotypes across all loci, concatenated in locus-insertion order."""
        result: list[Clonotype] = []
        for lr in self.loci.values():
            result.extend(lr.clonotypes)
        return result

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @property
    def is_sorted(self) -> bool:
        """True iff every per-locus repertoire satisfies :attr:`LocusRepertoire.is_sorted`."""
        return all(lr.is_sorted for lr in self.loci.values())

    def sort(self) -> None:
        """Sort every per-locus repertoire by ``duplicate_count`` descending."""
        for lr in self.loci.values():
            lr.sort()

    # ------------------------------------------------------------------
    # Polars I/O
    # ------------------------------------------------------------------

    def to_polars(self) -> pl.DataFrame:
        """Return all clonotypes as a single Polars DataFrame (AIRR schema).

        Per-locus DataFrames are concatenated; the ``locus`` column identifies
        the chain of each row.
        """
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
        """Deserialise a Polars DataFrame into a :class:`SampleRepertoire`.

        Rows are grouped by *locus_column*; each group becomes one
        :class:`LocusRepertoire`.  If *locus_column* is absent, all rows are
        placed in a single repertoire with an empty locus key.

        Parameters
        ----------
        df:
            DataFrame with AIRR-schema columns (plus *locus_column*).
        locus_column:
            Column used to group rows by locus.
        sample_id:
            Identifier for the resulting sample.
        sample_metadata:
            Metadata dict for the resulting sample.
        """
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
        """Return the :class:`LocusRepertoire` for *locus*."""
        return self.loci[locus]

    def __contains__(self, locus: str) -> bool:
        return locus in self.loci

    def __iter__(self) -> Iterator[LocusRepertoire]:
        return iter(self.loci.values())

    def __len__(self) -> int:
        """Number of distinct loci in this sample."""
        return len(self.loci)

    def __str__(self) -> str:
        parts = ", ".join(f"{k}: {v.clonotype_count}" for k, v in self.loci.items())
        return f"SampleRepertoire(id={self.sample_id!r}, loci={{{parts}}})"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

#: Alias for :class:`LocusRepertoire` kept for backward compatibility.
Repertoire = LocusRepertoire
