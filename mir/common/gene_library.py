"""Gene segment library — V/D/J allele entries and collection management.

Schema for the pre-built library files (``olga_gene_library.txt`` /
``imgt_gene_library.txt``)::

    species  locus  gene  allele  sequence

where *species* is ``"human"`` or ``"mouse"``, *locus* is ``"TRB"`` /
``"TRA"`` / etc., *gene* is ``"V"``, ``"D"``, or ``"J"``, *allele* is the
full IMGT name (e.g. ``"TRBV3-1*02"``), and *sequence* is the nucleotide
sequence (uppercase, no gaps).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mir import get_resource_path
from mir.basic.mirseq import translate_linear

_ALLOWED_LOCI = {'TRA', 'TRB', 'TRG', 'TRD', 'IGL', 'IGK', 'IGH'}
_ALLOWED_GENES = {'V', 'D', 'J', 'C'}

_GENE_LIBRARY_COLUMNS = ['species', 'locus', 'gene', 'allele', 'sequence']


class GeneEntry:
    """A single V/D/J/C gene entry with allele name, sequence, and annotations.

    Parameters
    ----------
    allele:
        Full IMGT allele name, e.g. ``"TRBV3-1*02"``.
    species:
        Canonical species name (``"human"``, ``"mouse"``, …).  Inferred
        as ``"Unknown"`` when not supplied.
    locus:
        Three-letter locus code (``"TRB"``, ``"TRA"``, …).  Inferred from
        the first three characters of *allele* when not supplied.
    gene:
        Gene type: ``"V"``, ``"D"``, ``"J"``, or ``"C"``.  Inferred from
        the fourth character of *allele* when not supplied.
    sequence:
        Nucleotide sequence (uppercase).  When provided and *sequence_aa*
        is absent, the amino-acid sequence is derived automatically via
        :func:`mir.basic.mirseq.translate_linear`.
    sequence_aa:
        Amino-acid sequence.  Auto-computed from *sequence* if omitted.
    refpoint:
        0-based reference point (right after Cys for V, right before
        F/W for J).  Reserved for future use; defaults to ``-1``.
    featnt:
        Named nucleotide feature intervals ``{name: (start, end)}``.
        Reserved for future use.
    feataa:
        Named amino-acid feature intervals.  Derived from *featnt* when
        absent.  Reserved for future use.
    """

    def __init__(self,
                 allele: str,
                 species: str = 'Unknown',
                 locus: str = None,
                 gene: str = None,
                 sequence: str = None,
                 sequence_aa: str = None,
                 refpoint: int = -1,
                 featnt: dict[str, tuple[int, int]] = {},
                 feataa: dict[str, tuple[int, int]] = {}):
        self.allele = allele
        self.species = species
        self.locus = locus if locus else allele[:3]
        if self.locus not in _ALLOWED_LOCI:
            raise ValueError(f'Bad locus {self.locus!r}')
        self.gene = gene if gene else (allele[3] if len(allele) > 3 else '?')
        if self.gene not in _ALLOWED_GENES:
            raise ValueError(f'Bad gene type {self.gene!r}')
        self.sequence = sequence
        if sequence_aa:
            self.sequence_aa = sequence_aa
        elif sequence:
            self.sequence_aa = translate_linear(sequence).rstrip('_')
        else:
            self.sequence_aa = None
        self.refpoint = refpoint
        self.featnt = {k: v for k, v in featnt.items() if v[1] > v[0]}
        self.feataa = {k: v for k, v in feataa.items() if v[1] > v[0]}
        if not feataa and self.featnt:
            self.feataa = {k: (v[0] // 3, v[1] // 3) for k, v in self.featnt.items()}

    def __str__(self) -> str:
        return self.allele

    def __repr__(self) -> str:
        if self.sequence_aa:
            if self.gene == 'V':
                seq = '..' + self.sequence_aa[-10:]
            elif self.gene == 'D':
                seq = '_' + self.sequence_aa + '_'
            else:
                seq = self.sequence_aa[:10] + '..'
        else:
            seq = '?'
        return f'{self.species} {self.allele}:{self.refpoint}:{seq}'


class GeneLibrary:
    """Collection of :class:`GeneEntry` objects keyed by allele name.

    Parameters
    ----------
    entries:
        Dict mapping allele name → :class:`GeneEntry`.
    complete:
        When ``True``, :meth:`get_or_create` raises for unknown alleles
        rather than creating minimal placeholder entries.
    """

    def __init__(self,
                 entries: dict[str, GeneEntry] = {},
                 complete: bool = False):
        self.entries = entries
        self.complete = complete

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _as_set(value) -> set:
        if isinstance(value, set):
            return value
        if isinstance(value, str):
            return {value}
        return set(value)

    @classmethod
    def load_default(cls,
                     loci: set[str] | list[str] | str = {'TRB', 'TRA'},
                     species: set[str] | list[str] | str = {'human'},
                     source: str = 'olga') -> 'GeneLibrary':
        """Load a gene library from a pre-built resource file.

        Parameters
        ----------
        loci:
            Locus codes to include (e.g. ``{'TRB', 'TRA'}``).
        species:
            Species names to include (e.g. ``{'human'}``).
        source:
            ``'olga'`` (default) loads ``olga_gene_library.txt``;
            ``'imgt'`` loads ``imgt_gene_library.txt``.

        Returns
        -------
        GeneLibrary
            Complete library restricted to the requested loci and species.
        """
        loci    = cls._as_set(loci)
        species = cls._as_set(species)
        fname   = f'{source}_gene_library.txt'
        path    = Path(get_resource_path(f'segments/{fname}'))
        entries: dict[str, GeneEntry] = {}
        with path.open(encoding='utf-8') as fh:
            next(fh)  # skip header
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 5:
                    continue
                sp, locus, gene, allele, sequence = parts
                if sp not in species or locus not in loci:
                    continue
                entries[allele] = GeneEntry(
                    allele=allele,
                    species=sp,
                    locus=locus,
                    gene=gene,
                    sequence=sequence,
                )
        return cls(entries, complete=True)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entries(self,
                    locus: str = None,
                    gene: str = None) -> list[GeneEntry]:
        """Return entries optionally filtered by locus and/or gene type."""
        return [e for e in self.entries.values()
                if (not locus or e.locus == locus)
                and (not gene or e.gene == gene)]

    def get_sequences_aa(self,
                         locus: str = None,
                         gene: str = None) -> list[tuple[str, str]]:
        """Return ``(allele, sequence_aa)`` pairs for the given filter."""
        return [(e.allele, e.sequence_aa)
                for e in self.get_entries(locus, gene)
                if e.sequence_aa]

    def get_sequences(self,
                      locus: str = None,
                      gene: str = None) -> list[tuple[str, str]]:
        """Return ``(allele, sequence)`` pairs for the given filter."""
        return [(e.allele, e.sequence)
                for e in self.get_entries(locus, gene)
                if e.sequence]

    def get_summary(self) -> Counter[tuple[str, str, str]]:
        """Return counts per ``(species, locus, gene)``."""
        return Counter((e.species, e.locus, e.gene) for e in self.entries.values())

    def get_species(self) -> set[str]:
        return {e.species for e in self.entries.values()}

    def get_loci(self) -> set[str]:
        return {e.locus for e in self.entries.values()}

    def get_genes(self) -> set[str]:
        return {e.gene for e in self.entries.values()}

    def __getitem__(self, allele: str) -> GeneEntry:
        return self.entries[allele]

    # ------------------------------------------------------------------
    # Entry creation / resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _allele_sort_key(allele: str) -> tuple[int, str]:
        suffix = str(allele).split('*', 1)[1]
        try:
            return (int(suffix), suffix)
        except ValueError:
            return (10 ** 9, suffix)

    def get_or_create(self,
                      entry: str | GeneEntry,
                      sequence_aa: str = None,
                      sequence: str = None) -> GeneEntry:
        """Return the existing entry or create a minimal placeholder."""
        if isinstance(entry, GeneEntry):
            existing = self.entries.get(entry.allele)
            if not existing:
                if self.complete:
                    raise ValueError(f'GeneEntry {entry} not found in complete library')
                self.entries[entry.allele] = entry
                return entry
            return existing
        allele = str(entry).strip()
        existing = self.entries.get(allele)
        if existing:
            return existing
        if self.complete:
            raise ValueError(f'GeneEntry {allele!r} not found in complete library')
        new = GeneEntry(allele, sequence=sequence, sequence_aa=sequence_aa)
        self.entries[allele] = new
        return new

    def get_or_create_noallele(self, allele_id: str) -> GeneEntry:
        """Resolve an allele-less identifier to a concrete entry.

        Tries, in order:
        1. If *allele_id* already contains ``*``, delegates to
           :meth:`get_or_create`.
        2. The lexicographically minimum allele with the same base name
           present in this library.
        3. Falls back to ``allele_id + "*01"`` via :meth:`get_or_create`.
        """
        if '*' in allele_id:
            return self.get_or_create(allele_id)
        candidates = sorted(
            (a for a in self.entries if a.startswith(allele_id + '*')),
            key=self._allele_sort_key,
        )
        if candidates:
            return self.get_or_create(candidates[0])
        return self.get_or_create(allele_id + '*01')

    def __repr__(self) -> str:
        sample = list(self.entries.values())[:5]
        return f'GeneLibrary({len(self.entries)} entries, complete={self.complete}, sample={sample})'


class _DefaultLibrary:
    """Lazy proxy for the default OLGA gene library singleton.

    Loads ``olga_gene_library.txt`` (human TRA+TRB by default) on first
    attribute access so that importing this module never triggers file I/O.
    ``complete`` is kept ``False`` so unknown allele names are accepted as
    placeholder entries rather than raising.
    """

    _lib: 'GeneLibrary | None' = None

    @classmethod
    def _load(cls) -> 'GeneLibrary':
        if cls._lib is None:
            cls._lib = GeneLibrary.load_default()
            cls._lib.complete = False
        return cls._lib

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


_GENE_LIBRARY_CACHE = _DefaultLibrary()
