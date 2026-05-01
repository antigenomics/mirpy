"""Immune-receptor clonotype as a single flat dataclass.

All gene-name fields store plain strings (IMGT allele names, e.g.
``"TRBV3-1*01"``).  Junction coordinates follow the AIRR Rearrangement
schema naming: *v_sequence_end*, *d_sequence_start*, *d_sequence_end*,
*j_sequence_start*.
"""

from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import ClassVar, NamedTuple

import polars as pl

from mir.basic.mirseq_compat import translate_bidi, is_coding as _c_is_coding, is_canonical as _c_is_canonical

_LOCUS_PREFIXES: frozenset[str] = frozenset({"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"})


# ---------------------------------------------------------------------------
# Junction boundary type (derived from Clonotype int fields)
# ---------------------------------------------------------------------------

class JunctionMarkup(NamedTuple):
    """V/D/J boundary positions within the junction sequence (AIRR names)."""
    v_sequence_end: int = -1
    d_sequence_start: int = -1
    d_sequence_end: int = -1
    j_sequence_start: int = -1


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass(unsafe_hash=True, slots=True)
class Clonotype:
    """Single immune-receptor rearrangement with AIRR-schema fields.

    All fields have safe defaults so instances can be constructed with any
    subset of information.  ``junction_aa`` is auto-translated from
    ``junction`` (via the C ``translate_bidi`` function) when omitted.
    Gene-name arguments may be plain strings or legacy ``GeneEntry`` objects
    (converted to ``str`` in ``__post_init__``).
    """

    sequence_id: str = ""
    duplicate_count: int = 0
    locus: str = ""
    junction: str = ""
    junction_aa: str = ""
    v_gene: str = ""
    j_gene: str = ""
    d_gene: str = ""
    c_gene: str = ""
    v_sequence_end: int = -1
    d_sequence_start: int = -1
    d_sequence_end: int = -1
    j_sequence_start: int = -1
    # Mutable metadata dict excluded from eq/repr/init
    clone_metadata: dict = field(default_factory=dict, repr=False,
                                  compare=False, init=False)
    # Pass _validate=False in performance-critical internal code to skip
    # junction_aa character and length checks (values are already trusted).
    _validate: InitVar[bool] = True

    def __post_init__(self, _validate: bool) -> None:
        if not _validate:
            # Hot path: caller (parser) guarantees all fields are already clean
            # strings and locus is already set.  Skip all normalisation.
            return
        # Auto-translate junction → junction_aa
        if self.junction and not self.junction_aa:
            self.junction_aa = translate_bidi(self.junction)
        # Normalise gene fields: accept GeneEntry objects or None
        for attr in ("v_gene", "d_gene", "j_gene", "c_gene"):
            val = getattr(self, attr)
            if val is None:
                setattr(self, attr, "")
            elif not isinstance(val, str):
                setattr(self, attr, str(val))
        # Infer locus from j_gene prefix when not explicitly set
        if not self.locus and self.j_gene:
            prefix = self.j_gene[:3].upper()
            if prefix in _LOCUS_PREFIXES:
                self.locus = prefix
        # Validate junction_aa: only the 20 standard amino-acid letters
        if self.junction_aa and not _c_is_coding(self.junction_aa):
            raise ValueError(
                f"junction_aa contains non-standard amino-acid characters: "
                f"{self.junction_aa!r}"
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Backward-compat alias for ``sequence_id``."""
        return self.sequence_id

    @property
    def junction_markup(self) -> JunctionMarkup:
        """V/D/J boundary positions derived from the four int fields."""
        return JunctionMarkup(
            self.v_sequence_end, self.d_sequence_start,
            self.d_sequence_end, self.j_sequence_start,
        )

    # ------------------------------------------------------------------
    # Classification helpers (delegate to C extension for speed)
    # ------------------------------------------------------------------

    def is_coding(self) -> bool:
        """True iff ``junction_aa`` contains only standard amino-acid letters."""
        return bool(_c_is_coding(self.junction_aa)) if self.junction_aa else False

    def is_canonical(self) -> bool:
        """True iff ``junction_aa`` starts with C and ends with F or W."""
        return bool(_c_is_canonical(self.junction_aa)) if self.junction_aa else False

    # ------------------------------------------------------------------
    # Size helper (supports int count or barcode lists)
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return read/cell count (alias for ``duplicate_count``)."""
        return self.duplicate_count

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Return all AIRR-schema fields as a plain ``dict``.

        The returned keys match :attr:`_POLARS_SCHEMA` exactly, making the
        output suitable for :func:`polars.from_dicts` or ``pd.DataFrame``.
        """
        return {
            "sequence_id":     self.sequence_id,
            "duplicate_count": self.duplicate_count,
            "locus":           self.locus,
            "junction":        self.junction,
            "junction_aa":     self.junction_aa,
            "v_gene":          self.v_gene,
            "d_gene":          self.d_gene,
            "j_gene":          self.j_gene,
            "c_gene":          self.c_gene,
            "v_sequence_end":  self.v_sequence_end,
            "d_sequence_start": self.d_sequence_start,
            "d_sequence_end":  self.d_sequence_end,
            "j_sequence_start": self.j_sequence_start,
        }

    # ------------------------------------------------------------------
    # Polars I/O
    # ------------------------------------------------------------------

    _POLARS_SCHEMA: ClassVar[dict[str, type]] = {
        "sequence_id":      pl.Utf8,
        "duplicate_count":  pl.Int64,
        "locus":            pl.Utf8,
        "junction":         pl.Utf8,
        "junction_aa":      pl.Utf8,
        "v_gene":           pl.Utf8,
        "d_gene":           pl.Utf8,
        "j_gene":           pl.Utf8,
        "c_gene":           pl.Utf8,
        "v_sequence_end":   pl.Int64,
        "d_sequence_start": pl.Int64,
        "d_sequence_end":   pl.Int64,
        "j_sequence_start": pl.Int64,
    }

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> list[Clonotype]:
        """Build a list of :class:`Clonotype` from a Polars DataFrame.

        Column names must use AIRR schema names (see :attr:`_POLARS_SCHEMA`).
        Rows without a ``sequence_id`` column receive incremental string IDs
        ``"0"``, ``"1"``, …
        """
        cols = set(df.columns)
        has_id = "sequence_id" in cols

        def _str(v) -> str:
            return "" if v is None else str(v)

        def _int(v, default: int = -1) -> int:
            if v is None:
                return default
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        clonotypes: list[Clonotype] = []
        for i, row in enumerate(df.iter_rows(named=True)):
            clonotypes.append(cls(
                sequence_id=    _str(row["sequence_id"]) if has_id else str(i),
                duplicate_count=_int(row.get("duplicate_count"), 0),
                locus=          _str(row.get("locus")),
                junction=       _str(row.get("junction")),
                junction_aa=    _str(row.get("junction_aa")),
                v_gene=         _str(row.get("v_gene")),
                d_gene=         _str(row.get("d_gene")),
                j_gene=         _str(row.get("j_gene")),
                c_gene=         _str(row.get("c_gene")),
                v_sequence_end= _int(row.get("v_sequence_end")),
                d_sequence_start=_int(row.get("d_sequence_start")),
                d_sequence_end= _int(row.get("d_sequence_end")),
                j_sequence_start=_int(row.get("j_sequence_start")),
                _validate=False,
            ))
        return clonotypes

    @staticmethod
    def to_polars(clonotypes: list[Clonotype]) -> pl.DataFrame:
        """Serialise a list of :class:`Clonotype` to a Polars DataFrame."""
        if not clonotypes:
            return pl.DataFrame(schema=Clonotype._POLARS_SCHEMA)
        return pl.from_dicts([c.serialize() for c in clonotypes],
                             schema_overrides=Clonotype._POLARS_SCHEMA)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"κ{self.sequence_id} {self.junction_aa}"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Backward-compat aliases  (no separate ClonotypeAA / ClonotypeNT classes)
# ---------------------------------------------------------------------------

ClonotypeAA = Clonotype
ClonotypeNT = Clonotype