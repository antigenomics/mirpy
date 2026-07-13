"""Reference prototype loading for TCREMP embeddings.

Prototype files are generated once by ``mir/resources/prototypes/generate_prototypes.py``
(sampling the VDJ rearrangement model via ``vdjtools.model.generate`` or a real repertoire
control) and committed to the repository. They are **never** regenerated at build or test time.

Each file contains up to :data:`N_PROTOTYPES` rows, de-duplicated on the
``(v_call, j_call, junction_aa)`` triple.

Typical usage::

    from mir.embedding.prototypes import load_prototypes

    df = load_prototypes("human", "TRB")                 # all 10 000 prototypes
    df_small = load_prototypes("human", "TRB", n=1000)   # first 1 000
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl

from mir import get_resource_path
from mir.aliases import normalize_locus_alias, normalize_species_alias

N_PROTOTYPES: int = 10_000
"""Maximum number of prototypes per species/locus combination."""

_RESOURCE_SUBDIR = "prototypes"
# AIRR columns (match vdjtools.io.schema V_CALL / J_CALL / JUNCTION_AA).
_COLS = ["v_call", "j_call", "junction_aa"]


def _prototypes_dir() -> Path:
    return Path(cast(str, get_resource_path(_RESOURCE_SUBDIR)))


def _prototype_path(species: str, locus: str) -> Path:
    return _prototypes_dir() / f"{species}_{locus}.tsv"


def list_available_prototypes() -> list[tuple[str, str]]:
    """Return sorted ``(species, locus)`` pairs with a bundled prototype file.

    Example:
        >>> ("human", "TRB") in list_available_prototypes()
        True
    """
    out: list[tuple[str, str]] = []
    for tsv in sorted(_prototypes_dir().glob("*.tsv")):
        stem = tsv.stem
        if "_" not in stem:
            continue
        species, locus = stem.split("_", 1)
        out.append((species, locus))
    return out


def load_prototypes(species: str, locus: str, n: int | None = None) -> pl.DataFrame:
    """Load prototype clonotypes for TCREMP distance-based embeddings.

    Reads the pre-generated TSV in ``mir/resources/prototypes/`` and returns the
    first *n* rows, preserving the fixed generation order so embeddings are
    reproducible across calls.

    Args:
        species: Species identifier; aliases (``'hsa'``, ``'Homo sapiens'``, …)
            are resolved via :func:`mir.aliases.normalize_species_alias`.
        locus: Receptor locus; aliases (``'beta'``, ``'T_beta'``, …) are resolved
            via :func:`mir.aliases.normalize_locus_alias`.
        n: Number of prototypes (first *n* rows). Must be ``<= N_PROTOTYPES``.
            Loads all available prototypes when ``None``.

    Returns:
        DataFrame with columns ``v_call``, ``j_call``, ``junction_aa`` in fixed
        generation order.

    Raises:
        ValueError: If ``n > N_PROTOTYPES``.
        FileNotFoundError: If no prototype file exists for the species/locus.

    Example:
        >>> df = load_prototypes("human", "TRB", n=100)
        >>> df.columns
        ['v_call', 'j_call', 'junction_aa']
        >>> len(df)
        100
    """
    if n is not None and n > N_PROTOTYPES:
        raise ValueError(
            f"n={n} exceeds the maximum number of prototypes ({N_PROTOTYPES}); "
            f"use n <= {N_PROTOTYPES} or None to load all."
        )

    species_c = normalize_species_alias(species)
    locus_c = normalize_locus_alias(locus)

    path = _prototype_path(species_c, locus_c)
    if not path.exists():
        raise FileNotFoundError(
            f"No prototype file for species={species_c!r}, locus={locus_c!r} "
            f"(expected {path}). Run mir/resources/prototypes/generate_prototypes.py."
        )

    df = pl.read_csv(path, separator="\t", columns=_COLS)
    return df.head(n) if n is not None else df
