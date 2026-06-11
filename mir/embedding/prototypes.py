"""Reference prototype loading for TCREMP embeddings.

Prototype files are generated once by running
``mir/resources/prototypes/generate_prototypes.py`` and committed to the
repository.  They are **never** regenerated at build or test time.

Each file contains up to :data:`N_PROTOTYPES` rows drawn from a real
repertoire control (when one is available for the species/locus) or from a
synthetic OLGA-generated control, de-duplicated on the
(v_call, j_call, junction_aa) triple.

Typical usage::

    from mir.embedding.prototypes import load_prototypes

    df = load_prototypes("human", "TRB")          # all 10 000 prototypes
    df_small = load_prototypes("human", "TRB", n=1000)  # first 1 000
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl

from mir import get_resource_path
from mir.basic.aliases import normalize_locus_alias, normalize_species_alias

N_PROTOTYPES: int = 10_000
"""Maximum number of prototypes per species/locus combination."""

_RESOURCE_SUBDIR = "prototypes"
_COLS = ["v_call", "j_call", "junction_aa"]


def _prototypes_dir() -> Path:
    """Return the bundled prototype resource directory."""
    return Path(cast(str, get_resource_path(_RESOURCE_SUBDIR)))


def _prototype_path(species: str, locus: str) -> Path:
    return _prototypes_dir() / f"{species}_{locus}.tsv"


def list_available_prototypes() -> list[tuple[str, str]]:
    """Return (species, locus) pairs with bundled prototype files.

    Returns:
        Sorted list of (species, locus) tuples for which a prototype TSV file
        exists in ``mir/resources/prototypes/``.

    Example:
        >>> pairs = list_available_prototypes()
        >>> ("human", "TRB") in pairs
        True
    """
    base = _prototypes_dir()
    out: list[tuple[str, str]] = []
    for tsv in sorted(base.glob("*.tsv")):
        stem = tsv.stem
        if "_" not in stem:
            continue
        species, locus = stem.split("_", 1)
        out.append((species, locus))
    return out


def load_prototypes(
    species: str,
    locus: str,
    n: int | None = None,
) -> pl.DataFrame:
    """Load prototype clonotypes for TCREMP distance-based embeddings.

    Reads from the pre-generated TSV file in ``mir/resources/prototypes/``.
    The first *n* rows are returned, preserving the fixed generation order so
    that embeddings are reproducible across calls.

    Args:
        species: Species identifier, e.g. ``'human'`` or ``'mouse'``.
            Aliases accepted by :func:`~mir.basic.aliases.normalize_species_alias`
            (``'hsa'``, ``'Homo sapiens'``, …) are resolved automatically.
        locus: Receptor locus, e.g. ``'TRB'`` or ``'TRA'``.
            Aliases accepted by :func:`~mir.basic.aliases.normalize_locus_alias`
            (``'beta'``, ``'T_beta'``, …) are resolved automatically.
        n: Number of prototypes to load (first *n* rows from the file).
            Must be ``<= N_PROTOTYPES`` (10,000).  Returns all available
            prototypes when ``None``.

    Returns:
        DataFrame with columns ``v_call``, ``j_call``, ``junction_aa``.
        Row order is fixed and matches the generation order from
        :mod:`mir.resources.prototypes.generate_prototypes`.

    Raises:
        ValueError: If ``n > N_PROTOTYPES``.
        FileNotFoundError: If no prototype file exists for the given
            species/locus combination.  Run
            ``mir/resources/prototypes/generate_prototypes.py`` to populate
            the resource directory.

    Example:
        >>> df = load_prototypes("human", "TRB", n=100)
        >>> df.columns
        ['v_call', 'j_call', 'junction_aa']
        >>> len(df)
        100
    """
    if n is not None and n > N_PROTOTYPES:
        raise ValueError(
            f"n={n} exceeds the maximum number of prototypes ({N_PROTOTYPES}). "
            f"Use n <= {N_PROTOTYPES} or None to load all available prototypes."
        )

    species_c = normalize_species_alias(species)
    locus_c = normalize_locus_alias(locus)

    path = _prototype_path(species_c, locus_c)
    if not path.exists():
        raise FileNotFoundError(
            f"No prototype file found for species={species_c!r}, locus={locus_c!r}. "
            f"Run mir/resources/prototypes/generate_prototypes.py to generate them. "
            f"Expected path: {path}"
        )

    df = pl.read_csv(path, separator="\t", columns=_COLS)

    if n is not None:
        df = df.head(n)

    return df
