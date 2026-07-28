"""Reference prototype loading for TCREMP embeddings.

Prototype files are generated once by ``mir/resources/prototypes/generate_prototypes.py`` and
committed to the repository. They are **never** regenerated at build or test time. Each of the nine
bundled files holds :data:`N_PROTOTYPES` = 10 000 rows of **real** receptors — a uniform random
sample (fixed ``seed=42``) of germline-resolvable, unique, productive clonotypes from
arda-annotated real repertoires, de-duplicated on the ``(v_call, j_call, junction_aa)`` triple.
Real repertoires, not model draws: synthetic P_gen junctions have degenerate lengths and embed
measurably worse (see ``SOURCES.md``).

**The default set.** ``replicate=0`` — the first ``n`` rows — is *the* prototype set. It is what
every default, preset, bundled codec and published number uses; leave it alone unless you are
deliberately measuring prototype-draw sensitivity.

**Replicates.** Because the file order is itself a uniform shuffle, any disjoint block of rows is an
independent draw from the same pool. ``replicate=r`` returns block ``r`` (rows ``r*n … (r+1)*n``),
so replicates are *disjoint* — independent by construction, not a with-replacement bootstrap. You
get :func:`n_replicates` = ``10000 // n`` of them: **10** at ``n=1000``, 5 at ``n=2000``. Use them to
answer "how much does my result depend on *which* prototypes I drew?"::

    from mir.embedding.tcremp import TCREmp

    scores = [score(TCREmp.from_defaults("human", "TRB", 1000, replicate=r).embed(df))
              for r in range(n_replicates("human", "TRB", 1000))]      # spread = draw sensitivity

Every replicate is a **different coordinate system**: distances within one are comparable, distances
across two are not. The prototype hash covers ``replicate``, so codecs, ``RepertoireSpace`` and
``DonorCohort`` all refuse to mix them — compare *summary statistics* across replicates, never raw
embeddings.

Typical usage::

    from mir.embedding.prototypes import load_prototypes, n_replicates

    df = load_prototypes("human", "TRB")                        # all 10 000
    df_small = load_prototypes("human", "TRB", n=1000)           # the default 1 000
    df_alt = load_prototypes("human", "TRB", n=1000, replicate=3)  # an independent 1 000
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


def _read_pool(species: str, locus: str) -> pl.DataFrame:
    """Read the full bundled prototype pool for a canonical species/locus."""
    path = _prototype_path(species, locus)
    if not path.exists():
        raise FileNotFoundError(
            f"No prototype file for species={species!r}, locus={locus!r} "
            f"(expected {path}). Run mir/resources/prototypes/generate_prototypes.py."
        )
    return pl.read_csv(path, separator="\t", columns=_COLS)


def n_replicates(species: str, locus: str, n: int) -> int:
    """How many disjoint prototype replicates of size *n* the bundled pool supports.

    ``pool_size // n`` — 10 at the common ``n=1000``, 5 at ``n=2000``. Replicate indices are
    ``range(n_replicates(...))``, with ``0`` the default set.

    Example:
        >>> n_replicates("human", "TRB", 1000)
        10
    """
    if n <= 0:
        raise ValueError(f"n={n} must be positive")
    pool = _read_pool(normalize_species_alias(species), normalize_locus_alias(locus))
    return pool.height // n


def load_prototypes(species: str, locus: str, n: int | None = None,
                    replicate: int = 0) -> pl.DataFrame:
    """Load prototype clonotypes for TCREMP distance-based embeddings.

    Reads the pre-generated TSV in ``mir/resources/prototypes/`` and returns *n* rows in the fixed
    generation order, so embeddings are reproducible across calls and machines.

    Args:
        species: Species identifier; aliases (``'hsa'``, ``'Homo sapiens'``, …)
            are resolved via :func:`mir.aliases.normalize_species_alias`.
        locus: Receptor locus; aliases (``'beta'``, ``'T_beta'``, …) are resolved
            via :func:`mir.aliases.normalize_locus_alias`.
        n: Number of prototypes. Must be ``<= N_PROTOTYPES``. Loads the whole pool when ``None``.
        replicate: Which disjoint block of *n* rows to take. ``0`` (default) is the canonical
            prototype set — the first *n* rows, what every preset and published number uses.
            ``r > 0`` returns rows ``r*n … (r+1)*n``, an **independent** draw from the same pool
            (the file order is a uniform shuffle), for measuring prototype-draw sensitivity.
            Requires an explicit *n*; see :func:`n_replicates` for the number available.

    Returns:
        DataFrame with columns ``v_call``, ``j_call``, ``junction_aa`` in fixed generation order.

    Raises:
        ValueError: If ``n`` is not in ``[1, N_PROTOTYPES]``, or ``replicate`` is negative, or
            ``replicate`` is given without ``n``, or the pool has no such block.
        FileNotFoundError: If no prototype file exists for the species/locus.

    Example:
        >>> df = load_prototypes("human", "TRB", n=100)
        >>> df.columns
        ['v_call', 'j_call', 'junction_aa']
        >>> len(df)
        100
        >>> alt = load_prototypes("human", "TRB", n=100, replicate=1)
        >>> set(df["junction_aa"]) & set(alt["junction_aa"])   # disjoint blocks
        set()
    """
    if n is not None and (n <= 0 or n > N_PROTOTYPES):
        raise ValueError(
            f"n={n} must be in [1, {N_PROTOTYPES}] or None to load all. A non-positive n would "
            f"silently change the prototype set — and thus its hash — via df.head(n), breaking "
            f"embedding comparability."
        )
    if replicate < 0:
        raise ValueError(f"replicate={replicate} must be >= 0 (0 is the default prototype set)")
    if replicate and n is None:
        raise ValueError("replicate= needs an explicit n: a replicate is a disjoint block of n rows")

    species_c = normalize_species_alias(species)
    locus_c = normalize_locus_alias(locus)
    df = _read_pool(species_c, locus_c)

    if n is None:
        return df
    if replicate == 0:
        return df.head(n)
    avail = df.height // n
    if replicate >= avail:
        raise ValueError(
            f"replicate={replicate} is out of range for {species_c}_{locus_c} at n={n}: the pool "
            f"holds {df.height} prototypes, so replicates 0..{avail - 1} exist "
            f"({avail} disjoint sets). Lower n for more replicates."
        )
    return df.slice(replicate * n, n)
