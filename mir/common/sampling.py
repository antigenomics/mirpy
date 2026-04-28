"""Repertoire downsampling utilities.

This module provides functions to downsample immune repertoires by randomly
sampling clonotypes according to their abundance, using multinomial sampling.
"""

from __future__ import annotations

import warnings
from copy import copy

import numpy as np

from mir.common.repertoire import LocusRepertoire, SampleRepertoire


def downsample_locus(
    repertoire: LocusRepertoire,
    downsample_count: int,
    random_seed: int | None = None,
) -> LocusRepertoire:
    """Downsample a LocusRepertoire to a target duplicate count.

    Uses multinomial sampling based on clonotype frequencies. Each clonotype's
    new duplicate count is drawn from a multinomial distribution with
    probabilities proportional to its initial abundance.

    Parameters
    ----------
    repertoire:
        The repertoire to downsample.
    downsample_count:
        Target total duplicate count for downsampled repertoire. Must be > 0.
    random_seed:
        Numpy random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    LocusRepertoire
        New repertoire with downsampled clonotypes. Clonotypes with zero
        new duplicate count are omitted.

    Warnings
    --------
    UserWarning
        If downsample_count >= repertoire.duplicate_count.

    Notes
    -----
    The total duplicate count of the returned repertoire is guaranteed to equal
    *downsample_count*.
    """
    if downsample_count <= 0:
        raise ValueError(f"downsample_count must be > 0, got {downsample_count}")

    total_duplicates = repertoire.duplicate_count

    if downsample_count >= total_duplicates:
        warnings.warn(
            f"downsample_count ({downsample_count}) >= total duplicate count ({total_duplicates}); "
            "no downsampling performed",
            UserWarning,
        )
        # Return a copy with the same clonotypes
        return LocusRepertoire(
            clonotypes=[copy(c) for c in repertoire.clonotypes],
            locus=repertoire.locus,
            repertoire_id=repertoire.repertoire_id,
            repertoire_metadata=dict(repertoire.repertoire_metadata),
        )

    # Compute frequencies
    frequencies = np.array(
        [c.duplicate_count / total_duplicates for c in repertoire.clonotypes],
        dtype=np.float64,
    )

    # Multinomial sampling
    if random_seed is not None:
        np.random.seed(random_seed)

    new_counts = np.random.multinomial(downsample_count, frequencies)

    # Build new repertoire, omitting zero-count clonotypes
    new_clonotypes = []
    for clonotype, new_count in zip(repertoire.clonotypes, new_counts):
        if new_count > 0:
            new_clonotype = copy(clonotype)
            new_clonotype.duplicate_count = int(new_count)
            new_clonotypes.append(new_clonotype)

    return LocusRepertoire(
        clonotypes=new_clonotypes,
        locus=repertoire.locus,
        repertoire_id=repertoire.repertoire_id,
        repertoire_metadata=dict(repertoire.repertoire_metadata),
    )


def downsample(
    repertoire: LocusRepertoire | SampleRepertoire,
    downsample_count: int,
    random_seed: int | None = None,
) -> LocusRepertoire | SampleRepertoire:
    """Downsample a repertoire to a target duplicate count.

    For :class:`LocusRepertoire`, applies downsampling directly.
    For :class:`SampleRepertoire`, applies downsampling to each locus independently.

    Parameters
    ----------
    repertoire:
        The repertoire to downsample.
    downsample_count:
        Target total duplicate count per locus.
    random_seed:
        Numpy random seed for reproducibility.

    Returns
    -------
    LocusRepertoire | SampleRepertoire
        Downsampled repertoire of the same type.
    """
    if isinstance(repertoire, LocusRepertoire):
        return downsample_locus(repertoire, downsample_count, random_seed=random_seed)

    # For SampleRepertoire, downsample each locus independently
    downsampled_loci = {
        locus: downsample_locus(lr, downsample_count, random_seed=random_seed)
        for locus, lr in repertoire.loci.items()
    }

    return SampleRepertoire(
        loci=downsampled_loci,
        sample_id=repertoire.sample_id,
        sample_metadata=dict(repertoire.sample_metadata),
    )
