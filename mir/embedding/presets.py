"""Recommended per-chain presets: prototype count and PCA dimensionality.

Data-driven from the bundled prototypes (``experiments/`` measurements):

* **n_prototypes** — the pairwise-distance geometry saturates (Pearson r vs a 5000-prototype
  reference ≥ 0.996 at 1000, ≥ 0.999 at 2000). Compact chains (IGK/IGL/TRG) saturate earlier;
  diverse chains (IGH/TR*) benefit from 2000. The paper's 3000 is safe but generous.
* **n_components** — PCA dimensions retaining ~95% variance (the paper's clustering regime).
* **n_components_recon** — PCA dimensions retaining ~99% variance, needed to *reconstruct*
  the longer, more diverse junctions with the inverse codec (Theory T5, arda coords: IGH
  exact-match 0.115 at 95% → 0.356 at 99%). Compact chains barely differ; IGH/TRD/TRA need many more.

Values are rounded up to the nearest 5; treat them as starting points, not hard limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from mir.aliases import normalize_locus_alias, normalize_species_alias


@dataclass(frozen=True)
class ChainPreset:
    """Recommended settings for one (species, locus)."""

    n_prototypes: int        # prototype count (embedding geometry saturated)
    n_components: int        # PCA dims for embedding / clustering (~95% variance)
    n_components_recon: int  # PCA dims for codec reconstruction (~99% variance)


# keyed by (canonical species, canonical locus)
CHAIN_PRESETS: dict[tuple[str, str], ChainPreset] = {
    ("human", "TRA"): ChainPreset(2000, 65, 220),
    ("human", "TRB"): ChainPreset(2000, 65, 260),
    ("human", "TRG"): ChainPreset(1000, 25, 100),
    ("human", "TRD"): ChainPreset(2000, 65, 280),
    ("human", "IGH"): ChainPreset(2000, 65, 300),
    ("human", "IGK"): ChainPreset(1000, 20, 65),
    ("human", "IGL"): ChainPreset(1000, 20, 65),
    ("mouse", "TRA"): ChainPreset(2000, 50, 150),
    ("mouse", "TRB"): ChainPreset(2000, 55, 225),
}

# generic fallback for a chain without a measured preset (diverse-chain defaults)
_FALLBACK = ChainPreset(2000, 60, 250)


def get_preset(species: str, locus: str) -> ChainPreset:
    """Return the :class:`ChainPreset` for a (species, locus), aliases resolved.

    Falls back to a generic diverse-chain preset for unmeasured combinations.

    Example:
        >>> get_preset("human", "IGH").n_components_recon
        300
        >>> get_preset("hsa", "beta").n_prototypes
        2000
    """
    key = (normalize_species_alias(species), normalize_locus_alias(locus))
    return CHAIN_PRESETS.get(key, _FALLBACK)


if __name__ == "__main__":
    assert get_preset("hsa", "beta").n_prototypes == 2000
    assert get_preset("human", "IGH").n_components_recon == 300
    assert get_preset("mouse", "TRA").n_components == 50
    assert get_preset("human", "TRD") is CHAIN_PRESETS[("human", "TRD")]
    print("mir.embedding.presets self-check OK")
    print(f"{'chain':<12}{'n_proto':>8}{'PCs(95%)':>9}{'PCs(99%)':>9}")
    for (sp, lo), p in CHAIN_PRESETS.items():
        print(f"{sp+'_'+lo:<12}{p.n_prototypes:>8}{p.n_components:>9}{p.n_components_recon:>9}")
