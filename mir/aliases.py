"""Species and locus alias normalization for user-facing entry points.

Users, notebooks, and other tools spell species and loci many ways (``beta``,
``T_beta``, ``hsa``, ``Homo sapiens``). Prototype resource files and germline
lookups are keyed by the canonical IMGT locus (``TRB``) and canonical species
(``human`` / ``mouse``), so normalize once at the boundary.
"""

from __future__ import annotations

_SPECIES_ALIASES: dict[str, str] = {
    "human": "human", "hsa": "human", "homosapiens": "human",
    "homo_sapiens": "human", "homo sapiens": "human",
    "mouse": "mouse", "mmu": "mouse", "musmusculus": "mouse",
    "mus_musculus": "mouse", "mus musculus": "mouse",
}

_LOCUS_ALIASES: dict[str, str] = {
    "TRA": "TRA", "TALPHA": "TRA", "T_ALPHA": "TRA", "ALPHA": "TRA",
    "TRB": "TRB", "TBETA": "TRB", "T_BETA": "TRB", "BETA": "TRB",
    "TRG": "TRG", "TGAMMA": "TRG", "T_GAMMA": "TRG", "GAMMA": "TRG",
    "TRD": "TRD", "TDELTA": "TRD", "T_DELTA": "TRD", "DELTA": "TRD",
    "IGH": "IGH", "BHEAVY": "IGH", "B_HEAVY": "IGH", "HEAVY": "IGH",
    "IGK": "IGK", "BKAPPA": "IGK", "B_KAPPA": "IGK", "KAPPA": "IGK",
    "IGL": "IGL", "BLAMBDA": "IGL", "B_LAMBDA": "IGL", "LAMBDA": "IGL",
}


def normalize_species_alias(species: str) -> str:
    """Return the canonical species (``'human'`` / ``'mouse'``) for an alias."""
    key = (species or "").strip().lower().replace("-", "_")
    if key not in _SPECIES_ALIASES:
        raise ValueError(
            f"Unknown species {species!r}; expected one of "
            f"{sorted(set(_SPECIES_ALIASES.values()))}"
        )
    return _SPECIES_ALIASES[key]


def normalize_locus_alias(locus: str) -> str:
    """Return the canonical IMGT locus (e.g. ``'TRB'``) for an alias."""
    key = (locus or "").strip().upper().replace("-", "_").replace(" ", "")
    if key not in _LOCUS_ALIASES:
        raise ValueError(
            f"Unknown locus {locus!r}; expected one of "
            f"{sorted(set(_LOCUS_ALIASES.values()))}"
        )
    return _LOCUS_ALIASES[key]


if __name__ == "__main__":
    assert normalize_species_alias("Homo sapiens") == "human"
    assert normalize_locus_alias("beta") == "TRB"
    assert normalize_locus_alias("T-alpha") == "TRA"
    for bad in ("frog", ""):
        try:
            normalize_species_alias(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    print("mir.aliases self-check OK")
