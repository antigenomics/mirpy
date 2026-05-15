"""Centralized species/locus alias utilities.

This module provides canonicalization helpers shared across control setup,
parser logic, and OLGA model resolution.
"""

from __future__ import annotations

_SPECIES_ALIASES: dict[str, str] = {
    "human": "human",
    "hsa": "human",
    "homosapiens": "human",
    "homo_sapiens": "human",
    "homo sapiens": "human",
    "mouse": "mouse",
    "mmu": "mouse",
    "musmusculus": "mouse",
    "mus_musculus": "mouse",
    "mus musculus": "mouse",
}

# Canonical IMGT locus aliases used by users/notebooks/tools.
_LOCUS_ALIASES: dict[str, str] = {
    "TRA": "TRA",
    "TALPHA": "TRA",
    "T_ALPHA": "TRA",
    "T-ALPHA": "TRA",
    "ALPHA": "TRA",
    "TRB": "TRB",
    "TBETA": "TRB",
    "T_BETA": "TRB",
    "T-BETA": "TRB",
    "BETA": "TRB",
    "TRG": "TRG",
    "TGAMMA": "TRG",
    "T_GAMMA": "TRG",
    "T-GAMMA": "TRG",
    "GAMMA": "TRG",
    "TRD": "TRD",
    "TDELTA": "TRD",
    "T_DELTA": "TRD",
    "T-DELTA": "TRD",
    "DELTA": "TRD",
    "IGH": "IGH",
    "BHEAVY": "IGH",
    "B_HEAVY": "IGH",
    "B-HEAVY": "IGH",
    "HEAVY": "IGH",
    "IGK": "IGK",
    "BKAPPA": "IGK",
    "B_KAPPA": "IGK",
    "B-KAPPA": "IGK",
    "KAPPA": "IGK",
    "IGL": "IGL",
    "BLAMBDA": "IGL",
    "B_LAMBDA": "IGL",
    "B-LAMBDA": "IGL",
    "LAMBDA": "IGL",
}

AIRR_LOCUS_ALIASES: dict[str, set[str]] = {
    "alpha": {"alpha", "tra"},
    "beta": {"beta", "trb"},
    "gamma": {"gamma", "trg"},
    "delta": {"delta", "trd"},
    "heavy": {"heavy", "igh"},
    "kappa": {"kappa", "igk"},
    "lambda": {"lambda", "igl"},
}

AIRR_ALIAS_TO_IMGT: dict[str, str] = {
    "alpha": "TRA", "tra": "TRA",
    "beta": "TRB", "trb": "TRB",
    "gamma": "TRG", "trg": "TRG",
    "delta": "TRD", "trd": "TRD",
    "heavy": "IGH", "igh": "IGH",
    "kappa": "IGK", "igk": "IGK",
    "lambda": "IGL", "igl": "IGL",
}

LOCUS_TO_OLGA_SUFFIX: dict[str, str] = {
    "TRA": "T_alpha",
    "TRB": "T_beta",
    "TRG": "T_gamma",
    "TRD": "T_delta",
    "IGH": "B_heavy",
    "IGK": "B_kappa",
    "IGL": "B_lambda",
}

OLGA_SUFFIX_TO_LOCUS: dict[str, str] = {
    "T_ALPHA": "TRA",
    "T_BETA": "TRB",
    "T_GAMMA": "TRG",
    "T_DELTA": "TRD",
    "B_HEAVY": "IGH",
    "B_KAPPA": "IGK",
    "B_LAMBDA": "IGL",
}

_LOCUS_SEARCH_TOKENS: dict[str, set[str]] = {
    "TRA": {"tra", "talpha"},
    "TRB": {"trb", "tbeta"},
    "TRG": {"trg", "tgamma"},
    "TRD": {"trd", "tdelta"},
    "IGH": {"igh", "bheavy"},
    "IGK": {"igk", "bkappa"},
    "IGL": {"igl", "blambda"},
}


def normalize_species_alias(species: str) -> str:
    key = (species or "").strip().lower().replace("-", "_")
    if key not in _SPECIES_ALIASES:
        raise ValueError(f"Unsupported species alias: {species!r}")
    return _SPECIES_ALIASES[key]


def normalize_locus_alias(locus: str) -> str:
    key = (locus or "").strip().upper().replace("-", "_")
    key = key.replace(" ", "")
    if key not in _LOCUS_ALIASES:
        raise ValueError(f"Unsupported locus alias: {locus!r}")
    return _LOCUS_ALIASES[key]


def normalize_airr_locus_value(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    return AIRR_ALIAS_TO_IMGT.get(raw, raw.upper()[:3])


def airr_aliases_for_locus(locus: str) -> set[str]:
    locus_norm = (locus or "").strip().lower()
    if locus_norm in AIRR_LOCUS_ALIASES:
        return AIRR_LOCUS_ALIASES[locus_norm]
    for aliases in AIRR_LOCUS_ALIASES.values():
        if locus_norm in aliases:
            return aliases
    return {locus_norm}


def locus_search_tokens(locus_imgt: str) -> set[str]:
    return set(_LOCUS_SEARCH_TOKENS.get(locus_imgt, {locus_imgt.lower()}))
