"""mir.signature — the geometry half of the portable repertoire signature.

The column contract lives in :mod:`vdjtools.signature`, which mirpy already depends on, so
there is one implementation of the layout, the transforms and the frozen reference rescaling
rather than two that can drift apart. This package supplies the ``rsig`` blocks: features of
the prototype-sum measure ``Φ(S) = Σ_σ w_σ z_σ``, each a linear functional, a norm, or a
mixture coefficient of it.
"""
# Re-exported so a caller of ``mir.signature`` never has to know the contract is implemented in
# vdjtools. One layout, two import paths — not two layouts.
from vdjtools.signature.layout import LOCI, TIERS, columns, describe, index, parse

from .assemble import rsig, signature, signature_cohort
from .blocks import (
    BANDS,
    CHUNK,
    ISOTYPE_BANDS,
    WEIGHTS,
    band_shares,
    contrast,
    depth_block,
    isotype_shares,
    prototype_sum,
    slots,
    weights,
)
from .scale import (
    MIN_N_OBS,
    ScaleReference,
    fit_scale,
    load_scale,
    measure_constants,
    save_scale,
)
from .reference import (
    DEFAULT_PATH,
    LocusReference,
    SignatureReference,
    load_reference,
    self_test,
)

__all__ = [
    "BANDS",
    "DEFAULT_PATH",
    "CHUNK",
    "LOCI",
    "TIERS",
    "columns",
    "describe",
    "index",
    "parse",
    "ISOTYPE_BANDS",
    "WEIGHTS",
    "band_shares",
    "contrast",
    "depth_block",
    "isotype_shares",
    "prototype_sum",
    "slots",
    "MIN_N_OBS",
    "LocusReference",
    "ScaleReference",
    "fit_scale",
    "load_scale",
    "measure_constants",
    "save_scale",
    "SignatureReference",
    "load_reference",
    "rsig",
    "self_test",
    "signature",
    "signature_cohort",
    "weights",
]
