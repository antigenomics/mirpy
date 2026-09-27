"""Repertoire signatures, geometry half (``rsig``).

The column contract lives in :mod:`vdjtools.signature.layout` -- mirpy depends on vdjtools and not
the reverse -- and is re-exported here so a caller of ``mir.signature`` need not know that. The
``rsig`` raw groups and channels are **declared** in :mod:`mir.signature.signature`, which registers
them into that same registry at import.

Two modules:

* :mod:`~mir.signature.features` -- the prototype-sum measure ``Phi`` and its functionals, a pure
  function of one sample plus the bundled prototype panel.
* :mod:`~mir.signature.signature` -- ``rsig`` / ``rsig_cohort`` / ``synthesize``, using
  :mod:`vdjtools.signature.corpus` for the fit, the winsorization and the artifact. The same
  machinery fits both halves, so they cannot drift into different notions of "standardised".
"""
from vdjtools.signature import (
    LOCI,
    NO_LOCUS,
    SUPPORTS,
    TRANSFORMS,
    Channel,
    Corpus,
    RawGroup,
    channel_columns,
    channels,
    parse,
    pc_columns,
    raw_columns,
    raw_groups,
    signature_columns,
    support_of,
)

from .features import BANDS, CHUNK, ISOTYPE_BANDS, K, band_shares, isotype_shares, prototype_sum
from .features import rao_of, slots, weights
from .signature import raw_and_channels, rsig, rsig_cohort, synthesize

__all__ = [
    "BANDS", "CHUNK", "Channel", "Corpus", "ISOTYPE_BANDS", "K", "LOCI", "NO_LOCUS", "RawGroup",
    "SUPPORTS", "TRANSFORMS", "band_shares", "channel_columns", "channels", "isotype_shares",
    "parse", "pc_columns", "prototype_sum", "rao_of", "raw_and_channels", "raw_columns",
    "raw_groups", "rsig", "rsig_cohort", "signature_columns", "slots", "support_of", "synthesize",
    "weights",
]
