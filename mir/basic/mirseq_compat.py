"""Compatibility layer for optional mirseq C-extension helpers.

Provides stable ``is_coding`` and ``is_canonical`` callables regardless of
whether the underlying extension exports them.
"""

from __future__ import annotations

from mir.basic.mirseq import translate_bidi

_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


try:
    from mir.basic.mirseq import is_coding as _is_coding
    from mir.basic.mirseq import is_canonical as _is_canonical
except ImportError:

    def _is_coding(junction_aa: str) -> bool:
        if not junction_aa:
            return False
        return all(c in _STANDARD_AA for c in junction_aa.upper())

    def _is_canonical(junction_aa: str) -> bool:
        if not junction_aa or len(junction_aa) < 2:
            return False
        aa = junction_aa.upper()
        return aa[0] == "C" and aa[-1] in {"F", "W"}


is_coding = _is_coding
is_canonical = _is_canonical

__all__ = ["translate_bidi", "is_coding", "is_canonical"]
