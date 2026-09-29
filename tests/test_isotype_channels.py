"""``isotype_masks`` / ``isotype_shares`` must not change when their implementation does.

Both built their masks by materialising the whole allele-stripped ``c_call`` column into a Python
list and running one ``in`` test per clonotype per band. The membership tests turned out **not** to
be the cost -- ``strip_allele`` was, at 100.7 ms of a 132.2 ms call on 200,000 rows, because the
expression resolves the allele suffix once per row while a ``c_call`` column carries a few dozen
distinct values. Both now share one pass through ``strip_allele_values``.

So what is pinned is the answer, on the inputs where a stripping change would show: comma
ambiguity, surrounding whitespace, nulls, a gene outside every band, and a band under the floor.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mir.signature.features import ISOTYPE_BANDS, isotype_masks, isotype_shares

GENES = ["IGHM", "IGHD", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2"]


def _frame(calls):
    n = len(calls)
    return pl.DataFrame({"c_call": pl.Series(calls, dtype=pl.Utf8),
                         "duplicate_count": np.arange(1, n + 1)})


def _w(df):
    a = df["duplicate_count"].to_numpy().astype(float)
    return a / a.sum()


def test_a_frame_with_no_c_call_is_empty_not_an_error():
    """An absent constant region is a TCR, or a BCR protocol that did not sequence it."""
    df = pl.DataFrame({"duplicate_count": [1, 2, 3]})
    assert isotype_masks(df) == {}
    assert isotype_shares(df, np.ones(3) / 3) == {}


def test_the_masks_are_the_declared_bands_and_nothing_else():
    df = _frame(["IGHM*01", "IGHG1*02", "IGHA2", "IGHE*01", None])
    m = isotype_masks(df)
    assert list(m) == list(ISOTYPE_BANDS)
    assert m["IgM"].tolist() == [True, False, False, False, False]
    assert m["IgG"].tolist() == [False, True, False, False, False]
    assert m["IgA"].tolist() == [False, False, True, False, False]


def test_an_allele_suffix_never_stops_a_call_matching_its_band():
    """``IGHG1*01`` matching nothing would report a plausible all-uncalled composition."""
    assert isotype_masks(_frame(["IGHG1*01"]))["IgG"].tolist() == [True]
    assert isotype_masks(_frame(["IGHG1"]))["IgG"].tolist() == [True]
    assert isotype_masks(_frame([" IGHG1*01 "]))["IgG"].tolist() == [True]


def test_a_null_or_unbanded_call_is_false_in_every_band():
    m = isotype_masks(_frame([None, "IGHE*01", "", "NOTAGENE"]))
    assert not any(v.any() for v in m.values())


def test_an_ambiguous_call_matches_only_when_the_stripped_form_does():
    """``strip_allele`` keeps a cross-gene tie as a comma-joined pair, which is in no band.

    That is the behaviour, not a defect: an ambiguous call is not evidence for either isotype, and
    an implementation that silently took the first gene would invent that evidence.
    """
    m = isotype_masks(_frame(["IGHG1*01,IGHG2*01", "IGHG1*02,IGHG1*04"]))
    assert m["IgG"].tolist() == [False, True], "an allele-level tie collapses; a gene-level one does not"


def test_shares_sum_to_one_with_the_uncalled_residual():
    df = _frame([g for g in GENES for _ in range(6)])
    s = isotype_shares(df, _w(df))
    assert set(s) == set(ISOTYPE_BANDS) | {"_uncalled"}
    assert s["_uncalled"] == pytest.approx(0.0, abs=1e-12)
    assert sum(s.values()) == pytest.approx(1.0, rel=1e-12)


def test_a_band_under_the_floor_is_dropped_and_falls_into_uncalled():
    """Declining to report a band is different from reporting it as zero."""
    df = _frame(["IGHM*01"] * 50 + ["IGHG1*01"] * 2)
    s = isotype_shares(df, _w(df), min_clonotypes=5)
    assert "IgM" in s and "IgG" not in s, s
    assert s["_uncalled"] > 0.0


def test_shares_and_masks_agree_about_which_rows_are_in_a_band():
    """They share one pass now; before, they were two copies of the same comprehension."""
    df = _frame([None] + [g + "*01" for g in GENES for _ in range(9)])
    w = _w(df)
    masks, shares = isotype_masks(df), isotype_shares(df, w)
    for name in ISOTYPE_BANDS:
        assert shares[name] == pytest.approx(float(w[masks[name]].sum()), rel=1e-12)
