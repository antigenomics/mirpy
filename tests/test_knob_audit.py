"""Every knob that moves a number must move only the numbers it is documented to move.

This file exists because of the class of bug that produced mirpy 3.20.0, not because of any one
bug in it. Twice now a knob has quietly changed emitted values with nothing in the matrix to say
so: a hard clip collapsed every sample past the bound onto one number, and a flat coverage
constant produced a whole diversity block at a level nobody established. Both were invisible
*in the output*, which is the only place a collaborator ever looks.

A unit fixture cannot catch that. The blast radius of a knob is a property of a **cohort** -- how
many samples land in a tail, how many loci a reference covers -- so these run on a seven-locus
cohort at a realistic clonotype spread. The assertions are all of one shape: turn the knob, take
the set of cells that moved, and check it against the set the knob is allowed to touch.
"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from mir.signature import rsig_cohort
from mir.signature.scale import load_scale

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
# Roughly the mix a bulk-RNA-seq blood cohort gives: IGH and TRB dominate, TRD is a rounding
# error. A TRB-only fixture cannot see a reference's B-cell coverage at all.
_LOCI = {"TRB": ("TRBV{}*01", "TRBJ{}*01", 28, 13, 0.27), "IGH": ("IGHV{}*01", "IGHJ{}*01", 7, 6, 0.30),
         "IGK": ("IGKV{}*01", "IGKJ{}*01", 6, 5, 0.17), "TRA": ("TRAV{}*01", "TRAJ{}*01", 40, 55, 0.13),
         "IGL": ("IGLV{}*01", "IGLJ{}*01", 5, 5, 0.10), "TRG": ("TRGV{}*01", "TRGJ{}*01", 6, 3, 0.02),
         "TRD": ("TRDV{}*01", "TRDJ{}*01", 3, 4, 0.01)}


def cohort(n_samples: int = 8, size: int = 900, seed: int = 0) -> dict:
    out = {}
    for i in range(n_samples):
        r = np.random.default_rng(seed + i)
        # a 39x spread between the shallow and deep samples, which is what the real cohort has
        depth = int(size * np.exp(r.normal(0, 0.9)))
        frames = {}
        for loc, (vf, jf, nv, nj, share) in _LOCI.items():
            n = max(12, int(depth * share))
            ln = r.integers(11, 18, n)
            frames[loc] = pl.DataFrame({
                "junction_aa": ["C" + "".join(_AA[r.integers(0, 20, k)]) + "F" for k in ln],
                "v_call": [vf.format(v) for v in r.integers(1, nv, n)],
                "j_call": [jf.format(v) for v in r.integers(1, nj, n)],
                "duplicate_count": np.ceil(r.zipf(1.4, n).clip(1, 3000)).astype(int).tolist()})
        out[f"S{i:02d}"] = frames
    return out


@pytest.fixture(scope="module")
def coh():
    return cohort()


def _emit(coh, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsig_cohort(coh, tier="standard", n_jobs=1, **kw)


def _moved(a, b):
    """``{(sample_id, column)}`` where two emitted frames disagree, nan-aware."""
    out = set()
    for c in a.columns:
        if c == "sample_id":
            continue
        for sid, x, y in zip(a["sample_id"], a[c].to_list(), b[c].to_list()):
            x = np.nan if x is None else x
            y = np.nan if y is None else y
            if not (x == y or (np.isnan(x) and np.isnan(y))):
                out.add((sid, c))
    return out


def test_the_cohort_actually_exercises_the_bound(coh):
    """A guard on the guard: with no saturation the tests below are vacuous."""
    ref = load_scale()
    sat = ref.saturation(_emit(coh, clip=8.0), clip=8.0)
    assert sat.height > 50, "too few scaled columns to say anything"
    assert float(sat["n_out"].sum()) > 0, "nothing lands outside the bound; widen the fixture"


def test_clip_moves_exactly_the_cells_saturation_reports(coh):
    """The whole "never silently" claim, as an equality rather than a slogan.

    Loosening the bound may only change cells that were beyond the tighter one -- and those are
    exactly the cells ``saturation`` names. If a knob can move a cell that no report mentions,
    the report is decoration.
    """
    ref = load_scale()
    tight, loose = _emit(coh, clip=2.0), _emit(coh, clip=8.0)
    moved = _moved(tight, loose)
    assert moved, "clip=2 vs clip=8 must move something on this cohort"

    flagged = set()
    sat_cols = set(ref.saturation(tight, clip=2.0)["column"])
    for c in sat_cols:
        for sid, v in zip(tight["sample_id"], tight[c].to_list()):
            if v is not None and np.isfinite(v) and abs(v) > 2.0:
                flagged.add((sid, c))
    assert moved <= flagged, f"{len(moved - flagged)} cells moved that saturation never named"


def test_the_squash_moves_only_cells_outside_the_bound(coh):
    soft, hard = _emit(coh, clip=8.0), _emit(coh, clip=8.0, squash="hard")
    moved = _moved(soft, hard)
    assert moved, "the two squashes must differ somewhere on this cohort"
    inside = {(sid, c) for (sid, c) in moved
              if abs(soft[c].to_list()[soft["sample_id"].to_list().index(sid)]) <= 8.0}
    assert not inside, f"{len(inside)} cells inside the bound moved; only the tail may"


def test_on_unscaled_hole_moves_only_columns_the_reference_never_scaled(coh):
    ref = load_scale()
    passed, holed = _emit(coh), _emit(coh, on_unscaled="hole")
    scaled = {c for c in ref.columns if ref.scale[ref.columns.index(c)] > 0}
    moved_cols = {c for _, c in _moved(passed, holed)}
    assert moved_cols, "some column in this tier must be unscaled for the test to mean anything"
    assert not (moved_cols & scaled), "a scaled column must not change"


def test_a_reference_swap_moves_only_scaled_columns_and_saturation_says_how_much(coh):
    """Switching references is the operation that cost a transfer model 0.31 AUC."""
    from mir.signature.scale import MODELS, _RES

    names = [n for n, f in MODELS.items() if (_RES / f).exists()]
    assert len(names) >= 2
    a, b = _emit(coh, scale=load_scale(names[0])), _emit(coh, scale=load_scale(names[1]))
    moved_cols = {c for _, c in _moved(a, b)}
    union = {c for n in names[:2] for c in load_scale(n).columns
             if load_scale(n).scale[load_scale(n).columns.index(c)] > 0}
    assert moved_cols <= union, "a column neither reference scales cannot move"

    # ...and the difference in truncation pressure is reported, not left to be discovered
    # downstream. This is `compare_references` doing the job the AUC drop did.
    from mir.signature import compare_references

    t = compare_references(_emit(coh, standardize="none"), names[:2])
    assert t.height == 2 and t["frac_out_of_bound"].to_list() == sorted(
        t["frac_out_of_bound"].to_list())


def test_n_jobs_moves_nothing(coh):
    """Parallelism that moves a number is a bug with a speedup. Seven loci, not one."""
    assert not _moved(_emit(coh), rsig_cohort(coh, tier="standard", n_jobs=3, clip=8.0))
