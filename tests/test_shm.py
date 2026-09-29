"""``shm_penalty_batch`` must equal the scalar ``shm_penalty`` exactly, on every input shape.

The batch said "vectorised" in its first docstring line and was a ``for`` loop calling the scalar
form per row -- re-parsing the mutation string and making one seqtree ``penalty`` call per
substitution, on the embedding path of every frame carrying mutation evidence. It is one polars
pass and one table gather now, and this module had **no tests at all** before that change, which
is the other half of why it went unnoticed.

Equality is asserted with ``==``, not a tolerance: every BLOSUM62 penalty is a small integer, so a
float64 sum of them is exact however it is ordered. A tolerance here would hide a mis-indexed
table, which is the one bug this rewrite could plausibly introduce.
"""
from __future__ import annotations

import numpy as np
import pytest

from mir.distances.shm import (
    V_REGION_AA,
    mean_shm_penalty,
    shm_penalty,
    shm_penalty_batch,
    substitution_penalty,
)

#: Every shape that can reach this function. Note the bare ints: the scalar path branches on
#: ``if mutations:``, so ``3`` is a spec that parses to nothing (cost 0) while ``0`` is falsy and
#: falls through to the identity path. Casting to string first would make ``"0"`` truthy and score
#: that row from the wrong branch -- plausibly, and with nothing about the number to show it had.
SPECS = [
    "A23V", "A23V,S31N", "a23v", "A23V;S31N", "A23V S31N", "  A23V , S31N ", "A23A",
    "", None, "AB", "A2V", "23V", "A23", "X5Y", "Z9B", "*3A", "A3*", "A23V,,S31N", ",",
    "A23V,AB,S31N", 3, 0, "A1B,C2D,E3F,G4H,I5K", "M1M",
]


def _reference(mutations, identity):
    lam = mean_shm_penalty()
    muts = mutations if mutations is not None else [None] * len(identity)
    iden = identity if identity is not None else [None] * len(mutations)
    return np.array([shm_penalty(m, identity=d, v_length=V_REGION_AA, lambda_scalar=lam)
                     for m, d in zip(muts, iden)])


def _same(a, b):
    """Exact equality, with ``nan`` required in the same places rather than compared."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.isnan(a), np.isnan(b)
    return np.array_equal(na, nb) and np.array_equal(a[~na], b[~nb])


def test_the_batch_equals_the_scalar_path_on_every_spec_shape():
    assert _same(shm_penalty_batch(SPECS), _reference(SPECS, None))


def test_a_spec_wins_over_an_identity_row_by_row():
    """Both arguments given: the mutation list is the better evidence wherever it exists."""
    iden = [None if i % 3 else 0.9 for i in range(len(SPECS))]
    assert _same(shm_penalty_batch(SPECS, iden), _reference(SPECS, iden))


def test_the_identity_only_path():
    iden = [1.0, 0.98, 0.9, None, float("nan"), 0.5, 1.5]
    got = shm_penalty_batch(None, iden)
    assert _same(got, _reference(None, iden))
    assert got[0] == 0.0, "identity 1.0 is germline and costs nothing"
    assert np.isnan(got[3]) and np.isnan(got[4]), "no evidence is nan, not zero"
    assert got[6] == 0.0, "identity above 1 is clamped, not negative"


@pytest.mark.parametrize("n", [1, 7, 4000])
def test_it_equals_the_scalar_path_at_scale(n):
    rng = np.random.default_rng(n)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    specs = [",".join(f"{aa[rng.integers(20)]}{rng.integers(1, 99)}{aa[rng.integers(20)]}"
                      for _ in range(rng.integers(0, 40))) for _ in range(n)]
    iden = rng.uniform(0.8, 1.0, n).tolist()
    assert _same(shm_penalty_batch(specs, iden), _reference(specs, iden))


@pytest.mark.parametrize("spec", ["J23V", "A23O", "U1A", "é23V"])
def test_a_residue_blosum62_does_not_define_still_raises(spec):
    """J, O and U are letters and are not in BLOSUM62; a lookup table must not score them 0.

    The scalar path raised because seqtree rejected the pair. A table with a silent default would
    return a finite, plausible penalty instead -- so both paths are pinned to raise together.
    """
    with pytest.raises(ValueError):
        shm_penalty(spec)
    with pytest.raises(ValueError):
        shm_penalty_batch([spec])


@pytest.mark.parametrize("spec", ["23V,A1V", "A23,A1V", "AB,A1V"])
def test_a_token_that_is_not_a_substitution_is_skipped_in_silence(spec):
    """Distinct from the case above: a non-letter is malformed input the parser drops by design.

    ``parse_mutations`` keeps only tokens whose first and last characters are letters, so these
    contribute nothing and the surviving ``A1V`` carries the whole penalty. Telling this case from
    the raising one is the only reason the alpha check survives at all.
    """
    assert shm_penalty_batch([spec])[0] == substitution_penalty("A", "V") == shm_penalty(spec)


def test_a_length_mismatch_raises_rather_than_scoring_the_wrong_rows():
    with pytest.raises(ValueError, match="entries"):
        shm_penalty_batch(["A1V", "A2V", "A3V"], [0.9])
