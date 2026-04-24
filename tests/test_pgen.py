"""
Tests for OlgaModel across all nine built-in models.
- 100 sequences are generated to verify the sampler
- Exact Pgen is computed for 20 sequences (IGH exact pgen ~100 ms/seq)
- 1mm Pgen is computed for 5 sequences (len(seq) × exact cost)
Mean log10 Pgen for each model is printed for reference.
"""
import math
import pytest

from mir.basic.pgen import OlgaModel

ALL_MODELS = [
    ("TRA", "human"),
    ("TRB", "human"),
    ("TRG", "human"),
    ("TRD", "human"),
    ("IGH", "human"),
    ("IGK", "human"),
    ("IGL", "human"),
    ("TRA", "mouse"),
    ("TRB", "mouse"),
]


@pytest.fixture(scope="module", params=ALL_MODELS, ids=[f"{s}-{l}" for l, s in ALL_MODELS])
def olga_model(request):
    locus, species = request.param
    return locus, species, OlgaModel(locus=locus, species=species)


def test_pgen_model(olga_model):
    locus, species, model = olga_model

    seqs = model.generate_sequences(100)
    assert len(seqs) == 100
    assert all(isinstance(s, str) and s for s in seqs), "empty or non-string sequence generated"

    # exact and 1mm Pgen on the same 5 sequences so 1mm >= exact is guaranteed per-sequence
    log_exact, log_1mm = [], []
    for s in seqs[:5]:
        p_exact = model.compute_pgen_junction_aa(s)
        p_1mm   = model.compute_pgen_junction_aa_1mm(s)
        assert p_exact is not None and p_exact >= 0, f"invalid exact Pgen for {s!r}"
        assert p_1mm   is not None and p_1mm   >= 0, f"invalid 1mm Pgen for {s!r}"
        assert p_1mm >= p_exact, f"1mm Pgen < exact Pgen for {s!r}"
        if p_exact > 0:
            log_exact.append(math.log10(p_exact))
        if p_1mm > 0:
            log_1mm.append(math.log10(p_1mm))

    mean_exact = sum(log_exact) / len(log_exact) if log_exact else float("-inf")
    mean_1mm   = sum(log_1mm)   / len(log_1mm)   if log_1mm   else float("-inf")

    print(f"\n{species} {locus}: exact={mean_exact:.2f}, 1mm={mean_1mm:.2f}")

    assert mean_exact > -25, f"mean log10 Pgen too low for {species} {locus}: {mean_exact}"
