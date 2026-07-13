import pytest

from mir.aliases import normalize_locus_alias, normalize_species_alias


@pytest.mark.parametrize(
    "raw,expected",
    [("human", "human"), ("hsa", "human"), ("Homo sapiens", "human"),
     ("mouse", "mouse"), ("Mus_musculus", "mouse")],
)
def test_species_aliases(raw, expected):
    assert normalize_species_alias(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("TRB", "TRB"), ("beta", "TRB"), ("T-beta", "TRB"),
     ("TRA", "TRA"), ("alpha", "TRA"), ("IGH", "IGH"), ("heavy", "IGH")],
)
def test_locus_aliases(raw, expected):
    assert normalize_locus_alias(raw) == expected


@pytest.mark.parametrize("bad", ["frog", "", "TRXX"])
def test_unknown_raises(bad):
    with pytest.raises(ValueError):
        normalize_locus_alias(bad)
    with pytest.raises(ValueError):
        normalize_species_alias(bad)
