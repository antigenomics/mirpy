from mir.alleles import allele_to_major, allele_with_default, strip_allele


def test_strip_allele():
    assert strip_allele("TRBV6-5*02") == "TRBV6-5"
    assert strip_allele("TRBV6-5") == "TRBV6-5"
    assert strip_allele("") == ""
    assert strip_allele(None) == ""


def test_allele_with_default():
    assert allele_with_default("TRBV6-5") == "TRBV6-5*01"
    assert allele_with_default("TRBV6-5*02") == "TRBV6-5*02"
    assert allele_with_default("TRBV6-5*") == "TRBV6-5*01"
    assert allele_with_default(None) == ""


def test_allele_to_major():
    assert allele_to_major("TRBV6-5*02") == "TRBV6-5*01"
    assert allele_to_major("TRBV6-5") == "TRBV6-5*01"
    assert allele_to_major("") == ""
