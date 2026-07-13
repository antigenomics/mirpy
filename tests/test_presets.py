import polars as pl

from mir.embedding import get_preset
from mir.embedding.presets import CHAIN_PRESETS, ChainPreset
from mir.embedding.tcremp import TCREmp


def test_get_preset_alias_resolution():
    a = get_preset("human", "TRB")
    b = get_preset("hsa", "beta")
    assert a == b
    assert isinstance(a, ChainPreset)
    assert a.n_prototypes == 2000
    assert a.n_components < a.n_components_recon   # 99% keeps more than 95%


def test_preset_covers_all_bundled_pairs():
    from mir.embedding.prototypes import list_available_prototypes

    for species, locus in list_available_prototypes():
        assert (species, locus) in CHAIN_PRESETS


def test_unknown_pair_falls_back():
    p = get_preset("human", "IGK")   # compact chain
    assert p.n_prototypes == 1000 and p.n_components == 20


def test_from_defaults_uses_preset():
    m = TCREmp.from_defaults("human", "IGK")   # n_prototypes=None -> preset
    assert m.n_prototypes == get_preset("human", "IGK").n_prototypes

    df = pl.DataFrame({"v_call": ["IGKV1-5*01"], "j_call": ["IGKJ1*01"],
                       "junction_aa": ["CQQYNSYSLTF"]})
    assert m.embed(df).shape == (1, 3 * m.n_prototypes)
