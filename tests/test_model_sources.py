"""Model selection contracts without model training or sequence generation."""
import inspect
from types import SimpleNamespace

import numpy as np
import polars as pl


def test_synthetic_defaults_and_explicit_olga():
    from mir.cli import build_parser
    from mir.signature import synthesize

    assert inspect.signature(synthesize).parameters['source'].default == 'isalgo'
    assert build_parser().parse_args(['corpus', '--corpus', 'naive']).source == 'isalgo'
    assert build_parser().parse_args(['corpus', '--corpus', 'naive', '--source', 'olga-legacy']).source == 'olga-legacy'
    for alias in ('olga', 'learned', 'arda'):
        assert build_parser().parse_args(['corpus', '--corpus', 'naive', '--source', alias]).source == alias


def test_naive_reference_uses_species_source_and_separate_cache(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.repertoire import naive_reference

    loaded = []

    def load(locus, *, source, organism):
        loaded.append((locus, source, organism))
        return source

    monkeypatch.setattr(models, 'load_bundled', load)
    sampled = []
    def sample(model, n, **kwargs):
        sampled.append(kwargs['engine'])
        return np.full((n, 2), 1. if model == 'isalgo' else 2.)
    monkeypatch.setattr(generate, 'generate', sample)
    space = SimpleNamespace(meta={'locus': 'TRA', 'species': 'mouse'}, _naive={},
        rff=SimpleNamespace(transform=lambda values: values), transform_clonotypes=lambda values: values)
    learned = naive_reference(space, n=3)
    assert naive_reference(space, n=3) is learned
    for alias in ("learned", "arda"):
        assert naive_reference(space, n=3, source=alias) is learned
    historical = naive_reference(space, n=3, source='olga-legacy')
    assert loaded == [('TRA', 'isalgo', 'mouse'), ('TRA', 'olga-legacy', 'mouse')]
    assert np.array_equal(learned, [1., 1.])
    assert np.array_equal(historical, [2., 2.])
    assert naive_reference(space, n=3, source='olga-legacy') is historical
    assert naive_reference(space, n=3, source='olga') is historical
    assert sampled == ['native', 'reference']


def test_background_forwards_isalgo_mouse_catalog(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.density import generate_background

    calls = []
    monkeypatch.setattr(models, 'load_bundled', lambda locus, **kwargs: calls.append((locus, kwargs)))
    monkeypatch.setattr(generate, 'generate', lambda *args, **kwargs:
        pl.DataFrame({'v_call': ['TRAV1'], 'j_call': ['TRAJ1'], 'junction_aa': ['CAVF']}))
    assert generate_background('TRA', 1, species='mouse').height == 1
    assert calls == [('TRA', {'source': 'isalgo', 'organism': 'mouse'})]


def test_background_defaults_use_native_for_isalgo_aliases_and_preserve_legacy_streams(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.density import generate_background

    calls = []
    monkeypatch.setattr(models, 'load_bundled', lambda *args, **kwargs: 'model')
    def sample(model, n, **kwargs):
        calls.append((model, n, kwargs))
        return pl.DataFrame({'v_call': ['TRAV1'], 'j_call': ['TRAJ1'], 'junction_aa': ['CAVF']})
    monkeypatch.setattr(generate, 'generate', sample)
    for source in ('isalgo', 'learned', 'arda', 'olga-legacy', 'olga'):
        generate_background('TRA', 1, source=source, threads=1)
    assert calls == [('model', 1, {'seed': 0, 'productive_only': True, 'engine': engine, 'threads': 1})
                     for engine in ('native', 'native', 'native', 'reference', 'reference')]


def test_naive_reference_separates_engines_and_reuses_thread_invariant_draw(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.repertoire import naive_reference

    calls = []
    monkeypatch.setattr(models, 'load_bundled', lambda *args, **kwargs: 'model')
    def sample(model, n, **kwargs):
        calls.append(kwargs)
        return np.full((n, 2), 1. if kwargs['engine'] == 'reference' else 2.)
    monkeypatch.setattr(generate, 'generate', sample)
    space = SimpleNamespace(meta={'locus': 'TRA', 'species': 'mouse'}, _naive={},
        rff=SimpleNamespace(transform=lambda values: values), transform_clonotypes=lambda values: values)
    historical = naive_reference(space, n=3, engine='reference', threads=1)
    native = naive_reference(space, n=3, threads=1)
    assert naive_reference(space, n=3, engine='native', threads=0) is native
    assert [call['engine'] for call in calls] == ['reference', 'native']
    assert [call['threads'] for call in calls] == [1, 1]
    assert np.array_equal(historical, [1., 1.]) and np.array_equal(native, [2., 2.])


def test_frozen_olga_aliases_keep_the_reference_stream():
    from mir.density import generate_background
    from vdjtools.model import load_bundled
    from vdjtools.model.generate import generate

    expected = generate(load_bundled('TRG', 'olga', collapse=True), 4, seed=5,
                        productive_only=True, engine='reference', threads=1)
    expected = expected.select('v_call', 'j_call', 'junction_aa')
    for source in ('olga', 'olga-legacy'):
        actual = generate_background('TRG', 4, source=source, seed=5, threads=1)
        assert actual.select(expected.columns).equals(expected)


def test_prepared_release_metadata_requires_published_parent_contracts():
    import tomllib
    from pathlib import Path
    import mir

    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())['project']
    assert mir.__version__ == '4.10.0'
    assert 'vdjtools>=5.4.0' in project['dependencies']
    assert 'vdjtools[model]>=5.4.0' in project['optional-dependencies']['build']
    assert 'arda-mapper>=2.38.2' in project['optional-dependencies']['build']
