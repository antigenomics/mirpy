"""Model selection contracts without model training or sequence generation."""
import inspect
from types import SimpleNamespace

import numpy as np
import polars as pl


def test_synthetic_defaults_and_explicit_olga():
    from mir.cli import build_parser
    from mir.signature import synthesize

    assert inspect.signature(synthesize).parameters['source'].default == 'learned'
    assert build_parser().parse_args(['corpus', '--corpus', 'naive']).source == 'learned'
    assert build_parser().parse_args(['corpus', '--corpus', 'naive', '--source', 'olga']).source == 'olga'


def test_naive_reference_uses_species_source_and_separate_cache(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.repertoire import naive_reference

    loaded = []

    def load(locus, *, source, organism):
        loaded.append((locus, source, organism))
        return source

    monkeypatch.setattr(models, 'load_bundled', load)
    monkeypatch.setattr(generate, 'generate', lambda model, n, **kwargs:
        np.full((n, 2), 1. if model == 'learned' else 2.))
    space = SimpleNamespace(meta={'locus': 'TRA', 'species': 'mouse'}, _naive={},
        rff=SimpleNamespace(transform=lambda values: values), transform_clonotypes=lambda values: values)
    learned = naive_reference(space, n=3)
    assert naive_reference(space, n=3) is learned
    historical = naive_reference(space, n=3, source='olga')
    assert loaded == [('TRA', 'learned', 'mouse'), ('TRA', 'olga', 'mouse')]
    assert np.array_equal(learned, [1., 1.])
    assert np.array_equal(historical, [2., 2.])
    assert naive_reference(space, n=3, source='olga') is historical


def test_background_forwards_learned_mouse_catalog(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.density import generate_background

    calls = []
    monkeypatch.setattr(models, 'load_bundled', lambda locus, **kwargs: calls.append((locus, kwargs)))
    monkeypatch.setattr(generate, 'generate', lambda *args, **kwargs:
        pl.DataFrame({'v_call': ['TRAV1'], 'j_call': ['TRAJ1'], 'junction_aa': ['CAVF']}))
    assert generate_background('TRA', 1, species='mouse').height == 1
    assert calls == [('TRA', {'source': 'learned', 'organism': 'mouse'})]


def test_background_forwards_native_engine_and_cpu_budget(monkeypatch):
    import vdjtools.model as models
    from vdjtools.model import generate
    from mir.density import generate_background

    calls = []
    monkeypatch.setattr(models, 'load_bundled', lambda *args, **kwargs: 'model')
    def sample(model, n, **kwargs):
        calls.append((model, n, kwargs))
        return pl.DataFrame({'v_call': ['TRAV1'], 'j_call': ['TRAJ1'], 'junction_aa': ['CAVF']})
    monkeypatch.setattr(generate, 'generate', sample)
    generate_background('TRA', 1, engine='native', threads=1)
    assert calls == [('model', 1, {'seed': 0, 'productive_only': True, 'engine': 'native', 'threads': 1})]


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
    historical = naive_reference(space, n=3, threads=1)
    native = naive_reference(space, n=3, engine='native', threads=1)
    assert naive_reference(space, n=3, engine='native', threads=0) is native
    assert [call['engine'] for call in calls] == ['reference', 'native']
    assert [call['threads'] for call in calls] == [1, 1]
    assert np.array_equal(historical, [1., 1.]) and np.array_equal(native, [2., 2.])
