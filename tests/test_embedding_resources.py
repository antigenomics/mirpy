"""RFF numerical-block and cohort lifetime limits; large runs are explicitly enabled."""
import gc
import os
import resource
import sys
import time
import weakref
from threading import Lock

import numpy as np
import polars as pl
import pytest

from mir.embedding.table import sample_embeddings, TablePCA
from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding, _sample_weights
from vdjtools.cores import available_cores, kernel_threads


def _spaces(loci=('TRB',), prototypes=16, n_rff=16):
    result = {}
    for loc in loci:
        m = TCREmp.from_defaults('human',loc,n_prototypes=prototypes,threads=0)
        df = pl.DataFrame({'v_call':m._proto_v,'j_call':m._proto_j,
                           'junction_aa':m._proto_junction})
        result[loc] = fit_repertoire_space(m,df,n_components=8,n_rff=n_rff,n_rff_second=8,seed=19)
    return result


def _reader(spaces, i, depth=8):
    result = {}
    for loc, sp in spaces.items():
        m = sp.clono.model
        indexes = np.arange(depth) % len(m._proto_v)
        result[loc] = pl.DataFrame({'v_call':np.asarray(m._proto_v)[indexes],
                                    'j_call':np.asarray(m._proto_j)[indexes],
                                    'junction_aa':np.asarray(m._proto_junction)[indexes],
                                    'duplicate_count':(np.arange(depth)+i)%17+1})
    return result


@pytest.mark.parametrize('workers',[1,2])
def test_only_active_workers_retain_input_frames(workers):
    if workers > available_cores():
        pytest.skip('needs two allocated cores')
    spaces = _spaces()
    state = {'live':0,'peak':0}
    lock = Lock()
    def released():
        with lock:
            state['live'] -= 1
    def read(i):
        assert kernel_threads(0) == 1
        frames = _reader(spaces,i)
        with lock:
            state['live'] += 1
            state['peak'] = max(state['peak'],state['live'])
        weakref.finalize(frames['TRB'],released)
        return frames
    out = sample_embeddings([(str(i),lambda i=i:read(i)) for i in range(100)],spaces,n_jobs=workers)
    gc.collect()
    assert out.height == 100
    assert state['live'] == 0
    assert state['peak'] <= workers


def test_chunked_rff_preserves_mean_second_and_diversity(monkeypatch):
    import mir.repertoire as rep
    space = _spaces()['TRB']
    df = _reader({'TRB':space},3,depth=97)['TRB']
    _,_,weights = _sample_weights(df,'log2p1')
    Z = space.transform_clonotypes(df)
    expected_mean = weights @ space.rff.transform(Z)
    psi = space.rff2.transform(Z)
    expected_second = psi.T @ (weights[:,None]*psi)
    reference = sample_embedding(space,df)
    monkeypatch.setattr(rep,'_RFF_BLOCK',7)
    seen = []
    transform = space.transform_clonotypes
    def checked(part):
        seen.append(part.height)
        return transform(part)
    monkeypatch.setattr(space,'transform_clonotypes',checked)
    got = sample_embedding(space,df)
    assert max(seen) <= 7 and sum(seen) == df.height
    # PCA coordinates are float32: changing BLAS batch shape changes rounding.
    # Bound the absolute normalized-RFF error to 16 float32 eps (~1.9e-6).
    tol = 16 * np.finfo(np.float32).eps
    np.testing.assert_allclose(got.mean,expected_mean,rtol=0,atol=tol)
    np.testing.assert_allclose(got.second,expected_second[np.triu_indices(8)],rtol=0,atol=tol)
    np.testing.assert_array_equal(got.diversity,reference.diversity)


@pytest.mark.benchmark
@pytest.mark.skipif(not os.getenv('RUN_BENCHMARK'),reason='set RUN_BENCHMARK=1')
def test_large_rff_cohort_memory():
    """Actual native embeddings + RFF + PCA, with no cached sample intermediates."""
    count = int(os.getenv('MIR_BENCH_SAMPLES','10000'))
    workers = int(os.getenv('MIR_BENCH_WORKERS','1'))
    spaces = _spaces(('TRA','TRB','TRG','TRD','IGH','IGK','IGL'),256,256)
    start = time.perf_counter()
    def read(i):
        from threadpoolctl import threadpool_info
        assert kernel_threads(0) == 1
        assert all(p['num_threads'] == 1 for p in threadpool_info())
        return _reader(spaces,i,depth=8+i%25)
    items = [(str(i),lambda i=i:read(i)) for i in range(count)]
    table = sample_embeddings(items,spaces,n_jobs=workers)
    assert table.height == count
    assert table.select(pl.col('^phi:.*$')).width == 7*256
    rotation = TablePCA.fit(table,n_components=min(32,count-1))
    out = rotation.transform(table)
    assert np.isfinite(out.select(pl.col('^pc.*$')).to_numpy()).all()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib = rss/(1024**3 if sys.platform == 'darwin' else 1024**2)
    print(f'RFF cohort: samples={count}, loci=7, RFF=256/locus, workers={workers}, '
          f'wall_s={time.perf_counter()-start:.3f}, peak_RSS_GiB={gib:.3f}',flush=True)
    assert gib < 16


@pytest.mark.benchmark
@pytest.mark.skipif(not os.getenv('RUN_BENCHMARK'),reason='set RUN_BENCHMARK=1')
def test_large_shared_basis_fit_memory():
    from mir.embedding.table import fit_sample_spaces
    base = _spaces(('TRB',),256,256)
    count = int(os.getenv('MIR_BASIS_SAMPLES','1200'))
    items = [(str(i),lambda i=i:_reader(base,i,depth=256)) for i in range(count)]
    start = time.perf_counter()
    spaces = fit_sample_spaces(items,{'TRB':base['TRB'].clono.model},
                              n_components=32,n_rff=256,n_rff_second=0)
    assert spaces['TRB'].clono.scaler.n_samples_seen_ == min(200000,count*256)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib = rss/(1024**3 if sys.platform == 'darwin' else 1024**2)
    print(f'Shared basis: samples={count}, clonotypes={count*256}, fit_rows={min(200000,count*256)}, '
          f'prototypes=256, wall_s={time.perf_counter()-start:.3f}, peak_RSS_GiB={gib:.3f}',flush=True)
    assert gib < 16
