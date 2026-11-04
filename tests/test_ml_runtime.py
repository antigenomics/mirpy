"""Torch entry points own a bounded phase and do not race shared fit RNGs."""
import threading

import numpy as np
import pytest

pytest.importorskip('torch')


def test_codec_owns_default_explicit_and_outer_worker_budgets(monkeypatch):
    import torch
    from mir.ml.train import ForwardEncoder
    from vdjtools.signature.cohort import parallel_rows
    from threadpoolctl import threadpool_info

    monkeypatch.setattr('vdjtools.cores.available_cores', lambda: 2)
    seen = []
    class Probe(torch.nn.Module):
        def forward(self, x):
            seen.append((torch.get_num_threads(), torch.get_num_interop_threads()))
            assert all(p['num_threads'] == torch.get_num_threads()
                       for p in threadpool_info() if p['user_api'] == 'openmp')
            return x[:, :1, 0]
    encoder = ForwardEncoder(Probe(), None, torch.device('cpu'), False)
    torch.set_num_threads(1)
    before = [(p['filepath'], p['num_threads']) for p in threadpool_info()]
    seqs = ['CASSQYF'] * 3
    encoder.code(seqs)
    assert seen[-1] == (2, 1)
    encoder.code(seqs, threads=1)
    assert seen[-1] == (1, 1)
    encoder.code(seqs, threads=2)
    assert seen[-1] == (2, 1)
    assert torch.get_num_threads() == 1
    assert before == [(p['filepath'], p['num_threads']) for p in threadpool_info()]
    seen.clear()
    parallel_rows([seqs, seqs], encoder.code, 2)
    assert seen == [(1, 1), (1, 1)]
    with pytest.raises(ValueError, match='threads'):
        encoder.code(seqs, threads=3)
    with pytest.raises(ValueError, match='threads'):
        parallel_rows([seqs, seqs], lambda rows: encoder.code(rows, threads=2), 2)
    class Broken(torch.nn.Module):
        def forward(self, x):
            assert torch.get_num_threads() == 2
            raise RuntimeError('model failure')
    encoder.model = Broken()
    with pytest.raises(RuntimeError, match='model failure'):
        encoder.code(seqs)
    assert torch.get_num_threads() == 1
    assert before == [(p['filepath'], p['num_threads']) for p in threadpool_info()]


@pytest.mark.parametrize('kind', ['forward', 'inverse', 'pgen', 'unified', 'diffusion', 'set'])
def test_seeded_fits_serialize_before_seeding(monkeypatch, kind):
    import torch
    from mir.ml import train, codec, diffusion, set_encoder
    from vdjtools.cores import serial_kernels

    monkeypatch.setattr('vdjtools.cores.available_cores', lambda: 2)
    first_seed, second_seed, release = threading.Event(), threading.Event(), threading.Event()
    original = train.seed_everything
    def seed(value):
        original(value)
        if threading.current_thread().name == 'first-fit':
            first_seed.set()
            assert release.wait(5)
        elif threading.current_thread().name == 'second-fit':
            second_seed.set()
    for module in [train, codec, diffusion, set_encoder]:
        monkeypatch.setattr(module, 'seed_everything', seed)
    seqs = ['CASS' + 'ACDEFGHIKLMNPQRSTVWY'[i] + 'QYF' for i in range(20)]
    x = np.random.default_rng(9).normal(size=(20, 3)).astype(np.float32)
    outputs, failures = {}, []
    def fit(value):
        options = dict(seed=value, epochs=1, device='cpu', verbose=False, threads=1)
        if kind == 'forward':
            return train.train_forward_encoder(seqs, x, batch=8, target_pca=None, **options)
        if kind == 'inverse':
            return train.train_inverse_decoder(x, seqs, batch=8, **options)
        if kind == 'pgen':
            return train.train_pgen_regressor(seqs, np.arange(20), batch=8, **options)
        if kind == 'unified':
            return codec.train_unified_codec(seqs, x, batch=8, **options)
        if kind == 'diffusion':
            return diffusion.train_diffusion(x, T=10, hidden=16, time_dim=8, **options)
        clouds = [(x[:5] + i / 10, np.full(5, .2, dtype=np.float32)) for i in range(20)]
        return set_encoder.train_set_encoder(clouds, np.arange(20, dtype=np.float32),
                                             d=16, n_seeds=2, **options)
    def worker(value):
        try:
            with serial_kernels():
                outputs[value] = fit(value)
        except BaseException as exc:
            failures.append(exc)
    first = threading.Thread(target=worker, args=(11,), name='first-fit')
    second = threading.Thread(target=worker, args=(12,), name='second-fit')
    first.start()
    try:
        assert first_seed.wait(5)
        second.start()
        assert not second_seed.wait(.1), 'second fit reset the first fit RNG'
    finally:
        release.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert not failures, failures
    assert second_seed.is_set()
    # A serialized concurrent call must train the same parameters as its ordinary call.
    def states(fitted):
        model = fitted[0]
        if kind == 'unified':
            return [model.encoder.state_dict(), model.decoder.state_dict()]
        return [model.model.state_dict()]
    for value in [11, 12]:
        expected = states(fit(value))
        for actual, reference in zip(states(outputs[value]), expected):
            for key in actual:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
