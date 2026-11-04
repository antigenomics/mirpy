"""Training must bound device staging and model calls, including held-out rows."""
import numpy as np
import pytest

pytest.importorskip('torch')


@pytest.mark.parametrize('kind', ['forward', 'inverse', 'pgen', 'unified', 'diffusion'])
def test_training_prepares_only_current_batch(monkeypatch, kind):
    import torch
    from mir.ml import train, codec
    from mir.ml.encoder import SequenceEncoder
    from mir.ml.decoder import SequenceDecoder
    from mir.ml.diffusion import DiffusionMLP, train_diffusion

    n, batch = 80, 8
    seqs = ['CASS' + 'ACDEFGHIKLMNPQRSTVWY'[i % 20] + 'QYF' for i in range(n)]
    codes = np.random.default_rng(7).normal(size=(n, 3)).astype(np.float32)
    staged = []
    token_batches = []
    encode_tokens = train.encode_indices

    def bounded_tokens(rows):
        token_batches.append(len(rows))
        assert len(rows) <= batch
        return encode_tokens(rows)

    monkeypatch.setattr(train, "encode_indices", bounded_tokens)
    original = torch.from_numpy

    def bounded(array):
        if array.ndim >= 2:
            staged.append(len(array))
            assert len(array) <= batch, 'whole split copied to the device'
        return original(array)

    monkeypatch.setattr(torch, 'from_numpy', bounded)
    for cls in [SequenceEncoder, SequenceDecoder, DiffusionMLP]:
        forward = cls.forward
        def checked(self, x, *args, _forward=forward, **kwargs):
            assert len(x) <= batch, 'held-out model call ignored batch'
            return _forward(self, x, *args, **kwargs)
        monkeypatch.setattr(cls, 'forward', checked)
    for module in [train, codec]:
        if not hasattr(module, "encode_onehot"):
            continue
        encode = module.encode_onehot
        def checked_encode(rows, _encode=encode):
            assert len(rows) <= batch, 'whole cohort expanded to one-hot'
            return _encode(rows)
        monkeypatch.setattr(module, 'encode_onehot', checked_encode)
    options = dict(epochs=1, batch=batch, val_frac=.27, test_frac=.23,
                   seed=11, device='cpu', verbose=False)
    if kind == 'forward':
        _, metrics = train.train_forward_encoder(seqs, codes, target_pca=None, **options)
    elif kind == 'inverse':
        _, metrics = train.train_inverse_decoder(codes, seqs, **options)
    elif kind == 'pgen':
        _, metrics = train.train_pgen_regressor(seqs, np.linspace(-10, -5, n), **options)
    elif kind == 'diffusion':
        options.pop('test_frac')
        _, metrics = train_diffusion(codes, ['a', 'b'] * (n // 2),
                                     T=10, hidden=16, time_dim=8, class_dim=4, **options)
    else:
        _, metrics = codec.train_unified_codec(seqs, codes, **options)
    assert metrics['n'] == n
    assert staged and max(staged) <= batch
    if kind != 'diffusion':
        assert sum(token_batches) == n
    assert all(np.isfinite(value) for value in metrics.values())
