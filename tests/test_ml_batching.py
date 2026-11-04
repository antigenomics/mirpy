"""The public batch size bounds tokenization, model calls and inverse transforms."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')


@pytest.mark.parametrize('kind', ['forward', 'regressor', 'unified'])
def test_prediction_prepares_only_one_batch(kind, monkeypatch):
    from mir.ml import codec, train
    from mir.ml.tokenize import FIXED_LEN, N_TOKENS, encode_onehot

    torch.manual_seed(7)
    width = 1 if kind == 'regressor' else 5
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(FIXED_LEN * N_TOKENS, width))
    device = torch.device('cpu')
    seqs = ['CASSLGQETQYF', 'CAVMDSNYQLIW'] * 4
    transformed = []

    def inverse(values):
        transformed.append(len(values))
        assert len(values) <= 3
        values *= 2
        values += 1
        return values

    transform = SimpleNamespace(inverse_transform=inverse)
    with torch.no_grad():
        expected = model(torch.from_numpy(encode_onehot(seqs))).numpy()
    raw_expected = expected.copy()
    captured = []
    model.register_forward_hook(lambda module, args, output: captured.append(output))
    if kind == 'forward':
        predictor = train.ForwardEncoder(model, transform, device, False).encode
        expected = 2 * expected + 1
    elif kind == 'regressor':
        predictor = train.PgenRegressor(model, transform, device).predict
        expected = (2 * expected + 1).ravel()
    else:
        predictor = codec.UnifiedCodec(model, torch.nn.Identity(), device).encode
    prepared = []

    def bounded(values):
        values = list(values)
        prepared.append(len(values))
        assert len(values) <= 3
        return encode_onehot(values)

    monkeypatch.setattr(train, 'encode_onehot', bounded)
    monkeypatch.setattr(codec, 'encode_onehot', bounded)
    actual = predictor(iter(seqs), batch=3)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.concatenate([value.numpy() for value in captured]),
                               raw_expected, rtol=1e-5, atol=1e-6)
    assert prepared == [3, 3, 2]
    assert transformed == ([] if kind == 'unified' else [3, 3, 2])
    with pytest.raises(ValueError):
        predictor(seqs, batch=0)
    with pytest.raises(ValueError):
        predictor([], batch=3)


def test_prediction_rejects_missing_rows():
    from mir.ml.train import _predict_batches

    with pytest.raises(ValueError, match='one prediction per sequence'):
        _predict_batches(lambda x: x[:1], ['CASSF'] * 3, torch.device('cpu'), 3)


@pytest.mark.parametrize('kind', ['inverse', 'unified'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_decoder_stages_only_one_batch(kind, dtype):
    from mir.ml.codec import UnifiedCodec
    from mir.ml.decoder import tokens_to_seq
    from mir.ml.train import InverseDecoder
    from mir.ml.tokenize import FIXED_LEN, N_TOKENS

    class Decoder(torch.nn.Module):
        def forward(self, values):
            indices = (values[:, :1].long() + torch.arange(FIXED_LEN)) % N_TOKENS
            return torch.nn.functional.one_hot(indices, N_TOKENS).float()

    values = np.arange(16, dtype=dtype).reshape(8, 2)
    original = values.copy()
    transformed = []

    class Codes:
        def __len__(self):
            return len(values)

        def __getitem__(self, index):
            return values[index]

        def __array__(self, *args, **kwargs):
            raise AssertionError('whole-cohort conversion')

    def transform(block):
        transformed.append(len(block))
        assert len(block) <= 3
        block += 1
        return block

    decoder = Decoder()
    expected_values = values.astype(np.float32) + (1 if kind == 'inverse' else 0)
    expected = [tokens_to_seq(row) for row in decoder(torch.from_numpy(expected_values)).argmax(-1).numpy()]
    if kind == 'inverse':
        predictor = InverseDecoder(decoder, SimpleNamespace(transform=transform), torch.device('cpu')).decode
    else:
        predictor = UnifiedCodec(torch.nn.Identity(), decoder, torch.device('cpu')).decode
    assert predictor(Codes(), batch=3) == expected
    np.testing.assert_array_equal(values, original)
    assert transformed == ([3, 3, 2] if kind == 'inverse' else [])
    with pytest.raises(ValueError):
        predictor(Codes(), batch=-1)


def test_vector_token_formatting_preserves_scalar_semantics():
    from mir.ml.decoder import tokens_to_seq, tokens_to_seqs
    values = np.array([[0, 1, 19, 20, -1, 21], [19, 0, 20, 3, 4, 20]])
    assert tokens_to_seqs(values) == [tokens_to_seq(row) for row in values]
    assert tokens_to_seqs(np.empty((2, 0), dtype=int)) == ['', '']
    with pytest.raises(ValueError):
        tokens_to_seqs(values[0])
