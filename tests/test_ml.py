import numpy as np
import pytest

from mir.ml.tokenize import AA, FIXED_LEN, N_TOKENS, encode_indices, encode_onehot


def test_tokenize_shape_and_onehot():
    oh = encode_onehot(["CASSIRSSYEQYF", "CAS", "C" * 60])
    assert oh.shape == (3, FIXED_LEN, N_TOKENS)
    assert np.allclose(oh.sum(axis=2), 1.0)      # one token per position


def test_tokenize_anchors_and_gap():
    idx = encode_indices(["CASSIRSSYEQYF"])[0]
    assert idx[0] == AA.index("C")               # N-terminal anchored
    assert idx[-1] == AA.index("F")              # C-terminal anchored
    assert (idx == N_TOKENS - 1).any()           # gap block present


def test_forward_encoder_learns():
    pytest.importorskip("torch")
    from mir.distances.junction import junction_distance_matrix
    from mir.embedding.prototypes import load_prototypes
    from mir.ml.train import train_forward_encoder

    protos = load_prototypes("human", "TRB", n=64)["junction_aa"].to_list()
    seqs = load_prototypes("human", "TRB", n=800)["junction_aa"].to_list()
    y = junction_distance_matrix(seqs, protos)
    enc, m = train_forward_encoder(seqs, y, epochs=5, verbose=False, seed=0)
    assert m["test_cosine"] > 0.3                # learns real signal
    Z = enc.encode(seqs[:10])
    assert Z.shape == (10, 64)
