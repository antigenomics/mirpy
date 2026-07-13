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


def test_inverse_decoder_learns():
    pytest.importorskip("torch")
    from sklearn.decomposition import PCA

    from mir.distances.junction import junction_distance_matrix
    from mir.embedding.prototypes import load_prototypes
    from mir.ml.train import train_inverse_decoder

    protos = load_prototypes("human", "TRB", n=300)["junction_aa"].to_list()
    seqs = load_prototypes("human", "TRB", n=1200)["junction_aa"].to_list()
    codes = PCA(n_components=0.95, whiten=True, random_state=0).fit_transform(
        junction_distance_matrix(seqs, protos))
    dec, m = train_inverse_decoder(codes, seqs, epochs=8, verbose=False, seed=0)
    assert m["token_acc"] > 0.5                  # learns to invert
    out = dec.decode(codes[:5])
    assert len(out) == 5 and all(isinstance(s, str) for s in out)


def test_pgen_regressor_learns():
    pytest.importorskip("torch")
    import numpy as np

    from mir.embedding.prototypes import load_prototypes
    from mir.ml.train import train_pgen_regressor

    seqs = load_prototypes("human", "TRB", n=1200)["junction_aa"].to_list()
    # learnable synthetic target: a function of CDR3 length
    target = np.array([-len(s) for s in seqs], dtype=np.float32)
    reg, m = train_pgen_regressor(seqs, target, epochs=8, verbose=False, seed=0)
    assert m["pearson"] > 0.5
    assert reg.predict(seqs[:5]).shape == (5,)


def test_unified_codec_learns():
    pytest.importorskip("torch")
    from sklearn.decomposition import PCA

    from mir.distances.junction import junction_distance_matrix
    from mir.embedding.prototypes import load_prototypes
    from mir.ml.codec import train_unified_codec

    protos = load_prototypes("human", "TRB", n=300)["junction_aa"].to_list()
    seqs = load_prototypes("human", "TRB", n=1500)["junction_aa"].to_list()
    codes = PCA(n_components=0.95, whiten=True, random_state=0).fit_transform(
        junction_distance_matrix(seqs, protos))
    codec, m = train_unified_codec(seqs, codes, epochs=8, verbose=False, seed=0)
    assert m["encode_cosine"] > 0.3               # geometry preserved
    rt = codec.roundtrip(seqs[:5])
    assert len(rt) == 5 and all(isinstance(s, str) for s in rt)


def test_codec_bundle_ships_prototypes_and_pca(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from mir.distances.junction import junction_distance_matrix
    from mir.embedding.prototypes import load_prototypes
    from mir.ml.bundle import CodecBundle
    from mir.ml.train import train_forward_encoder

    K = 200
    protos = load_prototypes("human", "TRB", n=K)["junction_aa"].to_list()
    seqs = load_prototypes("human", "TRB", n=1200)["junction_aa"].to_list()
    enc, _ = train_forward_encoder(seqs, junction_distance_matrix(seqs, protos),
                                   epochs=5, verbose=False)
    b = CodecBundle.from_forward(enc, "human", "TRB", K)
    p = tmp_path / "trb.codec"
    b.save(p)
    b2 = CodecBundle.load(p)
    assert b2.matches_current_prototypes()
    enc2 = b2.forward_encoder()
    assert np.allclose(enc.code(seqs[:10]), enc2.code(seqs[:10]), atol=1e-4)
    # incomparable prototype set must be refused
    b2.meta["prototype_hash"] = "deadbeef00000000"
    with pytest.raises(ValueError):
        b2.forward_encoder()
