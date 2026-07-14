"""Tests for mir.ml.set_encoder (learned repertoire track). Marked integration (torch + slower)."""

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.integration

_OLGA = "tests/assets/olga_humanTRB_1000.txt.gz"


def _clonotypes(n, offset=0):
    df = pl.read_csv(_OLGA, separator="\t", has_header=False,
                     new_columns=["junction_nt", "junction_aa", "v_call", "j_call"]) \
        .select(["junction_aa", "v_call", "j_call"]).slice(offset, n)
    return df.unique(subset=["junction_aa", "v_call", "j_call"])


def test_train_separates_public_cluster_toy():
    from mir.ml.set_encoder import train_set_encoder

    rng = np.random.default_rng(0)
    clouds, y = [], []
    for c in range(40):
        label = c % 2
        n = int(rng.integers(80, 160))
        Z = rng.standard_normal((n, 10)).astype(np.float32)
        if label:
            Z[:8] = np.float32(3.0) + 0.05 * rng.standard_normal((8, 10)).astype(np.float32)
        w = rng.random(n).astype(np.float32); w /= w.sum()
        clouds.append((Z, w)); y.append(float(label))
    _, metrics = train_set_encoder(clouds, np.array(y), task="classification",
                                   epochs=30, n_seeds=8, d=64, verbose=False)
    assert metrics["val_score"] > 0.6


def test_bundle_roundtrip_and_cross_basis_refusal(tmp_path):
    import torch

    from mir.embedding.tcremp import TCREmp
    from mir.ml.set_encoder import SetEncoderBundle, train_set_encoder
    from mir.repertoire import fit_repertoire_space

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=300)
    pool = _clonotypes(500)
    space = fit_repertoire_space(model, pool, n_rff=256, n_components=15, seed=0)

    rng = np.random.default_rng(0)
    clouds, y = [], []
    for s in range(16):
        df = pool.sample(80, seed=s).with_columns(
            pl.Series("duplicate_count", rng.integers(1, 50, 80).astype(float)))
        clouds.append(space.sample_cloud(df))
        y.append(float(s % 2))
    sem, _ = train_set_encoder(clouds, np.array(y), task="classification",
                               epochs=5, n_seeds=4, d=32, verbose=False)

    p = tmp_path / "enc.pt"
    SetEncoderBundle.from_model(sem, space, task="classification").save(p)
    pred_before = sem.predict(clouds[:3])

    b = SetEncoderBundle.load(p)
    assert b.meta["prototype_hash"] == space.meta["prototype_hash"]
    assert b.meta["task"] == "classification"
    assert np.isfinite(pred_before).all()

    d = torch.load(p, weights_only=False)
    d["meta"]["prototype_hash"] = "deadbeefdeadbeef"
    torch.save(d, p)
    with pytest.raises(ValueError, match="prototype hash mismatch"):
        SetEncoderBundle.load(p)
    SetEncoderBundle.load(p, verify=False)          # explicit override allowed
