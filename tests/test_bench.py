import os

import numpy as np
import pytest

from mir.bench.metrics import cluster, cluster_metrics


def test_cluster_metrics_perfect_separation():
    # two pure clusters + one noise point
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    antigens = ["A", "A", "A", "B", "B", "B", "A"]
    m = cluster_metrics(labels, antigens)
    assert m["A"].f1 == 1.0 and m["B"].f1 == 1.0
    assert m["A"].retention == 3 / 4        # 3 of 4 A's clustered
    assert m["B"].retention == 1.0
    assert m["A"].n == 4 and m["B"].n == 3


def test_cluster_metrics_mixed_cluster():
    # one cluster mixing A and B -> majority A -> B has zero precision as A
    labels = np.array([0, 0, 0, 0])
    antigens = ["A", "A", "A", "B"]
    m = cluster_metrics(labels, antigens)
    assert m["A"].precision == 3 / 4        # cluster predicted A, 3/4 correct
    assert m["B"].f1 == 0.0                  # B never predicted


def test_cluster_runs_on_embedding():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 8)), rng.normal(5, 0.1, (20, 8))])
    labels = cluster(X, eps=1.0, min_samples=3)
    assert labels.shape == (40,)
    assert set(labels[labels >= 0]) == {0, 1}   # two dense blobs recovered


@pytest.mark.parametrize("method", ["hdbscan", "optics"])
def test_cluster_alternative_methods(method):
    # HDBSCAN / OPTICS are drop-in: same (n,) shape, -1 noise convention, two blobs separated.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 8)), rng.normal(5, 0.1, (20, 8))])
    labels = cluster(X, min_samples=3, method=method)
    assert labels.shape == (40,)
    assert (labels == -1).sum() < 40                     # not everything is noise
    clustered = labels >= 0
    # the two well-separated blobs must not land in the same cluster
    assert labels[0] != labels[20] or not (clustered[0] and clustered[20])


def test_cluster_bad_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        cluster(np.zeros((5, 3)), method="kmeans")


def test_codec_losslessness_ceiling_and_recon():
    from mir.bench.theory import codec_losslessness

    rng = np.random.default_rng(0)
    seqs = ["CASSIRSSYEQYF", "CSARVSGYYGYTF", "CASSLAPGATNEKLFF", "CASSPGQGADTQYF"]
    codes = rng.standard_normal((4, 20))
    # distinct codes + perfect roundtrip: injective (ceiling 1), lossless recon
    r = codec_losslessness(codes, seqs, recon=seqs)
    assert r["collision_rate"] == 0.0 and r["exact_ceiling"] == 1.0
    assert r["exact_match"] == 1.0 and r["mean_edit"] == 0.0
    # one middle substitution: exact drops, middle_acc < anchor_acc
    bad = list(seqs)
    bad[0] = "CASSIRSAYEQYF"
    r2 = codec_losslessness(codes, seqs, recon=bad)
    assert r2["exact_match"] == 0.75 and r2["mean_edit"] == 0.25
    assert r2["middle_acc"] < r2["anchor_acc"] == 1.0
    # colliding codes for two DISTINCT sequences lower the ceiling; duplicate seqs do not
    codes2 = codes.copy()
    codes2[1] = codes2[0]
    assert codec_losslessness(codes2, seqs)["exact_ceiling"] < 1.0
    dup = codec_losslessness(np.vstack([codes, codes[:1]]), seqs + [seqs[0]])
    assert dup["n_unique"] == 4 and dup["collision_rate"] == 0.0


@pytest.mark.skipif(
    not os.path.exists("tests/assets/vdjdb.slim.txt.gz"),
    reason="VDJdb slim dump not present (gitignored local fixture)",
)
def test_load_vdjdb_schema():
    from mir.bench.vdjdb import antigen_subset, load_vdjdb

    df = load_vdjdb("tests/assets/vdjdb.slim.txt.gz")
    assert {"v_call", "j_call", "junction_aa", "locus", "epitope"} <= set(df.columns)
    trb = antigen_subset(df, "TRB", 300)
    assert trb.height > 0
    assert (trb["locus"] == "TRB").all()
