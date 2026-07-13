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
