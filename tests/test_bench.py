
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


def test_load_vdjdb_schema(tmp_path):
    # inline dump, not the gitignored tests/assets/vdjdb.slim.txt.gz: that skipif could never be
    # satisfied in CI, so the dotted-column mapping and the min_records filter went untested.
    import gzip

    from mir.bench.vdjdb import antigen_subset, load_vdjdb

    rows = [
        # cdr3, v.segm, j.segm, gene, antigen.epitope, mhc.class
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ2-7*01", "TRB", "GILGFVFTL", "MHCI"),
        ("CASSLGQAYEQFF", "TRBV28*01", "TRBJ2-1*01", "TRB", "GILGFVFTL", "MHCI"),
        ("CASSPGQGAYEQYF", "TRBV5-1*01", "TRBJ2-7*01", "TRB", "GILGFVFTL", "MHCI"),
        ("CASSIRSSYEQYF", "TRBV19*01", "TRBJ2-7*01", "TRB", "GILGFVFTL", "MHCI"),  # exact dup
        ("CAVRDSNYQLIW", "TRAV3*01", "TRAJ33*01", "TRA", "GILGFVFTL", "MHCI"),     # other locus
        ("CASSFGREQYF", "TRBV12-3*01", "TRBJ2-7*01", "TRB", "NLVPMVATV", "MHCI"),  # rare epitope
        ("CAS", "TRBV19*01", "TRBJ2-7*01", "TRB", "NLVPMVATV", "MHCI"),            # too short
        ("CASSQETQYF", "TRBV4-1*01", "TRBJ2-5*01", "TRB", "", "MHCI"),             # no epitope
    ]
    path = tmp_path / "vdjdb.slim.txt.gz"
    header = "cdr3\tv.segm\tj.segm\tgene\tantigen.epitope\tmhc.class\n"
    with gzip.open(path, "wt") as fh:
        fh.write(header + "".join("\t".join(r) + "\n" for r in rows))

    df = load_vdjdb(str(path))
    assert {"v_call", "j_call", "junction_aa", "locus", "epitope"} <= set(df.columns)
    assert df.height == 5                      # dup collapsed, short + epitope-less dropped
    assert df["junction_aa"].str.len_chars().min() >= 5

    trb = antigen_subset(df, "TRB", 3)
    assert (trb["locus"] == "TRB").all()
    assert set(trb["epitope"].unique()) == {"GILGFVFTL"}   # NLVPMVATV has < 3 records
    assert trb.height == 3


def test_codec_ceiling_counts_one_recoverable_sequence_per_collision_group():
    """Regression: `1 - collision_rate` charged for EVERY member of a colliding group.

    A decoder can still recover one sequence per mutually-confusable group -- the most frequent
    one -- so a single colliding pair among 10 sequences has a true ceiling of 0.9, not 0.8. The
    ceiling was also normalised by n_unique while exact_match is over all n, so a measured value
    could legitimately exceed its own "ceiling".
    """
    from mir.bench.theory import codec_losslessness

    rng = np.random.default_rng(0)
    seqs = [f"CASS{i:02d}YEQYF" for i in range(10)]
    codes = rng.standard_normal((10, 12)) * 100.0   # far apart
    codes[1] = codes[0]                             # exactly ONE colliding pair

    r = codec_losslessness(codes, seqs)
    assert r["collision_rate"] == pytest.approx(0.2)   # 2 of 10 have a within-eps neighbour
    assert r["exact_ceiling"] == pytest.approx(0.9)    # but only 1 of 10 is unrecoverable

    # three-way collision: 2 of the 3 are unrecoverable
    codes3 = codes.copy()
    codes3[2] = codes3[0]
    assert codec_losslessness(codes3, seqs)["exact_ceiling"] == pytest.approx(0.8)


def test_codec_ceiling_is_never_below_a_measurable_exact_match():
    """The ceiling and exact_match now share a base (all n), so the ceiling actually bounds."""
    from mir.bench.theory import codec_losslessness

    rng = np.random.default_rng(1)
    # duplicated sequences: n=6, n_unique=4, one colliding pair among the unique codes
    seqs = ["CASSAAYEQYF", "CASSBBYEQYF", "CASSCCYEQYF", "CASSDDYEQYF",
            "CASSAAYEQYF", "CASSAAYEQYF"]
    codes = rng.standard_normal((6, 12)) * 100.0
    codes[4] = codes[5] = codes[0]     # duplicates share their code, as they must
    codes[2] = codes[1]                # a real collision between two DISTINCT sequences

    # a real decoder is a function OF THE CODE, so the colliding pair must decode identically --
    # here both to "CASSBBYEQYF", which is what makes the ceiling a genuine bound.
    recon = ["CASSAAYEQYF", "CASSBBYEQYF", "CASSBBYEQYF", "CASSDDYEQYF",
             "CASSAAYEQYF", "CASSAAYEQYF"]
    r = codec_losslessness(codes, seqs, recon=recon)
    # the 3 copies of CASSAA are recoverable, CASSDD is, and of the colliding B/C pair only one is
    assert r["exact_ceiling"] == pytest.approx(5 / 6)
    assert r["exact_match"] == pytest.approx(5 / 6)
    assert r["exact_match"] <= r["exact_ceiling"] + 1e-12


def test_estimate_dbscan_eps_uses_the_kth_neighbour_excluding_self():
    """Regression: kneighbors() includes each point's own 0-distance self-match in column 0.

    Indexing [:, k-1] of a k-neighbour query therefore returned the (k-1)-NN curve, and a
    correspondingly small eps, everywhere the benchmark harness derives one.
    """
    from mir.bench.metrics import estimate_dbscan_eps

    # Points on a line at unit spacing. Excluding self, the sorted neighbour distances are
    # 1, 1, 2, 2, 3, 3, ... so the 1st is 1 and the 3rd is 2.
    X = np.arange(200, dtype=float).reshape(-1, 1)

    # k=1 is the sharpest probe: the old query asked for exactly ONE neighbour, got only the
    # self-match, and returned a curve of zeros -> eps == 0, a silently unusable DBSCAN radius.
    assert estimate_dbscan_eps(X, k=1) == pytest.approx(1.0)
    assert estimate_dbscan_eps(X, k=3) == pytest.approx(2.0)   # was 1.0 off by one neighbour
    # and eps is non-decreasing in k, as a k-NN distance curve must be
    eps = [estimate_dbscan_eps(X, k=k) for k in (1, 3, 5)]
    assert eps == sorted(eps)
