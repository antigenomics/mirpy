from __future__ import annotations

from scipy import stats

from mir.common.clonotype import Clonotype
from mir.common.pool import pool_samples
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset


def _c(
    sid: str,
    *,
    locus: str = "TRB",
    nt: str,
    aa: str,
    v: str,
    j: str,
    dup: int,
    meta: dict | None = None,
) -> Clonotype:
    c = Clonotype(
        sequence_id=sid,
        locus=locus,
        junction=nt,
        junction_aa=aa,
        v_gene=v,
        j_gene=j,
        duplicate_count=dup,
    )
    if meta:
        c.clone_metadata = dict(meta)
    return c


def test_pool_ntvj_weighted_selects_highest_weight_representative() -> None:
    s1 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire(
                clonotypes=[
                    _c("s1_a", nt="ATGC", aa="CASSF", v="TRBV1", j="TRBJ1", dup=10, meta={"tag": "A"}),
                    _c("s1_b", nt="ATGC", aa="CASRF", v="TRBV1", j="TRBJ1", dup=2, meta={"tag": "B"}),
                ],
                locus="TRB",
            )
        },
        sample_id="s1",
    )
    s2 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire(
                clonotypes=[
                    _c("s2_b", nt="ATGC", aa="CASRF", v="TRBV1", j="TRBJ1", dup=5, meta={"tag": "B"}),
                ],
                locus="TRB",
            )
        },
        sample_id="s2",
    )

    pooled = pool_samples([s1, s2], rule="ntvj", weighted=True, include_sample_ids=True)

    assert isinstance(pooled, LocusRepertoire)
    assert pooled.clonotype_count == 1
    clone = pooled.clonotypes[0]
    assert clone.duplicate_count == 17
    assert clone.clone_metadata["incidence"] == 2
    assert clone.clone_metadata["occurrences"] == 3
    assert clone.clone_metadata["tag"] == "A"
    assert clone.clone_metadata["sample_ids"] == ["s1", "s2"]


def test_pool_ntvj_unweighted_selects_most_frequent_row() -> None:
    s1 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire(
                clonotypes=[
                    _c("s1_a", nt="ATGC", aa="CASSF", v="TRBV1", j="TRBJ1", dup=100, meta={"tag": "A"}),
                    _c("s1_b", nt="ATGC", aa="CASRF", v="TRBV1", j="TRBJ1", dup=2, meta={"tag": "B"}),
                ],
                locus="TRB",
            )
        },
        sample_id="s1",
    )
    s2 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire(
                clonotypes=[
                    _c("s2_b", nt="ATGC", aa="CASRF", v="TRBV1", j="TRBJ1", dup=1, meta={"tag": "B"}),
                ],
                locus="TRB",
            )
        },
        sample_id="s2",
    )

    pooled = pool_samples([s1, s2], rule="ntvj", weighted=False)

    assert isinstance(pooled, LocusRepertoire)
    clone = pooled.clonotypes[0]
    assert clone.duplicate_count == 103
    assert clone.clone_metadata["tag"] == "B"
    assert clone.clone_metadata["occurrences"] == 3


def test_pool_nt_rule_merges_same_nt_across_different_vj() -> None:
    reps = [
        LocusRepertoire(
            clonotypes=[
                _c("r1", nt="ATGC", aa="CASSF", v="TRBV1", j="TRBJ1", dup=3),
                _c("r2", nt="ATGC", aa="CASSY", v="TRBV2", j="TRBJ2", dup=4),
            ],
            locus="TRB",
            repertoire_id="rA",
        ),
        LocusRepertoire(
            clonotypes=[
                _c("r3", nt="ATGC", aa="CASSY", v="TRBV2", j="TRBJ2", dup=5),
            ],
            locus="TRB",
            repertoire_id="rB",
        ),
    ]

    pooled = pool_samples(reps, rule="nt", weighted=True)

    assert isinstance(pooled, LocusRepertoire)
    assert pooled.clonotype_count == 1
    clone = pooled.clonotypes[0]
    assert clone.duplicate_count == 12
    assert clone.clone_metadata["incidence"] == 2
    assert clone.clone_metadata["occurrences"] == 3


def test_pool_sample_repertoires_returns_multi_locus_sample_pool() -> None:
    s1 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire(clonotypes=[_c("b1", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=1)], locus="TRB"),
            "TRA": LocusRepertoire(clonotypes=[_c("a1", locus="TRA", nt="GC", aa="CV", v="TRAV1", j="TRAJ1", dup=2)], locus="TRA"),
        },
        sample_id="s1",
    )
    s2 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire(clonotypes=[_c("b2", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=3)], locus="TRB"),
            "TRA": LocusRepertoire(clonotypes=[_c("a2", locus="TRA", nt="GC", aa="CV", v="TRAV1", j="TRAJ1", dup=4)], locus="TRA"),
        },
        sample_id="s2",
    )

    pooled = pool_samples([s1, s2], rule="aavj", weighted=True)

    assert isinstance(pooled, SampleRepertoire)
    assert pooled.sample_id == "pool"
    assert set(pooled.loci.keys()) == {"TRA", "TRB"}
    assert pooled.loci["TRB"].clonotypes[0].duplicate_count == 4
    assert pooled.loci["TRA"].clonotypes[0].duplicate_count == 6


def test_pool_dataset_input_supported() -> None:
    s1 = SampleRepertoire(
        loci={"TRB": LocusRepertoire(clonotypes=[_c("x1", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=2)], locus="TRB")},
        sample_id="s1",
        sample_metadata={"age": 10},
    )
    s2 = SampleRepertoire(
        loci={"TRB": LocusRepertoire(clonotypes=[_c("x2", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=3)], locus="TRB")},
        sample_id="s2",
        sample_metadata={"age": 80},
    )
    ds = RepertoireDataset(samples={"s1": s1, "s2": s2})

    pooled = pool_samples(ds, rule="ntvj", weighted=True)

    assert isinstance(pooled, LocusRepertoire)
    assert pooled.clonotypes[0].duplicate_count == 5


def test_pool_aa_rule_merges_by_junction_aa_only() -> None:
    reps = [
        LocusRepertoire(
            clonotypes=[
                _c("r1", nt="ATGC", aa="CASSF", v="TRBV1", j="TRBJ1", dup=3),
                _c("r2", nt="GGCC", aa="CASSF", v="TRBV2", j="TRBJ2", dup=4),
            ],
            locus="TRB",
            repertoire_id="r1",
        )
    ]
    pooled = pool_samples(reps, rule="aa", weighted=True)
    assert isinstance(pooled, LocusRepertoire)
    assert pooled.clonotype_count == 1
    assert pooled.clonotypes[0].duplicate_count == 7


def test_pool_result_keeps_incidence_occurrences_as_metadata() -> None:
    reps = [
        LocusRepertoire(
            clonotypes=[_c("r1", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=1)],
            locus="TRB",
            repertoire_id="s1",
        ),
        LocusRepertoire(
            clonotypes=[_c("r2", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=1)],
            locus="TRB",
            repertoire_id="s2",
        ),
    ]
    pooled = pool_samples(reps, rule="ntvj", weighted=False)
    clone = pooled.clonotypes[0]
    assert clone.clone_metadata["incidence"] == 2
    assert clone.clone_metadata["occurrences"] == 2


def test_pool_rule_validation() -> None:
    rep = LocusRepertoire(clonotypes=[_c("x", nt="AT", aa="CA", v="TRBV1", j="TRBJ1", dup=1)], locus="TRB")
    try:
        pool_samples([rep], rule="bad")  # type: ignore[arg-type]
    except ValueError as e:
        assert "rule must be one of" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid rule")
