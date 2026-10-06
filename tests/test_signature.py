"""The geometry half: raw features, the corpus fit, and the two halves joining.

Every test pins something that has gone wrong in this subsystem before, or an algebraic identity the
design rests on.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from vdjtools.signature import corpus as C
from vdjtools.signature import layout as L

from mir.signature import features as F
from mir.signature import raw_and_channels, rsig, rsig_cohort, synthesize

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


@pytest.fixture(scope="module")
def corpus():
    art, _ = synthesize("memory", loci=("TRG", "TRD"), n_samples=30, size=90, seed=6,
                        n_components=5)
    return art


@pytest.fixture(scope="module")
def held_out():
    samples = C.synthesize("memory", loci=("TRG", "TRD"), n_samples=3, size=90, seed=909,
                           fit_corpus=False)
    return {f"S{i}": s for i, s in enumerate(samples)}


def _frame(n, v, j, rng, c=None):
    return pl.DataFrame({
        "v_call": [v] * n, "j_call": [j] * n, "c_call": [c] * n,
        "junction_aa": ["C" + "".join(rng.choice(_AA, 12)) + "F" for _ in range(n)],
        "duplicate_count": np.ceil(rng.zipf(1.5, n).clip(1, 900)).astype(np.int64).tolist(),
    }, schema_overrides={"c_call": pl.Utf8})


# ------------------------------------------------------- 1. the identities Phi rests on


def test_the_slots_are_literal_strides_so_they_reconstruct_phi():
    """Exact by construction, which is what makes "how much of this is V?" answerable."""
    phi = np.random.default_rng(0).normal(size=3 * F.K)
    s = F.slots(phi)
    assert np.array_equal(s["phiv"], phi[0::3])
    assert np.array_equal(s["phij"], phi[1::3])
    assert np.array_equal(s["phic"], phi[2::3])
    rebuilt = np.empty_like(phi)
    rebuilt[0::3], rebuilt[1::3], rebuilt[2::3] = s["phiv"], s["phij"], s["phic"]
    assert np.array_equal(rebuilt, phi)


def test_the_mixture_identity_is_exact_so_band_shares_close():
    """Phi is linear in the clone-weight measure and the bands partition the clonotypes.

    An NNLS over overlapping parts is not a composition at all: its weights need not sum to one and
    one share can exceed it, which breaks every log-ratio coordinate downstream.
    """
    rng = np.random.default_rng(1)
    df = _frame(300, "TRBV20-1", "TRBJ2-2", rng)
    w = F.weights(df["duplicate_count"].to_numpy())
    assert abs(w.sum() - 1.0) < 1e-12
    sh = F.band_shares(df, w, min_clonotypes=1)
    assert abs(sum(sh.values()) - 1.0) < 1e-12, sh


def test_rao_agrees_with_the_independent_accumulator():
    """Two spellings of one telescoping identity, cross-checked rather than trusted."""
    from mir.embedding.tcremp import TCREmp
    from mir.repertoire import rao_dispersion

    rng = np.random.default_rng(2)
    df = _frame(150, "TRBV20-1", "TRBJ2-2", rng)
    w = F.weights(df["duplicate_count"].to_numpy())
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=32)
    phi, mean_sq = F.prototype_sum(df, model, w)
    n_eff = 1.0 / float(w @ w)
    mine = F.rao_of(phi, mean_sq, n_eff)
    theirs = float(rao_dispersion(np.asarray(model.embed(df), dtype=float), w))
    assert abs(mine - theirs) / max(mine, 1e-12) < 1e-9


def test_a_band_below_the_floor_is_absent_not_zero():
    """Zero is a measurement; absent is not, and a clr cannot tell them apart afterwards."""
    rng = np.random.default_rng(3)
    df = _frame(40, "TRBV20-1", "TRBJ2-2", rng)
    w = F.weights(df["duplicate_count"].to_numpy())
    strict = F.band_shares(df, w, min_clonotypes=10_000)
    assert set(strict) == {"_residual"} and strict["_residual"] == pytest.approx(1.0)


def test_the_chunk_size_bounds_memory_and_not_the_answer():
    """Phi and the Rao accumulator are running sums, so chunking must be exact."""
    from mir.embedding.tcremp import TCREmp

    rng = np.random.default_rng(4)
    df = _frame(200, "TRBV20-1", "TRBJ2-2", rng)
    w = F.weights(df["duplicate_count"].to_numpy())
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=32)
    a = F.prototype_sum(df, model, w, chunk=1000)
    b = F.prototype_sum(df, model, w, chunk=17)
    assert np.allclose(a[0], b[0], rtol=0, atol=1e-9)
    assert abs(a[1] - b[1]) < 1e-9


# --------------------------------------------- 2. the embedder must not change coordinates


def test_the_shm_columns_never_reach_the_embedder(corpus):
    """Their presence silently switches TCREmp to SHM-aware V distances.

    That is a different coordinate system under the same column names -- the numbers move and
    nothing says so, which is why they are dropped rather than trusted to be absent.
    """
    rng = np.random.default_rng(5)
    df = _frame(80, "TRGV9", "TRGJ1", rng)
    withshm = df.with_columns(v_identity=pl.lit(0.94), v_mutations=pl.lit(3))
    a, _ = raw_and_channels({"TRG": df}, corpus.vocab)
    b, _ = raw_and_channels({"TRG": withshm}, corpus.vocab)
    moved = [k for k in a if not (a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])))]
    assert moved == [], moved[:5]


def test_the_prototype_index_hoist_changes_no_raw_feature(corpus):
    """The pre-3.19.0 un-hoisted path must give bit-identical features."""
    from mir.embedding.tcremp import TCREmp

    rng = np.random.default_rng(6)
    frames = {"TRG": _frame(120, "TRGV9", "TRGJ1", rng)}
    hoisted, _ = raw_and_channels(frames, corpus.vocab)
    orig = TCREmp._proto_idx
    try:
        TCREmp._proto_idx = lambda self, comp: None
        plain, _ = raw_and_channels(frames, corpus.vocab)
    finally:
        TCREmp._proto_idx = orig
    moved = [k for k in hoisted
             if not (hoisted[k] == plain[k]
                     or (np.isnan(hoisted[k]) and np.isnan(plain[k])))]
    assert moved == [], moved[:5]


# ---------------------------------------------------- 3. the shared contract and registry


def test_rsig_registers_its_groups_into_the_vdjtools_registry():
    names = {g.name for g in L.raw_groups("rsig")}
    assert names == {"phiv", "phij", "phic", "depth", "band", "band_igh"}, names
    assert {c.name for c in L.channels("rsig")} == {"div", "disp", "mask", "qc"}
    # the two halves are disjoint, which is what lets them join on sample_id
    assert not ({g.name for g in L.raw_groups("vsig")} & {"phiv", "phij", "phic"})


def test_the_phi_slots_declare_a_nonneg_support_so_only_their_top_tail_is_trimmed():
    """Phi is a weighted mean of distances, so it cannot run away downward."""
    for slot in ("phiv", "phij", "phic"):
        assert L.support_of(f"rsig:{slot}:TRB:P001") == "nonneg"
        assert L.SUPPORTS["nonneg"] == (False, True)
    # while a log-ratio coordinate is two-sided
    assert L.support_of("rsig:band:TRB:top") == "real"
    assert L.support_of("rsig:depth:TRB:n_eff") == "nonneg"


def test_there_is_no_contrast_group_and_no_frozen_naive_vector():
    """The corpus centre IS the subtraction point now.

    Rotating through the ``naive`` corpus subtracts the median Phi of unselected repertoires, which
    is what ``contrast`` measured with a separately drawn 20,000-sequence reference. 231 columns and
    one whole failure mode -- a ``naive`` drawn against a different release of the recombination
    models -- replaced by choosing a corpus.
    """
    assert not any(g.name == "contrast" for g in L.raw_groups("rsig"))
    import mir.signature

    root = __import__("pathlib").Path(mir.signature.__file__).parent
    for gone in ("assemble.py", "blocks.py", "reference.py", "scale.py"):
        assert not (root / gone).exists(), gone
    res = root.parent / "resources" / "signature"
    # The directory is back, but only for corpus artifacts. What must never return is a FROZEN
    # reference: the prototype-cloud rotation, the naive vector and the six scale references were the
    # defect, and a corpus is the opposite of them -- fitted on repertoires, named in the output, and
    # gated on the germline it was drawn from.
    stale = sorted(q.name for q in res.glob("*")
                   if q.name.startswith(("scale_", "rsig_v", "kmer_spaces"))
                   or q.name == "build_rsig.py")
    assert stale == [], f"the old frozen artifacts are still installed: {stale}"
    # Artifacts are release assets fetched on first use since 4.2.0 -- ~110 MB across nine corpora and
    # both halves is not wheel payload -- so what the resources dir must carry is the INDEX. Any .npz
    # that is here is a dev convenience, not the contract; what ships is `corpora.json`.
    from vdjtools.signature.corpus import corpora_index

    idx = corpora_index(res)
    if idx:
        for name, entry in idx.items():
            assert len(entry["npz"]["sha256"]) == 64 and entry["npz"]["bytes"] > 100_000, name
        from mir.signature.signature import bundled_names

        assert set(idx) <= set(bundled_names())


# ------------------------------------------------------------- 4. holes, and the corpus


def test_an_absent_locus_is_a_hole_and_does_not_move_the_present_one(corpus, held_out):
    full = held_out["S0"]
    a = rsig(full, corpus)
    b = rsig({"TRG": full["TRG"]}, corpus)
    for c in a:
        if c.startswith("rsig:pc:TRG:"):
            assert a[c] == b[c], f"dropping TRD moved {c}"
    assert np.isnan(b["rsig:div:TRD:rao"])


def test_a_corpus_for_the_other_half_is_refused(corpus, held_out):
    with pytest.raises(ValueError, match="not 'rsig'"):
        rsig(held_out["S0"], C.Corpus(sig="vsig", name="x", vocab={}, fits={}))


def test_the_corpus_round_trips_and_verifies(corpus, held_out, tmp_path):
    path = corpus.save(tmp_path / "r")
    back = C.Corpus.load(path)
    a, b = rsig(held_out["S1"], corpus), rsig(held_out["S1"], back)
    assert set(a) == set(b)
    for c in a:
        if np.isnan(a[c]):
            assert np.isnan(b[c]), c
        else:
            assert abs(a[c] - b[c]) < 1e-4, c


def test_truncation_is_exact(corpus, held_out):
    full = rsig(held_out["S2"], corpus)
    short = rsig(held_out["S2"], corpus, n_components=2)
    for locus in corpus.fits:
        for i in (1, 2):
            c = f"rsig:pc:{locus}:PC{i:02d}"
            if c in short:
                assert full[c] == short[c], c


def test_rao_is_carried_through_untouched_by_the_rotation(corpus, held_out):
    """A diversity read-out in its own units. Rotating it would mix it with the geometry it sits
    beside, and the head-to-head against the statistics half's Hill numbers is the point."""
    raw, chan = raw_and_channels(held_out["S0"], corpus.vocab)
    out = rsig(held_out["S0"], corpus)
    for locus in ("TRG", "TRD"):
        assert out[f"rsig:div:{locus}:rao"] == chan[f"rsig:div:{locus}:rao"]


def test_whatever_the_bound_clamps_is_reported(corpus, held_out):
    assert rsig(held_out["S0"], corpus, mode="none")["rsig:qc:-:winsor_frac"] == 0.0
    raw, chan = raw_and_channels(held_out["S0"], corpus.vocab)
    raw[corpus.fits["TRG"].columns[0]] = 1e12
    assert C.apply(raw, chan, corpus, mode="features")["rsig:qc:-:winsor_frac"] > 0.0


# ------------------------------------------------------------------- 5. cohort and kwargs


def test_the_cohort_honours_its_kwargs_on_every_locus_and_every_sample(corpus):
    """One line in the old code carried three bugs at three scopes.

    Two ``kw.pop`` calls sat inside a per-locus comprehension over a dict belonging to the caller:
    the first locus consumed ``on_duplicate`` and every later locus silently fell back to
    ``"error"``; with ``sanitise=False`` the pop never ran and the key leaked into ``rsig()`` as a
    TypeError; and because ``kw`` is shared across samples, the first sample drained it for the
    rest. A single-locus, single-sample test passes against all three, so this uses two loci and
    four samples with the duplicate in the SECOND locus.
    """
    rng = np.random.default_rng(7)
    dup = _frame(3, "TRDV1", "TRDJ1", rng)
    dup = pl.concat([dup, dup])                          # a repeated amino-acid clonotype key
    samples = {f"S{i}": {"TRG": _frame(60, "TRGV9", "TRGJ1", rng), "TRD": dup}
               for i in range(4)}

    with pytest.raises(ValueError):
        rsig_cohort(samples, corpus)
    d = rsig_cohort(samples, corpus, on_duplicate="sum")
    assert d.height == 4
    # and with sanitise off the key must not leak through as a TypeError
    d2 = rsig_cohort(samples, corpus, sanitise=False)
    assert d2.height == 4


def test_a_deferred_sample_is_resolved_once_in_the_worker(corpus, held_out):
    import functools

    calls = {"n": 0}

    def read(sid):
        calls["n"] += 1
        return held_out[sid]

    d = rsig_cohort([(sid, functools.partial(read, sid)) for sid in held_out], corpus)
    assert d.height == 3 and calls["n"] == 3


def test_the_two_halves_join_on_sample_id(corpus, held_out):
    """There is no joined entry point any more, deliberately: each half has its own artifact, so
    the wrapper's only real job -- one scale reference over both -- no longer exists. Two calls and
    a polars join is the whole story, and this is that story asserted."""
    from vdjtools.signature import vsig_cohort

    vart, _ = C.synthesize("memory", loci=("TRG", "TRD"), n_samples=20, size=90, seed=6,
                           n_components=4)
    v = vsig_cohort(held_out, vart, cstar_target=0.5)
    r = rsig_cohort(held_out, corpus)
    joined = v.join(r, on="sample_id", how="inner")
    assert joined.height == 3
    assert set(v.columns) & set(r.columns) == {"sample_id"}, "the halves collide"
    assert joined.width == v.width + r.width - 1


# ------------------------------------------------- 6. removed interfaces stay removed


@pytest.mark.parametrize("kw", ["tier", "reference", "scale", "standardize", "clip", "squash",
                                "on_unscaled", "preset"])
def test_every_removed_keyword_raises(corpus, held_out, kw):
    with pytest.raises(TypeError):
        rsig(held_out["S0"], corpus, **{kw: 1})


def test_the_old_entry_points_are_gone():
    """``mir.signature.signature`` is the submodule, not the old joined function -- so this checks
    callability and the public surface rather than mere attribute presence."""
    import mir.signature as S

    for name in ("signature", "signature_cohort", "load_scale", "fit_scale", "load_reference",
                 "ScaleReference", "SignatureReference", "channel_spec"):
        assert name not in S.__all__, f"mir.signature.{name} is still exported"
        obj = getattr(S, name, None)
        assert obj is None or not callable(obj), f"mir.signature.{name} survived the purge"


def test_a_one_part_composition_is_a_hole_rather_than_a_crash(corpus):
    """Every band below the presence floor leaves only the closing residual.

    That is no composition at all, so there are no ratios to take: clr would raise on one part, and
    inventing a second would fabricate a coordinate. The hole is the answer.
    """
    rng = np.random.default_rng(11)
    tiny = _frame(3, "TRGV9", "TRGJ1", rng)
    raw, _ = raw_and_channels({"TRG": tiny}, corpus.vocab, min_clonotypes=10_000)
    assert np.isnan(raw["rsig:band:TRG:singleton"])
    assert np.isfinite(raw["rsig:phiv:TRG:P001"]), "the geometry itself is still measurable"


def test_rao_is_emitted_for_a_locus_the_corpus_does_not_model(corpus):
    """Rao needs the embedder and the sample, not the rotation.

    Gating it on the corpus vocabulary reported a hole for a number that was perfectly computable --
    the mirror image of the mistake the mask channels exist to prevent. Found by an end-to-end run:
    a TRA/TRB/IGH corpus returned nan for ``rsig:div:IGK:rao`` on samples that had IGK.
    """
    rng = np.random.default_rng(12)
    frames = {"TRG": _frame(60, "TRGV9", "TRGJ1", rng),
              "TRB": _frame(60, "TRBV20-1", "TRBJ2-2", rng)}       # TRB is not in this corpus
    assert "TRB" not in corpus.vocab, "fixture changed; pick a locus the corpus lacks"
    raw, chan = raw_and_channels(frames, corpus.vocab)
    assert np.isfinite(chan["rsig:div:TRB:rao"]), "a computable channel was reported as a hole"
    # ...while its raw features stay absent, because only those are indexed by the rotation
    assert not [c for c in raw if ":TRB:" in c]


def test_the_rsig_manifest_records_the_depth_range_it_was_drawn_across():
    """Both halves must be drawn across the same depths, so both must record which.

    The rsig manifest carried no depth field at all, which made the one thing a reader needs in
    order to know whether a corpus covers their samples -- the range of depths it saw -- readable
    only from the vsig half. `depth` and the five `pair:` log-ratios are estimated from the draw;
    nothing about them extrapolates to a cohort two decades deeper or shallower.
    """
    art, _ = synthesize("memory", loci=("TRG",), n_samples=24, size=80, seed=5, n_components=4,
                        depth_spread=50.0)
    assert art.meta["depth_spread"] == {"TRG": 50.0}
    assert art.meta["depth_spread_requested"] == 50.0

    default, _ = synthesize("memory", loci=("TRG",), n_samples=24, size=80, seed=5, n_components=4)
    assert default.meta["depth_spread"]["TRG"] == pytest.approx(94 / 22)
    assert default.meta["depth_spread_requested"] is None
    # the override is not cosmetic: a 50x draw has to reach the corpus's own depth ceiling
    i = art.fits["TRG"].columns.index("rsig:depth:TRG:n_eff")
    assert art.fits["TRG"].bounds[0.01][1][i] > default.fits["TRG"].bounds[0.01][1][i]


def test_the_rsig_half_of_a_cohort_corpus_records_the_same_bands_as_the_vsig_half():
    """A `vsig_<name>`/`rsig_<name>` pair is only joinable if both drew the same repertoires.

    Both halves resolve the corpus name through the one `corpus_plan`, and both write the manifest
    through the one `corpus_meta`, so the cohort, the per-locus depth spread, the singleton-fraction
    band and the germline fingerprint cannot disagree between them. They did have the room to: the
    rsig manifest carried no depth field at all before 4.1.0.
    """
    art, _ = synthesize("synthetic-tissue", loci=("TRG",), n_samples=24, n_components=4, seed=9)
    v, _ = C.synthesize("synthetic-tissue", loci=("TRG",), n_samples=24, n_components=4, seed=9)
    assert art.name == v.name == "synthetic-tissue"
    for key in ("regime", "cohort", "richness_band", "expanded_count_band", "singleton_frac_band",
                "depth_spread", "size_per_locus", "models", "seed"):
        assert art.meta[key] == v.meta[key], key
    assert art.meta["cohort"] == "tissue"
    assert art.meta["expanded_count_band"] == {"TRG": [2.85, 4.42, 6.82, 13.29, 102.00]}
    assert art.meta["mir_version"] and "vdjtools_version" not in art.meta


def test_dispersion_batches_preserve_all_accumulators():
    from mir.embedding.tcremp import TCREmp

    df = _frame(200, 'TRBV20-1', 'TRBJ2-2', np.random.default_rng(17))
    counts = df['duplicate_count'].to_numpy()
    w = F.weights(counts)
    model = TCREmp.from_defaults('human', 'TRB', n_prototypes=32, threads=1)
    masks = {'larger': counts > np.median(counts), 'smaller': counts <= np.median(counts)}
    whole = F.dispersion_pass(df, model, w, band_masks=masks, chunk=1000)
    batched = F.dispersion_pass(df, model, w, band_masks=masks, chunk=17)
    for field in ('phi', 'mean_sq', 'stride_sq', 'second', 'uni_phi', 'uni_mean_sq', 'n_eff'):
        np.testing.assert_allclose(getattr(batched, field), getattr(whole, field),
                                   rtol=1e-12, atol=1e-9)
    for band in masks:
        for a, b in zip(batched.bands[band], whole.bands[band]):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)
