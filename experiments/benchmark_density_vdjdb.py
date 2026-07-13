"""VDJdb density benchmark: epitope-specific ridges and mountains-per-epitope (Theory T6).

VDJdb labels TCRs with the epitope they recognise. Antigen-driven convergent selection makes
an epitope's TCRs pile up into one or more dense **ridges** (density level sets) in TCREMP
embedding space; each separated dense peak is a **mountain** (a convergent CDR3 motif). This
benchmark measures, per epitope, how much of its repertoire sits on enriched ridges (retention)
and how many mountains it splits into — the polyclonality of the response.

Two data-quality / realism knobs the user asked for:

* **Bystander removal by corroboration.** A VDJdb clonotype's ``reference.id`` is a comma-joined
  PMID list; requiring ``>= 2`` distinct references keeps only associations seen in independent
  studies, dropping single-study bystanders (~92k single-ref vs ~2k corroborated TRB rows). We
  report enrichment for both buckets — the filter should raise the signal.
* **Admixed noise from a real repertoire.** We dilute the labelled signal with unlabelled control
  clonotypes (``isalgo/airr_control``, a deep naive TRB repertoire) — the realistic detection
  problem where antigen-specific TCRs are a small fraction of a bystander sea.

Run against **two backgrounds** (appendix §T.6), which is the point of the noise admixture:

* **ALICE** — a generated P_gen background (Poisson). Real repertoires are pervasively convergent,
  so the naive control noise is *itself* denser than P_gen and gets over-flagged (~40%): poor
  specificity. This reproduces the T6 lesson.
* **TCRNET** — a *separate* control repertoire as background (binomial). The admixed naive noise now
  looks like background and drops out; only the extra antigen-convergence of the VDJdb ridges
  survives: high specificity. The biological control supplies what P_gen cannot.

Gates: (1) TCRNET signal enrichment rate >> admixed-noise rate (specificity); (2) the P_gen
background flags far more of the naive noise than the control background does (the over-flagging
contrast); (3) the well-studied epitopes GILGFVFTL / NLVPMVATV each recover >= 1 mountain.

Data: a VDJdb slim dump (default tests/assets/vdjdb.slim.txt.gz; --vdjdb for the full
2026-06-11 Zenodo release) + isalgo/airr_control TRB (cached). Needs [bench].
Run:  python experiments/benchmark_density_vdjdb.py [--vdjdb PATH] [--min-ref N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from _hf import fetch, load_repertoire

from mir.bench.metrics import cluster
from mir.density import (
    enriched_mask,
    fit_density_space,
    generate_background,
    neighbor_enrichment,
)
from mir.embedding.tcremp import TCREmp

CONTROL_REPO = "isalgo/airr_control"
CONTROL_FILE = "human.trb.aa.vdjtools.tsv.gz"
N_PROTO = 1000
N_COMPONENTS = 20
MIN_EP = 10          # min corroborated clonotypes for an epitope to enter the ridge analysis
NOISE_MULT = 3       # admixed control noise = NOISE_MULT x |signal|
BG_MULT = 5          # background size = BG_MULT x |obs|  (>= 5N for a stable ratio, §T.6)
_COLS = ("junction_aa", "v_call", "j_call")


def load_vdjdb_signal(path: str, chain: str, min_ref: int) -> pl.DataFrame:
    """VDJdb clonotypes for *chain*, one row per unique (junction, v, j) with a corroboration count.

    ``reference.id`` is a comma-joined PMID list, so the number of independent references is
    ``str.split(",").len()``. A clonotype seen under several epitopes keeps its best-corroborated
    epitope. Returns ``junction_aa, v_call, j_call, epitope, n_ref``.
    """
    raw = pl.read_csv(path, separator="\t", infer_schema_length=0)
    return (
        raw.filter(
            (pl.col("gene") == chain)
            & pl.col("cdr3").is_not_null()
            & pl.col("antigen.epitope").is_not_null()
            & pl.col("reference.id").is_not_null()
            & pl.col("cdr3").str.contains(r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$")
        )
        .select(
            junction_aa=pl.col("cdr3"),
            v_call=pl.col("v.segm"),
            j_call=pl.col("j.segm"),
            epitope=pl.col("antigen.epitope"),
            n_ref=pl.col("reference.id").str.split(",").list.len(),
        )
        # one point per clonotype: keep the best-corroborated epitope assignment
        .sort("n_ref", descending=True)
        .unique(subset=list(_COLS), keep="first", maintain_order=True)
        .filter(pl.col("n_ref") >= min_ref)
    )


def _enrich(model, obs: pl.DataFrame, bg: pl.DataFrame, test: str):
    """``(enriched-hit mask, obs embedding)`` for *obs* against *bg* in one shared coordinate system."""
    space, obs_emb, bg_emb = fit_density_space(
        model, obs, bg, n_components=N_COMPONENTS, space="full", pca_fit_cap=40000)
    res = neighbor_enrichment(obs_emb, bg_emb, lambda0=3.0, test=test)
    return enriched_mask(res), obs_emb


def _rate(mask: np.ndarray, bucket: np.ndarray, name: str) -> float:
    sel = bucket == name
    return float(mask[sel].mean()) if sel.any() else float("nan")


def main(vdjdb: str, min_ref: int) -> None:
    t0 = time.perf_counter()
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)

    # --- signal: corroborated (>=min_ref) vs single-reference (potential bystanders) ---
    all_sig = load_vdjdb_signal(vdjdb, "TRB", min_ref=1)
    keep_ep = (
        all_sig.filter(pl.col("n_ref") >= min_ref).group_by("epitope").len()
        .filter(pl.col("len") >= MIN_EP)["epitope"].to_list()
    )
    all_sig = all_sig.filter(pl.col("epitope").is_in(keep_ep))
    hi = all_sig.filter(pl.col("n_ref") >= min_ref)
    lo = all_sig.filter(pl.col("n_ref") < min_ref)
    if lo.height > hi.height:
        lo = lo.sample(hi.height, seed=0)          # matched single-ref comparison set
    sig_junctions = set(all_sig["junction_aa"].to_list())

    # --- noise + backgrounds: three disjoint draws from the deep control repertoire ---
    control = load_repertoire(fetch(CONTROL_REPO, CONTROL_FILE))
    control = control.filter(~pl.col("junction_aa").is_in(sig_junctions))  # clean buckets
    n_sig = hi.height + lo.height
    need = NOISE_MULT * n_sig
    obs_total = n_sig + need
    draw = control.sample(min(control.height, need + BG_MULT * obs_total), seed=1, shuffle=True)
    noise = draw.head(need)
    control_bg = draw.slice(need, BG_MULT * obs_total)      # TCRNET background, disjoint from noise
    gen_bg = generate_background("TRB", BG_MULT * obs_total, seed=0)  # ALICE background (P_gen)

    obs = pl.concat([
        hi.select(_COLS).with_columns(bucket=pl.lit("signal>=2ref")),
        lo.select(_COLS).with_columns(bucket=pl.lit("signal 1ref")),
        noise.select(_COLS).with_columns(bucket=pl.lit("noise")),
    ], how="vertical")
    bucket = obs["bucket"].to_numpy()
    epi = np.array(
        hi["epitope"].to_list() + [None] * lo.height + [None] * noise.height, dtype=object)

    print(f"VDJdb {vdjdb.split('/')[-1]}: signal {hi.height} (>= {min_ref}ref) + "
          f"{lo.height} (1ref) over {len(keep_ep)} epitopes; noise {noise.height}; "
          f"control-bg {control_bg.height}; P_gen-bg {gen_bg.height}")

    # --- enrichment under both backgrounds ---
    alice_mask, _ = _enrich(model, obs, gen_bg, test="poisson")
    tcrnet_mask, obs_emb = _enrich(model, obs, control_bg, test="binomial")

    print(f"\n{'bucket':<14}{'n':>7}{'ALICE(Pgen)':>14}{'TCRNET(ctrl)':>14}")
    for b in ("signal>=2ref", "signal 1ref", "noise"):
        n = int((bucket == b).sum())
        print(f"{b:<14}{n:>7}{_rate(alice_mask, bucket, b):>13.1%}{_rate(tcrnet_mask, bucket, b):>13.1%}")

    sig_hi_rate = _rate(tcrnet_mask, bucket, "signal>=2ref")
    noise_rate_ctrl = _rate(tcrnet_mask, bucket, "noise")
    noise_rate_pgen = _rate(alice_mask, bucket, "noise")
    lift = sig_hi_rate / noise_rate_ctrl if noise_rate_ctrl else float("inf")
    print(f"\nspecificity (TCRNET): signal>=2ref {sig_hi_rate:.1%} vs noise {noise_rate_ctrl:.1%} "
          f"= {lift:.1f}x lift")
    print(f"P_gen over-flags the naive noise: ALICE {noise_rate_pgen:.1%} vs TCRNET {noise_rate_ctrl:.1%} "
          f"(the T6 lesson — use a biological control)")

    # --- per-epitope ridges & mountains (TCRNET-enriched signal, one shared clustering) ---
    hi_idx = bucket == "signal>=2ref"
    hi_emb, hi_epi, hi_enr = obs_emb[hi_idx], epi[hi_idx], tcrnet_mask[hi_idx]
    labels = np.full(hi_emb.shape[0], -1, dtype=np.int64)
    if int(hi_enr.sum()) >= 5:
        labels[hi_enr] = cluster(np.ascontiguousarray(hi_emb[hi_enr]))  # DBSCAN in embedding space

    print(f"\n{'epitope':<14}{'n':>6}{'enr%':>7}{'mtns':>6}{'ret%':>7}")
    rows = []
    for ag in keep_ep:
        m = hi_epi == ag
        n = int(m.sum())
        enr = m & hi_enr
        n_enr = int(enr.sum())
        mtns = len(set(labels[enr].tolist()) - {-1})   # distinct density peaks this epitope occupies
        ret = n_enr / n if n else 0.0
        rows.append((ag, n, n_enr / n if n else 0.0, mtns, ret))
    for ag, n, enrf, mtns, ret in sorted(rows, key=lambda r: -r[1]):
        print(f"{ag:<14}{n:>6}{enrf:>6.0%}{mtns:>6}{ret:>6.0%}")

    mtn_counts = [r[3] for r in rows]
    print(f"\nmountains/epitope: median {int(np.median(mtn_counts))}, "
          f"range {min(mtn_counts)}-{max(mtn_counts)}, total {sum(mtn_counts)} across {len(rows)} epitopes")

    # --- gates ---
    ridge = {r[0]: r for r in rows}
    g1 = lift >= 3.0
    g2 = noise_rate_pgen > 1.5 * noise_rate_ctrl
    g3 = all(ridge.get(e, (None, 0, 0, 0, 0))[3] >= 1 for e in ("GILGFVFTL", "NLVPMVATV") if e in ridge)
    verdict = "PASS" if (g1 and g2 and g3) else "FAIL"
    print(f"\n[{verdict}] specificity {lift:.1f}x>=3 ({g1}); P_gen over-flags ({g2}); "
          f"immunodominant epitopes have mountains ({g3})")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vdjdb", default="tests/assets/vdjdb.slim.txt.gz",
                    help="VDJdb slim dump (dotted columns); default bundled, or the 2026-06-11 release")
    ap.add_argument("--min-ref", type=int, default=2, help="min distinct references (bystander filter)")
    a = ap.parse_args()
    main(a.vdjdb, a.min_ref)
