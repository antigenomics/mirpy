"""Spike VDJdb antigen-specific clones into shallow (SRA-like) repertoires and recover them.

The motivating use case is low-coverage bulk RNA-seq (`10²–10⁴` clonotypes/chain). Can the continuous-density
enrichment (mir.density, Theory T6) still find an antigen-specific response at that depth? We test it with a
controlled **spike-in**: a naive P_gen background of size ``N`` (the shallow "SRA-like" repertoire) plus ``K``
real VDJdb clones for one epitope (e.g. **NLVPMVATV**, the HLA-A*02 CMV pp65 response, n≈13k in VDJdb) — a
convergent public cluster. We enrich the spiked repertoire against a fresh P_gen background and measure
**recall** (fraction of spiked clones flagged, q<0.05) and the background **false-positive rate**, sweeping ``N``.

Because the naive background has no convergent structure, only the planted cluster should light up — so this is
a clean precision/recall test of "recover VDJdb motifs from RNA-seq-depth data". VDJdb is the ground truth
(`mir.bench.vdjdb`); the depths are the SRA regime (`SOURCES.md`).

Run:  python experiments/benchmark_repertoire_spikein.py [epitope] [K_spike]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from mir.density import enriched_mask, fit_density_space, generate_background, neighbor_enrichment
from mir.embedding.tcremp import TCREmp

VDJDB = "tests/assets/vdjdb.slim.txt.gz"
N_PROTO, N_COMPONENTS = 1000, 20
DEPTHS = (500, 1000, 3000, 10_000, 30_000)


def _vdjdb_clones(epitope: str) -> pl.DataFrame:
    """TRB clonotypes (v_call/j_call/junction_aa) for one VDJdb epitope, allele-stripped to gene."""
    raw = pl.read_csv(VDJDB, separator="\t", infer_schema_length=0)
    return (raw.filter((pl.col("gene") == "TRB") & (pl.col("antigen.epitope") == epitope))
            .select(junction_aa=pl.col("cdr3"),
                    v_call=pl.col("v.segm").str.replace(r"\*.*", ""),
                    j_call=pl.col("j.segm").str.replace(r"\*.*", ""))
            .filter(pl.col("v_call").str.starts_with("TRBV") & pl.col("j_call").str.starts_with("TRBJ"))
            .unique())


def _seq_convergent_core(pool: pl.DataFrame, k: int, radius: int = 1) -> pl.DataFrame:
    """The ``k`` most sequence-convergent clones of an epitope pool — a real CDR3 **motif family**.

    Selection uses **CDR3 Hamming distance on the raw strings** (within equal-length buckets) — an
    independent criterion from the BLOSUM-gapblock TCREMB embedding used later for detection, so
    recovery is a *cross-metric* validation, not the circular "select-and-detect in one space".
    Antigen specificity ≠ sequence convergence (a diffuse epitope sample has no near neighbours and is
    invisible to density); a motif family (mutually Hamming-close public clones) is the detectable core.
    """
    from collections import defaultdict

    juncs = pool["junction_aa"].to_list()
    deg = np.zeros(len(juncs))
    by_len = defaultdict(list)
    for i, s in enumerate(juncs):
        by_len[len(s)].append(i)
    for idxs in by_len.values():
        arr = [juncs[i] for i in idxs]
        for a, sa in enumerate(arr):
            deg[idxs[a]] = sum(a != b and sum(x != y for x, y in zip(sa, sb)) <= radius
                               for b, sb in enumerate(arr))
    return pool[np.argsort(-deg)[:k]]                               # most Hamming-≤radius neighbours


def main(epitope: str = "NLVPMVATV", k_spike: int = 50, spike_count: int = 20) -> None:
    t0 = time.perf_counter()
    spike_pool = _vdjdb_clones(epitope)
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    bg_ref = generate_background("TRB", 100_000, seed=1)                 # shared enrichment background
    print(f"epitope {epitope}: {spike_pool.height} VDJdb TRB clones available; spike K={k_spike} "
          f"convergent-core clones (each expanded to {spike_count} reads) into P_gen backgrounds of depth N\n")
    print(f"{'N':>7}{'recall':>9}{'ab_recall':>11}{'bg_FPR':>9}{'spike_fold':>12}")

    spike = _seq_convergent_core(spike_pool, k_spike).with_columns(
        pl.lit(float(spike_count)).alias("duplicate_count"))          # antigen clones are clonally expanded
    recalls = []
    for N in DEPTHS:
        naive = generate_background("TRB", N, seed=0).with_columns(pl.lit(1.0).alias("duplicate_count"))
        obs = pl.concat([naive.select(["v_call", "j_call", "junction_aa", "duplicate_count"]),
                         spike.select(["v_call", "j_call", "junction_aa", "duplicate_count"])])
        is_spike = np.r_[np.zeros(naive.height, bool), np.ones(spike.height, bool)]
        space, obs_emb, bg_emb = fit_density_space(model, obs, bg_ref, n_components=N_COMPONENTS,
                                                   space="full", pca_fit_cap=40_000)
        abund = obs["duplicate_count"].to_numpy()
        res = neighbor_enrichment(obs_emb, bg_emb, lambda0=5.0)                         # breadth only
        res_ab = neighbor_enrichment(obs_emb, bg_emb, lambda0=5.0, abundance=abund, weight="log1p")
        recall = enriched_mask(res, alpha=0.05)[is_spike].mean()
        recall_ab = enriched_mask(res_ab, alpha=0.05)[is_spike].mean()
        fpr = enriched_mask(res_ab, alpha=0.05)[~is_spike].mean()
        recalls.append(max(recall, recall_ab))
        print(f"{N:>7}{recall:>9.2%}{recall_ab:>11.2%}{fpr:>9.2%}{res_ab.fold[is_spike].mean():>12.2f}")

    best = max(recalls)
    verdict = "PASS" if best > 0.5 else "PARTIAL" if best > 0.2 else "FAIL"
    print(f"\n[{verdict}] best recall of spiked {epitope} clones = {best:.0%} (Hamming-selected motif family,"
          f" detected by the independent BLOSUM-embedding density — a cross-metric, non-circular test).")
    print("Caveats: (1) recovery is partial because a Hamming-tight family is not perfectly embedding-tight; "
          "(2) FPR is against a *clean* P_gen null — a real repertoire has its own convergent clusters, so use "
          "a biological differential control (T6) for a realistic false-positive rate.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(args[1] if len(args) > 1 else "NLVPMVATV",
         int(args[2]) if len(args) > 2 else 50,
         int(args[3]) if len(args) > 3 else 20)
