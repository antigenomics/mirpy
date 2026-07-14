"""ALICE-style YF benchmark: continuous-density enrichment (mir.density) recovers more
day-15 yellow-fever-expanded clones matching VDJdb LLWNGPMAV (the HLA-A*02 immunodominant
NS4b epitope) than at day 0 — and clonal abundance sharpens the recovery.

Each timepoint is enriched independently against a vdjtools P_gen background (the ALICE
regime), the enriched clones are matched to the LLW/A*02 reference, and the LLW-matching hit
counts are compared. Vaccination recruits *more distinct* LLW-specific clonotypes by day 15
(here 129 vs 79 present), so day 15 yields more LLW hits — the published ALICE result.

We run the enrichment two ways (appendix §T.6 ``sec:dens-abund``): the shipped **distinct** count
(``g≡1``), and the abundance-aware **weighted** mass ``S=Σ log(1+a_j)`` that adds the clonal-depth
channel. YF-specific clones are both convergent *and* clonally expanded, so folding in the read
counts (``duplicate_count``) recovers more of them — the point of incorporating counts.

Process the *full* repertoire (subsampling dilutes the sparse convergent clusters). A real
repertoire is pervasively convergent, so the P_gen enrichment flags many clones; the LLW
reference match supplies the antigen specificity and the day15-vs-day0 comparison the biology.

Data: isalgo/airr_yfv19 (AIRR TSV), LLW reference from bundled tests/assets/llwngpmav_trb_a02.tsv.gz.
Downloaded + cached on first run (needs [bench]).
Run:  python experiments/benchmark_density_yfv.py [donor] [cap]
"""

from __future__ import annotations

import sys
import time

import polars as pl

from _hf import fetch, load_repertoire

from mir.density import enriched_mask, fit_density_space, generate_background, neighbor_enrichment
from mir.embedding.tcremp import TCREmp

REPO = "isalgo/airr_yfv19"
N_PROTO = 1000
N_COMPONENTS = 20
PCA_FIT_CAP = 40000
BG_SIZE = 200000


def _llw_reference() -> list[str]:
    return pl.read_csv("tests/assets/llwngpmav_trb_a02.tsv.gz", separator="\t")["junction_aa"].to_list()


def main(donor: str = "Q1", cap: int = 0) -> None:
    t0 = time.perf_counter()
    llw = _llw_reference()
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    bg = generate_background("TRB", BG_SIZE, seed=0)  # shared P_gen background (ALICE regime)

    print(f"donor {donor}, background {bg.height} P_gen clones, "
          f"{len(llw)} LLW/A*02 reference CDR3s  (cap={cap or 'full'})")
    print(f"\n{'day':<5}{'method':<11}{'clones':>9}{'LLW_in':>8}{'hits':>9}{'LLW_hits':>10}{'recall':>8}")
    counts: dict[tuple[str, str], int] = {}
    for day in ("0", "15"):
        obs = load_repertoire(fetch(REPO, f"{donor}_{day}_F1.airr.tsv.gz"))
        if cap and obs.height > cap:
            obs = obs.sample(cap, seed=0)
        space, obs_emb, bg_emb = fit_density_space(
            model, obs, bg, n_components=N_COMPONENTS, space="full", pca_fit_cap=PCA_FIT_CAP)
        abund = obs["duplicate_count"].to_numpy()
        llw_in = obs.filter(pl.col("junction_aa").is_in(llw)).height
        runs = (  # ALICE (Poisson vs P_gen), distinct count vs abundance-weighted mass + orphan
            ("distinct", neighbor_enrichment(obs_emb, bg_emb, lambda0=5.0)),
            ("abundance", neighbor_enrichment(obs_emb, bg_emb, lambda0=5.0, abundance=abund, weight="log1p")),
        )
        for name, res in runs:
            hits = obs.filter(enriched_mask(res, alpha=0.05))
            llw_hits = hits.filter(pl.col("junction_aa").is_in(llw)).height
            counts[(day, name)] = llw_hits
            recall = llw_hits / llw_in if llw_in else 0.0
            print(f"d{day:<4}{name:<11}{obs.height:>9}{llw_in:>8}{hits.height:>9}{llw_hits:>10}{recall:>7.0%}")

    d15, d0 = counts[("15", "abundance")], counts[("0", "abundance")]
    boost = counts[("15", "abundance")] - counts[("15", "distinct")]
    verdict = "PASS" if d15 > d0 else "FAIL"
    print(f"\n[{verdict}] abundance-aware LLW hits: d15={d15} > d0={d0}; "
          f"counts boost d15 recovery by {boost:+d} vs distinct ({counts[('15','distinct')]})")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(args[1] if len(args) > 1 else "Q1", int(args[2]) if len(args) > 2 else 0)
