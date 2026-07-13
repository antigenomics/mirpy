"""Reproduce TCREMP supplementary S1-S3 with the v3 pipeline.

Validates the embedding's claimed properties against the published supplementary
(Kremlyakova et al., JMB 2025; suppl. revision 2):

* S2 / Theory T1 — Pearson(D_ij, d_ij), paper 0.56.
* S1 / Theory T4 — d_ij ~ Gamma (> Normal); D_ij ~ GEV/Frechet (> Normal), paper xi=+0.11.
* S3           — distances from real vs model prototypes, paper Pearson 0.96.

Run (needs the [bench] extra for the Smith-Waterman baseline):
    python experiments/reproduce_supplementary.py
"""

from __future__ import annotations

import time

import numpy as np

from mir.bench.theory import (
    fit_distributions,
    prototype_source_correlation,
    s2_dissimilarity_distance_correlation,
)
from mir.embedding.prototypes import load_prototypes

N = 3000


def _report_s1(tag: str, fits: dict) -> None:
    dg, dn, Dg, Dn = fits["d_gamma"], fits["d_normal"], fits["D_gev"], fits["D_normal"]
    dwin = "Gamma" if dg["aic"] < dn["aic"] else "Normal"
    Dwin = "GEV/Frechet" if Dg["aic"] < Dn["aic"] else "Normal"
    print(f"  [{tag}] d_ij -> {dwin} (Gamma AIC {dg['aic']:.0f} vs Normal {dn['aic']:.0f})")
    print(f"  [{tag}] D_ij -> {Dwin} (GEV KS {Dg['ks']:.3f} vs Normal {Dn['ks']:.3f}; "
          f"xi={fits['D_gev_xi']:+.3f})")


def main() -> None:
    t0 = time.perf_counter()
    cdr3 = load_prototypes("human", "TRB", n=N)["junction_aa"].to_list()

    print("=== S2 / T1: Pearson(D_ij, d_ij)  (paper 0.56) ===")
    res_gb = s2_dissimilarity_distance_correlation(cdr3, dissimilarity="gapblock")
    res_sw = s2_dissimilarity_distance_correlation(cdr3, dissimilarity="sw")
    print(f"  gapblock (v3): R = {res_gb.pearson:.3f}   ({res_gb.n_pairs:,} pairs)")
    print(f"  SW (paper):    R = {res_sw.pearson:.3f}")

    print("\n=== S1 / T4: distribution laws (paper d~Gamma, D~Frechet) ===")
    _report_s1("SW/paper", fit_distributions(res_sw.d, res_sw.D))
    _report_s1("gapblock", fit_distributions(res_gb.d, res_gb.D))

    print("\n=== S3: real vs model prototypes  (paper 0.96) ===")
    from vdjtools.model import generate, load_bundled

    model = generate.generate(load_bundled("TRB", source="learned"), 6000, seed=42,
                              productive_only=True)["junction_aa"] \
        .unique(maintain_order=True).to_list()[:N]
    query = load_prototypes("human", "TRB", n=6000)["junction_aa"].to_list()[N:N + 800]
    s3 = prototype_source_correlation(query, cdr3, model)
    print(f"  Pearson(D_real, D_model) = {s3['pearson']:.3f}   ({s3['n_pairs']:,} pairs)")
    print(f"\nTotal {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
