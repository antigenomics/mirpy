"""Export the TCREMP theory-appendix data as text files (supplementary S1-S3). 2026-07-13.

Build-time only (conda `mirpy` env + [bench] extra: BioPython for the SW baseline). Reuses
``mir.bench.theory`` so figures and reproduced numbers share one source. Writes plain TSVs +
a key/value stats file under ``appendix/data/``; the ``.gp`` gnuplot scripts render them (data
and plotting kept separate). Regenerate with:  python gen_theory_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from mir.bench.theory import (
    fit_distributions,
    s2_dissimilarity_distance_correlation,
)
from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import load_prototypes

DATA = Path(__file__).resolve().parent / "data"
N = 2500
_RNG = np.random.default_rng(0)


def _scatter(path: Path, x: np.ndarray, y: np.ndarray, header: str, k: int = 8000) -> None:
    idx = _RNG.choice(x.size, min(k, x.size), replace=False)
    idx.sort()
    np.savetxt(path, np.column_stack([x[idx], y[idx]]), fmt="%.4f",
               delimiter="\t", header=header, comments="# ")


def _hist(path: Path, data: np.ndarray, header: str, bins: int = 80) -> None:
    dens, edges = np.histogram(data, bins=bins, density=True)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    np.savetxt(path, np.column_stack([ctr, dens]), fmt="%.6g",
               delimiter="\t", header=header, comments="# ")


def _curves(path: Path, x: np.ndarray, cols: dict[str, np.ndarray], header: str) -> None:
    np.savetxt(path, np.column_stack([x, *cols.values()]), fmt="%.6g",
               delimiter="\t", header=header, comments="# ")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    cdr3 = load_prototypes("human", "TRB", n=N)["junction_aa"].to_list()

    # ---- S2 / T1: embedding distance D_ij vs dissimilarity d_ij (both metrics) ----
    res_sw = s2_dissimilarity_distance_correlation(cdr3, dissimilarity="sw")
    res_gb = s2_dissimilarity_distance_correlation(cdr3, dissimilarity="gapblock")
    _scatter(DATA / "s2_sw.tsv", res_sw.d, res_sw.D, "d_ij\tD_ij  (Smith-Waterman)")
    _scatter(DATA / "s2_gapblock.tsv", res_gb.d, res_gb.D, "d_ij\tD_ij  (gapblock v3)")

    # ---- S1 / T4: distribution laws, fit on the SW metric (as in the paper) ----
    fits = fit_distributions(res_sw.d, res_sw.D)
    d = res_sw.d[res_sw.d > 0]
    _hist(DATA / "s1_diss_hist.tsv", d, "center\tdensity  (d_ij)")
    xd = np.linspace(d.min(), d.max(), 300)
    _curves(DATA / "s1_diss_curves.tsv", xd,
            {"gamma": stats.gamma.pdf(xd, *fits["d_gamma"]["params"]),
             "normal": stats.norm.pdf(xd, *fits["d_normal"]["params"])},
            "x\tgamma\tnormal")
    D = res_sw.D
    _hist(DATA / "s1_dist_hist.tsv", D, "center\tdensity  (D_ij)")
    xD = np.linspace(D.min(), D.max(), 300)
    _curves(DATA / "s1_dist_curves.tsv", xD,
            {"gev": stats.genextreme.pdf(xD, *fits["D_gev"]["params"]),
             "normal": stats.norm.pdf(xD, *fits["D_normal"]["params"])},
            "x\tgev\tnormal")

    # ---- S3: real vs model prototype embedding distances ----
    from scipy.spatial.distance import pdist
    from vdjtools.model import generate, load_bundled

    model_p = generate.generate(load_bundled("TRB", source="olga"), 6000, seed=42,
                                productive_only=True)["junction_aa"] \
        .unique(maintain_order=True).to_list()[:N]
    query = load_prototypes("human", "TRB", n=N + 800)["junction_aa"].to_list()[N:N + 600]
    Dr = pdist(junction_distance_matrix(query, cdr3).astype(np.float64))
    Dm = pdist(junction_distance_matrix(query, model_p).astype(np.float64))
    r_s3 = float(np.corrcoef(Dr, Dm)[0, 1])
    _scatter(DATA / "s3.tsv", Dr, Dm, "D_real\tD_model")

    # ---- key/value stats (quoted by the .tex tables and gnuplot titles) ----
    st = {
        "r_sw": res_sw.pearson, "r_gb": res_gb.pearson,
        "ks_d_gamma": fits["d_gamma"]["ks"], "ks_d_normal": fits["d_normal"]["ks"],
        "aic_d_gamma": fits["d_gamma"]["aic"], "aic_d_normal": fits["d_normal"]["aic"],
        "ks_D_gev": fits["D_gev"]["ks"], "ks_D_normal": fits["D_normal"]["ks"],
        "xi": fits["D_gev_xi"], "r_s3": r_s3, "n": N,
    }
    with open(DATA / "theory_stats.txt", "w") as f:
        for k, v in st.items():
            f.write(f"{k}\t{v:.6g}\n")
    print("wrote", ", ".join(sorted(p.name for p in DATA.glob("*"))))
    print(f"S2 SW R={st['r_sw']:.3f} gapblock R={st['r_gb']:.3f} | "
          f"D GEV/Normal KS {st['ks_D_gev']:.3f}/{st['ks_D_normal']:.3f} xi={st['xi']:+.3f} | "
          f"S3 R={st['r_s3']:.3f}")


if __name__ == "__main__":
    main()
