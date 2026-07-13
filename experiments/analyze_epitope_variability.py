"""By-epitope variability of VDJdb ridges in TCREMP density space (companion to the T6 benchmark).

Where ``benchmark_density_vdjdb.py`` asks "can we detect the ridges in noise?", this asks "how do
the ridges differ *between* epitopes?". Each corroborated (>=2-ref) epitope is characterized on its
own clonotypes in one shared embedding with one global DBSCAN eps — so the dominant epitope
(GILGFVFTL) does not bias the others — yielding, per epitope:

* **convergence** — fraction of clonotypes that land on a ridge (a DBSCAN cluster);
* **mountains** — number of distinct ridges (motifs);
* **publicness** — the largest motif's share of the clustered clonotypes (dominance);
* **tightness** (median nearest-neighbour distance) and **diversity** (mean pairwise distance);
* **median marginalized Pgen** — the precursor-frequency axis Pogorelyy 2018 predicts drives it.

It then reports the spread of each metric across epitopes and their Spearman correlation with Pgen.
Finding on the bundled slim dump: ridge *shape* varies 3-5x more than raw CDR3 properties, and Pgen
predicts only CDR3 length (rho -0.65), not the ridge structure — convergence is epitope/MHC biology,
not precursor frequency.

Data: a VDJdb slim dump (default bundled; --vdjdb for the full 2026-06-11 Zenodo release). Human TRB
only (embedded against human prototypes). Needs [bench]. Run:
    python experiments/analyze_epitope_variability.py [--vdjdb PATH] [--min-ep N]
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from benchmark_density_vdjdb import load_vdjdb_signal, N_COMPONENTS, N_PROTO

from mir.bench.metrics import estimate_dbscan_eps
from mir.embedding.pca import pca_denoise
from mir.embedding.tcremp import TCREmp


def _metadata(path: str, chain: str) -> dict[str, tuple[str, str]]:
    """epitope -> (MHC class, antigen species), the modal value per epitope."""
    raw = pl.read_csv(path, separator="\t", infer_schema_length=0).filter(pl.col("gene") == chain)
    agg = raw.group_by("antigen.epitope").agg(
        pl.col("mhc.class").mode().first(), pl.col("antigen.species").mode().first())
    return {r["antigen.epitope"]: (r["mhc.class"] or "?", (r["antigen.species"] or "?")[:13])
            for r in agg.iter_rows(named=True)}


def main(vdjdb: str, min_ep: int) -> None:
    sig = load_vdjdb_signal(vdjdb, "TRB", min_ref=2)
    keep = sig.group_by("epitope").len().filter(pl.col("len") >= min_ep)["epitope"].to_list()
    sig = sig.filter(pl.col("epitope").is_in(keep))
    epi = np.array(sig["epitope"].to_list(), dtype=object)
    cdr3 = sig["junction_aa"].to_list()
    meta = _metadata(vdjdb, "TRB")

    # one shared embedding + one global DBSCAN eps (the paper's kneedle x 0.4)
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    emb = pca_denoise(model.embed(sig), n_components=N_COMPONENTS)
    eps = estimate_dbscan_eps(emb) * 0.4

    # marginalized (CDR3-only) Pgen — the precursor-frequency axis, gene-namespace-independent
    from vdjtools.model import load_bundled, native
    pg = np.asarray(native.pgen_aa_batch(load_bundled("TRB", "learned"), cdr3, mismatches=0, threads=0))
    logpg = np.log10(np.clip(pg, 1e-20, None))

    rng = np.random.default_rng(0)
    rows = []
    for ag in keep:
        m = np.where(epi == ag)[0]
        X = emb[m]
        labs = DBSCAN(eps=eps, min_samples=2).fit_predict(X)      # mountains = clusters of >=2
        clus = labs[labs >= 0]
        top = (np.bincount(clus).max() / len(clus)) if len(clus) else 0.0  # largest motif's share
        nnd = float(np.median(NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]))
        take = m if len(m) <= 60 else rng.choice(m, 60, replace=False)
        rows.append(dict(ag=ag, n=len(m), conv=len(clus) / len(m), mtns=len(set(clus.tolist())),
                         top=top, nnd=nnd, div=float(pdist(emb[take]).mean()),
                         lpg=float(np.median(logpg[m])),
                         clen=float(np.mean([len(cdr3[i]) for i in m])),
                         mhc=meta.get(ag, ("?", "?"))[0], agsp=meta.get(ag, ("?", "?"))[1]))

    print(f"{sig.height} corroborated human TRB clonotypes over {len(keep)} epitopes; shared eps={eps:.0f}\n")
    hdr = (f"{'epitope':<15}{'n':>6}{'conv%':>6}{'mtns':>5}{'top%':>6}{'nnd':>6}{'div':>6}"
           f"{'medPgen':>9}{'len':>5}  {'MHC':<7}{'antigen'}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["conv"]):
        print(f"{r['ag']:<15}{r['n']:>6}{r['conv']*100:>5.0f}%{r['mtns']:>5}{r['top']*100:>5.0f}%"
              f"{r['nnd']:>6.0f}{r['div']:>6.0f}{r['lpg']:>9.1f}{r['clen']:>5.1f}  {r['mhc']:<7}{r['agsp']}")

    def spread(key):
        v = np.array([r[key] for r in rows], float)
        return v.min(), np.median(v), v.max(), v.std() / (abs(v.mean()) + 1e-9)
    print("\n=== spread across epitopes (min / median / max / CV) ===")
    for k, lab in [("conv", "convergence (conv%)"), ("mtns", "mountains"), ("top", "publicness (top%)"),
                   ("nnd", "tightness (nnd)"), ("div", "diversity"), ("lpg", "median log10 Pgen"),
                   ("clen", "CDR3 len")]:
        lo, md, hi, cv = spread(k)
        print(f"  {lab:<22}{lo:8.2f}{md:8.2f}{hi:8.2f}   CV={cv:.2f}")

    print("\n=== Spearman vs median log10 Pgen (precursor frequency), across epitopes ===")
    lpg_e = [r["lpg"] for r in rows]
    for k, lab in [("conv", "convergence"), ("top", "publicness"), ("nnd", "tightness(nnd)"),
                   ("div", "diversity"), ("clen", "CDR3 length"), ("mtns", "mountains")]:
        rho, p = spearmanr(lpg_e, [r[k] for r in rows])
        print(f"  logPgen vs {lab:<15} rho={rho:+.2f}  p={p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vdjdb", default="tests/assets/vdjdb.slim.txt.gz",
                    help="VDJdb slim dump; default bundled, or the 2026-06-11 Zenodo release")
    ap.add_argument("--min-ep", type=int, default=10, help="min corroborated clonotypes per epitope")
    a = ap.parse_args()
    main(a.vdjdb, a.min_ep)
