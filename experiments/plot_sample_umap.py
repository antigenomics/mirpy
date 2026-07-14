"""UMAP of the sample-level repertoire embedding — *see* age / HLA / batch / COVID / CMV structure.

Each point is one repertoire, laid out by UMAP of its kernel-mean block Φ₁ (which *is* the MMD geometry:
``‖Φ₁(S)−Φ₁(S')‖ ≈ MMD``). Panels recolor the same layout by different metadata, so you can read off
what the embedding organizes by — and, for the batch-confounded cohorts, whether **batch** dominates the
map (the prop:batch story: a raw layout clusters by sequencing run; residualizing Φ₁ on batch dissolves it
so the biological label re-emerges).

Cohorts (each fits ONE RepertoireSpace on its pooled cloud, then embeds every sample):
  covid  — airr_covid19 (TRB): COVID status, batch, HLA-A*02 carriage, IgG serostatus. Two rows: raw Φ₁
           and batch-residualized Φ₁ (watch batch dissolve, status sharpen).
  aging  — airr_benchmark Britanova (TRB): age (continuous), sequencing batch.
  hip    — airr_hip Emerson 2017 (TRB): CMV serostatus, HLA-A*02, age.

Data: local ~/hf or HF fallback (needs [bench] + umap-learn from [examples]).
Run:  python experiments/plot_sample_umap.py [cohort] [n] [downsample_reads]
      cohort ∈ {covid, aging, hip} (default covid). Writes PNG + PDF to experiments/figures/.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _cohort import pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048


def _phi1(space, frames) -> np.ndarray:
    """Kernel-mean block Φ₁ (the MMD geometry), standardized for UMAP."""
    M = np.stack([sample_embedding(space, f, blocks=("mean",)).mean for f in frames])
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


def _umap(X: np.ndarray) -> np.ndarray:
    import umap
    k = min(15, X.shape[0] - 1)
    return umap.UMAP(n_neighbors=k, min_dist=0.1, random_state=0).fit_transform(X)


def _scatter(ax, coords, labels, title, *, continuous=False):
    if continuous:
        vals = np.array([np.nan if v is None or v == "" else float(v) for v in labels])
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="viridis", s=16, alpha=0.85)
        plt.colorbar(sc, ax=ax, shrink=0.75)
    else:
        cats = sorted(set(map(str, labels)))
        cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
        for i, c in enumerate(cats):
            m = np.array([str(v) == c for v in labels])
            ax.scatter(coords[m, 0], coords[m, 1], s=16, alpha=0.85, color=cmap(i % cmap.N), label=c)
        ax.legend(fontsize=7, markerscale=1.3, loc="best", framealpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def _figure(rows, suptitle, stem):
    """rows = list of (row_label, Φ matrix, [(col_title, labels, continuous), ...]); one UMAP per row."""
    ncol = max(len(cols) for _, _, cols in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.2 * ncol, 4.0 * len(rows)), squeeze=False)
    for ri, (row_label, X, cols) in enumerate(rows):
        coords = _umap(X)
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci < len(cols):
                title, labels, cont = cols[ci]
                _scatter(ax, coords, labels, f"{row_label}: {title}", continuous=cont)
            else:
                ax.axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{stem}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUTDIR}/{stem}.png (+.pdf)")


def _fit(frames, locus="TRB"):
    model = TCREmp.from_defaults("human", locus, n_prototypes=N_PROTO)
    return fit_repertoire_space(model, pooled_clonotypes([(None, f) for f in frames]),
                                n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)


def run_covid(n_donors: int, downsample_to: int) -> None:
    from _covid import batch_of, carries, load_covid, residualize
    rows, frames = load_covid(n_donors, downsample_to, statuses=("COVID", "healthy", "precovid"))
    space = _fit(frames)
    X = _phi1(space, frames)
    batch = batch_of(rows)
    panels = [("COVID status", [r["COVID_status"] for r in rows], False),
              ("batch", batch, False),
              ("HLA-A*02", ["A*02+" if carries(r, "A*02") else "A*02−" for r in rows], False),
              ("IgG", [r.get("COVID_IgG") or "NA" for r in rows], False)]
    Xr = residualize(X, batch)
    _figure([("raw Φ₁", X, panels),
             ("batch-residualized Φ₁", Xr, panels)],
            f"airr_covid19 sample UMAP ({len(rows)} donors, ≤{downsample_to} reads)", "umap_covid")


def run_aging(cap: int, downsample_to: int) -> None:
    from _cohort import load_cohort
    _, samples = load_cohort("isalgo/airr_benchmark", "vdjtools/metadata_aging.txt",
                             prefix="vdjtools/", suffix=".gz", downsample_to=downsample_to, cap_samples=cap or None)
    frames = [df for _, df in samples]
    space = _fit(frames)
    X = _phi1(space, frames)
    ages = [r["age"] for r, _ in samples]
    batch = [r["sample_id"].split("-")[0] for r, _ in samples]
    _figure([("Φ₁", X, [("age", ages, True), ("batch", batch, False)])],
            f"aging (Britanova) sample UMAP ({len(frames)} donors, ≤{downsample_to} reads)", "umap_aging")


def run_hip(n_per_class: int, downsample_to: int) -> None:
    from _cohort import load_cohort
    from benchmark_repertoire_cmvhla import _age_matched_cmv, STRATIFIER
    allow = _age_matched_cmv(n_per_class)
    _, samples = load_cohort("isalgo/airr_hip", "metadata.txt", downsample_to=downsample_to,
                             only=allow)
    frames = [df for _, df in samples]
    space = _fit(frames)
    X = _phi1(space, frames)
    cmv = [{"+": "CMV+", "-": "CMV−"}.get(r["cmv"], "NA") for r, _ in samples]
    a02 = ["A*02+" if STRATIFIER in (r["hla"] or "") else "A*02−" for r, _ in samples]
    ages = [r["age"] if r.get("age") not in (None, "NA") else None for r, _ in samples]
    _figure([("Φ₁", X, [("CMV", cmv, False), ("HLA-A*02", a02, False), ("age", ages, True)])],
            f"airr_hip (Emerson 2017) sample UMAP ({len(frames)} donors, ≤{downsample_to} reads)", "umap_hip")


def main() -> None:
    a = sys.argv
    cohort = a[1] if len(a) > 1 else "covid"
    n = int(a[2]) if len(a) > 2 else (300 if cohort == "covid" else 60)
    ds = int(a[3]) if len(a) > 3 else 20_000
    t0 = time.perf_counter()
    {"covid": run_covid, "aging": run_aging, "hip": run_hip}[cohort](n, ds)
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
