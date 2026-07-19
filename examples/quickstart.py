# mirpy — TCREMP embedding quickstart.
# Reactive marimo app: pick a mode and prototype count, embed a TCR set, PCA-denoise,
# and see the 2-D UMAP + DBSCAN clustering. Uses a local VDJdb dump for antigen colours
# when present (tests/assets/vdjdb.slim.txt.gz), else the bundled prototypes.
# Run with:  marimo edit notebooks/quickstart.py
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from mir.bench.metrics import cluster, cluster_metrics
    from mir.bench.vdjdb import antigen_subset, load_vdjdb
    from mir.embedding.pca import pca_denoise
    from mir.embedding.prototypes import load_prototypes
    from mir.embedding.tcremp import TCREmp
    return (antigen_subset, cluster, cluster_metrics, load_prototypes,
            load_vdjdb, np, os, pca_denoise, pl, plt, TCREmp)


@app.cell
def _(mo):
    mo.md(
        """
        # TCREMP embedding quickstart
        Each TCR is embedded as its **distances to a fixed prototype set**; Euclidean distance in
        this space approximates pairwise alignment distance. Pick the embedding mode and the number
        of prototypes below — everything recomputes reactively.
        """
    )
    return


@app.cell
def _(mo):
    mode = mo.ui.dropdown(["vjcdr3", "cdr123"], value="vjcdr3", label="embedding mode")
    n_proto = mo.ui.slider(100, 3000, value=1000, step=100, label="# prototypes")
    mo.hstack([mode, n_proto], justify="start", gap=2)
    return mode, n_proto


@app.cell
def _(antigen_subset, load_prototypes, load_vdjdb, os, pl):
    # antigen-labelled TCRs when a VDJdb dump is available, else unlabelled prototypes
    _vdjdb = "tests/assets/vdjdb.slim.txt.gz"
    if os.path.exists(_vdjdb):
        _df = antigen_subset(load_vdjdb(_vdjdb), "TRB", 300)
        query = _df.group_by("epitope", maintain_order=True).head(400)
        labels_true = query["epitope"].to_list()
        source = "VDJdb TRB (antigen-labelled)"
    else:
        query = load_prototypes("human", "TRB", n=5000)[3000:4200]  # disjoint from prototypes
        labels_true = None
        source = "bundled prototypes (unlabelled)"
    return labels_true, query, source


@app.cell
def _(TCREmp, mode, n_proto, pca_denoise, query):
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=int(n_proto.value),
                                 mode=mode.value)
    X = model.embed(query)
    Xp = pca_denoise(X, n_components=50)
    return X, Xp, model


@app.cell
def _(Xp, np):
    import umap
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xp)
    return (coords,)


@app.cell
def _(coords, labels_true, mode, plt, query, source, X):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if labels_true is not None:
        cats = sorted(set(labels_true))
        idx = {c: i for i, c in enumerate(cats)}
        col = [idx[l] for l in labels_true]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=col, cmap="tab20", s=6, alpha=0.7)
        ax.set_title(f"UMAP of TCREMP({mode.value}) embedding — {len(cats)} antigens")
    else:
        lens = [len(s) for s in query["junction_aa"].to_list()]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=lens, cmap="viridis", s=6, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="CDR3 length")
        ax.set_title(f"UMAP of TCREMP({mode.value}) embedding")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.text(0.02, 0.98, f"{X.shape[0]} TCRs × {X.shape[1]} features\n{source}",
            transform=ax.transAxes, va="top", fontsize=8)
    fig
    return


@app.cell
def _(cluster, cluster_metrics, labels_true, mo, Xp):
    if labels_true is None:
        _out = mo.md("*Provide a VDJdb dump for antigen-cluster F1 / retention metrics.*")
    else:
        _lab = cluster(Xp, min_samples=3)
        _m = cluster_metrics(_lab, labels_true)
        _rows = [
            {"epitope": ag, "n": v.n, "F1": round(v.f1, 2), "retention": round(v.retention, 2)}
            for ag, v in sorted(_m.items(), key=lambda kv: -kv[1].n)
        ]
        _out = mo.ui.table(_rows, label="DBSCAN antigen-cluster metrics")
    _out
    return


if __name__ == "__main__":
    app.run()
