# mirpy — the portable repertoire signature: why the geometry half is transformed differently,
# and how many components are worth keeping.
#
# Reactive marimo app. Runs on repertoires SAMPLED FROM THE BUNDLED vdjtools MODELS — no download,
# no cohort. The subject is the feature machinery, and generated repertoires exercise it exactly
# as real ones do.
#
# Three sections:
#   1. the live transform table, read from the registry rather than written down here;
#   2. why `rsig` coordinates take `transform="none"` while `vsig` proportions take arcsine/logit;
#   3. how many principal components survive a group-disjoint refit -- and why "90% of the
#      variance" is the wrong question.
#
# Run with:  marimo edit examples/signature.py
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import polars as pl

    import mir.signature as S
    from vdjtools.model import load_bundled
    from vdjtools.model.generate import generate
    return S, generate, load_bundled, np, pl


@app.cell
def _(mo):
    mo.md(
        """
        # The portable signature — geometry half

        `mir.signature` contributes `rsig`: coordinates, norms and mixture coefficients of the
        prototype-sum measure $\\Phi(S) = \\sum_\\sigma w_\\sigma z_\\sigma$. Its partner is
        `vdjtools.signature` (`vsig`), which contributes statistics of the clone-size vector.
        They share one column contract and one transform registry.
        """
    )
    return


@app.cell
def _(S, pl):
    # Read from the registry, not from a table typed into this notebook: if a block's transform
    # changes, this cell changes with it.
    S.describe(tier="full").group_by(["sig", "transform"]).len().sort(["sig", "len"],
                                                                     descending=[False, True])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Two halves, opposite tables

        `vsig` is dominated by `arcsine`, `clr` and `logit`. `rsig` is dominated by `none`. That is
        not an oversight — it follows from what the columns are.

        A `vsig` proportion is bounded, discrete, and **mean–variance coupled**: for a binomial
        share the variance is $p(1-p)/m$, a function of the value itself. A variance-stabilising
        transform is exactly the right tool.

        An `rsig` coordinate is a linear functional of a weighted mean of *fixed* embedding
        vectors. It is signed, roughly symmetric, has no boundary to compress against, and its
        variance is $\\approx \\sigma^2 / n_\\text{eff}$ — which depends on depth but **not on the
        coordinate's own value**. There is nothing for a stabiliser to stabilise, and `log`,
        `logit` and `arcsine` are all undefined on half its range anyway.
        """
    )
    return


@app.cell
def _(S, pl):
    S.describe(tier="full").filter(pl.col("sig") == "rsig").group_by(
        ["block", "transform", "magnitude"]).len().sort(["block", "transform"])
    return


@app.cell
def _(mo):
    mo.md(
        """
        Where `rsig` *does* transform, the quantity has stopped being a coordinate: the block
        **norms** ($\\lVert\\Phi\\rVert$, Rao dispersion) are non-negative right-skewed magnitudes,
        so they take `log1p`; `band` is a genuine closed composition, so it takes `clr` and ships
        $k-1$ parts.

        And one block breaks the pattern on purpose. `contrast` — $\\Psi = \\text{mass}\\cdot(\\Phi -
        \\text{naive})$ — carries `magnitude=True`: **one frozen scalar RMS for the whole block, and
        no centring at all**. Per-column z-scoring would give every coordinate unit variance, which
        makes a sample sitting near zero — an immune desert, a repertoire that has barely moved
        from naive — indistinguishable from a typical one. How far a repertoire is from naive *is*
        what that block exists to carry.
        """
    )
    return


@app.cell
def _(mo):
    n_donors = mo.ui.slider(30, 240, value=90, step=30, label="synthetic donors")
    n_donors
    return (n_donors,)


@app.cell
def _(S, generate, load_bundled, n_donors, np, pl):
    model = load_bundled("TRB", source="olga")

    def clones(n, seed):
        """`generate` emits one rearrangement per row; a clonotype frame counts the duplicates."""
        return (generate(model, n, seed=seed)
                .group_by(["junction_aa", "v_call", "j_call"]).len()
                .rename({"len": "duplicate_count"})
                .with_columns((pl.col("duplicate_count")
                               / pl.col("duplicate_count").sum()).alias("frequency")))

    # Ten "labs" of nine donors each. Nothing distinguishes them biologically -- they are draws
    # from one generative model -- so any component that fails to reproduce across a lab-disjoint
    # refit below is failing on sampling noise alone, with no batch effect to blame.
    per_lab = max(n_donors.value // 10, 3)
    rows, lab = [], []
    for _l in range(10):
        for _d in range(per_lab):
            r = S.rsig({"TRB": clones(400, 1000 + _l * 100 + _d)}, tier="standard")
            rows.append(r)
            lab.append(_l)
    cols = [c for c in rows[0] if c.startswith("rsig:") and np.isfinite(rows[0][c])]
    X = np.array([[r[c] for c in cols] for r in rows])
    lab = np.array(lab)
    X = X[:, np.isfinite(X).all(0) & (X.std(0) > 0)]
    f"{X.shape[0]} donors x {X.shape[1]} finite rsig columns, {len(np.unique(lab))} labs"
    return X, clones, cols, lab, model, per_lab, rows


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. How many components survive a refit

        Three criteria, applied to the same matrix. They do not agree, and the disagreement is the
        lesson.

        * **cumulative variance** — how much of *this* corpus you reproduce.
        * **per-component correlation across a group-disjoint refit** — is this *axis* identified?
          Both halves score the same held-out donors, so the number answers "would a collaborator's
          refit put my samples in the same place".
        * **subspace overlap** $\\lVert V_a V_b^\\top\\rVert_F^2 / k$ — is this *subspace*
          identified? Invariant to any rotation within the retained span, so unlike the
          per-component number it does not punish two near-degenerate components for swapping
          order. Chance level is $k/p$, not 0.
        """
    )
    return


@app.cell
def _(X, lab, np, pl):
    def robust_z(A, clip=8.0):
        med = np.median(A, 0)
        mad = 1.4826 * np.median(np.abs(A - med), 0)
        mad[mad <= 0] = A.std(0)[mad <= 0]
        mad[mad <= 0] = 1.0
        return np.clip((A - med) / mad, -clip, clip)

    def refit_stability(Z, groups, k, seeds=8, seed=0):
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        r_acc, o_acc = [], []
        for _ in range(seeds):
            a, b, c = np.array_split(rng.permutation(uniq), 3)
            ia, ib, ic = (np.isin(groups, x) for x in (a, b, c))
            sc, bs = [], []
            for i in (ia, ib):
                M = Z[i].mean(0)
                V = np.linalg.svd(Z[i] - M, full_matrices=False)[2][:k]
                sc.append((Z[ic] - M) @ V.T)
                bs.append(V)
            r_acc.append([abs(np.corrcoef(sc[0][:, j], sc[1][:, j])[0, 1]) for j in range(k)])
            C = bs[0] @ bs[1].T
            o_acc.append([np.sum(C[:j + 1, :j + 1] ** 2) / (j + 1) for j in range(k)])
        return np.nanmean(r_acc, 0), np.nanmean(o_acc, 0)

    Z = robust_z(X)
    # Each refit sees a third of the donors, so its basis has at most that rank -- ask for more
    # components than a third can support and the SVD simply does not return them.
    K = min(12, X.shape[1] - 1, X.shape[0] // 3 - 2)
    ev = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    r_comp, overlap = refit_stability(Z, lab, K)
    stability = pl.DataFrame({
        "k": np.arange(1, K + 1),
        "cumulative variance": np.cumsum(ev)[:K] / ev.sum(),
        "per-component |r|": r_comp,
        "subspace overlap": overlap,
        "chance overlap": np.arange(1, K + 1) / X.shape[1],
    })
    stability
    return K, Z, ev, overlap, r_comp, refit_stability, robust_z, stability


@app.cell
def _(mo, np, r_comp, stability):
    _k95 = int(np.argmax(r_comp < 0.95)) if (r_comp < 0.95).any() else len(r_comp)
    _var = stability["cumulative variance"][max(_k95 - 1, 0)]
    mo.md(
        f"""
        These donors are draws from **one** generative model — there is no biology and no batch
        effect for a component to find. Yet the leading components carry
        **{stability["cumulative variance"][-1]:.0%}** of the variance, while
        **{_k95}** of them reach $|r| \\ge 0.95$ across a lab-disjoint refit and the subspace
        overlap sits near its chance level. Explained variance is not evidence of structure; it is
        a description of the matrix you happen to have.

        On the real 14,553-sample × 1,369-column emitted matrix (182 studies) the same measurement
        gives:

        | criterion | components |
        |---|---|
        | 90% cumulative variance | 394 |
        | Horn parallel analysis | 241 |
        | participation ratio (effective rank) | 144 |
        | per-component $|r| \\ge 0.95$, study-disjoint | **1** |

        Per-component correlation there was 0.949 for PC1, 0.614 for PC2 and 0.15–0.32 from PC3 on,
        while eigenvalues 2–12 sat at 73, 56, 51, 48, 44, 42, 38, 34, 32, 31, 30 — near-degenerate,
        so the axes swap between refits even where the subspace is stable. That is exactly what the
        overlap column above separates.

        **What to do with this**

        * Do not interpret an individual PC beyond the first as a named feature. It is a coordinate
          of the corpus that fitted it.
        * Select rank by out-of-group reproducibility, not explained variance — and prefer the
          subspace criterion, which does not fail on degeneracy alone.
        * Refit the rotation inside every cross-validation fold. Fitting once on the whole task
          lets the components see the test groups' covariance.
        * Null anything you chose by looking at the labels: a maximum over 64 components reached
          AUC 0.84 by chance on a 26-vs-7 contrast ($p = 0.20$).

        This is also why no corpus-fitted rotation ships in the artifact: the `phiv` / `phij` /
        `phic` bases come from the prototype cloud — zero samples, so no corpus to be unstable
        with respect to.
        """
    )
    return


if __name__ == "__main__":
    app.run()
