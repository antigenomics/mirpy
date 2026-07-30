# mirpy — exposure trajectory, generative loop, digital twin.
# Reactive marimo app demonstrating three new repertoire-level capabilities: a PhenoPath-style
# covariate-disentangled exposure trajectory (mir.track), a fitted generative density over
# repertoire descriptors with sample/evolve (mir.generate), and the digital-twin glue that lets
# one donor be perturbed or resampled (mir.twin). Self-contained on the bundled human_TRB
# prototypes (a synthetic cohort stands in for a real exposure cohort) — no downloads.
# Run with:  marimo edit examples/trajectory_and_twin.py
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from mir.embedding.prototypes import load_prototypes
    from mir.embedding.tcremp import TCREmp
    from mir.explain import ChannelBuilder
    from mir.generate import evolve, fit_descriptor_density
    from mir.repertoire import fit_repertoire_space, sample_descriptor, sample_embedding
    from mir.track import fit_exposure_trajectory
    from mir.twin import make_twins
    return (ChannelBuilder, TCREmp, evolve, fit_descriptor_density, fit_exposure_trajectory,
            fit_repertoire_space, load_prototypes, make_twins, np, pl, plt, sample_descriptor,
            sample_embedding)


@app.cell
def _(mo):
    mo.md(
        """
        # Exposure trajectory, generative loop, digital twin

        A repertoire cohort with a **known covariate** (here: an HLA-like binary group) and an
        **unknown severity/progression axis** — the setup `mir.track.fit_exposure_trajectory`
        targets (a PhenoPath-style model, Campbell & Yau 2018). Below: a synthetic cohort where
        both an expanding convergent family *and* a covariate-dependent diversity response are
        injected, so the fitted trajectory and its channel interactions have a known answer to
        recover.
        """
    )
    return


@app.cell
def _(load_prototypes, np, pl):
    # 30 synthetic samples: each a subsample of the bundled pool + an injected expanding family
    # whose size scales with a latent "severity" (the trajectory to recover). Half the cohort is
    # an "HLA+" group where the diversity channel additionally contracts with severity -- a
    # covariate x trajectory interaction fit_exposure_trajectory should surface.
    protos = load_prototypes("human", "TRB", n=3000)
    rng = np.random.default_rng(0)
    n_samples = 30
    hla = np.array([i % 2 == 0 for i in range(n_samples)])          # known covariate (numeric form)
    hla_label = ["HLA+" if h else "HLA-" for h in hla]                # same covariate, as plain labels
    severity_true = rng.uniform(0, 1, n_samples)                     # unknown trajectory (ground truth)

    samples = []
    for i in range(n_samples):
        base = protos.sample(200, seed=i)
        seed_row = protos.row(i, named=True)
        fam_n = int(5 + 40 * severity_true[i])                       # bigger family = more severe
        fam_count = int(10 + 200 * severity_true[i])
        counts = list(rng.integers(1, 5, base.height))
        if hla[i]:
            # HLA+ : diversity contracts faster with severity (fewer distinct clones at high severity)
            base = base.slice(0, max(20, int(base.height * (1 - 0.6 * severity_true[i]))))
            counts = counts[:base.height]
        df = base.with_columns(pl.Series("duplicate_count", counts))
        fam = pl.DataFrame({"junction_aa": [seed_row["junction_aa"]] * fam_n,
                           "v_call": [seed_row["v_call"]] * fam_n,
                           "j_call": [seed_row["j_call"]] * fam_n,
                           "duplicate_count": [fam_count] * fam_n})
        samples.append(pl.concat([df.select(["junction_aa", "v_call", "j_call", "duplicate_count"]),
                                  fam]))
    return hla, hla_label, samples, severity_true


@app.cell
def _(TCREmp, fit_repertoire_space, pl, samples):
    # one shared RepertoireSpace (the comparability contract), then per-sample Φ(S)
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=500)
    space = fit_repertoire_space(model, pl.concat(samples), n_rff=256, n_components=15, seed=0)
    return model, space


@app.cell
def _(ChannelBuilder, np, sample_embedding, samples, space):
    # a small, named channel matrix: diversity (Hill numbers) + the mean-block's norm (a scalar
    # proxy for "how far this sample's clonal composition sits from the origin"). fit_exposure_
    # trajectory works on ANY (n_samples, n_channels) matrix -- ChannelBuilder or stack_embeddings.
    embs = [sample_embedding(space, s, blocks=("mean", "diversity")) for s in samples]
    diversity = np.stack([e.diversity for e in embs])
    mean_norm = np.array([np.linalg.norm(e.mean) for e in embs])
    X, spec = (ChannelBuilder()
              .add("diversity", diversity)
              .add("mean_norm", mean_norm)
              .build())
    return X, embs, spec


@app.cell
def _(X, fit_exposure_trajectory, hla, mo, np, severity_true, spec):
    fit = fit_exposure_trajectory(X, hla.astype(float), channel_names=[
        f"{name}{j}" if len(spec.columns(name)) > 1 else name
        for name in spec.names for j in range(len(spec.columns(name)))
    ])
    _corr = float(np.corrcoef(fit.tau, severity_true)[0, 1])
    mo.md(
        f"""
        **Recovered trajectory vs. planted severity**: correlation = `{abs(_corr):.2f}`
        (sign is arbitrary — a trajectory's orientation, like a PC's, is fixed by convention
        alone).

        **Top covariate x trajectory interactions** (which channels' response to the trajectory
        differs by HLA group — the planted diversity-contraction effect should surface here):
        """
    )
    return (fit,)


@app.cell
def _(fit):
    fit.top_interactions(top=5)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Generative loop + digital twin

        Fit a density over each sample's `RepertoireDescriptor` (mass-preserving: infiltration /
        diversity / clonality / identity), conditioned on the same HLA covariate — then perturb
        one donor's descriptor ("what if this donor's repertoire were hotter") and draw brand-new
        synthetic donor states from the fitted generator.
        """
    )
    return


@app.cell
def _(fit_descriptor_density, hla_label, mo, samples, sample_descriptor, space):
    descriptors = [sample_descriptor(space, s) for s in samples]
    density = fit_descriptor_density(descriptors, labels=hla_label)
    d0 = descriptors[0]
    mo.md(f"donor 0 (HLA+): infiltration={d0.metrics()['infiltration']:.2f}, "
          f"diversity={d0.metrics()['diversity']:.1f}, clonality={d0.metrics()['clonality']:.3f}")
    return d0, density, descriptors


@app.cell
def _(d0, density, evolve, mo):
    hotter = evolve(density, d0, coordinate="infiltration", delta=2.0, condition="HLA+")  # d0 is donor 0 (HLA+)
    mo.md(
        f"""
        **`evolve`** (a +2.0 infiltration move, coupling propagated via the fitted covariance):
        diversity {d0.metrics()['diversity']:.1f} → {hotter.metrics()['diversity']:.1f},
        clonality {d0.metrics()['clonality']:.3f} → {hotter.metrics()['clonality']:.3f}.
        """
    )
    return


@app.cell
def _(density, descriptors, hla_label, make_twins, mo):
    # DonorTwin: descriptor + covariate in one perturbable/simulatable object
    twins = make_twins(descriptors, conditions=hla_label, donor_ids=[f"S{i}" for i in range(len(descriptors))])
    synthetic_peers = twins[0].simulate(density, n=5, seed=1)  # 5 new synthetic "donors like donor 0"
    mo.md(f"5 synthetic peers of donor {twins[0].donor_id} (condition={twins[0].condition}): "
          f"infiltration range [{min(m['infiltration'] for m in synthetic_peers):.2f}, "
          f"{max(m['infiltration'] for m in synthetic_peers):.2f}]")
    return


if __name__ == "__main__":
    app.run()
