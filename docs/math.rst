Mathematical foundations
========================

Every object mirpy computes, with its definition, its derivation, and the call that produces it.
The organising idea is one chain of quotients: a *sequence* becomes a *point*, a *sample* becomes a
*measure* on those points, and a measure becomes a *vector* through a characteristic kernel. Each
step is chosen so that a named property survives it, and each property licenses one class of
downstream operation. Where a step is stated as a proposition, the label in parentheses
(``prop:kme``, ``eq:rff``, …) is the label it carries in the theory appendix of the companion
manuscript, so a docstring, this page and the appendix all name the same result.

Numbers quoted as *measured* are empirical results from the benchmark programme, reproduced here
because they decide a design question; the scripts of record live in the companion analysis
repository.

.. contents:: On this page
   :local:
   :depth: 2


Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - symbol
     - meaning
   * - :math:`\sigma`
     - a clonotype: the triple (``junction_aa``, ``v_call``, ``j_call``)
   * - :math:`a_\sigma`
     - its clone size (``duplicate_count``); :math:`N = \sum_\sigma a_\sigma` the read total
   * - :math:`S`
     - a repertoire — the multiset :math:`\{(\sigma, a_\sigma)\}`; :math:`n = |S|` its richness
   * - :math:`\varphi(\sigma) = z_\sigma \in \mathbb{R}^p`
     - the clonotype embedding (:class:`~mir.embedding.tcremp.TCREmp`)
   * - :math:`g`
     - the clone-size transform; :math:`w_\sigma = g(a_\sigma)/\sum_\tau g(a_\tau)`
   * - :math:`\rho_S`
     - the sample measure :math:`\sum_\sigma w_\sigma \delta_{z_\sigma}`
   * - :math:`k, \psi`
     - the kernel and its random-Fourier feature map, :math:`k(z,z') \approx \psi(z)^\top\psi(z')`
   * - :math:`\Phi_1(S)`
     - the kernel mean embedding :math:`\sum_\sigma w_\sigma \psi(z_\sigma)`
   * - :math:`f_1, f_2, f_{3+}`
     - the number of clonotypes seen exactly once, twice, three or more times
   * - :math:`M_0`
     - the missing mass — total frequency of the clonotypes never drawn
   * - :math:`{}^qD`
     - the Hill number of order :math:`q`; :math:`n_{\text{eff}} = (\sum_\sigma w_\sigma^2)^{-1}`


The pipeline
------------

.. graphviz::

   digraph pipeline {
     rankdir=LR;
     bgcolor="transparent";
     node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#8899a6",
           fontname="Helvetica", fontsize=10];
     edge [color="#8899a6", fontname="Helvetica", fontsize=9];

     seq  [label="sequence\nσ = (junction, V, J)"];
     z    [label="point\nz = φ(σ) ∈ ℝᵖ", fillcolor="#e8f0fe"];
     rho  [label="measure\nρ_S = Σ w δ_z", fillcolor="#e8f0fe"];
     phi  [label="vector\nΦ(S) = mean ‖ div ‖ 2nd", fillcolor="#e8f0fe"];

     dens [label="clone-level\nenrichment E(z)"];
     read [label="cohort read-outs\nMMD · channels · survival"];
     gen  [label="generation\ndescriptor · diffusion · codec"];

     seq -> z   [label=" prototype distances "];
     z   -> rho [label=" clone-size weights "];
     rho -> phi [label=" characteristic kernel "];
     z   -> dens [label=" balloon density "];
     phi -> read;
     phi -> gen;

     {rank=same; dens; read; gen;}
   }

Two scales run in parallel and share one coordinate system: the **clone level**
(:mod:`mir.density`, where each clonotype is tested against a background) and the **sample level**
(:mod:`mir.repertoire`, where the whole repertoire is one vector). Both consume the same
:math:`z_\sigma`, which is what makes a clone-level hit and a sample-level channel comparable
statements about the same space.


From sequence to point
----------------------

The prototype embedding
~~~~~~~~~~~~~~~~~~~~~~~

Fix a panel of :math:`K` reference clonotypes :math:`\{\pi_1,\dots,\pi_K\}`. A query clonotype is
represented by its **distances to the panel**, in three separate geometries concatenated
(``eq:embedding``):

.. math::

   \varphi(\sigma) \;=\; \big(\, d_{\mathrm{junc}}(\sigma, \pi_k),\;
                              d_{V}(\sigma, \pi_k),\;
                              d_{J}(\sigma, \pi_k) \,\big)_{k=1}^{K} \;\in\; \mathbb{R}^{3K}.

The junction term is the gap-block alignment score of :mod:`seqtree` (BLOSUM62 Gram penalty, gap
placements :math:`(3,4,-4,-3)`); the V and J terms are baked germline region distances (CDR1/CDR2
included) looked up per allele with a cascade fallback. Because every coordinate is a distance,
:math:`\varphi` is **all-positive** — a fact with a practical consequence, see
:ref:`math-invariants`.

Why distances-to-landmarks is a legitimate coordinate system rather than a trick: for a metric
:math:`d`, the map :math:`\sigma \mapsto (d(\sigma,\pi_k))_k` is 1-Lipschitz in each coordinate
(``prop:lipschitz``), so

.. math::

   \frac{1}{\sqrt{K}}\,\|\varphi(\sigma) - \varphi(\tau)\|_2 \;\le\; d(\sigma,\tau)
   \quad\text{and}\quad
   |d(\sigma,\tau)| \;\ge\; \max_k \big| d(\sigma,\pi_k) - d(\tau,\pi_k) \big| ,

i.e. Euclidean distance in embedding space is a lower bound on, and empirically a close proxy for,
the alignment distance it replaces. The panel does not have to be biologically special — it only has
to *span*. Measured: the pairwise geometry induced by two independent prototype draws agrees at
:math:`R = 0.922` at :math:`K = 100`, :math:`0.990` at :math:`K = 500` and :math:`0.997` at
:math:`K = 2000`, and real-versus-model panels agree at :math:`R = 0.965`.

Squared versus root metric
~~~~~~~~~~~~~~~~~~~~~~~~~~

``metric="squared"`` (the default, and the published space) uses :math:`d`; ``metric="sqrt"`` uses
:math:`\rho = \sqrt{d}`. The latter is the metrically tidier choice — :math:`\sqrt{d}` satisfies the
triangle inequality for a wider class of :math:`d` and makes the space of negative type, hence
Hilbert-embeddable (``prop:schoenberg``) — and benchmarked as a wash, so the default stays put.

**Call.** :meth:`mir.embedding.tcremp.TCREmp.embed` → ``(n, 3K)``;
:func:`mir.distances.junction.junction_distance_matrix` for the junction block alone.


From point cloud to measure
---------------------------

The quotient that removes order *and* length
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A repertoire is an unordered set of variable size, so raw inputs live in

.. math::

   \bigsqcup_{n \ge 1} (\mathbb{R}^p)^n / \mathfrak{S}_n ,

on which no fixed-dimensional map is even defined. Passing to the **empirical measure**
(``def:sampmeasure``, ``eq:sampmeasure``)

.. math::

   \rho_S \;=\; \sum_{\sigma \in S} w_\sigma\, \delta_{z_\sigma},
   \qquad w_\sigma = \frac{g(a_\sigma)}{\sum_\tau g(a_\tau)},

is exactly the quotient that kills both nuisances at once: :math:`\rho_S` is a single point of
:math:`\mathcal{P}(\mathbb{R}^p)` whatever :math:`n` is. A sample embedding is order-invariant **iff
it factors through** :math:`\rho_S` (``prop:sampinv``), which is the whole design constraint.

Cardinality is not thrown away by the quotient — it re-enters explicitly as richness
:math:`{}^0D = n` in the diversity block, so the embedding *sees* how many clonotypes there are
without letting that count set its dimension.

The clone-size transform
~~~~~~~~~~~~~~~~~~~~~~~~

Clone sizes are Zipf-heavy. A concave :math:`g` tames the tail so no single hyperexpanded clone owns
the vector:

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - ``weight=``
     - :math:`g(a)`
     - character
   * - ``"log2p1"`` *(default)*
     - :math:`\log_2(1+a)`
     - concave; one clone at :math:`10^6` reads outweighs a singleton :math:`{\sim}20\times`, not :math:`10^6\times`
   * - ``"log1p"``
     - :math:`\ln(1+a)`
     - same shape, natural base
   * - ``"anscombe"``
     - :math:`\sqrt{a + 3/8}`
     - variance-stabilising for Poisson counts
   * - ``"duplicate_count"``
     - :math:`a`
     - linear; the frequency measure, tail-dominated
   * - ``"distinct"``
     - :math:`1`
     - presence only; :math:`n_{\text{eff}} = {}^0D`

Normalising to :math:`\sum_\sigma w_\sigma = 1` is what buys the **mixture property**: pooling two
samples gives :math:`\Phi(S \cup T) = (N_S\Phi_S + N_T\Phi_T)/(N_S+N_T)`, exactly linear
interpolation, so a point *between* two sample embeddings is the embedding of a real pooled
repertoire. That is what makes an axis navigable and :mod:`mir.twin` meaningful. The unnormalised sum
has no such property — its midpoint is the embedding of nothing. Measured, normalise-or-not is
exactly one scalar (:math:`\Phi_{\text{sum}} = N\,\Phi_{\text{mean}}`), so keeping the mean and
carrying :math:`\log N` as its own channel strictly dominates the plain sum.

**Call.** :meth:`mir.repertoire.RepertoireSpace.sample_cloud` → :math:`(Z, w)`.


The kernel mean embedding
-------------------------

An empirical characteristic function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take a shift-invariant kernel. By **Bochner's theorem** it is the Fourier transform of a finite
measure :math:`\Lambda`,

.. math::

   k(z,z') \;=\; \int e^{i\omega^\top (z-z')}\, d\Lambda(\omega),

and the random-Fourier map samples that integral at frequencies :math:`\omega_j \sim \Lambda`
(``eq:rff``):

.. math::

   \psi(z) \;=\; \sqrt{\tfrac{2}{D}}\,
   \big(\cos(\omega_j^\top z + b_j)\big)_{j=1}^{D},
   \qquad \omega_j \sim \mathcal{N}(0, \ell^{-2} I),\;\; b_j \sim \mathcal{U}[0,2\pi],

so that :math:`\mathbb{E}[\psi(z)^\top\psi(z')] = k(z,z')` (Rahimi–Recht). The mean feature is then
an empirical characteristic function evaluated at the sampled frequencies (``eq:ecf``):

.. math::

   \sum_\sigma w_\sigma e^{i\omega_j^\top z_\sigma}
   \;=\; \int e^{i\omega_j^\top z}\, d\rho_S(z) \;=\; \widehat{\rho_S}(\omega_j).

Why the characteristic function and not a moment-generating one: clone sizes are heavy-tailed, so an
:math:`e^{t^\top z}` feature would be dominated by the largest clone and need not converge, whereas
:math:`|e^{i\omega^\top z}| = 1` always exists and is bounded — and by Lévy uniqueness it is a
*complete* invariant, which is precisely the requirement that the kernel be **characteristic**.

The bandwidth :math:`\ell` is calibrated to the one-substitution embedding scale :math:`r_1`
(:func:`mir.density.calibrate_radius`), so the kernel resolves about one CDR3 mutation.

Depth robustness
~~~~~~~~~~~~~~~~

The kernel mean converges to the population mean map at the Monte-Carlo rate in the *effective*
sample size (``prop:kme``, ``eq:neff``):

.. math::

   \mathbb{E}\big\|\Phi_1(S) - \Phi_1(P)\big\|_{\mathcal{H}}
   \;\le\; \frac{C}{\sqrt{n_{\text{eff}}}},
   \qquad n_{\text{eff}} = \Big(\sum_\sigma w_\sigma^2\Big)^{-1},

with :math:`n_{\text{eff}}` itself a Hill number, :math:`{}^2D \le n_{\text{eff}} \le {}^0D`
(``prop:antag``). Measured slope of :math:`\log\|\Delta\Phi_1\|` against :math:`\log` depth:
:math:`-0.55`, against the predicted :math:`-0.5`. This is a *generic* KME rate rather than something
peculiar to this embedding — which is exactly why it can be relied on.

MMD, biased and unbiased
~~~~~~~~~~~~~~~~~~~~~~~~

Distance between repertoires is the maximum mean discrepancy (``eq:kme``):

.. math::

   \mathrm{MMD}(S,S') \;=\; \big\|\Phi_1(S) - \Phi_1(S')\big\| .

The plain (V-statistic) form carries a positive bias from the :math:`k(z,z)` diagonal of order
:math:`1/n_{\text{eff}}`, so a low-diversity sample has its distances **inflated by construction** —
and when diversity is itself the variable of interest, that bias masquerades as signal with the wrong
sign. Removing the diagonal analytically (``rem:unbiasedmmd``; Gretton et al. 2012), with
:math:`s = 1/n_{\text{eff}} = \sum_\sigma w_\sigma^2` and :math:`k(z,z)\approx 1`:

.. math::

   \widetilde{\|\mu\|^2} \;=\; \frac{\|\mu\|^2 - s}{1 - s},
   \qquad
   \widetilde{\mathrm{MMD}}^2(S,S') \;=\;
   \widetilde{\|\mu_S\|^2} + \widetilde{\|\mu_{S'}\|^2} - 2\,\mu_S^\top\mu_{S'} .

Undefined for a point mass (:math:`n_{\text{eff}} \le 1`), which the implementation refuses rather
than silently returns. Measured consequence of ignoring this: an apparent age-divergence signal
carried the sign of *clonal expansion* rather than richness until the diagonal was removed.

**Call.** :func:`mir.repertoire.sample_embedding`, :func:`mir.repertoire.mmd_distance`,
:func:`mir.repertoire.mmd_matrix`; ``unbiased=True`` for the diagonal-removed form.


Diversity: abundance and function
---------------------------------

Hill numbers and what they cannot see
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :math:`p_i = a_i/\sum_j a_j`,

.. math::

   {}^qD \;=\; \Big(\sum_i p_i^{\,q}\Big)^{1/(1-q)},
   \qquad
   {}^0D = n, \quad
   {}^1D = e^{-\sum_i p_i \log p_i}, \quad
   {}^2D = \Big(\sum_i p_i^2\Big)^{-1}.

Every Hill number is a functional of the clone-size distribution **alone**, hence invariant to
permuting which receptor carries which abundance: permute the sequence labels and every diversity
statistic is unchanged while the biology is destroyed. Diversity describes the *shape* of the
abundance distribution and is blind to composition. Coverage standardisation to a common
Good–Turing coverage :math:`\hat{C}^\star` (``prop:coverage``, via :mod:`vdjtools.stats.inext`)
removes the depth dependence of the estimate itself.

Rao's quadratic entropy is the norm of the kernel mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rao's quadratic entropy uses the *dissimilarity between receptors*, so it occupies exactly the blind
spot above. With the kernel dissimilarity :math:`d(z,z') = 1 - k(z,z')`:

.. math::

   Q(S) \;=\; \sum_{\sigma,\tau} w_\sigma w_\tau \big(1 - k(z_\sigma, z_\tau)\big)
         \;=\; \underbrace{\sum_{\sigma,\tau} w_\sigma w_\tau}_{=\,1}
             - \sum_{\sigma,\tau} w_\sigma w_\tau\, \psi(z_\sigma)^\top \psi(z_\tau)
         \;=\; 1 - \|\Phi_1(S)\|^2 .

So the **norm of the kernel mean is a functional diversity**, computable without ever forming a Gram
matrix. Verified against an explicit Gram to machine precision (max relative error :math:`\sim
10^{-16}`). Measured: this one scalar recovers :math:`R^2` 0.74–0.85 of classical diversity, while
embedding derivatives reach :math:`R^2` 0.974–0.994 for Shannon and 0.985–0.9999 for richness — which
is the sense in which the embedding *carries* the classical statistics rather than competing with
them.

Two caveats, both consequences of the construction rather than defects. The kernel is a random-Fourier
approximation, so :math:`k(z,z) = 1` holds only to :math:`O(D^{-1/2})` and a single-clone sample reads
:math:`Q \sim 10^{-2}` rather than exactly 0. And :math:`Q` is defined by a **norm**, so it is valid
only for the true, uncentred :math:`\Phi_1`: centring or PCA-projecting preserves *differences* (MMD
survives) but not norms.

The second-moment block
~~~~~~~~~~~~~~~~~~~~~~~

Co-occurrence structure — which clonotypes appear *together*, the shape an HLA imprint takes — is a
second-order property invisible to any mean (``prop:interact``):

.. math::

   \Sigma_S \;=\; \sum_\sigma w_\sigma\, \psi_2(z_\sigma)\psi_2(z_\sigma)^\top ,

a codebook-free Fisher vector on a small feature map :math:`\psi_2` (dimension :math:`D_2`), stored as
its :math:`D_2(D_2+1)/2` upper triangle. The opt-in ``n_eigs=r`` alternative keeps the top :math:`r`
eigenvalues instead — compact and rotation-invariant, and **measured lossy for exactly the signal the
block exists for**: an HLA imprint lives in *which* clones co-occur (a directional fact), so the
rotation-invariant spectrum reached :math:`\le 0.55` AUC against 0.593 for the full triangle. The
triangle stays the default.

**Call.** ``sample_embedding(..., blocks=("mean","diversity","second"))``;
:func:`mir.repertoire.rao_q`.


The deficient measure
---------------------

The premise that fails
~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi_1` is the kernel mean of a *probability* measure: the weights sum to 1, so every sample
asserts one full unit of confidence about its repertoire. At RNA-seq depth this is false. Measured
unique-clonotype counts put the median tissue TRB sample at **21** clonotypes (blood TRB 254, 1st
percentile 1), so :math:`w_\sigma = a_\sigma/N` is :math:`1/n` for a *technical draw size*, not a
clonal frequency — true frequencies live at :math:`10^{-5}` to :math:`10^{-8}`, and one singleton's
weight was measured to span a factor of **21,454** across blood TRB purely from sample size. What
:math:`\Phi_1` actually is at that depth: the centroid of a captured point set, an unbiased estimator
of :math:`\mathbb{E}[z]` under the capture process with **variance** :math:`\propto 1/n`.

Good–Turing decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

Let the true repertoire have :math:`S_{\text{true}}` clonotypes at frequencies :math:`p_\sigma`. The
target is :math:`\Phi_{\text{true}} = \sum_\sigma p_\sigma z_\sigma`; split the sum at what was seen:

.. math::
   :label: eq-gt

   \Phi_{\text{true}} \;=\; (1 - M_0)
     \underbrace{\sum_{\text{seen}} \frac{p_\sigma}{\sum_{\text{seen}} p}\, z_\sigma}_{\Phi_{\text{seen}}}
     \;+\; M_0\, \mathbb{E}[z \mid \text{unseen}],
   \qquad M_0 = \sum_{\text{unseen}} p_\sigma .

Every weighting scheme in play is this one identity with a different estimator of :math:`M_0` — that
is the unification.

**Turing.** :math:`\hat M_0 = f_1/N`: the classical read-weighted answer.

**Chao, from a boundary assumption.** Chao1 estimates the *count* of missing clonotypes and says
nothing directly about their mass. The bridge is one modelling assumption: *an unseen clonotype is
rare, so had it been drawn it would carry at most one read, and unseen clonotypes do not overlap.*
That pins each unseen clone's frequency at the detection boundary :math:`1/(N + S_u)`, so

.. math::

   \hat M_0 \;=\; S_u \cdot \frac{1}{N + S_u} \;=\; \frac{S_u}{N + S_u},
   \qquad S_u \;=\; \frac{f_1(f_1-1)}{2(f_2+1)} ,

and substituting into :eq:`eq-gt` collapses the observed term to

.. math::

   (1 - \hat M_0)\cdot \frac{a_\sigma}{N}
   \;=\; \frac{N}{N + S_u}\cdot\frac{a_\sigma}{N}
   \;=\; \frac{a_\sigma}{N + S_u} .

**This is also why the units work.** :math:`N` counts reads and :math:`S_u` counts clonotypes; adding
them would normally be a category error, and is legitimate *only* because the one-read assumption
makes each unseen clone contribute exactly one read, so :math:`N + S_u` is the read total of the
completed table. The formula is a pseudo-count completion: sample slightly deeper, catch every
missing clone exactly once.

Use the **bias-corrected** :math:`S_u = f_1(f_1-1)/(2(f_2+1))`, never the classical
:math:`f_1^2/2f_2`: at these depths a sample with no clonotype seen exactly twice is common, and the
classical form is undefined there. The two estimators rank samples nearly identically (measured
:math:`r` = 0.93 and 0.76 in two views) but differ in level — Chao runs above Turing when singletons
dominate doubletons and below it when doubletons abound. Both limits check out: a deep sample gives
:math:`\hat M_0 = 0.0002` and reduces to the ordinary embedding, an all-singleton sample gives 0.96,
the honest statement that it measured almost nothing.

Sub-probability, not negative probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Could the missing mass be a *negative* probability? No, and it need not be. :math:`\Phi`'s value is
that :math:`\|\Phi_P - \Phi_Q\|` **is** the MMD — a metric between distributions — and that a convex
combination of two :math:`\Phi`'s is the :math:`\Phi` of a real pooled repertoire. A measure allowed
to go negative on a set is not a probability measure: MMD stops being a metric on it and the midpoint
of two embeddings stops being the embedding of anything. A **sub-probability** measure costs neither:

.. math::

   \mu_S \;=\; \sum_{\text{seen}} w_\sigma\,\delta_{z_\sigma},
   \qquad \text{total mass } = 1 - M_0 \;\le\; 1 .

Renormalising *forces* a 5-clonotype tumour to assert a full unit of confidence; a deficient measure
lets it say *I hold 0.08 of a repertoire's worth of evidence*.

Where negativity legitimately lives is a **difference of two probability measures**:

.. math::

   \Psi_S \;=\; (1 - M_0)\big(\Phi_{\text{seen}} - \Phi_{\text{naive}}\big)
          \;=\; \Phi_{\text{true}} - \Phi_{\text{naive}} ,

whose coordinates go negative wherever the sample is **depleted** relative to unselected V(D)J
output. It is an ordinary RKHS element and :math:`\|\Psi_S\|` is exactly
:math:`\mathrm{MMD}(S, \text{naive})` — signed structure with the metric intact. The second equality
is immediate from :eq:`eq-gt` and is the reason the two constructions are one object:
:math:`\Psi` *is* the reference-centred sub-probability embedding.

.. graphviz::

   digraph deficient {
     rankdir=LR;
     bgcolor="transparent";
     node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#8899a6",
           fontname="Helvetica", fontsize=10];
     edge [color="#8899a6", fontname="Helvetica", fontsize=9];

     counts [label="counts a_σ"];
     m0     [label="M₀  (Turing | Chao)", fillcolor="#fdf0e3"];
     seen   [label="Φ_seen\nkernel mean of what was drawn", fillcolor="#e8f0fe"];
     naive  [label="Φ_naive\nkernel mean of germline draws", fillcolor="#e8f0fe"];
     psi    [label="Ψ = (1−M₀)(Φ_seen − Φ_naive)\nmagnitude = confidence × deviation",
             fillcolor="#e6f4ea"];
     origin [label="immune desert / vague sample\nM₀ → 1  ⇒  Ψ → 0", shape=note,
             fillcolor="#ffffff"];

     counts -> m0    [label=" f₁, f₂, N "];
     counts -> seen  [label=" w = g(a)/Σg "];
     m0     -> psi;
     seen   -> psi;
     naive  -> psi;
     psi    -> origin [style=dashed, arrowhead=none];
   }

Where the unseen lives
~~~~~~~~~~~~~~~~~~~~~~

The unseen block only changes the *direction* of :math:`\Phi` if it is actually filled with a vector,
and the choice decides whether the object is a shrinkage estimator toward a **meaningful** point.
Three candidates, measured:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - prior for the unseen
     - verdict
   * - the sample's own singletons
     - begs the question — no new information enters
   * - the corpus mean
     - James–Stein toward the centroid, and it measurably **hurt**: shallow samples pile into a dense
       ball that is itself depth-correlated
   * - **a germline draw**
     - the one that works

Filling the unseen block with :math:`\sim 20{,}000` naive V(D)J recombinations
(:func:`mir.repertoire.naive_reference`, ~8 s) took :math:`R^2(\mathrm{PC}_1, \text{depth})` from
0.259 to **0.001** in one view and 0.067 to **0.006** in another, with kNN label entropy unchanged or
better. Checked against the degenerate explanation: the reference vector is finite, PC1's *explained
variance* was unchanged (0.309 → 0.334) so PC1 is a genuinely different direction rather than a
collapsed one, and depth left the whole leading block (best of PC1–5: 0.253 → **0.047**).

Mechanistically :math:`\Phi = (1-M_0)\Phi_{\text{seen}} + M_0 z_{\text{naive}}` *is* shrinkage — a
shallow sample has large :math:`M_0` and collapses toward a fixed point, a deep one keeps its own
:math:`\Phi`. The difference from the failed James–Stein arm is entirely the **target**: shrinking an
unmeasurable sample toward the corpus centroid creates a depth-correlated ball, while shrinking it
toward the germline says the honest thing — *a repertoire we did not measure looks like unselected
recombination output* — and lands it at a biologically meaningful location.

**Call.** ``sample_embedding(..., missing_mass="turing"|"chao")`` sets
:attr:`~mir.repertoire.SampleEmbedding.mass`; :func:`mir.repertoire.missing_mass`,
:func:`mir.repertoire.naive_reference`, :func:`mir.repertoire.contrast_embedding`.


Depth as variance, not bias
---------------------------

The damage depth does is **not** bias — it is variance :math:`\propto 1/n` over an :math:`n` spanning
four orders of magnitude, and in a neighbour graph that heteroscedasticity *is* a depth axis. A
21-clonotype sample's :math:`\Phi` is :math:`\sim 16\times` noisier than a 5,780-clonotype one's.
Decompose the observed spread by regressing each sample's squared distance from the cohort centroid on
:math:`1/n`:

.. math::

   \mathbb{E}\big\|\Phi_S - \bar\Phi\big\|^2 \;\approx\; \tau^2 + \frac{\sigma^2}{n},
   \qquad \boxed{\;\kappa = \sigma^2/\tau^2\;}

with :math:`\tau^2` the between-sample (biological) variance and :math:`\sigma^2` the within-sample
sampling variance, so :math:`\kappa` is the size at which the two are **equal** — below it a sample's
:math:`\Phi` is more noise than signal. Measured :math:`\kappa \approx 40`–70 clonotypes across four
independent views, with 23–69% of samples below it.

This is an estimate, not a taste: report :math:`\kappa` for the cohort in front of you instead of
importing a cutoff. The library deliberately applies **no floor** — in blood a low clonotype count is
shallow sequencing, in a tumour it is low infiltration, i.e. the phenotype of interest, and a floor
applied in tumour deletes the stratum the analysis exists to predict. Carry :math:`\kappa` and
:attr:`~mir.repertoire.SampleEmbedding.mass` and weight instead of filtering.

**Call.** :func:`mir.repertoire.depth_threshold`, :func:`mir.repertoire.sample_statistics`,
:func:`mir.cohort.depth_report`.


Mixture algebra
---------------

Compartments and the dilution factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi_1` is linear in the clone-weight measure, so for any partition into sub-populations
:math:`c` with weight shares :math:`\pi_c`,

.. math::

   \rho_S = \sum_c \pi_c \rho_c
   \quad\Longrightarrow\quad
   \Phi_1(S) = \sum_c \pi_c\, \Phi_1(c),
   \qquad \pi_c = \frac{\sum_{\sigma \in c} g(a_\sigma)}{\sum_\tau g(a_\tau)} ,

verified numerically to :math:`\sim 10^{-17}`. Two consequences.

*Averaging is the wrong operation for a minority signal.* Write the repertoire as naive plus
expanded, :math:`\rho_S = (1-\pi)\rho_N + \pi\rho_E`. A disease effect confined to the expanded
compartment, :math:`\Delta = \Phi_1(E_{\text{case}}) - \Phi_1(E_{\text{ctrl}})`, reaches
:math:`\Phi_1` attenuated to :math:`\pi\Delta`, while the *noise* is supplied by the naive
compartment that owns most of the clonotypes. So the whole-repertoire average has a
signal-to-noise ratio worse by roughly :math:`\pi` — and :math:`\pi` is small precisely in the
samples that matter. This is a statement about the estimator, not the data.

*The shares are recoverable.* Since the identity is exact, a non-negative least squares of the whole
on its compartments is well-posed rather than a heuristic fit:

.. math::

   \hat\pi \;=\; \arg\min_{\pi \ge 0} \Big\| \Phi_1(S) - \sum_c \pi_c \Phi_1(c) \Big\|^2 .

Measured on IGH isotypes: the class-switched IgG compartment — affinity-matured, T-dependent, the
fraction most likely to carry antigen-specific signal — carries a median :math:`\pi` of **0.070**,
against unswitched IgM 0.230, IgA 0.176 and 0.520 unaccounted for (the uncalled-isotype share, in
ballpark agreement with the :math:`\sim 0.43` counted from reads — different denominators, embedding
*weight* versus *reads*). That number is a **power calculation**: a subset carrying :math:`\pi \approx
0.001` cannot be detected by any aggregate distance on :math:`\Phi`, and the per-clonotype witness is
the sensitive route.

.. graphviz::

   digraph bands {
     rankdir=LR;
     bgcolor="transparent";
     node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#8899a6",
           fontname="Helvetica", fontsize=10];
     edge [color="#8899a6", fontname="Helvetica", fontsize=9];

     S [label="repertoire S", fillcolor="#e8f0fe"];
     a [label="singleton\ncount = 1"];
     b [label="expanded\ncount ≥ 2"];
     c [label="top\ntop 1%, clipped"];
     m [label="igm  IGHM/IGHD"];
     g [label="igg  IGHG1–4"];
     i [label="iga  IGHA1–2"];
     nnls [label="NNLS on Φ₁(S)\nπ per compartment", shape=box, fillcolor="#e6f4ea"];

     S -> a; S -> b; S -> c;
     S -> m [style=dashed]; S -> g [style=dashed]; S -> i [style=dashed];
     a -> nnls; b -> nnls; c -> nnls; m -> nnls; g -> nnls; i -> nnls;

     label="dashed: IGH only, via c_call; rows with a null c_call belong to no isotype band";
     labelloc="b"; fontname="Helvetica"; fontsize=9; fontcolor="#667788";
   }

What the decomposition measured, in the direction the theory predicts: the ``singleton`` band
reproduces a clinical-covariates-only score almost exactly (0.600 vs 0.599) — pure dilution — while
``expanded`` alone reproduces the *whole-repertoire* score (0.634 vs 0.634), so the entire prognostic
content of :math:`\Phi_1` lives in the expanded compartment. What it did **not** do is lift that
content above what the clone-size distribution already encodes: on survival endpoints, zero of 22
block × endpoint cells cleared a pre-registered bar against a diversity reference, and the isotype cut
failed identically. Where banding *does* win is where abundance classes are different biology —
unmutated IgM singletons against class-switched expansions — with a measured kNN entropy of 0.3875
against 0.4686 for the pooled arm in tissue IGH.

Rarefaction, and the exact Rao gap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because :math:`\Phi` is linear in the measure, the mean of :math:`\Phi` over :math:`R` independent
multinomial subsamples is itself a kernel mean embedding — of the mixture distribution over
subsamples — so MMD, Rao and mixture linearity all survive (verified to :math:`\sim 10^{-15}`). This
is the **only** depth correction that preserves the kernel-mean semantics exactly: an orthogonal
projection breaks the norm identity, and a per-coordinate location-scale rescale breaks both.

The excess diversity of the average is exactly the replicate variance. With
:math:`\bar\Phi = R^{-1}\sum_r \Phi_r` and :math:`v_{\text{rep}} = R^{-1}\sum_r \|\Phi_r -
\bar\Phi\|^2`, the parallel-axis identity :math:`R^{-1}\sum_r\|\Phi_r\|^2 = \|\bar\Phi\|^2 +
v_{\text{rep}}` gives

.. math::

   Q(\bar\Phi) \;=\; 1 - \|\bar\Phi\|^2
   \;=\; 1 - \Big(R^{-1}\!\sum_r \|\Phi_r\|^2 - v_{\text{rep}}\Big)
   \;=\; \frac{1}{R}\sum_r Q(\Phi_r) \;+\; v_{\text{rep}} .

So :math:`v_{\text{rep}}` is a free, per-sample estimate of exactly the sampling noise that
:math:`\kappa` measures cohort-wide — and the identity is a warning as well as a tool: averaging
commutes with :math:`\Phi` but **not** with nonlinear functionals of it, so
:math:`\mathrm{mean}(Q) \ne Q(\mathrm{mean})` and the gap must be reported rather than assumed away.
Measured gaps ran 0.02% to 0.93%, monotone in target depth.

Rarefaction is not a default: rarefying a cohort to the depth of its shallowest useful samples
discards the deep samples' entire advantage. Reach for it when two groups must be compared *at matched
depth* and the comparison has to stay an MMD.

**Call.** :func:`mir.repertoire.band_frames`, :func:`mir.repertoire.band_embeddings`,
:func:`mir.repertoire.mixture_weights`, :func:`mir.repertoire.rarefy_embedding`.


Clone-level density: continuous TCRNET / ALICE
----------------------------------------------

Neighbour enrichment without a graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Antigen-driven convergence shows up as local over-density of receptors. Classical methods build an
exact- or 1-mismatch graph; the continuous form estimates the same ratio directly in embedding space
(``sec:dens-balloon``, ``prop:balloon``). With a **balloon** (adaptive-radius) estimator, radius
:math:`r(z)` chosen so the background ball holds a target count,

.. math::

   E(z) \;=\; \frac{\hat f_{\text{obs}}(z)}{\hat f_{\text{gen}}(z)}
         \;=\; \frac{n_{\text{obs}}(z)\,/\,N_{\text{obs}}}{n_{\text{gen}}(z)\,/\,N_{\text{gen}}} ,

which is a density-ratio estimate (``sec:dens-dre``, ``lem:lsif``) and is volume-free: the ball
volume cancels between numerator and denominator, so no bandwidth normalisation is needed and the
estimator is usable in :math:`p` where a kernel density estimate is not.

Significance per clonotype
~~~~~~~~~~~~~~~~~~~~~~~~~~

Two nulls, matching the two background types (``prop:poissontest``, ``prop:ctest``):

.. math::

   n_{\text{obs}}(z) \sim \mathrm{Poisson}\big(\lambda(z)\big),
   \quad \lambda(z) = n_{\text{gen}}(z)\cdot \tfrac{N_{\text{obs}}}{N_{\text{gen}}}
   \qquad\text{(generative background: ALICE)}

.. math::

   n_{\text{obs}}(z) \sim \mathrm{Binomial}\big(n_{\text{obs}}+n_{\text{gen}},\; p_0\big),
   \quad p_0 = \tfrac{N_{\text{obs}}}{N_{\text{obs}}+N_{\text{gen}}}
   \qquad\text{(control background: TCRNET)}

with Benjamini–Hochberg control across clonotypes, plus a **water-level** calibration
(``eq:waterlevel``) that raises the null level until the empirical false-discovery profile is
consistent — necessary because in the naive regime the background is itself pervasively convergent.

The empirical lesson matters more than the test: real repertoires are so convergent that a
:math:`P_{\text{gen}}` background flags :math:`\sim 40\%` of clones. Use a **biological control**
(day 15 vs day 0, allele-positive vs negative, seropositive vs control) for specificity, and process
the *full* repertoire — subsampling dilutes exactly the sparse antigen clusters being looked for. Made
quantitative on a spiked benchmark: admixed noise clones were over-flagged 43% under
:math:`P_{\text{gen}}` against 1% under a control background, a 46× signal-to-noise lift.

Abundance-aware enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Counting *distinct* neighbours ignores that an expanded clone is stronger evidence than a singleton
(``sec:dens-abund``, ``prop:abund``). Replace the in-ball count with the variance-stabilised mass
:math:`S(z) = \sum_{j \in B(z)} g(a_j)`; under a compound-Poisson model the tail is Gamma with
dispersion

.. math::

   \varphi \;=\; \frac{\mathbb{E}[g(A)^2]}{\mathbb{E}[g(A)]} ,

and a per-clonotype orphan/depth channel :math:`P(A \ge a_j)` is Fisher-combined with the breadth
term, so a clone can be called on being *large where it should be small* as well as on having many
neighbours. ``abundance=None`` recovers the distinct count exactly (:math:`g \equiv 1`).

**Call.** :func:`mir.density.fit_density_space`, :func:`mir.density.neighbor_enrichment`,
:func:`mir.density.enriched_mask`, :func:`mir.density.denoise_and_cluster`,
:func:`mir.density.generate_background`.


Cohort-level modelling
----------------------

Named channels and ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A channel is a named group of columns of the sample matrix. Given a caller-supplied scorer
:math:`f` (higher is better) and a reference score :math:`b`:

.. math::

   \delta_{\text{in}}(c) = f\big(X_{:,c}\big) - b,
   \qquad
   \delta_{\text{out}}(c) = f\big(X\big) - f\big(X_{:,\neg c}\big) .

:math:`\delta_{\text{in}}` is *marginal* — inflated by correlation between channels, so two redundant
channels both look important. :math:`\delta_{\text{out}}` is *conditional* and deflated by the same
correlation. Reported together they separate the claims: high in and high out means irreplaceable,
high in with :math:`\delta_{\text{out}} \approx 0` is the **redundancy** signature. Significance,
when asked for, is a row permutation of the block — the scorer holds :math:`y` in row order, so
shuffling the block's rows breaks the association exactly as shuffling :math:`y` would, and it is the
only scorer-agnostic null available.

Leave-one-*in* is the default because the scorers in practice reduce their input (an in-fold PCA), so
dropping one narrow channel lets the reduction re-mix the rest and reconstruct nearly the same
components — leave-one-out is structurally near-blind to exactly the 1-column channels that often win.

The witness
~~~~~~~~~~~

The MMD witness function turns a group difference back into sequences (``prop:witness``,
``eq:witness``). With :math:`w = \mu_{\text{pos}} - \mu_{\text{neg}}` in feature space, a clonotype
scores

.. math::

   s(\sigma) \;=\; \big\langle \mu_{\text{pos}} - \mu_{\text{neg}},\; \psi(\varphi(\sigma)) \big\rangle ,

so the top-scoring candidates are the discriminative public clones — the supervised way to find
motifs that the unsupervised bulk mean cannot surface, because there it is swamped by the naive
background. Only a channel with a clonotype pre-image (a kernel mean) can be attributed this way; a
Hill number has no pre-image, and asking which clones drive it is a category error.

Recoverability, not competition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scoring rule for an embedding is not "does :math:`\Phi` beat the one-number marker" — that is a
competition between an object and one of its own coordinates. It is whether each basic statistic is
**carried inside** it: a grouped-CV ridge from the embedding's PCs back to richness, Shannon,
top-clone fraction, singleton fraction, unseen fraction and library size, reported as :math:`R^2`.
Renormalising to mass 1 **deletes the magnitude**, so coverage- and richness-like statistics are
unrecoverable from :math:`\Phi` by construction — which is why the deficient measure should win this
question as a design consequence rather than an empirical accident. Report it beside the *increment*:
the endpoint score for the embedding, for the statistic alone, and for both together.

Batch offsets and why the obvious correction backfires
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A first-order batch effect is an additive offset, so subtracting each batch's mean removes it
(``prop:batch``) — in-sample. Out-of-sample the estimation error dominates. With :math:`\hat\mu_b`
fitted from :math:`n_b` samples in :math:`d` dimensions,

.. math::

   \mathbb{E}\|\hat\mu_b - \mu_b\|^2 \;=\; \frac{\sigma^2 d}{n_b} ,

which for :math:`n_b \approx 8`–15 and :math:`d \approx 1{,}280` is a vector of norm :math:`\approx
16` against a true offset of norm 7–24: **subtracting it injects a batch-constant vector as large as
the one it removes.** Measured, evaluated leave-one-batch-out plus a donor-level split inside each
batch, batch-identity AUC went 0.863 raw → **0.985** after per-batch centring and 0.978 after ComBat.
The positive-part James–Stein estimator shrinks the offset by its own reliability,

.. math::

   \hat\mu_b^{\text{JS}} \;=\; c_b\,\hat\mu_b,
   \qquad
   c_b \;=\; \Big(1 - \frac{(d-2)\,\hat\sigma^2/n_b}{\|\hat\mu_b\|^2}\Big)_+ ,

so a batch whose apparent offset is no bigger than its estimation error is left alone. Measured, it
recovered most of the damage: 0.985 → **0.889**. The cluster-aware alternative removes the offset
*per soft cluster* with a batch-diversity penalty (Harmony-style), which corrects a batch confounded
with biology instead of erasing that biology, and reduces exactly to plain residualisation at one
cluster.

Three conditions any correction must meet, learned the hard way: evaluate **out-of-sample** (in-sample
the injected error is invisible by construction); report the **geometry cost** (within-batch distance
correlation, idempotence, symmetry) and not only the batch metric; and check **mixture additivity**,
since the band and isotype decompositions depend on it.

**Call.** :class:`mir.explain.ChannelBuilder`, :func:`mir.explain.channel_report`,
:func:`mir.explain.channel_drivers`, :func:`mir.repertoire.class_witness`,
:func:`mir.bench.recovery_report`, :func:`mir.cohort.residualize`,
:func:`mir.repertoire.correct_batch`.


Trajectory, descriptor, generation
----------------------------------

Covariate-disentangled trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cohort often has a **known covariate** (allele, batch, arm) and an **unknown** progression axis
(days since exposure, severity). The PhenoPath-style model (``eq:phenopath``,
``prop:trajectory-fit``) infers the latent axis while separating out which channels respond to it
differently by covariate:

.. math::

   Y_{ng} \;=\; c_g \;+\; \alpha_g^\top x_n \;+\; \big(\kappa_g + \gamma_g^\top x_n\big)\,\tau_n
              \;+\; \varepsilon_{ng} ,

with :math:`\tau_n` the trajectory position of sample :math:`n`, :math:`\kappa_g` a channel's
baseline response and :math:`\gamma_g` the **covariate × trajectory interaction** — the term that
answers "does this channel move differently in carriers". Fitted by alternating closed-form
per-channel ridge regressions and a GLS update for :math:`\tau`,

.. math::

   \tau_n \;=\; \frac{\sum_g \beta_{ng}\,\big(Y_{ng} - c_g - \alpha_g^\top x_n\big)/s_g^2}
                     {\sum_g \beta_{ng}^2/s_g^2},
   \qquad \beta_{ng} = \kappa_g + \gamma_g^\top x_n ,

with ARD-style iteratively-reweighted shrinkage on :math:`\gamma` (``rem:phenopath-approx``; a
simplified closed-form approximation to PhenoPath's CAVI inference, not a reimplementation of it).

The derivable descriptor
~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi` normalises the total mass away, which is exactly the infiltration signal in tissue. The
descriptor keeps it as a coordinate (``rem:descriptor``):

.. math::

   \mathrm{desc}(S) \;=\; \Big(\underbrace{\log\textstyle\sum_\sigma a_\sigma}_{\text{infiltration}},\;
   \underbrace{-\log \textstyle\sum_\sigma w_\sigma^2}_{\log n_{\text{eff}}},\;
   \underbrace{\textstyle\sum_\sigma w_\sigma^2}_{\text{clonality}},\;
   \underbrace{\Phi_1(S)}_{\text{identity}}\Big),

every coordinate smooth (no integer richness), every named metric readable back off the vector
analytically. That is what makes it **simulatable**: fit a density over the cohort's descriptors and
the coordinate distribution is a generative manifold.

In-silico evolution
~~~~~~~~~~~~~~~~~~~

Fit a (optionally class-conditional) Gaussian with Ledoit–Wolf shrinkage,
:math:`\mathcal{N}(m, \Sigma)`, over descriptor vectors. Perturbing one coordinate and letting the
rest follow is the Gaussian conditional mean (``prop:evolve``): splitting into the perturbed
coordinate :math:`1` and the rest :math:`2`,

.. math::

   \mathbb{E}\big[x_2 \mid x_1 = m_1 + \delta\big]
   \;=\; m_2 + \Sigma_{21}\Sigma_{11}^{-1}\,\delta ,

so "make this donor hotter" propagates through every coupled coordinate with the cohort's own
covariance. Measured couplings match immunobiology: hotter ⇒ diversity :math:`+0.84`, class-switch
:math:`+0.52`, T-versus-B :math:`-0.63`. The non-linear alternative is a compact conditional
DDPM/DDIM over the same descriptor space with classifier-free guidance, sharing the
``sample(n, condition=…)`` call shape so either generator drops in unchanged.

**Call.** :func:`mir.track.fit_exposure_trajectory`, :func:`mir.repertoire.sample_descriptor`,
:class:`mir.generate.DescriptorDensity`, :class:`mir.twin.DonorTwin`, :mod:`mir.ml.diffusion`.


Sequence reconstruction
-----------------------

The junction embedding is compressible: PCA retaining 95% of the variance is enough for *geometry*
(clustering, MMD), while *reconstruction* needs 99% — the chain-adaptive lesson. A codec learns
:math:`\text{seq} \to \text{code} \to \text{seq}` with a geometry anchor, so the code stays close to
the true embedding rather than drifting to whatever is easiest to decode:

.. math::

   \mathcal{L} \;=\; \underbrace{\mathcal{L}_{\text{recon}}}_{\text{token cross-entropy}}
   \;+\; \lambda_{\text{embed}}\,\underbrace{\big\|\,\text{code} - \text{PCA}(\varphi(\sigma))\,\big\|^2}
   _{\text{distances preserved}} .

Reconstruction is **training-data-limited, not architecture-limited**: the same one-shot decoder goes
from exact-match 0.885 at :math:`n = 20`k to 0.941 at 50k to 0.958 at 100k. Levers in order: more
data (free) > more PCs (to ~99% variance) > :math:`K \to \sim 2000` (it saturates; :math:`K = 10{,}000`
*regresses*) > an autoregressive decoder for the last percent.

Note the direction of the information flow. The distance-to-prototypes code is an **expansion**
(:math:`\sim 10` kbit against a :math:`\sim 63`-bit sequence), so for archival losslessness one stores
the string; the codec inverse exists for ML and generation — decoding a *synthetic* point that no
observed clonotype occupies.

Embeddings are comparable only if the prototype panel **and** the PCA rotation match, which is why a
codec ships as a bundle carrying both plus a prototype hash, and refuses to load against a different
panel.

**Call.** :mod:`mir.ml.codec`, :mod:`mir.ml.bundle`, :func:`mir.bench.theory.codec_losslessness`.


.. _math-invariants:

Which transformation preserves what
-----------------------------------

Every property above licenses a class of operation, and every transformation applied to :math:`\Phi`
must state which properties it keeps. This table is the contract.

.. list-table::
   :header-rows: 1
   :widths: 30 16 16 16 22

   * - transformation
     - MMD :math:`=\|\Delta\Phi\|`
     - Rao :math:`=1-\|\Phi\|^2`
     - mixture linearity
     - use
   * - replicate averaging (rarefaction)
     - ✔
     - ✔ (with the exact :math:`v_{\text{rep}}` gap)
     - ✔
     - matched-depth comparison
   * - global scalar rescale
     - ✔ (up to scale)
     - ✘
     - ✔
     - magnitude-carrying blocks
   * - centring / PCA projection
     - ✔
     - ✘
     - ✔ (affine, same shift per band)
     - visualisation, Cox input
   * - per-column standardisation
     - ✘
     - ✘
     - ✘
     - **never** on a magnitude block
   * - orthogonal projection
     - ✔ (on the retained subspace)
     - ✘
     - ✔
     - nuisance removal
   * - sub-probability scaling :math:`(1-M_0)`
     - ✔ (a signed contrast)
     - ✘
     - ✔
     - deficient coverage

The one to internalise: **per-column standardisation forces every coordinate to unit variance across
samples**, so a matrix in which half the rows sit at the origin comes out looking exactly like one in
which none do — it deletes the deficiency it was meant to preserve. This already invalidated one
experiment, whose deficient arm scored 0.6481 against a 0.6179 baseline purely from the rescaling.
Any block whose point is that magnitude carries information must be scaled by **one global scalar**:
``ChannelBuilder.add(..., preserve_magnitude=True)``.

A second, subtler one: on all-positive TCREmp coordinates, the shared profile puts *any* two vectors
at raw cosine :math:`\approx 0.999`, and subtracting each vector's own scalar mean does **not** fix it
(measured: 0.9993 raw, 0.9985 self-centred, :math:`-0.14` cohort-centred). Only cohort centring
removes the shared profile. Read a cosine between two TCREmp-derived vectors only after cohort
centring.


Reading order
-------------

For the full derivations, the theory appendix of the companion manuscript is the reference; the
labels used in parentheses throughout this page are its labels. For the empirical record — which arm
won, on which cohort, with which interval — see the benchmark documents in the companion analysis
repository. This page is the middle layer: enough mathematics to know what each call computes and
what it is allowed to be used for.
