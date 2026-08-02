Mathematical foundations
========================

This page is one argument, read top to bottom. It starts from the reason a set of immune receptors
has no natural vector representation, builds the one it does have in three steps, shows what that
object knows and what it cannot know, then repairs the one premise that fails on real data. Each
section ends with the calls that compute what was just derived, so the page doubles as a map of the
library — but the order is the order of the *derivation*, not the order of the modules.

Where a step is stated as a proposition, the label in parentheses (``prop:kme``, ``eq:rff``, …) is the
label it carries in the theory appendix of the companion manuscript, so a docstring, this page and the
appendix all name the same result. Numbers quoted as *measured* are empirical results from the
benchmark programme, reproduced here because they decide a design question.

.. contents:: On this page
   :local:
   :depth: 2


The argument in one page
------------------------

#. A repertoire is an **unordered set of variable size**, so no fixed-dimensional map is even
   defined on it. Everything downstream follows from fixing that. → :ref:`math-problem`
#. A sequence becomes a **point** by its distances to a fixed panel of reference receptors. The map
   is Lipschitz, so Euclidean geometry in that space tracks alignment distance.
   → :ref:`math-step1`
#. A sample becomes a **measure** on those points. This single quotient removes order *and* length at
   once, and it is the only way an embedding can be order-invariant. → :ref:`math-step2`
#. Normalising the measure to total mass 1 buys **mixture linearity**: a point between two sample
   embeddings is the embedding of a real pooled repertoire. Everything in
   :ref:`math-surviving-algebra` and every generative operation rests on this one property.
   → :ref:`math-mixture-property`
#. A measure becomes a **vector** through a characteristic kernel. The kernel mean is an empirical
   characteristic function, its distance *is* the MMD, and it converges at the Monte-Carlo rate in
   effective sample size. → :ref:`math-step3`
#. That vector already **contains** the classical statistics: richness and Shannon are recoverable
   from it, and Rao's quadratic entropy is *exactly* ``1 − ‖Φ₁‖²``. The right question is therefore
   recoverability, not competition. → :ref:`math-knows`
#. But claim 3 assumed the weights are **frequencies**. At RNA-seq depth they are not — they are
   ``1/n`` for a technical draw size. Repairing that gives a **sub-probability** measure, a
   principled location for what was never observed, and a signed contrast.
   → :ref:`math-deficient`
#. The failure is **variance, not bias**, and its scale is estimable: ``κ = σ²/τ²`` is the sample size
   at which sampling noise equals biological signal. → :ref:`math-depth`
#. Mixture linearity then licenses two further operations exactly — **compartment decomposition** and
   **rarefaction** — the second with a closed-form identity for its own noise.
   → :ref:`math-surviving-algebra`
#. The same points support a **clone-level** test (local density against a background), and the same
   vectors support **cohort-level** read-outs and **generation**. → :ref:`math-density`,
   :ref:`math-answers`, :ref:`math-back-to-biology`
#. Every transformation applied to ``Φ`` keeps some of these properties and breaks others. The final
   table is the contract. → :ref:`math-invariants`


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


.. _math-problem:

The problem, and the shape of the solution
------------------------------------------

A repertoire is a set of receptors with clone sizes: unordered, and of a size that varies by orders
of magnitude between samples. Raw inputs therefore live in

.. math::

   \bigsqcup_{n \ge 1} (\mathbb{R}^p)^n / \mathfrak{S}_n ,

the disjoint union over all lengths of tuples-up-to-reordering. **No fixed-dimensional map is even
defined there**, so there is nothing to feed a model — and the two obstructions are different in kind:
order is a symmetry, length is a dimension mismatch. Any fix has to dispose of both.

The construction below disposes of them in three steps, and the point of splitting it into three is
that each step preserves a *named* property, and each property is what licenses one class of later
operation:

.. graphviz::

   digraph spine {
     rankdir=LR;
     bgcolor="transparent";
     node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#8899a6",
           fontname="Helvetica", fontsize=10];
     edge [color="#8899a6", fontname="Helvetica", fontsize=9];

     seq  [label="sequence\nσ = (junction, V, J)"];
     z    [label="point\nz = φ(σ) ∈ ℝᵖ", fillcolor="#e8f0fe"];
     rho  [label="measure\nρ_S = Σ w δ_z", fillcolor="#e8f0fe"];
     phi  [label="vector\nΦ(S)", fillcolor="#e8f0fe"];

     p1 [label="Lipschitz:\ngeometry ≈ alignment", shape=note, fillcolor="#ffffff"];
     p2 [label="order- and length-free;\nmixture linearity", shape=note, fillcolor="#ffffff"];
     p3 [label="‖ΔΦ‖ = MMD;\nRao = 1 − ‖Φ‖²", shape=note, fillcolor="#ffffff"];

     seq -> z   [label=" step 1: prototype distances "];
     z   -> rho [label=" step 2: clone-size weights "];
     rho -> phi [label=" step 3: characteristic kernel "];

     z   -> p1 [style=dotted, arrowhead=none];
     rho -> p2 [style=dotted, arrowhead=none];
     phi -> p3 [style=dotted, arrowhead=none];
   }

Read the dotted notes as the reason each step exists. Step 1 makes distance meaningful, step 2 makes
the object well-defined and *interpolable*, step 3 makes comparison a statistical distance. Lose any
one property and a whole class of operation goes with it — which is why the last section of this page
is a table of exactly what each downstream transformation keeps.


.. _math-step1:

Step 1 — a sequence becomes a point
-----------------------------------

*Before a set can be summarised, its elements need coordinates in which nearness means biological
similarity.*

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
included), looked up per allele with a cascade fallback. Note for later that every coordinate is a
distance, so :math:`\varphi` is **all-positive** — a fact with a practical consequence in
:ref:`math-invariants`.

Why this is a coordinate system and not a trick
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a metric :math:`d`, the map :math:`\sigma \mapsto (d(\sigma,\pi_k))_k` is 1-Lipschitz in each
coordinate (``prop:lipschitz``), giving a two-sided grip on the original distance:

.. math::

   \frac{1}{\sqrt{K}}\,\|\varphi(\sigma) - \varphi(\tau)\|_2 \;\le\; d(\sigma,\tau)
   \quad\text{and}\quad
   d(\sigma,\tau) \;\ge\; \max_k \big| d(\sigma,\pi_k) - d(\tau,\pi_k) \big| .

So Euclidean distance in embedding space is a lower bound on — and empirically a close proxy for — the
alignment distance it replaces. The consequence that matters in practice: the panel does not have to
be biologically special, it only has to *span*. Measured, the pairwise geometry induced by two
independent prototype draws agrees at :math:`R = 0.922` at :math:`K = 100`, :math:`0.990` at
:math:`K = 500` and :math:`0.997` at :math:`K = 2000`; real and model-generated panels agree at
:math:`R = 0.965`. From :math:`K \approx 500` up, *which* prototypes were drawn is not what a result
rests on.

Two prototype panels are nevertheless two different coordinate systems, and distances across them are
not comparable. That is why every serialisable object in the library carries a **prototype hash** and
refuses to load against a different panel.

Squared versus root metric
~~~~~~~~~~~~~~~~~~~~~~~~~~

``metric="squared"`` (the default, and the published space) uses :math:`d`; ``metric="sqrt"`` uses
:math:`\rho = \sqrt{d}`. The latter is the metrically tidier choice — :math:`\sqrt{d}` satisfies the
triangle inequality for a wider class of :math:`d` and makes the space of negative type, hence
Hilbert-embeddable (``prop:schoenberg``) — and benchmarked as a wash, so the default stays put.

**Call.** :meth:`mir.embedding.tcremp.TCREmp.embed` → ``(n, 3K)``;
:func:`mir.distances.junction.junction_distance_matrix` for the junction block alone.


.. _math-step2:

Step 2 — a sample becomes a measure
-----------------------------------

*We have a point per receptor. A sample is a set of them, and a set of variable size still has no
vector. One quotient fixes both obstructions at once.*

The quotient that removes order *and* length
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass from the tuple to the **empirical measure** (``def:sampmeasure``, ``eq:sampmeasure``):

.. math::

   \rho_S \;=\; \sum_{\sigma \in S} w_\sigma\, \delta_{z_\sigma},
   \qquad w_\sigma = \frac{g(a_\sigma)}{\sum_\tau g(a_\tau)} .

:math:`\rho_S` is a *single point* of :math:`\mathcal{P}(\mathbb{R}^p)` whatever :math:`n` is — the
dimension mismatch is gone because the object no longer has a dimension that depends on :math:`n` —
and it is manifestly unchanged by reordering. The converse is the design constraint: a sample
embedding is order-invariant **iff it factors through** :math:`\rho_S` (``prop:sampinv``). Anything
that does not factor through the measure is reading the input order.

Cardinality is not lost in the quotient, only relocated: it re-enters explicitly as richness
:math:`{}^0D = n` inside the diversity block. The embedding *sees* how many clonotypes there are
without letting that count set its dimension.

The clone-size transform
~~~~~~~~~~~~~~~~~~~~~~~~

Clone sizes are Zipf-heavy, so the choice of :math:`g` decides whether one hyperexpanded clone owns
the vector. A concave :math:`g` tames the tail:

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - ``weight=``
     - :math:`g(a)`
     - character
   * - ``"log2p1"`` *(default)*
     - :math:`\log_2(1+a)`
     - concave; a clone at :math:`10^6` reads outweighs a singleton :math:`{\sim}20\times`, not
       :math:`10^6\times`
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
     - presence only; then :math:`n_{\text{eff}} = {}^0D`

.. _math-mixture-property:

Why normalise: the mixture property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dividing by :math:`\sum_\tau g(a_\tau)` looks like a cosmetic choice and is not. Because the
construction is **linear in the clone-weight measure**, a partition of a repertoire into
sub-populations :math:`c` with weight shares :math:`\pi_c` satisfies

.. math::
   :label: eq-mixture

   \rho_S = \sum_c \pi_c\, \rho_c
   \quad\Longrightarrow\quad
   \Phi(S) = \sum_c \pi_c\, \Phi(c),
   \qquad \pi_c = \frac{\sum_{\sigma \in c} g(a_\sigma)}{\sum_\tau g(a_\tau)} ,

verified numerically to :math:`\sim 10^{-17}`. Two corollaries used repeatedly below. Pooling two
samples gives :math:`\Phi(S \cup T) = (N_S\Phi_S + N_T\Phi_T)/(N_S+N_T)` — exactly linear
interpolation, so a point *between* two sample embeddings is the embedding of a real pooled
repertoire. And any partition can be decomposed after the fact, which is what makes
:ref:`math-compartments` and :ref:`math-rarefaction` exact rather than approximate.

The unnormalised sum has no such property: its midpoint is the embedding of nothing. Nor does one lose
anything by normalising, since normalise-or-not is exactly one scalar
(:math:`\Phi_{\text{sum}} = N\,\Phi_{\text{mean}}`) — so keeping the mean and carrying :math:`\log N`
as its own channel strictly dominates the plain sum, and measured, it does.

**Call.** :meth:`mir.repertoire.RepertoireSpace.sample_cloud` → :math:`(Z, w)`.


.. _math-step3:

Step 3 — a measure becomes a vector
-----------------------------------

*A measure is well-defined but infinite-dimensional. To compare two of them with a number, embed them
in a Hilbert space where the inner product is a kernel.*

An empirical characteristic function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take a shift-invariant kernel. By **Bochner's theorem** it is the Fourier transform of a finite
measure :math:`\Lambda`,

.. math::

   k(z,z') \;=\; \int e^{i\omega^\top (z-z')}\, d\Lambda(\omega),

so sampling frequencies :math:`\omega_j \sim \Lambda` approximates the kernel by an explicit feature
map (``eq:rff``):

.. math::

   \psi(z) \;=\; \sqrt{\tfrac{2}{D}}\,
   \big(\cos(\omega_j^\top z + b_j)\big)_{j=1}^{D},
   \qquad \omega_j \sim \mathcal{N}(0, \ell^{-2} I),\;\; b_j \sim \mathcal{U}[0,2\pi],

with :math:`\mathbb{E}[\psi(z)^\top\psi(z')] = k(z,z')` (Rahimi–Recht). The :math:`\cos(\omega^\top z
+ b)` form is real-inner-product bookkeeping; what is actually being computed is transparent
(``eq:ecf``):

.. math::

   \sum_\sigma w_\sigma e^{i\omega_j^\top z_\sigma}
   \;=\; \int e^{i\omega_j^\top z}\, d\rho_S(z) \;=\; \widehat{\rho_S}(\omega_j),

the sample's **empirical characteristic function**, evaluated at :math:`D` random frequencies. So
:math:`\Phi_1(S) = \sum_\sigma w_\sigma \psi(z_\sigma)` is a finite sketch of a complete description
of :math:`\rho_S`.

That the summary is codebook-free follows from the same line: no clustering, no vocabulary, no
:math:`K` to tune (``prop:codebook``). The bandwidth :math:`\ell` is calibrated to the
one-substitution embedding scale :math:`r_1` (:func:`mir.density.calibrate_radius`), so the kernel
resolves about one CDR3 mutation — and the same :math:`r_1` reappears as the radius in
:ref:`math-density`.

Why the characteristic function and not a moment-generating one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the step where the heavy tail could have destroyed everything. An :math:`e^{t^\top z}` feature
would be dominated by the single largest clone and need not even converge on a Zipf distribution,
whereas :math:`|e^{i\omega^\top z}| = 1` is bounded, always exists, and is robust to a hyperexpanded
clone. And by Lévy uniqueness the characteristic function is a **complete invariant**: two measures
with the same one are the same measure. That is exactly the requirement that the kernel be
*characteristic*, and it is what makes the next paragraph a metric statement rather than a heuristic.

Distance is the MMD
~~~~~~~~~~~~~~~~~~~

.. math::

   \mathrm{MMD}(S,S') \;=\; \big\|\Phi_1(S) - \Phi_1(S')\big\| \qquad (\texttt{eq:kme})

— not "a distance between feature vectors" but the maximum mean discrepancy between the two
underlying distributions, zero iff they coincide.

The plain (V-statistic) form carries a positive bias from the :math:`k(z,z)` diagonal of order
:math:`1/n_{\text{eff}}`, so a low-diversity sample has its distances **inflated by construction** —
and when diversity is itself the variable of interest, that bias masquerades as signal *with the wrong
sign*. Removing the diagonal analytically (``rem:unbiasedmmd``; Gretton et al. 2012), with
:math:`s = 1/n_{\text{eff}} = \sum_\sigma w_\sigma^2` and :math:`k(z,z)\approx 1`:

.. math::

   \widetilde{\|\mu\|^2} \;=\; \frac{\|\mu\|^2 - s}{1 - s},
   \qquad
   \widetilde{\mathrm{MMD}}^2(S,S') \;=\;
   \widetilde{\|\mu_S\|^2} + \widetilde{\|\mu_{S'}\|^2} - 2\,\mu_S^\top\mu_{S'} .

Undefined for a point mass (:math:`n_{\text{eff}} \le 1`), which the implementation refuses rather
than silently returns. Measured consequence of ignoring it: an apparent age-divergence signal carried
the sign of *clonal expansion* rather than richness until the diagonal was removed.

How much data the estimate needs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel mean converges to the population mean map at the Monte-Carlo rate in the **effective**
sample size (``prop:kme``, ``eq:neff``):

.. math::

   \mathbb{E}\big\|\Phi_1(S) - \Phi_1(P)\big\|_{\mathcal{H}}
   \;\le\; \frac{C}{\sqrt{n_{\text{eff}}}},
   \qquad n_{\text{eff}} = \Big(\sum_\sigma w_\sigma^2\Big)^{-1},

where :math:`n_{\text{eff}}` is itself a Hill number, :math:`{}^2D \le n_{\text{eff}} \le {}^0D`
(``prop:antag``) — the *effective* number of clonotypes, which is what a heavy tail leaves you.
Measured slope of :math:`\log\|\Delta\Phi_1\|` against :math:`\log` depth: :math:`-0.55`, against the
predicted :math:`-0.5`. This is a *generic* kernel-mean rate rather than anything peculiar to this
embedding, which is precisely why it can be relied on — and it is also the crack through which
:ref:`math-depth` enters, because a rate in :math:`n_{\text{eff}}` is a promise about *variance*, not
a promise that any particular sample is adequate.

**Call.** :func:`mir.repertoire.fit_repertoire_space` (one basis per cohort — see the comparability
note above), :func:`mir.repertoire.sample_embedding`, :func:`mir.repertoire.mmd_distance`,
:func:`mir.repertoire.mmd_matrix` (``unbiased=True`` for the diagonal-removed form).


.. _math-knows:

What the vector already knows
-----------------------------

*Before asking what to add to* :math:`\Phi`, *it is worth establishing what is already inside it —
because a statistic that can be read back out does not need to be bolted on beside it.*

Hill numbers, and their blind spot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :math:`p_i = a_i/\sum_j a_j`,

.. math::

   {}^qD \;=\; \Big(\sum_i p_i^{\,q}\Big)^{1/(1-q)},
   \qquad
   {}^0D = n, \quad
   {}^1D = e^{-\sum_i p_i \log p_i}, \quad
   {}^2D = \Big(\sum_i p_i^2\Big)^{-1} .

Every Hill number is a functional of the clone-size distribution **alone**, hence invariant to
permuting which receptor carries which abundance: permute the sequence labels and every diversity
statistic is unchanged while the biology is destroyed. Diversity describes the *shape* of the
abundance distribution and is blind to composition. Coverage standardisation to a common Good–Turing
coverage :math:`\hat{C}^\star` (``prop:coverage``, via :mod:`vdjtools.stats.inext`) removes the depth
dependence of the estimate itself.

Rao's quadratic entropy is the norm of the kernel mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rao's quadratic entropy weights each pair of receptors by how *different* they are, so it occupies
exactly the blind spot above. Take the kernel dissimilarity :math:`d(z,z') = 1 - k(z,z')`:

.. math::

   Q(S) \;=\; \sum_{\sigma,\tau} w_\sigma w_\tau \big(1 - k(z_\sigma, z_\tau)\big)
         \;=\; \underbrace{\sum_{\sigma,\tau} w_\sigma w_\tau}_{=\,1}
             \;-\; \sum_{\sigma,\tau} w_\sigma w_\tau\, \psi(z_\sigma)^\top \psi(z_\tau)
         \;=\; 1 - \|\Phi_1(S)\|^2 .

The **norm of the kernel mean is a functional diversity** — no Gram matrix is ever formed, and the
identity is exact (verified against an explicit Gram to a maximum relative error of
:math:`\sim 10^{-16}`). Note what this says about the geometry: two samples' *directions* carry
composition and their *radii* carry functional diversity, so a transformation that preserves
differences but not norms keeps MMD and silently destroys :math:`Q`.

Two caveats, both structural rather than defects. The kernel is a random-Fourier approximation, so
:math:`k(z,z) = 1` holds only to :math:`O(D^{-1/2})` and a single-clone sample reads :math:`Q \sim
10^{-2}` rather than exactly 0. And because :math:`Q` is defined by a norm, it is valid only for the
true, **uncentred** :math:`\Phi_1`.

The second-moment block
~~~~~~~~~~~~~~~~~~~~~~~

Which clonotypes appear *together* — the shape an HLA imprint takes — is a second-order property
invisible to any mean (``prop:interact``):

.. math::

   \Sigma_S \;=\; \sum_\sigma w_\sigma\, \psi_2(z_\sigma)\psi_2(z_\sigma)^\top ,

a codebook-free Fisher vector on a small feature map :math:`\psi_2` of dimension :math:`D_2`, stored
as its :math:`D_2(D_2+1)/2` upper triangle. The opt-in ``n_eigs=r`` alternative keeps the top
:math:`r` eigenvalues instead — compact and rotation-invariant, and **measured lossy for exactly the
signal the block exists for**: an HLA imprint lives in *which* clones co-occur, a directional fact, so
the rotation-invariant spectrum reached :math:`\le 0.55` AUC against 0.593 for the full triangle. The
triangle stays the default.

Recoverability, not competition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three subsections above make a scoring rule available that is better than the obvious one. The
question is *not* "does :math:`\Phi` beat the one-number marker" — that is a competition between an
object and one of its own coordinates. It is whether each basic statistic is **carried inside** it: a
grouped-CV ridge from the embedding's principal components back to richness, Shannon, top-clone
fraction, singleton fraction, unseen fraction and library size, reported as :math:`R^2`. High means
nothing has to be bolted on beside the embedding.

Measured, the embedding's derivatives reach :math:`R^2` 0.974–0.994 for Shannon and 0.985–0.9999 for
richness, and Rao's :math:`Q` **alone** — one scalar — reaches 0.74–0.85. The interesting failure is
the one that is structural: renormalising every sample to mass 1 *deletes the magnitude*, so every
coverage- and richness-like statistic is unrecoverable from :math:`\Phi` **by construction**. That
observation is what the next section is about.

**Call.** ``sample_embedding(..., blocks=("mean","diversity","second"))``,
:func:`mir.repertoire.rao_q`, :func:`mir.bench.recovery_report`.


.. _math-deficient:

When the sample is not a probability distribution
-------------------------------------------------

*Step 2 wrote* :math:`w_\sigma = g(a_\sigma)/\sum g` *and called it a frequency. On real data at
RNA-seq depth, it is not one. This section repairs that premise without giving up anything derived
from it.*

The premise that fails
~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi_1` is the kernel mean of a *probability* measure: the weights sum to 1, so every sample
asserts one full unit of confidence about its repertoire. Measured unique-clonotype counts put the
median tissue TRB sample at **21** clonotypes (blood TRB 254, 1st percentile 1). A weight computed
from 21 observations is not an estimate of a clonal frequency — it is :math:`1/n` for a *technical
draw size*, while true frequencies live at :math:`10^{-5}` to :math:`10^{-8}`. Directly measured: one
singleton's weight spans a factor of **21,454** across blood TRB, entirely from sample size.

So :math:`\hat P_S` is nowhere near :math:`P_S`, and "kernel mean embedding of the repertoire
distribution" is not what :math:`\Phi` is at this depth. What it *is*: the centroid of a captured point
set — an unbiased estimator of :math:`\mathbb{E}[z]` under the capture process, with **variance**
:math:`\propto 1/n`. Both halves of that sentence get a treatment: the missing mass here, the variance
in :ref:`math-depth`.

The Good–Turing decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let the true repertoire have :math:`S_{\text{true}}` clonotypes at frequencies :math:`p_\sigma`. The
target is :math:`\Phi_{\text{true}} = \sum_\sigma p_\sigma z_\sigma`. Split the sum at what was seen:

.. math::
   :label: eq-gt

   \Phi_{\text{true}} \;=\; (1 - M_0)
     \underbrace{\sum_{\text{seen}} \frac{p_\sigma}{\sum_{\text{seen}} p}\, z_\sigma}_{\Phi_{\text{seen}}}
     \;+\; M_0\, \mathbb{E}[z \mid \text{unseen}],
   \qquad M_0 = \sum_{\text{unseen}} p_\sigma .

This is an identity, not a model, and it is the unification: **every weighting scheme in play is
this one identity with a different estimator of** :math:`M_0`. The current default is the special case
:math:`M_0 = 0` — the assumption that nothing was missed.

Two estimators of the missing mass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Turing.** :math:`\hat M_0 = f_1/N`: the classical read-weighted answer, the total frequency of
clonotypes seen exactly once.

**Chao, via a boundary assumption.** Chao1 estimates the *count* of missing clonotypes and says
nothing directly about their mass, so a bridge is needed. It is one assumption: *an unseen clonotype
is rare, so had it been drawn it would carry at most one read, and unseen clonotypes do not overlap.*
That pins each unseen clone's frequency at the detection boundary :math:`1/(N + S_u)`, hence

.. math::

   \hat M_0 \;=\; S_u \cdot \frac{1}{N + S_u} \;=\; \frac{S_u}{N + S_u},
   \qquad S_u \;=\; \frac{f_1(f_1-1)}{2(f_2+1)} ,

and substituting into :eq:`eq-gt` collapses the observed term to a single pseudo-count completion:

.. math::

   (1 - \hat M_0)\cdot \frac{a_\sigma}{N}
   \;=\; \frac{N}{N + S_u}\cdot\frac{a_\sigma}{N}
   \;=\; \frac{a_\sigma}{N + S_u} .

**And this is why the units work.** :math:`N` counts reads while :math:`S_u` counts clonotypes; adding
them would normally be a category error, and is legitimate *only* because the one-read assumption
makes each unseen clone contribute exactly one read — so :math:`N + S_u` is the read total of the
completed table. Read the formula as: sample slightly deeper, catch every missing clone exactly once.

Use the **bias-corrected** :math:`S_u = f_1(f_1-1)/(2(f_2+1))`, never the classical
:math:`f_1^2/2f_2`: at these depths a sample with no clonotype seen exactly twice is common, and the
classical form is undefined there. The two estimators rank samples nearly identically (measured
:math:`r` = 0.93 and 0.76 in two views) but differ in level — Chao runs above Turing when singletons
dominate doubletons, below when doubletons abound. Both limits behave: a deep sample gives
:math:`\hat M_0 = 0.0002` and reduces to the ordinary embedding; an all-singleton sample gives 0.96,
the honest statement that it measured almost nothing.

Sub-probability, not negative probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural question at this point is whether the missing mass could be a *negative* probability. It
should not be, and it need not be. :math:`\Phi`'s value is that :math:`\|\Phi_P - \Phi_Q\|` **is** the
MMD — a metric between distributions — and that a convex combination of two :math:`\Phi`'s is the
:math:`\Phi` of a real pooled repertoire (:eq:`eq-mixture`). A measure allowed to go negative on a set
is not a probability measure: MMD stops being a metric on it, and the midpoint of two embeddings stops
being the embedding of anything. Both losses are severe and neither is necessary.

Dropping the renormalisation instead costs nothing. Let the measure carry mass :math:`1 - M_0`:

.. math::

   \mu_S \;=\; \sum_{\text{seen}} w_\sigma\,\delta_{z_\sigma},
   \qquad \text{total mass } = 1 - M_0 \;\le\; 1 .

Sub-probability measures are standard objects and nothing in the RKHS construction breaks.
Interpretively this is the whole point: renormalising **forces** a 5-clonotype tumour to assert a full
unit of confidence, whereas a deficient measure lets it say *I hold 0.08 of a repertoire's worth of
evidence*.

Where signed structure legitimately lives is a **difference of two probability measures**:

.. math::

   \Psi_S \;=\; (1 - M_0)\big(\Phi_{\text{seen}} - \Phi_{\text{naive}}\big)
          \;=\; \Phi_{\text{true}} - \Phi_{\text{naive}} ,

whose coordinates go negative wherever the sample is **depleted** relative to unselected V(D)J output
— a real statement. It is an ordinary RKHS element and :math:`\|\Psi_S\|` is exactly
:math:`\mathrm{MMD}(S, \text{naive})`, so signed structure comes at no cost to the metric. The second
equality is immediate from :eq:`eq-gt`, and it is why the two constructions are one object rather than
two: :math:`\Psi` *is* the reference-centred sub-probability embedding.

Reading the result: **magnitude = confidence × deviation-from-naive**. An immune desert has
:math:`M_0 \to 1` and lands at the **origin**, the correct location for "no infiltrate detected"
rather than an arbitrary point on the unit sphere. A vague, shallow blood sample lands there too and
says so *by its norm* instead of by being dropped from the cohort.

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

One term in :eq:`eq-gt` is still unspecified, and it is not cosmetic: :math:`M_0` alone only rescales
:math:`\Phi`, so the unseen block changes the *direction* of the embedding only if it is actually
filled with a vector. The choice of that vector decides whether the whole construction is a shrinkage
estimator toward a **meaningful** point. Three candidates, measured:

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

The mechanism is worth stating because it explains the failure above. :math:`\Phi = (1-M_0)\Phi_{\text{seen}}
+ M_0 z_{\text{naive}}` *is* shrinkage: a shallow sample has large :math:`M_0` and collapses toward a
fixed point, a deep one keeps its own :math:`\Phi`. The difference from the failed James–Stein arm is
entirely the **target**. Shrinking an unmeasurable sample toward the corpus centroid creates a dense
ball that is itself depth-correlated; shrinking it toward the germline says the honest thing — *a
repertoire we did not measure looks like unselected recombination output* — and lands it at a
biologically meaningful location.

**Call.** ``sample_embedding(..., missing_mass="turing"|"chao")`` sets
:attr:`~mir.repertoire.SampleEmbedding.mass`; :func:`mir.repertoire.missing_mass`,
:func:`mir.repertoire.naive_reference`, :func:`mir.repertoire.contrast_embedding`.


.. _math-depth:

How much of a sample can be trusted
-----------------------------------

*The missing mass handled what was not seen. The other half of the sentence in* :ref:`math-deficient`
*was variance — and unlike bias, variance has a scale that can be estimated from the cohort itself.*

The damage depth does to a kernel mean is **not bias** but variance :math:`\propto 1/n`, over an
:math:`n` spanning four orders of magnitude — and in a nearest-neighbour graph that
heteroscedasticity *is* a depth axis, because a noisier point has systematically different neighbours.
A 21-clonotype sample's :math:`\Phi` is :math:`\sim 16\times` noisier than a 5,780-clonotype one's.
Measured, a sampling fingerprint (log reads, log unique, singleton fraction, Good–Turing missing mass,
top-clone fraction) explained :math:`R^2 = 0.257` of PC1 in one view and 0.465 in another.

Decompose the observed spread. Each sample's squared distance from the cohort centroid has an
expectation with two parts — real between-sample spread, plus sampling noise that shrinks with depth:

.. math::

   \mathbb{E}\big\|\Phi_S - \bar\Phi\big\|^2 \;\approx\; \underbrace{\tau^2}_{\text{signal}}
   \;+\; \underbrace{\frac{\sigma^2}{n}}_{\text{noise}},
   \qquad \boxed{\;\kappa \;=\; \sigma^2/\tau^2\;}

so a straight line fitted to :math:`\|\Phi_S - \bar\Phi\|^2` against :math:`1/n` gives :math:`\tau^2`
as its intercept, :math:`\sigma^2` as its slope, and :math:`\kappa` as the sample size at which the two
are **equal**. Below :math:`\kappa`, a sample's :math:`\Phi` is more sampling noise than signal.

Measured :math:`\kappa \approx 40`–70 clonotypes across four independent views, with 23–69% of samples
below it. The number is stable enough to be worth knowing and variable enough to be worth estimating
rather than importing.

What to do with it is deliberately *not* "apply a floor". The library applies none, because the same
low clonotype count means opposite things in different tissues: in blood it is shallow sequencing, in a
tumour it is low infiltration — the phenotype of interest. Measured, a floor applied in tumour deleted
the stratum the analysis existed to predict. Carry :math:`\kappa` and
:attr:`~mir.repertoire.SampleEmbedding.mass` and **weight instead of filtering**; use :math:`\kappa`
to define the trusted subset a diagnostic is re-read inside, not to decide who is in the cohort.

**Call.** :func:`mir.repertoire.depth_threshold`, :func:`mir.repertoire.sample_statistics` /
:func:`~mir.repertoire.cohort_statistics` (the fingerprint), :func:`mir.cohort.depth_report` (R² of
the leading PCs on the fingerprint, with a trusted-subset arm — read beside the returned
``explained_variance``, which is what separates *a different direction* from *a collapsed one*).


.. _math-surviving-algebra:

The algebra that survives
-------------------------

*Mixture linearity* :eq:`eq-mixture` *was established in step 2 and has been used once, to justify
normalising. It licenses two further operations exactly — and "exactly" is what makes them worth
having, because both would otherwise be heuristics.*

.. _math-compartments:

Compartments and the dilution factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi_1` is a weighted **average**, which is the right operation for estimating a population mean
and the wrong one for detecting a **minority** signal. Make that precise. Write the repertoire as
naive plus expanded, :math:`\rho_S = (1-\pi)\rho_N + \pi\rho_E`; then by :eq:`eq-mixture` a difference
confined to the expanded compartment,
:math:`\Delta = \Phi_1(E_{\text{case}}) - \Phi_1(E_{\text{ctrl}})`, reaches the whole-repertoire
embedding attenuated to

.. math::

   \Phi_1(S_{\text{case}}) - \Phi_1(S_{\text{ctrl}}) \;=\; \pi\,\Delta ,

while the *noise* is supplied by the naive compartment, which contributes most of the clonotypes. The
signal-to-noise ratio of the average is therefore worse than the compartment's own by roughly
:math:`\pi` — and :math:`\pi` is small precisely in the samples that matter. This is a statement about
the **estimator**, not about the data, which is why it predicts a fix: embed the compartments
separately.

Because :eq:`eq-mixture` is exact, the shares are also recoverable after the fact by non-negative
least squares — well-posed, not a heuristic fit:

.. math::

   \hat\pi \;=\; \arg\min_{\pi \ge 0} \Big\| \Phi_1(S) - \sum_c \pi_c \Phi_1(c) \Big\|^2 .

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

Two implementation constraints follow from the mathematics rather than from taste. Every compartment
must be embedded through the **same** basis — refitting per band would put each in its own geometry
and make band-to-band and cohort-to-cohort distances meaningless. And a compartment with a handful of
clonotypes is recorded **absent**, not embedded: a 3-clone "repertoire" is not an estimate, and a hole
is a value the assembly layer already understands.

Measured on IGH isotypes, the class-switched **IgG compartment carries a median** :math:`\pi` **of
0.070** of :math:`\Phi_1(\text{IGH})` (unswitched IgM 0.230, IgA 0.176, 0.520 unaccounted — the
uncalled-isotype share, in ballpark agreement with the :math:`\sim 0.43` counted from reads;
embedding *weight* and *reads* are different denominators). Since IgG is the affinity-matured,
T-dependent fraction most likely to carry antigen-specific signal, that single number is the dilution
factor :math:`\pi` measured rather than assumed — and it doubles as a **power calculation**: a subset
carrying :math:`\pi \approx 0.001` cannot be detected by any aggregate distance on :math:`\Phi`, so
the per-clonotype witness (:ref:`math-answers`) is the sensitive route. Check :math:`\pi` before
spending compute on the aggregate.

What the decomposition measured, in the direction the theory predicts: the ``singleton`` band
reproduces a clinical-covariates-only score almost exactly (0.600 vs 0.599) — pure dilution — while
``expanded`` alone reproduces the *whole-repertoire* score (0.634 vs 0.634), so the entire prognostic
content of :math:`\Phi_1` lives in the expanded compartment. What it did **not** do is lift that
content above what the clone-size distribution already encodes: on survival endpoints, zero of 22
pre-registered block × endpoint cells cleared the bar against a diversity reference, and the isotype
cut failed identically. Where banding *does* win is where abundance classes are different biology —
unmutated IgM singletons against class-switched expansions — with a measured kNN entropy of 0.3875
against 0.4686 for the pooled arm in tissue IGH.

.. _math-rarefaction:

Rarefaction, and the exact Rao gap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second consequence answers "how do I compare two groups at matched depth without leaving the
metric". Because :math:`\Phi` is linear in the measure, the mean of :math:`\Phi` over :math:`R`
independent multinomial subsamples is itself a kernel mean embedding — of the *mixture* distribution
over subsamples — so MMD, Rao and mixture linearity all survive (verified to :math:`\sim 10^{-15}`).
It is the **only** depth correction that does: an orthogonal projection breaks the norm identity, and
a per-coordinate location-scale rescale breaks both.

The cost is also exactly computable, which is the appealing part. With :math:`\bar\Phi = R^{-1}\sum_r
\Phi_r` and :math:`v_{\text{rep}} = R^{-1}\sum_r \|\Phi_r - \bar\Phi\|^2`, the parallel-axis identity
:math:`R^{-1}\sum_r\|\Phi_r\|^2 = \|\bar\Phi\|^2 + v_{\text{rep}}` and :math:`Q = 1 - \|\Phi\|^2` give

.. math::

   Q(\bar\Phi) \;=\; 1 - \|\bar\Phi\|^2
   \;=\; 1 - \Big(\tfrac{1}{R}\!\sum_r \|\Phi_r\|^2 - v_{\text{rep}}\Big)
   \;=\; \frac{1}{R}\sum_r Q(\Phi_r) \;+\; v_{\text{rep}} .

Read it in either direction. As interpretation: :math:`\bar\Phi` embeds the mixture over subsamples,
which is genuinely more diverse than any single subsample, so **the excess diversity of the average is
the replicate variance**. As a tool: :math:`v_{\text{rep}}` is a free per-sample estimate of exactly
the sampling noise that :math:`\kappa` measures cohort-wide. And as a warning: averaging commutes with
:math:`\Phi` but **not** with nonlinear functionals of it, so :math:`\mathrm{mean}(Q) \ne
Q(\mathrm{mean})` and the gap must be reported rather than assumed away. Measured gaps ran 0.02% to
0.93%, monotone in target depth.

Rarefaction is not a default: rarefying a cohort to the depth of its shallowest useful samples
discards the deep samples' entire advantage. Reach for it when two groups must be compared *at matched
depth* and the comparison has to stay an MMD.

**Call.** :func:`mir.repertoire.band_frames`, :func:`mir.repertoire.band_embeddings`,
:func:`mir.repertoire.mixture_weights`, :func:`mir.repertoire.rarefy_embedding`.


.. _math-density:

Back to the single clone: local density
---------------------------------------

*Everything so far aggregated the point cloud. The other question one can ask of the same points is
local: is this receptor in a neighbourhood that is denser than chance? The kernel and its radius are
already defined, so this needs no new geometry.*

Enrichment as a density ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Antigen-driven convergence appears as local over-density of receptors. Classical methods build an
exact- or 1-mismatch sequence graph; the continuous form estimates the same ratio directly in
embedding space (``sec:dens-balloon``, ``prop:balloon``). With a **balloon** (adaptive-radius)
estimator whose radius :math:`r(z)` is chosen so the background ball holds a target count,

.. math::

   E(z) \;=\; \frac{\hat f_{\text{obs}}(z)}{\hat f_{\text{gen}}(z)}
         \;=\; \frac{n_{\text{obs}}(z)\,/\,N_{\text{obs}}}{n_{\text{gen}}(z)\,/\,N_{\text{gen}}} .

This is a density-*ratio* estimate (``sec:dens-dre``, ``lem:lsif``), and the ratio form is what makes
it usable in high :math:`p`: the ball volume cancels between numerator and denominator, so no
bandwidth normalisation is needed and no density has to be estimated on its own — which in :math:`p`
of this size a kernel density estimate could not do. As :math:`r \to 0` it recovers the graph limit,
confirmed numerically (:math:`\rho` 0.37 → −0.05 as the radius grows).

Two nulls, two background types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The significance test depends on what the background *is* (``prop:poissontest``, ``prop:ctest``):

.. math::

   n_{\text{obs}}(z) \sim \mathrm{Poisson}\big(\lambda(z)\big),
   \quad \lambda(z) = n_{\text{gen}}(z)\cdot \tfrac{N_{\text{obs}}}{N_{\text{gen}}}
   \qquad\text{(generative background)}

.. math::

   n_{\text{obs}}(z) \sim \mathrm{Binomial}\big(n_{\text{obs}}+n_{\text{gen}},\; p_0\big),
   \quad p_0 = \tfrac{N_{\text{obs}}}{N_{\text{obs}}+N_{\text{gen}}}
   \qquad\text{(control background)}

with Benjamini–Hochberg control across clonotypes, plus a **water-level** calibration
(``eq:waterlevel``) that raises the null level until the empirical false-discovery profile is
consistent — needed because in the naive regime the background is itself pervasively convergent.

That last point is the empirical lesson, and it matters more than the choice of test: real repertoires
are so convergent that a generation-probability background flags :math:`\sim 40\%` of clones. Use a
**biological control** (day 15 versus day 0, allele-positive versus negative, seropositive versus
control) for specificity, and process the *full* repertoire — subsampling dilutes exactly the sparse
antigen clusters being looked for. Made quantitative on a spiked benchmark: admixed noise clones were
over-flagged 43% under a generative background against 1% under a control background, a 46×
signal-to-noise lift.

Abundance-aware enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Counting *distinct* neighbours ignores that an expanded clone is stronger evidence than a singleton
(``sec:dens-abund``, ``prop:abund``). Replace the in-ball count with the variance-stabilised mass
:math:`S(z) = \sum_{j \in B(z)} g(a_j)`; under a compound-Poisson model its tail is Gamma with
dispersion

.. math::

   \varphi \;=\; \frac{\mathbb{E}[g(A)^2]}{\mathbb{E}[g(A)]} ,

and a per-clonotype orphan/depth channel :math:`P(A \ge a_j)` is Fisher-combined with the breadth
term — so a clone can be called on being *large where it should be small* as well as on having many
neighbours. Setting ``abundance=None`` recovers the distinct count exactly (:math:`g \equiv 1`).

**Call.** :func:`mir.density.fit_density_space`, :func:`mir.density.neighbor_enrichment`,
:func:`mir.density.enriched_mask`, :func:`mir.density.denoise_and_cluster`,
:func:`mir.density.generate_background`.


.. _math-answers:

From vectors to answers
-----------------------

*A cohort is now a matrix of vectors. Two things are needed to report a result from it: a way to say
which part of the vector carried the signal, and a correction for the nuisance structure that will
otherwise carry it instead.*

Named channels and ablation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A **channel** is a named group of columns. Given a caller-supplied scorer :math:`f` (higher is better)
and a reference score :math:`b`, there are exactly two questions one can ask of a channel, and they
are not the same question:

.. math::

   \delta_{\text{in}}(c) = f\big(X_{:,c}\big) - b,
   \qquad
   \delta_{\text{out}}(c) = f\big(X\big) - f\big(X_{:,\neg c}\big) .

:math:`\delta_{\text{in}}` is *marginal* — inflated by correlation between channels, so two redundant
channels both look important. :math:`\delta_{\text{out}}` is *conditional* — deflated by the same
correlation. Reported together they separate the claims: high in **and** high out means
irreplaceable, while high in with :math:`\delta_{\text{out}} \approx 0` is the **redundancy**
signature. Leave-one-*in* is the default for a concrete reason: real scorers reduce their input (an
in-fold PCA), so dropping one narrow channel lets the reduction re-mix the rest and reconstruct nearly
the same components, making leave-one-out structurally near-blind to exactly the 1-column channels
that often win.

Significance, when asked for, is a **row permutation of the block**: the scorer holds :math:`y` in row
order, so shuffling the block's rows breaks the association exactly as shuffling :math:`y` would, and
it is the only null available to a function that never sees the labels.

The witness: from a difference back to sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An MMD between two groups is a number; the witness function turns it back into receptors
(``prop:witness``, ``eq:witness``). The direction :math:`\mu_{\text{pos}} - \mu_{\text{neg}}` in
feature space scores each candidate clonotype by

.. math::

   s(\sigma) \;=\; \big\langle \mu_{\text{pos}} - \mu_{\text{neg}},\; \psi(\varphi(\sigma)) \big\rangle ,

so the top-scoring candidates are the discriminative public clones. This is the supervised route to
motifs that the unsupervised bulk mean cannot surface, because there they are swamped by the naive
background — the same dilution argument as :ref:`math-compartments`, which is why a small :math:`\pi`
sends you here.

Note what can and cannot be attributed this way. Only a channel with a clonotype **pre-image** — a
kernel mean — has a witness; a Hill number is a summary of the count distribution and has no
pre-image, so asking which clones drive it is a category error rather than a hard problem.

Batch offsets, and why the obvious correction backfires
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A first-order batch effect is an additive offset, so subtracting each batch's mean removes it
(``prop:batch``) — in-sample. Out-of-sample the *estimation error* dominates. With
:math:`\hat\mu_b` fitted from :math:`n_b` samples in :math:`d` dimensions,

.. math::

   \mathbb{E}\|\hat\mu_b - \mu_b\|^2 \;=\; \frac{\sigma^2 d}{n_b} ,

which for :math:`n_b \approx 8`–15 and :math:`d \approx 1{,}280` is a vector of norm :math:`\approx
16` against a true offset of norm 7–24: **subtracting it injects a batch-constant vector as large as
the one it removes.** Measured, evaluated leave-one-batch-out plus a donor-level split inside each
batch, batch-identity AUC went 0.863 raw → **0.985** after per-batch centring and 0.978 after ComBat.
Both classical corrections made the batch *easier* to read.

The fix is to shrink the offset by its own reliability. The positive-part James–Stein estimator,

.. math::

   \hat\mu_b^{\text{JS}} \;=\; c_b\,\hat\mu_b,
   \qquad
   c_b \;=\; \Big(1 - \frac{(d-2)\,\hat\sigma^2/n_b}{\|\hat\mu_b\|^2}\Big)_+ ,

leaves alone any batch whose apparent offset is no larger than its estimation error, and measured, it
recovered most of the damage: 0.985 → **0.889**. The cluster-aware alternative removes the offset *per
soft cluster* with a batch-diversity penalty (Harmony-style), which corrects a batch confounded with
biology instead of erasing that biology, and reduces exactly to plain residualisation at one cluster.

Three conditions any correction must meet, learned the hard way: evaluate **out-of-sample**, since
in-sample the injected error is invisible by construction; report the **geometry cost** (within-batch
distance correlation, idempotence, symmetry) and not only the batch metric; and check **mixture
additivity** :eq:`eq-mixture`, because the compartment decompositions depend on it.

**Call.** :class:`mir.explain.ChannelBuilder`, :func:`mir.explain.channel_report`,
:func:`mir.explain.channel_drivers`, :func:`mir.repertoire.class_witness`,
:func:`mir.cohort.residualize` (``shrink=True`` for the James–Stein form),
:func:`mir.repertoire.correct_batch`, and the scorers in :mod:`mir.bench.eval`.


.. _math-back-to-biology:

From vectors back to biology
----------------------------

*The construction so far is one-directional: a repertoire in, a vector out. The remaining question is
whether the arrow can be reversed — whether a point in the space can be turned back into a repertoire
state, a trajectory, or a sequence.*

The derivable descriptor
~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Phi` normalises the total mass away, and in tissue that mass *is* the infiltration signal. The
descriptor keeps it as a coordinate (``rem:descriptor``):

.. math::

   \mathrm{desc}(S) \;=\; \Big(\underbrace{\log\textstyle\sum_\sigma a_\sigma}_{\text{infiltration}},\;
   \underbrace{-\log \textstyle\sum_\sigma w_\sigma^2}_{\log n_{\text{eff}}},\;
   \underbrace{\textstyle\sum_\sigma w_\sigma^2}_{\text{clonality}},\;
   \underbrace{\Phi_1(S)}_{\text{identity}}\Big) .

Every coordinate is smooth — no integer richness — and every named metric is readable back off the
vector analytically. Smoothness is the requirement that makes the object **simulatable**: fit a density
over a cohort's descriptors and the coordinate distribution is a generative manifold rather than a
scatter of labels.

In-silico evolution
~~~~~~~~~~~~~~~~~~~

Fit a (optionally class-conditional) Gaussian :math:`\mathcal{N}(m,\Sigma)` with Ledoit–Wolf shrinkage
over descriptor vectors. Then "perturb one coordinate and let the rest follow" is not a heuristic but
the Gaussian conditional mean (``prop:evolve``): splitting into the perturbed coordinate :math:`1` and
the rest :math:`2`,

.. math::

   \mathbb{E}\big[x_2 \mid x_1 = m_1 + \delta\big]
   \;=\; m_2 + \Sigma_{21}\Sigma_{11}^{-1}\,\delta ,

so "make this donor hotter" propagates through every coupled coordinate using the cohort's own
covariance. Measured couplings match immunobiology: hotter ⇒ diversity :math:`+0.84`, class-switch
:math:`+0.52`, T-versus-B :math:`-0.63`. The non-linear alternative is a compact conditional DDPM/DDIM
over the same descriptor space with classifier-free guidance, sharing the ``sample(n, condition=…)``
call shape so either generator drops in unchanged.

Covariate-disentangled trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cohort often has a **known covariate** (allele, batch, arm) and an **unknown** progression axis
(days since exposure, severity). The PhenoPath-style model (``eq:phenopath``,
``prop:trajectory-fit``) infers the latter while separating out which channels respond to it
differently by covariate:

.. math::

   Y_{ng} \;=\; c_g \;+\; \alpha_g^\top x_n \;+\; \big(\kappa_g + \gamma_g^\top x_n\big)\,\tau_n
              \;+\; \varepsilon_{ng} ,

with :math:`\tau_n` the trajectory position of sample :math:`n`, :math:`\kappa_g` a channel's baseline
response, and :math:`\gamma_g` the **covariate × trajectory interaction** — the term that answers
"does this channel move differently in carriers". Fitting alternates closed-form per-channel ridge
regressions with a GLS update for :math:`\tau`,

.. math::

   \tau_n \;=\; \frac{\sum_g \beta_{ng}\,\big(Y_{ng} - c_g - \alpha_g^\top x_n\big)/s_g^2}
                     {\sum_g \beta_{ng}^2/s_g^2},
   \qquad \beta_{ng} = \kappa_g + \gamma_g^\top x_n ,

with ARD-style iteratively-reweighted shrinkage on :math:`\gamma` (``rem:phenopath-approx``; a
simplified closed-form approximation to PhenoPath's CAVI inference, not a reimplementation of it).

Sequence reconstruction
~~~~~~~~~~~~~~~~~~~~~~~

The last arrow to reverse is step 1. The junction embedding is compressible: PCA retaining 95% of the
variance suffices for *geometry* (clustering, MMD), while *reconstruction* needs 99% — the
chain-adaptive lesson. A codec learns :math:`\text{seq} \to \text{code} \to \text{seq}` with a geometry
anchor, so the code stays near the true embedding instead of drifting to whatever is easiest to decode:

.. math::

   \mathcal{L} \;=\; \underbrace{\mathcal{L}_{\text{recon}}}_{\text{token cross-entropy}}
   \;+\; \lambda_{\text{embed}}\,\underbrace{\big\|\,\text{code} - \text{PCA}(\varphi(\sigma))\,\big\|^2}
   _{\text{distances preserved}} .

Reconstruction turned out to be **training-data-limited, not architecture-limited**: the same one-shot
decoder goes from exact-match 0.885 at :math:`n = 20`k to 0.941 at 50k to 0.958 at 100k. Levers in
order: more data (free) > more PCs (to ~99% variance) > :math:`K \to \sim 2000` (it saturates;
:math:`K = 10{,}000` *regresses*) > an autoregressive decoder for the last percent.

Note the direction of information flow before using this. The distance-to-prototypes code is an
**expansion** (:math:`\sim 10` kbit against a :math:`\sim 63`-bit sequence), so for archival
losslessness one stores the string; the codec inverse exists for ML and generation — decoding a
*synthetic* point that no observed clonotype occupies. And as in step 1, embeddings are comparable only
if the prototype panel **and** the PCA rotation match, which is why a codec ships as a bundle carrying
both plus a prototype hash, and refuses to load against a different panel.

**Call.** :func:`mir.repertoire.sample_descriptor`, :class:`mir.generate.DescriptorDensity`,
:class:`mir.twin.DonorTwin`, :mod:`mir.ml.diffusion`, :func:`mir.track.fit_exposure_trajectory`,
:mod:`mir.ml.codec`, :mod:`mir.ml.bundle`.


.. _math-invariants:

The contract: which transformation preserves what
-------------------------------------------------

*Each of the three steps bought a property, and each property licensed a class of operation. The
practical consequence is that no transformation of* :math:`\Phi` *is neutral: it keeps some and breaks
others, and which ones decides what may still be computed downstream.*

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
     - ✔ (as a signed contrast)
     - ✘
     - ✔
     - deficient coverage

Two rows deserve to be read twice.

**Per-column standardisation.** It forces every coordinate to unit variance across samples, so a
matrix in which half the rows sit at the origin comes out looking exactly like one in which none do —
it deletes the deficiency it was meant to preserve. This already invalidated one experiment, whose
deficient arm scored 0.6481 against a 0.6179 baseline purely from the rescaling. Any block whose point
is that **magnitude carries information** must be scaled by one global scalar:
``ChannelBuilder.add(..., preserve_magnitude=True)``.

**Centring.** It keeps differences and therefore MMD, but not norms — so Rao's :math:`Q` cannot be read
off a centred or PCA-reduced identity block, and any norm-based quantity has to be recomputed from the
true :math:`\Phi_1`. A related trap follows from step 1's all-positive coordinates: the shared profile
puts *any* two TCREmp-derived vectors at raw cosine :math:`\approx 0.999`, and subtracting each
vector's own scalar mean does **not** fix it (measured: 0.9993 raw, 0.9985 self-centred, :math:`-0.14`
cohort-centred). Only **cohort** centring removes the shared profile, so read a cosine between two
such vectors only after it.


Where to go next
----------------

The spine of this page was: coordinates (:ref:`math-step1`) → a well-defined object
(:ref:`math-step2`) → a comparable vector (:ref:`math-step3`) → what it contains
(:ref:`math-knows`) → the premise that fails and its repair (:ref:`math-deficient`,
:ref:`math-depth`) → what the surviving algebra licenses (:ref:`math-surviving-algebra`) → the three
directions out (:ref:`math-density`, :ref:`math-answers`, :ref:`math-back-to-biology`) → what may be
done to :math:`\Phi` at all (:ref:`math-invariants`).

For the full derivations, the theory appendix of the companion manuscript is the reference, and the
labels used in parentheses throughout this page are its labels. For the empirical record — which arm
won, on which cohort, with which interval — see the benchmark documents in the companion analysis
repository. For the calls themselves, each section ends with them, and the :doc:`API reference <api>`
has the signatures.
