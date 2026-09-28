Repertoire signatures
=====================

A *signature* is a fixed-order, name-addressed feature vector for one repertoire. ``mirpy`` emits the
**geometry** half, ``rsig``; the **statistics** half, ``vsig``, comes from `vdjtools
<https://github.com/antigenomics/vdjtools>`_ and shares the same contract, the same corpus machinery
and the same column grammar.

Everything in this half rests on one object:

.. math::

   \Phi(S) = \sum_\sigma w_\sigma\, z_\sigma

``z_\sigma`` is a clonotype's vector of distances to a fixed, bundled prototype panel — ``K = 256``
receptors per locus, embedded by germline V/J distance plus junction gapblock alignment — and
``w_\sigma`` is its normalised clone weight. ``TCREmp.embed`` interleaves the three components per
prototype as ``[V, J, junction]``, so ``Φ[0::3]``, ``Φ[1::3]`` and ``Φ[2::3]`` are the exact V / J /
junction slots. Literal column strides, not an attribution model, which is what makes "how much of
this distance is V?" answerable without SHAP, sampling or a surrogate.

Quickstart
----------

.. code-block:: bash

   # four corpora ship with the wheel; name one, no build and no cohort needed
   mir signature --corpus synthetic-blood cohort/*.tsv.gz -o rsig.tsv

   # or build your own -- still uses no samples from anybody's cohort
   mir corpus --corpus naive --smoke -o naive_rsig.npz

   mir signature --corpus naive_rsig.npz cohort/*.tsv.gz -o rsig.tsv
   mir signature --corpus naive_rsig.npz --components 32 --describe

.. code-block:: python

   from mir.signature import rsig, rsig_cohort, Corpus

   corpus = Corpus.load("naive_rsig.npz")
   row = rsig({"TRB": trb, "IGH": igh}, corpus)
   frame = rsig_cohort({"S1": s1, "S2": s2}, corpus, n_jobs=0)

Joining the two halves
----------------------

There is **no joined entry point**, deliberately. Each half has its own artifact, so the wrapper's
only real job — applying one scale reference over both — no longer exists. Two calls and a polars
join is the whole story:

.. code-block:: python

   from mir.signature import rsig_cohort
   from vdjtools.signature import vsig_cohort
   from vdjtools.signature.corpus import Corpus

   v = vsig_cohort(samples, Corpus.load("vsig_naive.npz"))
   r = rsig_cohort(samples, Corpus.load("rsig_naive.npz"))
   full = v.join(r, on="sample_id", how="inner")

Use the **same corpus name and seed** for both halves. ``vdjtools corpus`` and ``mir corpus`` draw
the same synthetic repertoires from the same seeds, which is what makes the join meaningful rather
than merely type-correct.

What changed in 4.0, and why
----------------------------

.. important::

   **The old rotation was fitted on the wrong unit.** ``build_rsig.py`` fitted ``R_V``, ``R_J`` and
   ``R_C`` on ``(10_000, 768)`` — one row per **clonotype**, from the bundled prototype panel — while
   every one of the 399 PC columns it produced was a **repertoire** statistic, obtained by projecting
   a clone-weighted *mean* through those axes. PCA over 10,000 receptors evaluated at a mean is not
   PCA of repertoires: the two maximise variance in different units and give different axes. A
   sample's coordinate averages ~400 effective clones, so variance between samples along those axes
   is of order :math:`1/\sqrt{400}` of the variance being maximised.

   The artifact's ``centre`` and ``scale`` were fitted on **zero rows** and arrived from a separate
   corpus of real samples. Two independent fits, stitched — which is how one shipped reference came
   to pair a centre of exactly ``0.0`` with a scale plainly fitted from data, putting a
   corpus-typical sample **81 robust deviations** out, with 885 of 885 samples of an independent
   cohort outside the bound.

   Now the rotation, the bounds, the centre and the scale all come out of **one pass over one matrix
   of repertoires**, per locus.

**There is no ``contrast`` group any more, and nothing was lost.** It was ``Ψ = mass·(Φ − naive)``,
with ``naive`` a separately drawn 20,000-sequence reference, because the rotation was fit-free and
needed an explicit subtraction point. The corpus centre now *is* that point: rotating through the
``naive`` corpus subtracts the median ``Φ`` of unselected repertoires, which is what the contrast
measured. 231 columns, one frozen vector, and one whole failure mode — a ``naive`` drawn against a
different release of the recombination models, which moved every contrast column by 0.1–1.6% per
locus — replaced by choosing a corpus. ``mass`` remains a feature in its own right, so the rotation
still sees it.

Raw feature groups
------------------

Every group at a locus is concatenated into one feature vector and rotated together.

.. list-table::
   :header-rows: 1
   :widths: 14 12 74

   * - group
     - width
     - what
   * - ``phiv``
     - 256
     - V-germline slot of ``Φ``: distance from each clonotype's V gene to each prototype's, clone-weighted
   * - ``phij``
     - 256
     - J-germline slot
   * - ``phic``
     - 256
     - Junction slot — the gapblock alignment distance
   * - ``depth``
     - 2
     - ``n_eff = 1/Σw²`` and ``mass = 1 − M₀``. A Hill number *of the weights the geometry actually
       uses*, so it predicts how noisy this sample's ``Φ`` is
   * - ``band``
     - 2
     - Clone-size compartment shares of ``Φ``, in clr coordinates
   * - ``band_igh``
     - 3
     - IGH only: isotype shares of ``Φ(IGH)``

``p_L`` is 772 per locus, 775 at IGH.

.. note::

   ``rsig:phiv`` is **not V-gene usage**, and should not be described as such. It is the clone-weighted
   mean of the clonotypes' V-germline-similarity profiles: it encodes usage only softly — a sample
   dominated by one V sits near that V's row in germline space — and it is not a histogram over V
   genes. Explicit V and J usage are ``vsig`` features.

The compartment shares are exact, not fitted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Φ`` is linear in the clone-weight measure and the compartments *partition* the clonotypes, so

.. math::

   \Phi(S) = \sum_c \pi_c\, \Phi(c), \qquad \pi_c = \sum_{\sigma \in c} w_\sigma

holds exactly and the shares are read off the weights in closed form. Solving a non-negative least
squares for the same quantity is both slower and worse posed: over overlapping compartments the
weights need not sum to one and one share can exceed it, which breaks every log-ratio coordinate
downstream.

A compartment below ``min_clonotypes`` is recorded **absent** — dropped from the composition — rather
than set to zero. Zero is a measurement; absent is not, and a clr cannot tell them apart afterwards.
When *every* compartment falls below the floor only the closing residual is left, which is no
composition at all, so the coordinates are holes rather than an invented ratio.

.. warning::

   The compartment shares are **depth-fragile and deliberately uncorrected**. A compartment's share
   genuinely moves with sequencing depth: the singleton fraction grows as rarer clones are sampled,
   and a 1% quantile selects 20 clonotypes in a 2,000-clonotype sample against 1,000 in a
   100,000-clonotype one. Measured on one repertoire across a 67x depth range, ``band:top`` spans
   about 6.9 in log-ratio coordinates. Bounding the quantile to a clonotype count was tried and
   merely relocated the discontinuity.

   The answer to a depth-fragile column is to **carry the covariate**, not to correct it — which is
   why ``depth`` is in the rotation and ``cov:*:cstar`` is a channel on the other half.

Channels
--------

Never rotated, never clamped. See :doc:`channels`.

* ``rsig:div:<locus>:rao`` — Rao quadratic entropy in embedding coordinates, self-pair corrected. It
  sees that two clonotypes are one substitution apart, which no Hill number can, and it telescopes
  out of the same chunked pass that computes ``Φ``: ``Q = 2(Σw‖z‖² − ‖Φ‖²)``. Carried in its own
  units because the head-to-head against the statistics half's Hill numbers is the point.
* ``rsig:qc:-:winsor_frac`` — what fraction of this row the corpus's bounds clamped.

Supports, and why ``Φ`` is one-sided
------------------------------------

Every ``Φ`` coordinate is a weighted mean of distances, so it is non-negative and its tail runs
upward only: the slots declare ``support="nonneg"`` and only their **top** tail is trimmed. ``n_eff``
is a count, likewise ``nonneg``. The compartment shares are clr coordinates of a composition and are
therefore two-sided, ``real``. The full support table is in `vdjtools' signature reference
<https://docs.isalgo.dev/vdjtools/signature.html#winsorization-by-percentile-and-one-sided-where-the-metric-is>`_ — one table for both halves, since one module applies it.

Building a corpus
-----------------

.. code-block:: bash

   mir corpus --corpus synthetic-blood  -o synthetic-blood_rsig.npz
   mir corpus --corpus synthetic-tissue -o synthetic-tissue_rsig.npz
   mir corpus --corpus naive  -o naive_rsig.npz
   mir corpus --corpus memory --size n_eff --components 0.95 -o memory_rsig.npz
   mir corpus --smoke -o /tmp/smoke.npz
   mir corpus --corpus naive -j 8 -o naive_rsig.npz     # 8 worker processes

Four corpora ship, all synthetic. ``synthetic-blood`` and ``synthetic-tissue`` are the ones to reach
for: each repertoire is a naive/memory **mixture** drawn from three quantile ladders measured per
locus on the cohort it is named after -- richness, reads per expanded clone, and the singleton
fraction that stands in for the naive compartment -- through that cohort's measured rank
correlations. ``naive`` and ``memory`` are the pure regimes, and neither varies what a cohort varies:
``memory`` at a fixed size has the same read count in every sample. The construction, the
acceptance table and why reads per *expanded* clone is the drawable quantity are in `the vdjtools
half <https://docs.isalgo.dev/vdjtools/signature-methods.html#the-two-cohort-corpora-three-measured-ladders-per-locus>`_,
since one module draws for both.

Both halves must be built from the **same corpus name and seed**: they resolve the name through one
``corpus_plan`` and write the manifest through one ``corpus_meta``, so the cohort, the three ladders,
the rank correlations and the germline fingerprint cannot disagree between ``vsig_<name>`` and
``rsig_<name>`` -- which is what makes joining them on ``sample_id`` meaningful.

The build runs across worker **processes**, one contiguous block of samples each; ``-j/--jobs``
defaults to every core the process may use (``-j 1`` stays in-process). Both halves share one
builder, so a sample is the same repertoire in either -- its generator is seeded from
``(seed, locus, index)``, which is also why the artifact is identical at every ``--jobs``.

.. note::

   ``--jobs`` is worker **processes**; ``POLARS_MAX_THREADS`` / ``OMP_NUM_THREADS`` are **kernel
   threads**. mirpy shipped a ``--threads`` wired straight to the process count for several
   releases -- 16 processes each claiming 16 threads -- so the distinction is stated wherever a
   concurrency flag appears.

The build is a deterministic function of ``(corpus, loci, samples, size, seed, source)`` and
vdjtools' bundled recombination models, and is required to be byte-identical across processes,
worker counts and thread counts:

.. code-block:: bash

   mir corpus --corpus naive --loci TRG,TRD --samples 24 --size 80 -o /tmp/a.npz
   OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
     mir corpus --corpus naive --loci TRG,TRD --samples 24 --size 80 -o /tmp/b.npz
   cmp /tmp/a.npz /tmp/b.npz          # must be identical

Measured on a small TRG/TRD corpus, 5 components reach **0.91–0.93** of the variance — far more
compressible than the statistics half, where the same count reaches 0.34–0.48, because the 256
``Φ`` coordinates are highly correlated distances to one panel while the ``vsig`` groups are
genuinely heterogeneous.

Traps
-----

* **The SHM columns must never reach the embedder.** ``v_identity`` and ``v_mutations`` silently
  switch ``TCREmp.embed`` to SHM-aware V distances, which is a different coordinate system under the
  same column names — the numbers move and nothing says so. They are dropped before embedding.
* **``chunk`` bounds memory, not the answer.** ``Φ`` and the Rao accumulator are running sums, so the
  full ``(n, 3K)`` matrix is never held and the result is chunk-independent.
* **``--jobs`` is processes.** The embedder already threads inside one sample, so a pool worker
  deliberately takes one kernel thread. ``n_jobs=1`` is therefore **not** serial, and four workers
  measured 0.79x one in-process pass — which is why the parallelism test checks *where the work ran*
  (by PID) rather than how long it took.
* **A pool that cannot start raises.** It does not fall back to one process: a correctness-preserving
  fallback turned a dead pool into a merely slow one and hid a 20x regression here for months.

API
---

.. automodule:: mir.signature
   :members:
   :imported-members:

.. automodule:: mir.signature.features
   :members:

.. automodule:: mir.signature.signature
   :members:
