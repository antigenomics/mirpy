Channels
========

A **channel** is a named family of columns emitted in its own units, never passed through the
rotation. The full vocabulary -- including the ``vsig`` channels -- is documented in
`vdjtools' channel reference <https://docs.isalgo.dev/vdjtools/channels.html>`_; this page covers
mirpy's and the reason the split exists.

Why a channel is not rotated
----------------------------

A provenance number that has been mixed with the measurements it was supposed to qualify is no longer
provenance. If ``winsor_frac`` went through a rotation, the coordinate carrying it would also carry
geometry, and a caller could no longer ask "is this row comparable to ours at all?" — which is the
only question that column exists to answer.

The split is by *role*, not by cost: a **measurement of the repertoire** is a raw feature, transformed
and winsorized and rotated and scaled; a **statement about the row itself** is a channel.

The rsig channels
-----------------

Carried rather than rotated because the head-to-head against the statistics half's
coverage-standardised Hill numbers is the result, not a redundancy.

``div`` -- embedding diversity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every one of these is a metric-space analogue of a classical index, and each sees something a
clonotype-counting index cannot: that two clonotypes are one substitution apart. To a Hill number
they are two species, as distinct from each other as from anything else.

All of them come out of the same chunked pass that already computes ``Phi``, so the family costs
accumulators rather than a second embedding.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - channel
     - what it is
   * - ``rsig:div:<L>:rao``
     - Rao quadratic entropy, ``log1p``-stabilised and self-pair corrected. Simpson, with distance.
   * - ``rsig:div:<L>:q_v`` / ``q_j`` / ``q_c``
     - The same restricted to the V, J and junction strides of ``Phi``. The strides are literal
       column offsets, so "how much of this is V-driven" needs no attribution model. The three sum
       to ``rao``.
   * - ``rsig:div:<L>:q_frac_v`` / ``q_frac_j`` / ``q_frac_c``
     - Each stride's share of the total dispersion, in clr coordinates -- a composition of *where*
       the diversity sits.
   * - ``rsig:div:<L>:evenness``
     - ``Q(w) / Q(uniform)``: the clone-size weighting's effect with the composition held fixed.
       Bounded, and much less depth-fragile than richness.
   * - ``rsig:div:<L>:eff_dim``
     - ``exp(H(lambda))`` over the weighted covariance spectrum -- **richness**, as the number of
       independent directions of receptor space occupied. A thousand clones inside one convergent
       cluster occupy few; a thousand unrelated clones occupy many.
   * - ``rsig:div:<L>:eff_dim_pr``
     - ``(sum lambda)^2 / sum lambda^2``, the order-2 version, led by the dominant directions.
   * - ``rsig:div:<L>:q_top`` / ``q_singleton``
     - Rao **inside** a clone-size compartment, weights renormalised within it -- the diversity *of*
       the expanded compartment rather than its share, which ``band`` already carries.
   * - ``rsig:div:<L>:q_ratio_top``
     - ``q_top / q_singleton``.

``disp`` -- displacements between compartments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A displacement is a quantity no index has: *where* a compartment sits rather than how large it is.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - channel
     - what it is
   * - ``rsig:disp:<L>:top_singleton`` / ``cos_top_singleton``
     - Distance and cosine between the expanded and singleton centroids.
   * - ``rsig:disp:IGH:IgG_IgM`` / ``IgA_IgM`` and their cosines
     - The same between isotype compartments -- class-switch geometry.
   * - ``rsig:disp:<L>:norm``
     - ``||Phi||``.

.. warning::

   **Only ever within one locus.** Each locus has its own panel of ``K`` prototype receptors, so
   ``Phi(TRA)`` and ``Phi(TRB)`` are vectors in different spaces and a distance between them is
   arithmetic without a meaning. Cross-locus comparison belongs in ``vsig:pair``, a ratio of
   scalars.

``mask`` -- which loci this donor resolved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - channel
     - loci
     - what it says
   * - ``rsig:mask:<locus>:present``
     - 7
     - Whether the locus had any usable clonotype at all.
   * - ``rsig:mask:<locus>:estimable``
     - 7
     - Whether it cleared ``min_clonotypes``. ``present=1`` with ``estimable=0`` means the locus is
       there and too shallow for its ``div``/``disp`` family to be a measurement.

.. important::

   ``mask`` is a **feature block**, not diagnostics -- which loci a donor resolved is biology. See
   `the vdjtools page <https://docs.isalgo.dev/vdjtools/channels.html#channels-mask>`_ for the
   measurement: adding the presence flags moved an external ROC-AUC from 0.6243 to 0.6676.

   It is also what makes holing the diversity family safe. Rao of a one-clonotype locus is
   arithmetically ``0.0`` and is **not** a diversity measurement -- it is the presence mask in
   different units, and read as a measurement it sat about 29 robust deviations below the 1st
   percentile of the real values. A Cox screen over 261 channels returned a hazard ratio of 739 per
   standard deviation at p = 7e-155 for ``rsig:div:IGH:rao`` on 838 patients -- off a single
   extreme point, against a Spearman correlation with survival of -0.023. The family is now a hole
   below the floor, and ``mask`` says why.

   The *geometry* is still computed there: ``Phi`` is perfectly measurable from three clonotypes
   even when its dispersion is not.

``qc``
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - channel
     - loci
     - what it says
   * - ``rsig:qc:-:winsor_frac``
     - --
     - What fraction of this row's finite values the corpus's bounds clamped. A value near 1.0 means
       this sample does not belong to this corpus -- not that it is unusual.

``depth`` is **not** a channel
------------------------------

``rsig:depth:<locus>:n_eff`` and ``:mass`` are raw features and go through the rotation, even though
they look like provenance. They are measurements of the repertoire: ``n_eff`` is a Hill number of the
clone weights the geometry actually uses, and ``mass`` is the share of the repertoire ever drawn. The
compartment shares are depth-fragile on purpose and these two are the covariates that make that
adjustable, so a rotation that could not see them would be worse, not cleaner.

Reading a row
-------------

.. code-block:: python

   from mir.signature import Corpus, rsig

   corpus = Corpus.load("naive_rsig.npz")
   row = rsig(sample, corpus)

   row["rsig:qc:-:winsor_frac"]    # did the corpus's bounds edit this sample?
   row["rsig:div:TRB:rao"]         # sequence-aware diversity, in its own units
   row["rsig:depth:TRB:n_eff"]     # how many clones are effectively behind this Phi?

Declaring a channel
-------------------

``rsig`` registers its groups and channels into vdjtools' registry at import of
:mod:`mir.signature.signature`, with
:func:`~vdjtools.signature.layout.register_raw` and
:func:`~vdjtools.signature.layout.register_channel`. The contract lives in vdjtools because mirpy
depends on vdjtools and not the reverse; nothing in vdjtools imports ``mir``.

.. code-block:: python

   from mir.signature import channel_columns, channels, raw_groups, support_of

   [g.name for g in raw_groups("rsig")]   # phiv, phij, phic, depth, band, band_igh
   [c.name for c in channels("rsig")]     # ['div', 'disp', 'disp', 'mask', 'qc']
   support_of("rsig:phiv:TRB:P001")       # 'nonneg' -- a mean of distances
   support_of("rsig:band:TRB:top")        # 'real'   -- a clr coordinate

.. note::

   **A channel is pass-through, so adding one does not invalidate a fitted corpus.** The rotation is
   indexed by :func:`~vdjtools.signature.layout.raw_columns` and nothing else, and
   :func:`~vdjtools.signature.corpus.apply` fills every registered channel from the sample. The
   ``div``, ``disp`` and ``mask`` families are therefore emitted by every artifact already
   published, with no refit and no new download.
