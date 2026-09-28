Glossary
========

Terms as this documentation uses them. The repertoire-immunology vocabulary -- clonotype, junction,
locus, Pgen -- is defined in
`vdjtools' glossary <https://docs.isalgo.dev/vdjtools/glossary.html>`_; this page covers what mirpy
adds on top.

.. glossary::
   :sorted:

   ALICE
      Neighbourhood enrichment measured against a V(D)J **generation model**: a receptor with more
      similar neighbours than its generation probability predicts is a candidate for antigen-driven
      expansion. Contrast :term:`TCRNET`, which uses a control repertoire instead. In
      :mod:`mir.density` both are the same test with a different background.

   balloon estimator
      A density estimator whose bandwidth adapts per query point -- it grows the neighbourhood until
      a target count is reached, rather than fixing a radius. Used in :mod:`mir.density` because
      receptor density varies over orders of magnitude across the embedding space, so a fixed radius
      is either empty in sparse regions or saturated in dense ones.

   basis
      The fitted coordinate system a set of repertoire vectors lives in: the PCA rotation plus the
      random features. Fitted **once** per cohort by ``fit_repertoire_space``. Two samples embedded
      through different bases are not comparable, and the containers here refuse to mix them rather
      than doing it silently.

   clone-size transform
      The function ``g`` turning a clone's abundance into its weight in a repertoire vector, with
      weights normalised to sum to one. ``log2p1``, the default, is concave, so one hyperexpanded
      clone cannot dominate the fingerprint; ``duplicate_count`` is linear; ``distinct`` ignores
      abundance entirely. The choice is part of the definition of the vector, so it must match across
      a cohort.

   codec
      A neural encoder-decoder over the embedding space (:mod:`mir.ml`, ``[ml]`` extra). Forward maps
      a sequence to a vector, inverse recovers a sequence from a vector. Reconstruction needs more
      PCA components than clustering does -- the preset's 99-percent dimension rather than its
      95-percent one -- because sequence detail lives in the low-variance directions.

   coordinate system
      What the prototype embedding produces, and the reason the word matters: distances between
      points are meaningful, so any method that consumes vectors works, without needing a sequence
      alignment of its own. A :term:`replicate` is a *different* coordinate system, not another
      sample of the same one.

   corpus
      A fitted reference artifact for :term:`signature` columns -- winsorization bounds, a per-locus
      rotation, and per-component centre and scale, all estimated in one pass over many repertoires.
      Shared with vdjtools, which fits it. See :doc:`signature`.

   digital donor
      One donor's repertoires across several chains, fused into a single hash-verified matrix row
      (:mod:`mir.cohort`). The unit a cohort-level model is fitted on, and the level at which batch
      residualisation applies.

   digital twin
      A perturbed or resampled copy of one donor's state, produced by pushing that donor's vector
      through a generator (:mod:`mir.twin`). Used to ask what would change if one aspect of a
      repertoire were different.

   Fisher vector
      The second-moment block of a repertoire vector: it records how pairs of embedding directions
      co-occur within a donor, which is where HLA-linked public structure shows up. The kernel mean
      alone cannot see it, because a mean is blind to correlation among the points it averages.

   Hill diversity profile
      A block of the repertoire vector carrying coverage-standardised
      `Hill numbers <https://docs.isalgo.dev/vdjtools/glossary.html>`_ at several orders, so the
      vector knows how even the repertoire is and not only where its mass sits.

   kernel mean
      The main block of a repertoire vector: the average of a random-feature map over the
      receptors of a repertoire, weighted by :term:`clone-size transform`. It sketches the whole
      empirical distribution of receptors in one fixed-length vector, with no codebook and no
      clustering step. Its norm is Rao's quadratic entropy.

   MMD
      Maximum mean discrepancy: the distance between two repertoires, computed as the norm of the
      difference of their :term:`kernel mean` s. The **unbiased** estimator removes the
      self-similarity term analytically and is the one to use whenever samples differ in depth or
      diversity -- the biased estimator carries a positive term of about ``1/n_eff``, which inflates
      distances for low-diversity samples and shows up as signal with the wrong sign if diversity is
      what you are studying. It can return an exact zero for two samples from the same distribution,
      which is correct rather than a rounding artefact.

   motif witness
      A clonotype, or a set of them, identified as separating two groups of repertoires -- the
      readout that turns a significant distance back into receptors you can look at.
      ``class_witness`` in :mod:`mir.repertoire`.

   n_eff
      The effective number of clonotypes in a repertoire under its weighting -- the reciprocal of the
      sum of squared weights. What the :term:`MMD` self-term is scaled by, and what makes a
      single-clonotype sample degenerate: the unbiased estimator is undefined there and raises rather
      than returning a number.

   prototype
      One of a fixed set of real receptor sequences that define the :term:`coordinate system`. A
      receptor is embedded as its vector of distances to every prototype. mirpy bundles 10,000 real,
      unique, productive, germline-resolvable receptors per chain, sampled at a fixed seed from
      arda-annotated repertoires. Real rather than model-generated, because synthetic junctions have
      degenerate lengths and embed measurably worse.

   prototype hash
      A fingerprint of the prototype set, including its :term:`replicate` index, stored in every
      container that holds embeddings. What makes mixing two coordinate systems an error you see
      rather than a result you do not.

   replicate
      A disjoint block of ``n`` prototypes from the bundled pool -- ten of them at ``n=1000``, five at
      ``n=2000``. Because the file order is itself a uniform shuffle, each block is an independent
      draw from the same pool, so comparing a result across replicates measures how much it depends
      on the draw. **Distances across two replicates are not comparable**: compare summary
      statistics such as AUC or cluster count, never raw embeddings. Sweeping ``n_prototypes`` is a
      different question, because those draws are nested.

   RFF
      Random Fourier features. A finite-dimensional map whose inner products approximate a kernel, so
      that a kernel mean can be stored and compared as an ordinary vector. What makes the repertoire
      embedding fixed-length and codebook-free.

   signature
      One fixed-width, named, already-standardised feature vector per repertoire, standardised
      against a published :term:`corpus` so that two people computing it independently get the same
      coordinate system. ``rsig`` is mirpy's geometry half, ``vsig`` is vdjtools' statistics half,
      and they join on ``sample_id``. Distinct from ``Phi(S)``, which is fitted on your own cohort
      and therefore not portable. See :doc:`signature`.

   sub-probability embedding
      A repertoire vector that records a deficient total mass instead of renormalising to one, for a
      sample that was never fully observed. The missing mass is estimated by Good-Turing or Chao, and
      the signed contrast against a naive reference is ``Psi = mass * (Phi - naive)``. Asserting full
      confidence in a partially observed sample is the error this exists to avoid.

   TCRNET
      Neighbourhood enrichment measured against a **control repertoire**. Contrast :term:`ALICE`,
      which uses a generation model. Prefer a biological control when you have one -- pre- against
      post-vaccination, patient against healthy -- because differential enrichment cancels generic
      public convergence and isolates the antigen-specific response.

   TCREMP
      The embedding this library implements: represent a receptor by its alignment distances to a
      fixed set of :term:`prototype` s, so that Euclidean distance in the resulting space
      approximates pairwise alignment distance.

   water level
      The baseline neighbour density a naive repertoire produces simply because receptors with high
      generation probability have many near neighbours by chance. Enrichment testing has to calibrate
      it away, or every public receptor looks antigen-driven.
