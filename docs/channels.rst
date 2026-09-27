Channels
========

A **channel** is a named family of columns emitted in its own units, never passed through the
rotation. The full vocabulary — including the thirteen ``vsig`` channels — is documented in
`vdjtools' channel reference <https://docs.isalgo.dev/vdjtools/channels.html>`_; this page covers
mirpy's two and the reason the split exists.

Why a channel is not rotated
----------------------------

A provenance number that has been mixed with the measurements it was supposed to qualify is no longer
provenance. If ``winsor_frac`` went through a rotation, the coordinate carrying it would also carry
geometry, and a caller could no longer ask "is this row comparable to ours at all?" — which is the
only question that column exists to answer.

The split is by *role*, not by cost: a **measurement of the repertoire** is a raw feature, transformed
and winsorized and rotated and scaled; a **statement about the row itself** is a channel.

The two rsig channels
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - channel
     - loci
     - what it says
   * - ``rsig:div:<locus>:rao``
     - 7
     - Rao quadratic entropy in embedding coordinates, ``log1p``-stabilised and self-pair corrected.
       A *sequence-aware* diversity: it sees that two clonotypes are one substitution apart, which no
       Hill number can. Carried rather than rotated because the head-to-head against the statistics
       half's coverage-standardised Hill numbers is the result, not a redundancy.
   * - ``rsig:qc:-:winsor_frac``
     - —
     - What fraction of this row's finite values the corpus's bounds clamped. A value near 1.0 means
       this sample does not belong to this corpus — not that it is unusual.

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
   [c.name for c in channels("rsig")]     # div, qc
   support_of("rsig:phiv:TRB:P001")       # 'nonneg' -- a mean of distances
   support_of("rsig:band:TRB:top")        # 'real'   -- a clr coordinate
