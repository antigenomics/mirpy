mir.biomarkers package
======================

This package contains repertoire-level biomarker and motif-enrichment tools.

Practical notes
---------------

- ``mir.biomarkers.token_stats`` is the canonical module name for token/k-mer
   enrichment (the previous ``kmer_stats`` name has been retired).
- For GLIPH-style runs on large controls, prefer
   ``extract_gliph_artifacts_batch_from_repertoire(..., build_mappings=False, chunk_size=...)``
   and keep identical ``trim_first``/``trim_last`` settings for sample and control.
- ALICE runs neighborhood search and Pgen in two sequential parallel phases
   (trie search first, then OLGA Pgen) to avoid thread-pool contention.

ALICE vs TCRNET
~~~~~~~~~~~~~~~

Both modules detect antigen-driven CDR3 clusters in T-cell or B-cell repertoires,
but differ in how they estimate the background:

**ALICE** (``mir.biomarkers.alice``) uses OLGA generation probability (Pgen) as
the background: ``λ = N × pgen_1mm`` (Pogorelyy et al. *PLoS Biol.* 2019).

- A 10 M synthetic MC pool estimates ``pgen_1mm`` rapidly; sequences with < 2
  pool matches fall back to OLGA analytical 1mm Pgen.
- The paper's original pool size is 100 M and uses V+J gene matching
  (``match_mode="vj"``).  Our default is ``match_mode="none"`` for speed.
- Use ``pgen_mode="mc"`` for production; ``"exact"`` underestimates λ.
- Default ``match_mode="vj"`` (V+J gene restriction, matching the original paper).

**TCRNET** (``mir.biomarkers.tcrnet``) is a **purely MC-control** algorithm:
no OLGA Pgen is computed.  Neighbor density is compared between sample and a
provided control using a binomial (or beta-binomial) test.

- Works with any control: real repertoire (captures V/J and length biases
  automatically) or a synthetic :class:`~mir.basic.pgen.McPgenPool`.
- When using a *synthetic* control, apply ``q_factor ≈ 3–5`` to correct for
  pre-selection bias (OLGA sequences are at recombination, not post-thymic,
  frequencies).  Estimate Q as ``median(pgen_real / pgen_olga)`` using
  :meth:`~mir.basic.pgen.McPgenPool.build_real` on a real control.
- To fully reproduce the original ALICE paper using TCRNET: use a 100 M
  synthetic pool, ``match_mode="vj"``, ``pvalue_mode="beta-binomial"``, and
  ``q_factor=Q``.
- Swap sample and control to detect **neighbor-depleted** sequences (clones
  present in the control but lost in the sample).

Submodules
----------

mir.biomarkers.token_stats module
---------------------------------

.. automodule:: mir.biomarkers.token_stats
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.tcrnet module
----------------------------

.. automodule:: mir.biomarkers.tcrnet
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.alice module
---------------------------

.. automodule:: mir.biomarkers.alice
   :members:
   :undoc-members:
   :show-inheritance:

mir.biomarkers.gliph module
---------------------------

.. automodule:: mir.biomarkers.gliph
   :members:
   :undoc-members:
   :show-inheritance:

