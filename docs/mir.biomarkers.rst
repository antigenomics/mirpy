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

