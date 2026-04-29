mir.common package
==================

Submodules
----------

mir.common.clonotype module
---------------------------

.. automodule:: mir.common.clonotype
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.clonotype_dataset module
-----------------------------------

.. automodule:: mir.common.clonotype_dataset
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.filter module
------------------------

.. automodule:: mir.common.filter
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.io_parallel module
-----------------------------

.. automodule:: mir.common.io_parallel
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.repertoire_dataset module
-----------------------------------

.. automodule:: mir.common.repertoire_dataset
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.normalize_gene_usage module
-------------------------------------

.. automodule:: mir.common.normalize_gene_usage
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Default And Fallback Policy
------------------------------------

- Default mode uses parallel parsing with 4 workers.
- Sequential fallback is used when any of these are true:

   - n_jobs is set to 1.
   - Parsed row count is below 10,000 (parallel_min_rows default).
   - The file fits in one chunk (n_rows <= chunk_size).

- Practical estimate from bundled sample files:

   - tests/assets/yfv_s1_d0_f1.airr.tsv.gz is about 3,000 rows at about 0.07 MB gz.
   - tests/assets/yfv_s1_d15_f1.airr.tsv.gz is about 3,000 rows at about 0.07 MB gz.
   - This is approximately 43,000 rows per MB gz for similarly narrow AIRR tables.
   - Under this approximation, 10,000 rows corresponds to roughly 0.23 MB gz.

- Rule of thumb:

   - If a gzipped AIRR file is substantially below about 0.23 MB, sequential loading is typically chosen.
   - If it is above about 0.23 MB, parallel loading is typically beneficial and selected by default.

mir.common.parser module
------------------------

.. automodule:: mir.common.parser
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.repertoire module
----------------------------

.. automodule:: mir.common.repertoire
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.segments module
--------------------------

.. automodule:: mir.common.segments
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: mir.common
   :members:
   :undoc-members:
   :show-inheritance:
