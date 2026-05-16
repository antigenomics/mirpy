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

mir.common.filter module
------------------------

.. automodule:: mir.common.filter
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.alleles module
-------------------------

.. automodule:: mir.common.alleles
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.gene\_library module
--------------------------------

.. automodule:: mir.common.gene_library
   :members:
   :undoc-members:
   :show-inheritance:

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

mir.common.single\_cell module
------------------------------

.. automodule:: mir.common.single_cell
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.single\_cell\_parser module
---------------------------------------

.. automodule:: mir.common.single_cell_parser
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.single\_cell\_repair module
---------------------------------------

.. automodule:: mir.common.single_cell_repair
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.single\_cell\_util module
-------------------------------------

.. automodule:: mir.common.single_cell_util
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.sampling module
--------------------------

.. automodule:: mir.common.sampling
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.pool module
----------------------

.. automodule:: mir.common.pool
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.control module
-------------------------

.. automodule:: mir.common.control
   :members:
   :undoc-members:
   :show-inheritance:

mir.common.io\_parallel module
------------------------------

.. automodule:: mir.common.io_parallel
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Default And Fallback Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Default mode uses parallel parsing with 4 workers.
- Sequential fallback is used when any of these are true:

   - n_jobs is set to 1.
   - Parsed row count is below 10,000 (parallel_min_rows default).
   - The file fits in one chunk (n_rows <= chunk_size).

- Practical estimate for typical AIRR tables:

   - Small to medium AIRR files (~3,000 rows at ~0.07 MB gz) represent approximately
     43,000 rows per MB gz for similarly narrow AIRR tables.
   - Under this approximation, 10,000 rows corresponds to roughly 0.23 MB gz.

- Rule of thumb:

   - If a gzipped AIRR file is substantially below about 0.23 MB, sequential loading is typically chosen.
   - If it is above about 0.23 MB, parallel loading is typically beneficial and selected by default.

mir.common.repertoire\_dataset module
--------------------------------------

.. automodule:: mir.common.repertoire_dataset
   :members:
   :undoc-members:
   :show-inheritance:

TSV And Parquet I/O Layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repertoire classes provide Polars-first TSV/Parquet I/O helpers with
roundtrip-safe schemas:

- LocusRepertoire:

   - to_tsv / from_tsv
   - to_parquet / from_parquet

- SampleRepertoire:

   - single-file: one TSV/Parquet with a locus column
   - split-loci: one file per locus via split_loci=True

- RepertoireDataset:

   - per_sample_locus layout: one file per sample and locus
   - single_file layout: one combined file with sample_id and locus columns
      plus separate metadata.tsv

All dataset loaders operate with worker tasks on individual samples.

Module contents
---------------

.. automodule:: mir.common
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
