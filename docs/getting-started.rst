Getting Started
===============

Installation
============

``mirpy`` targets Python 3.11+ and includes a compiled extension for distance
calculations, so a working C/C++ toolchain is recommended when installing from
source.

.. code-block:: bash

   pip install .

For editable development installs:

.. code-block:: bash

   pip install -e .

Core Concepts
=============

mirpy works with a small set of core data abstractions.

Clonotype
---------

The smallest unit is a clonotype. Parsers convert rows from tabular repertoire
formats into ``ClonotypeAA`` or ``ClonotypeNT`` objects with sequence and
annotation fields attached.

Repertoire
----------

A repertoire is a collection of clonotypes from one sample together with
metadata and summary counts such as ``clonotype_count`` and
``duplicate_count``.

Internally, repertoire data may remain as clonotype objects, lazy parser
columns, or a Polars table. Tabular conversion is lazy for list-based
construction and only materializes when needed.

RepertoireDataset
-----------------

A repertoire dataset is a collection of repertoires loaded together for
multi-sample analysis, comparison, resampling, and cohort-level workflows.

ClonotypeDataset
----------------

``ClonotypeDataset`` is an additional dataset-level abstraction used when the
analysis is centered on clonotypes themselves rather than on repertoire objects,
for example in matching, biomarker, or graph-oriented workflows.

Typical Workflow
================

1. Parse a file with one of the supported repertoire parsers.
2. Wrap clonotypes into a ``Repertoire`` or ``RepertoireDataset``.
3. Compute repertoire-level summaries such as diversity or segment usage.
4. Move to matching, graph, or embedding utilities if deeper analysis is needed.

For batch-corrected gene usage workflows, use
``compute_batch_corrected_gene_usage`` and consume ``pfinal`` as the corrected
probability output (already normalized per sample/locus).

If you compute only ``scope='vj'`` and need V- or J-marginal views for plots,
reuse ``marginalize_batch_corrected_gene_usage(..., scope='v'|'j')`` from the
same module instead of ad-hoc notebook ``groupby`` code.

Pooling Repertoires Across Samples
==================================

Use ``pool_samples`` to combine clonotypes across samples with explicit
identity rules.

.. code-block:: python

   from mir.common.pool import pool_samples

   # Pool two samples by nucleotide CDR3 + V/J genes.
   pooled = pool_samples([sample_rep_1, sample_rep_2], rule="ntvj", weighted=True)

   # Pool a dataset by amino-acid CDR3 + V/J genes and keep sample ids per pooled clone.
   pooled_ds = pool_samples(dataset, rule="aavj", include_sample_ids=True)

Supported pooling rules:

* ``ntvj``: key ``(junction, v_gene, j_gene)``
* ``nt``: key ``(junction,)``
* ``aavj``: key ``(junction_aa, v_gene, j_gene)``
* ``aa``: key ``(junction_aa,)``

Each pooled clonotype stores:

* ``duplicate_count`` as the sum over grouped clonotypes,
* ``incidence`` in clonotype metadata (number of unique samples containing the key),
* ``occurrences`` in clonotype metadata (number of grouped rows).

Neighborhood Enrichment and Clonotype Similarity
=================================================

Use ``compute_neighborhood_stats`` to find clonotypes similar to each other
based on edit distance in the CDR3 junction region. This is useful for
TCRnet and ALICE algorithms.

.. code-block:: python

   from mir.graph import compute_neighborhood_stats

   # Count neighbors for each clonotype within edit distance 1
   stats = compute_neighborhood_stats(
       repertoire,
       metric="hamming",
       threshold=1,
       match_v_gene=True,
   )

   # stats["clonotype_id"] = {
   #     "neighbor_count": 15,
   #     "potential_neighbors": 200,
   # }

Supported options:

* ``metric``: ``"hamming"`` or ``"levenshtein"`` for junction_aa comparison
* ``threshold``: Maximum edit distance to consider a clonotype a neighbor
* ``match_v_gene``: If True, only count neighbors with matching V gene
* ``match_j_gene``: If True, only count neighbors with matching J gene

You can compute neighborhood stats against an explicit background repertoire:

.. code-block:: python

   # Query against background (adds +1 pseudocount in background mode)
   bg_stats = compute_neighborhood_stats(
      repertoire,
      background=background_repertoire,
      metric="hamming",
      threshold=1,
   )

To attach parent-vs-background neighborhood enrichment metadata in one call:

.. code-block:: python

   from mir.graph import add_neighborhood_enrichment_metadata

   add_neighborhood_enrichment_metadata(
      repertoire,
      background=background_repertoire,
      metric="hamming",
      threshold=1,
      metadata_prefix="neighborhood",
   )

This writes parent/background counts, potentials, densities, and
``neighborhood_enrichment`` for each clonotype.

Control Data Setup (Synthetic / Real)
=====================================

Background controls are expensive to build/download and are managed explicitly
through ``mir.common.control``.

.. code-block:: python

   from mir.common.control import ControlManager

   mgr = ControlManager()  # default: ~/.cache/mirpy/controls (or MIRPY_CONTROL_DIR)

   # Build synthetic OLGA control (default n=10_000_000)
   mgr.ensure_synthetic_control("human", "TRB", n=1_000_000)

   # Download real control from HuggingFace dataset and convert to pickle
   mgr.ensure_real_control("hsa", "Tbeta")

   # Load normalized ntvj table (duplicate_count, junction, junction_aa, v_gene, j_gene)
   df_control = mgr.load_control_df("synthetic", "human", "TRB")

   # Or build/fetch on demand when a workflow needs a control immediately
   df_real = mgr.ensure_and_load_control_df("real", "human", "TRB")

You can also prebuild controls via CLI:

.. code-block:: bash

   mirpy-control-setup --type synthetic --species human,mouse --loci TRA,TRB --n 1000000

Benchmark coverage includes both synthetic generation and real-control
download/build paths (HuggingFace), with cache-hit timing diagnostics in
``tests/test_control_benchmark.py``.

Available aliases include species ``human/hsa/HomoSapiens`` and
``mouse/mmu/MusMusculus``; loci aliases include IMGT names and forms such as
``Talpha``/``Tbeta``.

Control setup is concurrency-safe: when multiple workers (for example GNU
Parallel or Slurm jobs) request the same control simultaneously, one process
builds while others wait on a per-control lock and then reuse the produced
artifact.

You can also add neighborhood stats directly to clonotype metadata:

.. code-block:: python

   from mir.graph import add_neighborhood_metadata

   add_neighborhood_metadata(repertoire, metric="hamming", threshold=1)
   # Adds neighborhood_count and neighborhood_potential to each clonotype's metadata

Next Steps
==========

* Start with :doc:`examples` for the full notebook gallery.
* Browse :doc:`modules` for API documentation.
