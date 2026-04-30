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

Next Steps
==========

* Start with :doc:`examples` for the full notebook gallery.
* Browse :doc:`modules` for API documentation.
