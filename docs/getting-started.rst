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

Segment library
---------------

The segment library stores V, D, and J reference segments and is used by
parsers and alignment code.

.. code-block:: python

   from mir.common.segments import SegmentLibrary

   lib = SegmentLibrary.load_default(
       genes={"TRA", "TRB"},
       organisms={"HomoSapiens"},
   )

Clonotypes
----------

Parsers convert rows from tabular repertoire formats into ``ClonotypeAA`` or
``ClonotypeNT`` objects with segment annotations attached.

Repertoires
-----------

A repertoire is an iterable collection of clonotypes together with metadata and
summary counts such as clone number and read depth.

Typical Workflow
================

1. Load a segment library.
2. Parse a file with one of the format-specific parsers.
3. Wrap clonotypes into a ``Repertoire``.
4. Compute repertoire-level summaries such as diversity or segment usage.
5. Move to matching, graph, or embedding utilities if deeper analysis is needed.

Next Steps
==========

* Start with :doc:`examples` for copy-pasteable workflows.
* Browse :doc:`modules` for API details.
