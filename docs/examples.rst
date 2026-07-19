Examples
========

mirpy ships three runnable `marimo <https://marimo.io>`_ notebooks under ``examples/``. They are
self-contained (they run on the bundled prototypes / test assets — no downloads) and double as
living documentation for the three tiers. Install the extra and open one:

.. code-block:: bash

   pip install "mirpy-lib[examples]"     # marimo, matplotlib, umap-learn
   marimo edit examples/quickstart.py   # interactive; or `marimo run …` for read-only

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Notebook
     - What it shows
   * - ``examples/quickstart.py``
     - Clonotype embedding end-to-end: ``TCREmp.embed`` a VDJdb-style set, PCA-denoise, cluster
       antigen-specific TCRs, and a UMAP coloured by epitope.
   * - ``examples/density.py``
     - Background subtraction (Theory T6): fit a density space, run balloon ``neighbor_enrichment``
       against a P_gen / control background, and pull out the enriched convergent family.
   * - ``examples/theory.py``
     - Reproduces the supplementary results S1–S3 on bundled data — the distance laws
       (Gamma / extreme-value), the D↔d correlation, and prototype-source robustness.

The full benchmark suite (VDJdb Table S1, density, repertoire / TCGA cohorts) and its result docs
live in the companion `2026-mirpy-analysis <https://github.com/antigenomics>`_ repository; this repo
keeps the library, its tests, and these bundled-data examples.
