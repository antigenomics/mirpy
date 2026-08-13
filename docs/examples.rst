Examples
========

mirpy ships six runnable `marimo <https://marimo.io>`_ notebooks under ``examples/``. They are
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
       antigen-specific TCRs, and a UMAP coloured by epitope. The epitope colouring and the F1
       table need a local VDJdb slim dump at ``tests/assets/vdjdb.slim.txt.gz`` (gitignored — see
       ``SOURCES.md``); without it the notebook falls back to the bundled prototypes and colours
       by CDR3 length.
   * - ``examples/density.py``
     - Background subtraction (Theory T6): fit a density space, run balloon ``neighbor_enrichment``
       against a P_gen / control background, and pull out the enriched convergent family.
   * - ``examples/theory.py``
     - Reproduces the supplementary results S1–S3 on bundled data — the distance laws
       (Gamma / extreme-value), the D↔d correlation, and prototype-source robustness.
   * - ``examples/trajectory_and_twin.py``
     - Exposure trajectory, generative loop, digital twin: fit a PhenoPath-style covariate-
       disentangled trajectory (``mir.track``) on a synthetic cohort with a known covariate and a
       planted severity axis, fit a generative density over ``RepertoireDescriptor`` vectors
       (``mir.generate``), perturb one donor ("what if hotter") and draw new synthetic donor states
       via the digital-twin glue (``mir.twin``).
   * - ``examples/feature_vectors.py``
     - **Start here to turn repertoires into a model-ready table.** Pick a ranked preset, see
       exactly what it contains and why, run it over a whole dataset in parallel, and check the
       result against the ``nuisance`` control — if a model on depth/masks/QC alone scores as well,
       the real model is reading library prep.
   * - ``examples/signature.py``
     - The portable signature's geometry half: the live transform table read from the registry,
       why ``rsig`` coordinates take ``transform="none"`` where ``vsig`` proportions take arcsine
       and logit, why ``contrast`` is a ``magnitude`` block that is never centred, and a runnable
       rank-selection measurement — on donors drawn from one generative model, twelve components
       carry ~59% of the variance and *none* of them reproduces across a group-disjoint refit.

The full benchmark suite (VDJdb Table S1, density, repertoire / TCGA cohorts) and its result docs
live in the companion `2026-mirpy-analysis <https://github.com/antigenomics>`_ repository; this repo
keeps the library, its tests, and these bundled-data examples.
