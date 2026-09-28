Worked examples
===============

Six runnable notebooks under ``examples/``. Each one is a complete piece of work with its data,
its commands and its conclusions in one file, so it can be read start to finish or edited and
re-run against your own samples.

They are `marimo <https://marimo.io>`_ notebooks, which are **plain Python files** with
``@app.cell`` decorators rather than JSON. That means they diff and review like source code, and
they run three ways:

.. code-block:: bash

   pip install 'mirpy-lib[examples]'
   marimo edit examples/signature_pipeline.py    # interactive, cells re-run as you edit
   marimo run  examples/signature_pipeline.py    # read-only app, no code shown
   python      examples/signature_pipeline.py    # plain script; prints, no UI

Data is fetched from Hugging Face on first run and cached under ``examples/.data/``, so a fresh
``pip install`` is enough --- no local paths and no pre-staged files. If you already hold a copy,
drop it under ``./data_dump/`` (gitignored) and it is used instead of downloading.

Start here
----------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Signature pipeline
      :link: https://github.com/antigenomics/mirpy/blob/master/examples/signature_pipeline.py

      **Read this one first.** A directory of AIRR TSVs to one row per sample, joined to your own
      metadata sheet --- the commands, the join, and how to read the result. Runs on 1,764 SRA
      samples. If you want to use the library rather than understand it, this is the whole page
      you need.

   .. grid-item-card:: Quickstart
      :link: https://github.com/antigenomics/mirpy/blob/master/examples/quickstart.py

      Clonotypes to vectors. What it means to place a receptor sequence at a point, what the
      coordinates stand for, and the distance approximation the placement rests on. Clusters
      antigen-specific TCRs and colours a UMAP by epitope.

Going further
-------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Signature internals
      :link: https://github.com/antigenomics/mirpy/blob/master/examples/signature.py

      Why the geometry half is transformed differently from the statistics half --- read live
      from the registry, not typed into the notebook --- and how many principal components
      survive a refit on a disjoint set of donors. The answer is smaller than "90% of the
      variance" suggests.

   .. grid-item-card:: Density and enrichment
      :link: https://github.com/antigenomics/mirpy/blob/master/examples/density.py

      Which receptors have more similar neighbours than a background model predicts. Fits a
      density over sequence space, runs neighbourhood enrichment against a generation-probability
      background, and reads out the convergent family that drives the signal.

   .. grid-item-card:: Trajectories and twins
      :link: https://github.com/antigenomics/mirpy/blob/master/examples/trajectory_and_twin.py

      Tracking one donor's repertoire through time with a covariate-disentangled trajectory,
      fitting a generative model over repertoire states, and drawing a synthetic donor from it ---
      a "what if this donor had been sicker" counterfactual.

   .. grid-item-card:: Theory
      :link: https://github.com/antigenomics/mirpy/blob/master/examples/theory.py

      The results the library rests on, each with the measurement that supports it: the
      distance laws, the correspondence between embedding distance and sequence distance, and
      how much the choice of reference panel matters. Runs on bundled data.

.. note::

   The full benchmark suite --- VDJdb specificity tables, density benchmarks, the repertoire and
   TCGA cohorts --- lives in a companion analysis repository, not here. This repository keeps the
   library, its tests, and these examples.
