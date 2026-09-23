Notebooks
=========

Worked examples as `marimo <https://marimo.io>`_ notebooks --- **plain Python files** with
``@app.cell`` decorators, so they diff and review like source rather than like JSON, and they run
three ways:

.. code-block:: bash

   pip install 'mirpy-lib[examples]'
   marimo edit examples/signature_pipeline.py    # interactive
   marimo run  examples/signature_pipeline.py    # read-only app
   python      examples/signature_pipeline.py    # plain script; prints, no UI

Each one bootstraps its own data from Hugging Face and caches it under ``examples/.data/``, so a
fresh ``pip install`` is enough --- no local paths and no pre-staged files. Drop a copy under
``./data_dump/`` (gitignored) and it is used instead of downloading.

Start here
----------

.. raw:: html

   <div class="proj-card-grid">
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/signature_pipeline.py">
       <h3>Signature pipeline</h3>
       <p>A folder of AIRR TSVs to one table joined with your metadata, in one command. Runs on
       1,764 SRA samples from <code>isalgo/airr_benchmark</code>. Read this one first.</p>
     </a>
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/quickstart.py">
       <h3>Quickstart</h3>
       <p>Clonotypes to vectors: the prototype embedding, what the coordinates mean, and the
       distance approximation it rests on.</p>
     </a>
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/signature.py">
       <h3>Signature internals</h3>
       <p>Why the geometry half is transformed differently from the statistics half, and how many
       principal components survive a group-disjoint refit.</p>
     </a>
   </div>

Going further
-------------

.. raw:: html

   <div class="proj-card-grid">
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/feature_vectors.py">
       <h3>Feature vectors</h3>
       <p>The repertoire-level embedding in full: bases, weights, and what varies between donors.</p>
     </a>
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/density.py">
       <h3>Density and enrichment</h3>
       <p>Neighbourhood density, TCRnet-style enrichment, and reading out what drives a signal.</p>
     </a>
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/trajectory_and_twin.py">
       <h3>Trajectories and twins</h3>
       <p>Tracking a repertoire through time, and comparing donors who should look alike.</p>
     </a>
     <a class="proj-card" href="https://github.com/antigenomics/mirpy/blob/master/examples/theory.py">
       <h3>Theory</h3>
       <p>The results T1-T7 the library rests on, each with the measurement that supports it.</p>
     </a>
   </div>
