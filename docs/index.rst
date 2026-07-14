mirpy
=====

.. raw:: html

   <div class="proj-intro">
     <div>
       <p class="proj-intro__eyebrow">mirpy-lib &middot; import <code>mir</code></p>
       <p class="proj-intro__lead">ML-oriented embeddings for immune receptor repertoires (TCR/BCR):
       prototype (TCREMP) embeddings, neural codecs, continuous-density background subtraction, and a
       sample-level repertoire embedding. Pure-Python, built on the antigenomics ecosystem
       (<a href="https://github.com/antigenomics/vdjtools">vdjtools</a>, seqtree, arda).</p>
       <p class="proj-intro__links">
         <a href="https://github.com/antigenomics/mirpy">GitHub</a>
         <span>&middot;</span>
         <a href="mir.html">API reference</a>
       </p>
     </div>
   </div>

   <div class="proj-card-grid">
     <a class="proj-card" href="mir.embedding.html">
       <h3>Clonotype embedding</h3>
       <p>TCREMP prototype embedding + PCA denoise, on <code>seqtree.gapblock</code>.</p>
     </a>
     <a class="proj-card" href="mir.html#module-mir.density">
       <h3>Density &amp; background subtraction</h3>
       <p>Graph-free TCRNET/ALICE neighbour enrichment in embedding space (T6).</p>
     </a>
     <a class="proj-card" href="mir.html#module-mir.repertoire">
       <h3>Repertoire embedding</h3>
       <p>One vector per sample — kernel mean ‖ diversity ‖ second moment; MMD (T7).</p>
     </a>
   </div>

Install
-------

.. code-block:: bash

   pip install mirpy-lib            # import mir
   pip install "mirpy-lib[ml]"      # + neural codecs (torch)
   pip install "mirpy-lib[bench]"   # + benchmark harness

Requires ``vdjtools>=2.3.0`` and ``seqtree>=0.3.0`` (native code ships in their wheels; ``mir`` itself
is a pure-Python ``py3-none-any`` wheel). See the project ``README`` for a quick start, ``THEORY.md`` for
the mathematical theory (T1–T7), and ``BENCHMARKS.md`` for recorded results.

API reference
-------------

.. toctree::
   :maxdepth: 3

   mir
