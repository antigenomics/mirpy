Embedding your repertoire
=========================

This guide is the shortest path from a clean install to a TCREmp embedding of
your own repertoire. It assumes no prior mirpy setup.

Install from the ``dev`` branch
-------------------------------

mirpy is a source build (compiles small C++ extensions), so you need a C/C++
toolchain and CMake. Inside a fresh virtual environment or conda environment::

    # conda (recommended for an isolated check)
    conda create -y -n mirpy python=3.12
    conda activate mirpy

    # install the development version directly from GitHub
    pip install "git+https://github.com/antigenomics/mirpy.git@dev"

    # optional: UMAP / scanpy extras used by the example notebooks
    pip install "mirpy-lib[sc] @ git+https://github.com/antigenomics/mirpy.git@dev"

.. note::

   Embedding needs **only** the bundled gene library and prototypes. You do
   **not** need `arda <https://github.com/antigenomics/arda>`_ or ``mmseqs2`` —
   those are a build-time-only extra (``pip install "mirpy-lib[arda]"``) used
   solely to regenerate region annotations or annotate a custom gene library.

Verify the install
------------------

Run the smoke test shipped in the repository (or paste its body into a shell)::

    python tests/integration/colleague_smoke.py

It should print ``colleague smoke test PASSED`` after loading the region
annotations and embedding a few clonotypes with both feature modes.

Embed a repertoire
------------------

Any table with V gene, J gene and CDR3/junction amino-acid columns works. For an
AIRR rearrangement file (columns ``locus``, ``v_call``, ``j_call``,
``junction_aa``)::

    from mir.common.parser import AIRRParser
    from mir.embedding.tcremp import TCREmp

    chain = "TRB"                       # TRA | TRB | TRG | TRD | IGH | IGK | IGL
    clonotypes = AIRRParser(locus=chain).parse("my_repertoire.airr.tsv")

    model = TCREmp.from_defaults("human", chain, n_prototypes=3000, mode="cdr123")
    X = model.embed(clonotypes)         # shape: (n_clonotypes, 3 * 3000)

``X`` is a dense ``float32`` matrix ready for PCA / UMAP / clustering.

Choosing a feature mode
-----------------------

``TCREmp.from_defaults`` accepts ``mode``:

* ``"vjcdr3"`` (default) — V-gene, J-gene and CDR3/junction distances.
* ``"cdr123"`` — CDR1, CDR2 (germline V-gene-determined, precomputed from the
  bundled region annotations) and CDR3/junction distances.

See :doc:`examples` → **TCREmp feature comparison** for a side-by-side benchmark
of the two modes, and **Embed any chain** for an end-to-end template (with
OLGA-generated mock data) you can adapt to your data.

Template notebook
-----------------

``notebooks/embed_any_chain.ipynb`` runs the full flow out of the box on mock
data and contains a clearly marked cell for swapping in your own repertoire.
Point the environment variable ``REPERTOIRE_PATH`` at an AIRR file to embed it.
