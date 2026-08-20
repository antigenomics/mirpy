Data pre-processing
===================

mirpy does its own pre-processing, and one part of it is not negotiable.

.. contents::
   :local:
   :depth: 2


The short version
-----------------

**Non-productive rearrangements are removed on every dataset read, and that cannot be turned off.**

Everything else — format conversion, error correction, depth normalisation, length and frequency
and segment filtering — lives in vdjtools' own `Data pre-processing
<https://antigenomics.github.io/vdjtools/preprocessing.html>`_ page, and mirpy calls it.
mirpy does not reimplement any of it.


Why the productive filter is mandatory here
-------------------------------------------

This is the one place mirpy deliberately differs from vdjtools, where the same filter is optional.
The reason is specific to embedding.

A stop codon is **in** the amino-acid alphabet the distance code uses. So an unfiltered frame does
not crash — it returns a finite, entirely meaningless distance, and that number then flows into
:math:`\Phi`, into every principal component, and into whatever model you fit downstream. Nothing
raises, nothing warns, and the result looks exactly like a real one.

That is strictly worse than crashing. A crash you notice.

So mirpy states its contract as: **it cannot embed a non-productive rearrangement meaningfully**,
which is a stronger and more useful claim than *cannot embed it at all*. It can. That is the
problem.

.. code-block:: python

   from mir.embedding.tcremp import TCREmp

   model = TCREmp.from_defaults("human", "TRB")
   model.embed(frame_with_a_stop_codon)
   # ValueError: junction_aa has 1 of 20 non-productive value(s), e.g. ['CASS*YEQYF'].
   # '*' is in seqtree's alphabet, so this does NOT raise downstream -- it embeds to a finite,
   # meaningless distance and contaminates the geometry silently. mirpy therefore refuses it
   # with no opt-out. Filter first with vdjtools.preprocess.filter_productive(); if you want
   # the non-productive fraction itself, that is a vdjtools question, not a mirpy one.

There is no ``allow_nonstandard`` parameter. There was one; it was removed, because a guard you can
switch off is not a guard against something that fails silently.


What happens on a read
----------------------

:func:`mir.cli._read` — and every CLI command that loads clonotypes — applies
:func:`vdjtools.preprocess.filter_productive` and reports what it dropped:

.. code-block:: text

   [mir] dropped 1,412 non-productive rearrangement(s) of 12,004 (productivity read from
         junction_aa); mirpy always does this

The evidence is named because it matters: where a file carries the AIRR ``productive``,
``stop_codon`` or ``vj_in_frame`` columns those are authoritative, and productivity is only derived
from ``junction_aa`` when the file states nothing. See vdjtools' pre-processing page.

Frequencies are renormalised over the survivors by default. Pass
``recompute_frequencies=False`` to leave the file's own frequencies untouched.

Asking for it not to happen is an error:

.. code-block:: python

   _read("sample.tsv", productive_only=False)
   # ValueError: mirpy cannot embed non-productive rearrangements meaningfully ...

.. code-block:: bash

   mir embed clonotypes sample.tsv --no-filter-functional
   # [mir] --no-filter-functional is not available in mirpy. ... Use `vdjtools filter
   #       --nonproductive` if the non-productive fraction is what you want.


Three axes, and mirpy only mandates one
---------------------------------------

The full hierarchy is documented on vdjtools' pre-processing page. In brief:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - axis
     - the question
     - mirpy's position
   * - **parseable**
     - is ``junction_aa`` readable at all?
     - raises — a damaged table, never silently filtered
   * - **productive** (AIRR)
     - does the rearrangement encode a chain?
     - **filtered on every read, no opt-out**
   * - **functional genes** (IMGT F/ORF/P)
     - is the germline gene real?
     - your choice — call vdjtools before handing frames to mirpy

Note the third one is *not* mandated. A pseudogene V that rearranged in frame with no stop codon
embeds perfectly well; whether it should be in your analysis is a biological question, not a
numerical one, so mirpy leaves it to you.

.. code-block:: python

   from vdjtools.preprocess import filter_functional_genes, filter_length

   df = filter_functional_genes(df, keep=("F", "ORF"))
   df = filter_length(df, min_len=5, max_len=60)     # inclusive bounds
   # ... then hand it to mirpy


References ship pre-filtered
----------------------------

Every frozen reference artifact mirpy ships — the geometry reference and the fitted scale
reference — was fitted on **productive-only** data. That is the same population your samples are
reduced to on read, which is what makes the standardisation meaningful: a sample and the reference
it is scored against must describe the same thing.

The scripts that refit those references are not distributed. They read a private corpus, and the
artifact rather than the recipe is what ships.
