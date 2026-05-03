mir.graph package
=================

Search behavior
---------------

Graph and neighborhood edit-distance searches use ``tcrtrie`` as the primary
engine. If trie search raises an error for a query, code falls back to a
length-constrained brute-force comparison:

- ``hamming``: only equal-length sequences are compared.
- ``levenshtein``: only candidates with length difference ``<= threshold`` are
   compared.

Submodules
----------

mir.graph.distance\_utils module
---------------------------------

.. automodule:: mir.graph.distance_utils
   :members:
   :undoc-members:
   :show-inheritance:

mir.graph.edit\_distance\_graph module
--------------------------------------

.. automodule:: mir.graph.edit_distance_graph
   :members:
   :undoc-members:
   :show-inheritance:

mir.graph.neighborhood\_enrichment module
------------------------------------------

.. automodule:: mir.graph.neighborhood_enrichment
   :members:
   :undoc-members:
   :show-inheritance:

mir.graph.token\_graph module
-----------------------------

.. automodule:: mir.graph.token_graph
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: mir.graph
   :members:
   :undoc-members:
   :show-inheritance:
