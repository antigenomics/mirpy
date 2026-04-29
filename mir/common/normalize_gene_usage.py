"""Compatibility shim for batch gene-usage normalization.

Primary implementation lives in :mod:`mir.basic.gene_usage`.
"""

from mir.basic.gene_usage import compute_batch_corrected_gene_usage

__all__ = ["compute_batch_corrected_gene_usage"]
