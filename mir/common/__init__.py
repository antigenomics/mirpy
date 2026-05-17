'''Common classes and routines'''

from .gene_library import GeneEntry, GeneLibrary
from .clonotype import JunctionMarkup, Clonotype, ClonotypeAA, ClonotypeNT
from .filter import filter_functional, filter_canonical
from .sampling import downsample, downsample_locus, resample_to_gene_usage, select_top
from .pool import pool_samples
from .control import ControlManager, control_setup_cli
from .repertoire_dataset import RepertoireDataset
from .alleles import allele_to_major
from .single_cell import (
	LOCUS_PAIR_TO_LOCI,
	PairedRepertoire,
	PairedClonotype,
	PairedLocusRepertoire,
	SingleCellRepertoire,
	SingleCellSample,
	build_tenx_sample_from_cell_clonotypes,
	build_tenx_donor_from_cell_clonotypes,
	load_10x_vdj_v1_sample,
	load_10x_vdj_v1_donor,
	load_10x_vdj_v1_citeseq_sample,
	validate_citeseq_binders_against_vdjdb_10x,
)
from .single_cell_parser import load_10x_vdj_v1_cell_clonotypes, load_10x_vdj_v1_cell_clonotypes_donor
from .single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from .diversity import (
	CountField,
	DiversitySummary,
	RarefactionResult,
	build_abundance_table,
	hill_curve,
	rarefaction_curve,
	summarize_counts,
	summaries_to_polars,
)
from .parser import VDJdbFullPairedParser
from mir.graph.single_cell_pairing import PairingGraph, build_pairing_graph
from mir.basic.gene_usage import compute_batch_corrected_gene_usage

__all__ = [
	'GeneEntry',
	'GeneLibrary',
	'JunctionMarkup',
	'Clonotype',
	'ClonotypeAA',
	'ClonotypeNT',
	'filter_functional',
	'filter_canonical',
	'downsample',
	'downsample_locus',
	'resample_to_gene_usage',
	'select_top',
	'pool_samples',
	'ControlManager',
	'control_setup_cli',
	'RepertoireDataset',
	'allele_to_major',
	'LOCUS_PAIR_TO_LOCI',
	'PairedRepertoire',
	'PairedClonotype',
	'PairedLocusRepertoire',
	'SingleCellRepertoire',
	'SingleCellSample',
	'build_tenx_sample_from_cell_clonotypes',
	'build_tenx_donor_from_cell_clonotypes',
	'load_10x_vdj_v1_sample',
	'load_10x_vdj_v1_donor',
	'load_10x_vdj_v1_citeseq_sample',
	'validate_citeseq_binders_against_vdjdb_10x',
	'load_10x_vdj_v1_cell_clonotypes',
	'load_10x_vdj_v1_cell_clonotypes_donor',
	'cleanup_cell_clonotypes',
	'impute_missing_chains',
	'CountField',
	'DiversitySummary',
	'RarefactionResult',
	'build_abundance_table',
	'hill_curve',
	'rarefaction_curve',
	'summarize_counts',
	'summaries_to_polars',
	'VDJdbFullPairedParser',
	'PairingGraph',
	'build_pairing_graph',
	'compute_batch_corrected_gene_usage',
	'Segment',
	'SegmentLibrary',
]

# Compatibility aliases for older notebooks/import paths without a dedicated
# mir.common.segments shim module.
Segment = GeneEntry
SegmentLibrary = GeneLibrary
# from .parser import ClonotypeTableParser, VDJdbSlimParser, OlgaParser, VDJtoolsParser, ImmrepParser
# from .repertoire import Repertoire, RepertoireDataset
