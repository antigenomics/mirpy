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
	PairedClonotype,
	PairedLocusRepertoire,
	SingleCellRepertoire,
	TenXVdjV1DonorData,
	build_tenx_donor_from_cell_clonotypes,
	load_10x_vdj_v1_donor,
)
from .single_cell_parser import load_10x_vdj_v1_cell_clonotypes
from .single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from .single_cell_util import PairingGraph, build_pairing_graph
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
	'PairedClonotype',
	'PairedLocusRepertoire',
	'SingleCellRepertoire',
	'TenXVdjV1DonorData',
	'build_tenx_donor_from_cell_clonotypes',
	'load_10x_vdj_v1_donor',
	'load_10x_vdj_v1_cell_clonotypes',
	'cleanup_cell_clonotypes',
	'impute_missing_chains',
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
