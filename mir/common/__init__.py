'''Common classes and routines'''

from .gene_library import GeneEntry, GeneLibrary
from .clonotype import JunctionMarkup, Clonotype, ClonotypeAA, ClonotypeNT
from .filter import filter_functional, filter_canonical
from .sampling import downsample, downsample_locus, resample_to_gene_usage, select_top
from .pool import pool_samples
from .control import ControlManager, control_setup_cli
from .repertoire_dataset import RepertoireDataset
from mir.basic.gene_usage import compute_batch_corrected_gene_usage

# Compatibility aliases for older notebooks/import paths without a dedicated
# mir.common.segments shim module.
Segment = GeneEntry
SegmentLibrary = GeneLibrary
# from .parser import ClonotypeTableParser, VDJdbSlimParser, OlgaParser, VDJtoolsParser, ImmrepParser
# from .repertoire import Repertoire, RepertoireDataset
