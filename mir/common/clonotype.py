import re
import typing as t

from mir.basic.mirseq import translate_linear
from mir.common.gene_library import GeneEntry, _GENE_LIBRARY_CACHE

_CODING_AA = re.compile('^[ARNDCQEGHILKMFPSTWYV]+$')
_CANONICAL_AA = re.compile('^C[ARNDCQEGHILKMFPSTWYV]+[FW]$')


class ClonotypePayload:
    """Optional read/sample count metadata attached to a clonotype."""

    def __init__(self) -> None:
        self.number_of_reads = None
        self.number_of_samples = None


class Clonotype:
    __slots__ = 'id', 'duplicate_count', 'payload', 'clone_metadata'

    def __init__(self,
                 id: int | str,
                 duplicate_count: int | list[str] = 1,
                 payload: t.Any = None):
        self.id = id
        self.duplicate_count = duplicate_count
        self.payload = payload
        self.clone_metadata = {}

    def size(self) -> int:
        """Return the number of cells/reads associated with this clonotype."""
        if isinstance(self.duplicate_count, int):
            return self.duplicate_count
        else:
            return len(self.duplicate_count)

    def serialize(self) -> dict:
        return {'id': self.id,
                'duplicate_count': self.duplicate_count}

    def __str__(self):
        return 'κ' + str(self.id)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return Clonotype(self.id, self.duplicate_count, self.payload)


class ClonotypeAA(Clonotype):
    """Clonotype with amino-acid junction sequence (AIRR: junction_aa)."""
    __slots__ = 'junction_aa', 'v_gene', 'd_gene', 'j_gene'

    def __init__(self, junction_aa: str,
                 v_gene: str | GeneEntry = None,
                 d_gene: str | GeneEntry = None,
                 j_gene: str | GeneEntry = None,
                 id: int | str = -1,
                 duplicate_count: int | list[str] = 1,
                 payload: t.Any = None):
        """
        Parameters
        ----------
        junction_aa:
            Junction amino-acid sequence (AIRR ``junction_aa``).
        v_gene, d_gene, j_gene:
            V/D/J gene entries or allele-name strings.  Strings are
            resolved via the default OLGA gene-library cache.
        id:
            Clonotype identifier (integer row index or custom string).
        duplicate_count:
            Read / cell count, or a list of barcode strings for single-cell
            (AIRR ``duplicate_count``).
        payload:
            Arbitrary metadata attached to this clonotype.
        """
        super().__init__(id, duplicate_count, payload)
        self.junction_aa = junction_aa
        if isinstance(v_gene, str):
            v_gene = _GENE_LIBRARY_CACHE.get_or_create(v_gene)
        self.v_gene = v_gene
        if isinstance(d_gene, str):
            d_gene = _GENE_LIBRARY_CACHE.get_or_create(d_gene)
        self.d_gene = d_gene
        if isinstance(j_gene, str):
            j_gene = _GENE_LIBRARY_CACHE.get_or_create(j_gene)
        self.j_gene = j_gene

    def is_coding(self):
        """Return True if junction_aa consists only of standard amino-acid characters."""
        return _CODING_AA.match(self.junction_aa)

    def is_canonical(self):
        """Return True if junction_aa starts with C and ends with F or W."""
        return _CANONICAL_AA.match(self.junction_aa)

    def serialize(self) -> dict:
        return {'id': self.id,
                'duplicate_count': self.duplicate_count,
                'junction_aa': self.junction_aa,
                'v_gene': str(self.v_gene) if self.v_gene else None,
                'd_gene': str(self.d_gene) if self.d_gene else None,
                'j_gene': str(self.j_gene) if self.j_gene else None}

    def __str__(self):
        return super().__str__() + ' ' + self.junction_aa

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return ClonotypeAA(self.junction_aa, self.v_gene, self.d_gene, self.j_gene,
                           self.id, self.duplicate_count, self.payload)

    def __getitem__(self, segment):
        if segment == 'v':
            return self.v_gene
        elif segment == 'd':
            return self.d_gene
        elif segment == 'j':
            return self.j_gene
        else:
            raise Exception('You can get only v/d/j gene by itemization')


class JunctionMarkup:
    """Stores V/D/J boundary positions within the junction sequence."""
    __slots__ = 'vend', 'dstart', 'dend', 'jstart'

    def __init__(self,
                 vend: int = None,
                 dstart: int = None,
                 dend: int = None,
                 jstart: int = None):
        self.vend = vend
        self.dstart = dstart
        self.dend = dend
        self.jstart = jstart


class ClonotypeNT(ClonotypeAA):
    """Clonotype with nucleotide junction sequence (AIRR: junction)."""
    __slots__ = 'junction', 'junction_markup'

    def __init__(self,
                 junction: str,
                 junction_markup: JunctionMarkup = None,
                 junction_aa: str = None,
                 v_gene: str | GeneEntry = None,
                 d_gene: str | GeneEntry = None,
                 j_gene: str | GeneEntry = None,
                 id: int | str = -1,
                 duplicate_count: int | list[str] = 1,
                 payload: t.Any = None):
        """
        Parameters
        ----------
        junction:
            Junction nucleotide sequence (AIRR ``junction``).  ``junction_aa``
            is auto-translated when not provided.
        junction_markup:
            Optional V/D/J boundary markup (positions within the junction).
        junction_aa:
            Junction amino-acid sequence.  Derived from *junction* if omitted.
        v_gene, d_gene, j_gene:
            V/D/J gene entries or allele-name strings.
        """
        if not junction_aa:
            junction_aa = translate_linear(junction).rstrip('_')
        super().__init__(junction_aa, v_gene, d_gene, j_gene, id, duplicate_count, payload)
        self.junction = junction
        self.junction_markup = junction_markup

    def serialize(self) -> dict:
        return {'id': self.id,
                'duplicate_count': self.duplicate_count,
                'junction_aa': self.junction_aa,
                'junction': self.junction,
                'v_gene': str(self.v_gene) if self.v_gene else None,
                'd_gene': str(self.d_gene) if self.d_gene else None,
                'j_gene': str(self.j_gene) if self.j_gene else None}

    def __str__(self):
        return super().__str__() + ' ' + self.junction

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return ClonotypeNT(self.junction, self.junction_markup, self.junction_aa,
                           self.v_gene, self.d_gene, self.j_gene,
                           self.id, self.duplicate_count, self.payload)


class PairedChainClone(Clonotype):
    """Paired alpha+beta (or heavy+light) clone."""
    def __init__(self,
                 chainA: Clonotype,
                 chainB: Clonotype,
                 id: int | str = -1,
                 duplicate_count: int | list[str] = 1,
                 payload: t.Any = None):
        super().__init__(id, duplicate_count, payload)
        self.chainA = chainA
        self.chainB = chainB

    def __str__(self):
        return 'alpha ' + self.chainA.__str__() + ' beta ' + self.chainB.__str__()

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return PairedChainClone(self.chainA, self.chainB)


class ClonalLineage:
    def __init__(self, clonotypes: list[Clonotype]):
        self.clonotypes = clonotypes
