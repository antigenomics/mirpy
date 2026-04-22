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
    __slots__ = 'id', 'cells', 'payload', 'clone_metadata'

    def __init__(self,
                 id: int | str,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        self.id = id
        self.cells = cells
        self.payload = payload
        self.clone_metadata = {}

    def size(self) -> int:
        """Return the number of cells/reads associated with this clonotype."""
        if isinstance(self.cells, int):
            return self.cells
        else:
            return len(self.cells)

    def serialize(self) -> dict:
        """Return a plain-dict representation of this clonotype.
        """
        return {'id': self.id,
                'cells': self.cells}

    def __str__(self):
        return 'κ' + str(self.id)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return Clonotype(self.id, self.cells, self.payload)


class ClonotypeAA(Clonotype):
    """
    The clonotype which stores the amino acid sequence. Recommended to use everywhere instead of usual `Clonotype`.
    """
    __slots__ = 'cdr3aa', 'v', 'd', 'j'

    def __init__(self, cdr3aa: str,
                 v: str | GeneEntry = None,
                 d: str | GeneEntry = None,
                 j: str | GeneEntry = None,
                 id: int | str = -1,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        """
        Parameters
        ----------
        cdr3aa:
            CDR3 amino-acid sequence.
        v, d, j:
            V/D/J gene entries or allele-name strings.  Strings are
            resolved via the default OLGA gene-library cache.
        id:
            Clonotype identifier (integer row index or custom string).
        cells:
            Read / cell count, or a list of barcode strings for single-cell.
        payload:
            Arbitrary metadata attached to this clonotype.
        """
        super().__init__(id, cells, payload)
        self.cdr3aa = cdr3aa
        if isinstance(v, str):
            v = _GENE_LIBRARY_CACHE.get_or_create(v)
        self.v = v
        if isinstance(d, str):
            d = _GENE_LIBRARY_CACHE.get_or_create(d)
        self.d = d
        if isinstance(j, str):
            j = _GENE_LIBRARY_CACHE.get_or_create(j)
        self.j = j

    def is_coding(self):
        """Return True if CDR3 consists only of standard amino-acid characters."""
        return _CODING_AA.match(self.cdr3aa)

    def is_canonical(self):
        """Return True if CDR3 starts with C and ends with F or W."""
        return _CANONICAL_AA.match(self.cdr3aa)

    def serialize(self) -> dict:
        """Return a plain-dict representation of this clonotype."""
        return {'id': self.id,
                'cells': self.cells,
                'cdr3aa': self.cdr3aa,
                'v': self.v,
                'd': self.d,
                'j': self.j}

    def __str__(self):
        return super().__str__() + ' ' + self.cdr3aa

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return ClonotypeAA(self.cdr3aa, self.v, self.d, self.j, self.id, self.cells, self.payload)

    def __getitem__(self, segment):
        if segment == 'v':
            return self.v
        elif segment == 'd':
            return self.d
        elif segment == 'j':
            return self.j
        else:
            raise Exception('You can get only v/d/j gene by itemization')


class JunctionMarkup:
    """
    The junction markup object. Stores the end of V, beginning and end of D and beginning of J
    """
    __slots__ = 'vend', 'dstart', 'dend', 'jstart'

    def __init__(self,
                 vend: int=None,
                 dstart: int=None,
                 dend: int=None,
                 jstart: int=None):
        self.vend = vend
        self.dstart = dstart
        self.dend = dend
        self.jstart = jstart


class ClonotypeNT(ClonotypeAA):
    """
    The clonotype which stores the nucleotide sequence. Recommended to use everywhere instead of usual `Clonotype`.
    """
    __slots__ = 'cdr3nt', 'junction'
    def __init__(self,
                 cdr3nt: str,
                 junction: JunctionMarkup = None,
                 cdr3aa: str = None,
                 v: str | GeneEntry = None,
                 d: str | GeneEntry = None,
                 j: str | GeneEntry = None,
                 id: int | str = -1,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        """
        Parameters
        ----------
        cdr3nt:
            CDR3 nucleotide sequence.  ``cdr3aa`` is auto-translated when
            not provided.
        junction:
            Optional V/D/J boundary markup (positions in AA space).
        cdr3aa:
            CDR3 amino-acid sequence.  Derived from *cdr3nt* if omitted.
        v, d, j:
            V/D/J gene entries or allele-name strings (resolved via the
            default OLGA gene-library cache).
        """
        if not cdr3aa:
            cdr3aa = translate_linear(cdr3nt).rstrip('_')
        super().__init__(cdr3aa, v, d, j, id, cells, payload)
        self.cdr3nt = cdr3nt
        self.junction = junction

    def serialize(self) -> dict:
        return {'id': self.id,
                'cells': self.cells,
                'cdr3aa': self.cdr3aa,
                'cdr3nt': self.cdr3nt,
                'v': self.v,
                'd': self.d,
                'j': self.j}

    def __str__(self):
        return super().__str__() + ' ' + self.cdr3nt

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return ClonotypeNT(self.cdr3nt, self.junction, self.cdr3aa, self.v, self.d, self.j, self.id, self.cells,
                           self.payload)


# TODO
class PairedChainClone(Clonotype):
    """
    The object which stores a clone (alpha+beta chains or heavy+light and so on).
    """
    def __init__(self,
                 chainA: Clonotype,
                 chainB: Clonotype,
                 id: int | str = -1,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        super().__init__(id, cells, payload)
        self.chainA = chainA
        self.chainB = chainB

    def __str__(self):
        return 'alpha ' + self.chainA.__str__() + ' beta ' + self.chainB.__str__()

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return PairedChainClone(self.chainA, self.chainB)


# TODO
class ClonalLineage:
    def __init__(self, clonotypes: list[Clonotype]):
        self.clonotypes = clonotypes
