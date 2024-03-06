from collections import defaultdict

import pandas as pd

from mir.common.clonotype import ClonotypeAA
from mir.common.parser import ClonotypeTableParser


class Repertoire:
    """
    The main object in the library is `Repertoire`. It stores the information of the immune repertoire for a \
    single sample. By default it can contain any number of chains in it (e.g. TCRB clonotypes and TCRA clonotypes), \
    but some code would only work if the repertoires contains a single gene data.
    You can read the info, subsample, filter, resample, sort, evaluate segment usage and so on for the `Repertoire` \
    object.
    """
    def __init__(self,
                 clonotypes: list[ClonotypeAA],
                 sorted: bool = False,
                 metadata: dict[str, str] | pd.Series = dict(),
                 gene: str = None):
        """
        The initializing function for the repertoire which creates an object using a list of clonotypes and other param
        :param clonotypes: the list of clonotypes to create an object from. You *should do everything you can*\
        to not use `Clonotype` class here. Please use one of: `ClonotypeAA`, `ClonotypeNT`, `PairedChainClonotype`
        :param sorted: whether the repertoire clonotypes are sorted by usage or not
        :param metadata: the metadata which was given along with clonotypes information. Usually contains \
        age/sex/disease_status/HLA info
        :param gene: TRA/TRB/IGH/IGL... By defauly can be None. In this case it can contain any number of chains \
        in it (e.g. TCRB clonotypes and TCRA clonotypes), but some code would only work if the repertoires contains \
        a single gene data.
        """
        if gene is not None:
            self.clonotypes = [x for x in clonotypes if gene in x.v.gene and gene in x.j.gene  and (
                    x.d is None or gene in x.d.gene)]
        else:
            self.clonotypes = clonotypes
        self.sorted = sorted
        self.metadata = metadata
        self.segment_usage = None
        self.number_of_clones = len(self.clonotypes)
        self.number_of_reads = sum([x.size() for x in self.clonotypes])
        self.gene = gene

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: dict[str, str] | pd.Series = dict(),
             path: str = None,
             n: int = None,
             sample: bool = False,
             gene: str = None):
        """
        The initializing function for the repertoire which creates an object using a list of clonotypes and other param
        to not use `Clonotype` class here. Please use one of: `ClonotypeAA`, `ClonotypeNT`, `PairedChainClonotype`
        :param metadata: the metadata which was given along with clonotypes information. Usually contains \
        age/sex/disease_status/HLA info
        :param gene: TRA/TRB/IGH/IGL... By defauly can be None. In this case it can contain any number of chains \
        in it (e.g. TCRB clonotypes and TCRA clonotypes), but some code would only work if the repertoires contains \
        a single gene data.
        :param parser: the parser which would parse the initial file and create the clonotype list
        :param path: the path to a file that we should parse
        :param n: number of rows to parse or None (parse everything)
        :param sample: whether to sample random rows from the initial file rows or not
        """
        if not path:
            if 'path' not in metadata:
                raise ValueError("'path' is missing in metadata")
            path = metadata['path']
        else:
            metadata['path'] = path
        return cls(clonotypes=parser.parse(path, n=n, sample=sample), metadata=metadata, gene=gene)

    def __copy__(self):
        return Repertoire(self.clonotypes, self.sorted, self.metadata)

    def sort(self):
        """
        The function which sorts the clonotypes by `cells` column
        """
        self.sorted = True
        self.sorted_by = 'cells'
        self.clonotypes.sort(key=lambda x: x.size(), reverse=True)

    def sort_by_clone_metadata(self, sort_by: str, reverse=False):
        """
        The function which sorts the clonotypes by any metadata column
        :param sort_by: the parameter to sort by
        :param reverse: whether to sort low to high or high to low
        """
        self.sorted = True
        self.sorted_by = sort_by
        for clone in self.clonotypes:
            if sort_by not in clone.clone_metadata:
                raise Exception(f'Cannot sort by {sort_by}, {clone} has no such metadata!')
        self.clonotypes.sort(key=lambda x: x.clone_metadata[sort_by], reverse=reverse)

    def top(self, n: int = 100):
        """
        Get `n` top used clonotypes in a repertoire
        :param n: number of clonotypes
        :return: a list of top clonotypes
        """
        #TODO should we change to return the Repertoire with sampled clonotypes?
        if not sorted:
            self.sort()
        return self.clonotypes[0:n]

    @property
    def total(self):
        """
        returns the total size of all the clonotypes (number of reads/number of cells)
        :return:
        """
        return sum(c.size() for c in self.clonotypes)

    @property
    def evaluate_segment_usage(self):
        """
        Evaluates the segment usage for the repertoire.
        Creates the segment usage dictionary property in a `Repertoire` and returns it
        :return: Creates the segment usage dictionary
        """
        if self.segment_usage is None:
            self.segment_usage = defaultdict(int)
            for c in self.clonotypes:
                # TODO add type of clonotype assertion
                self.segment_usage[c.v.id] += 1
                self.segment_usage[c.j.id] += 1
                if c.d is not None:
                    self.segment_usage[c.d.id] += 1
        return self.segment_usage

    def serialize(self):
        """
        Performs serialization of the object.
        Returns a `pd.DataFrame` where each row is a sample and each column represents the usage of a clonotype
        :return:
        """
        serialization_dct = defaultdict(list)
        for i, clonotype in enumerate(self.clonotypes):
            serialization_res = clonotype.serialize()
            keys_to_process = set(serialization_res.keys()).union(serialization_dct.keys())
            for k in keys_to_process:
                if k in serialization_res:
                    serialization_dct[k].append(serialization_res[k])
                else:
                    serialization_dct[k].append(None)
        return pd.DataFrame(serialization_dct)

    def __getitem__(self, idx):
        return self.clonotypes[idx]

    def __len__(self):
        return len(self.clonotypes)

    def __str__(self):
        return f'Repertoire of {self.__len__()} clonotypes and {self.total} cells:\n' + \
               '\n'.join([str(x) for x in self.clonotypes[0:5]]) + \
               '\n' + str(self.metadata) + '\n...'

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.clonotypes)

    def __add__(self, other):
        if not isinstance(other, Repertoire):
            raise ValueError('Can only sum objects of class Repertoire')
        new_metadata = dict(self.metadata)  # or orig.copy()
        new_metadata.update(dict(other.metadata))
        new_clonotypes = self.clonotypes + other.clonotypes
        return Repertoire(
            clonotypes=new_clonotypes,
            sorted=False,
            metadata=new_metadata)
    # TODO subsample
    # TODO aggregate redundant
    # TODO group by and aggregate
