import pandas as pd

# class PublicClonotypesSelectionMethod(Enum):
from mir.common.clonotype import ClonotypeAA


class ClonotypeUsageTable:
    def __init__(self, clonotype_matrix, repertoire_dataset, public_clonotypes):
        self.clonotype_matrix = clonotype_matrix
        self.repertoire_dataset = repertoire_dataset
        self.public_clonotypes = public_clonotypes

    @classmethod
    def load_from_repertoire_dataset(cls, repertoire_dataset,
                                     clonotypes_count_for_public_extraction=2,
                                     method_for_public_extraction='unique-occurence',
                                     mismatch_max=1, threads=32):
        public_clonotypes = ClonotypeUsageTable.extract_public_clonotypes_for_directory(
            repertoire_dataset=repertoire_dataset,
            method=method_for_public_extraction,
            count_of_clones=clonotypes_count_for_public_extraction)

        from mir.comparative.match import MultipleRepertoireDenseMatcher
        dense_repertoire_matcher = MultipleRepertoireDenseMatcher(mismatch_max=mismatch_max)
        clonotype_matrix = dense_repertoire_matcher.create_clonotype_matrix_for_clones(
            public_clonotypes, repertoire_dataset, threads)

        return cls(clonotype_matrix, repertoire_dataset, public_clonotypes)

    @staticmethod
    def extract_public_clonotypes_for_directory(repertoire_dataset,
                                                method='unique-occurence',
                                                count_of_clones=2,
                                                ):
        datasets_to_concat = []
        for run in repertoire_dataset:
            cur_data = pd.DataFrame({'cdr3aa': [x.cdr3aa for x in run.clonotypes], 'count': 1})
            datasets_to_concat.append(cur_data)
        full_data = pd.concat(datasets_to_concat)
        full_data = full_data[full_data.cdr3aa.str.isalpha()]
        print(len(full_data))
        top = full_data.groupby(['cdr3aa'], as_index=False).count()

        if method == 'top':
            top = top.sort_values(by=['count'], ascending=False).head(count_of_clones)
        elif method == 'random-roulette':
            top = top.sample(n=count_of_clones, random_state=42, weights='count')
        elif method == 'random-uniform':
            top = top.sample(n=count_of_clones, random_state=42)
        elif method == 'unique-occurence':
            top = top[top['count'] >= count_of_clones]
        return list(top.cdr3aa)

    def get_clone_usage(self, clonotype):
        if isinstance(clonotype, ClonotypeAA):
            clonotype = clonotype.cdr3aa
        if clonotype not in self.public_clonotypes:
            return 0
        return self.clonotype_matrix[clonotype].sum()
