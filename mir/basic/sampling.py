import numpy as np
from copy import copy

from mir.basic.segment_usage import SegmentUsageTable
from mir.common.repertoire import Repertoire


class RepertoireSampling:
    """
    A class which can resample the repertoire. The main method of the class is `sample`.
    This method uses a single repertoire and resamples it using the initial segment usage matrix and the usage matrix \
    you want your resampled repertoire to correspond to.
    If you don't want to change the segment usage distribution you can simply skip the `new_usage_matrix` parameter.
    Parameter `n` is the number of clones in the new repertoire. Skip it if you want to have the same number of reads.
    """
    def __init__(self, random_seed: int = 42):
        """
        An initializing method. Just creates an object

        :param random_seed: the random seed for sampling. 42 as default
        """
        self.random_seed = random_seed

    def sample(self, repertoire: Repertoire, old_usage_matrix: list[SegmentUsageTable],
               new_usage_matrix: list[SegmentUsageTable] = None, n: int = None):
        """
        A function which samples a new repertoire based on the given one and its usage matrix

        :param repertoire: a `Repertoire` object which is the base for resampling
        :param old_usage_matrix: the V(D)J gene usage matrix describing the segment usage in the initial repertoire
        :param new_usage_matrix: the V(D)J gene usage matrix describing the desired segment usage in the new repertoire;\
        do not fill this parameter if you do not want to change the segment usage distribution
        :param n: number of reads to be sampled (samples the same size if None)
        :return: a resampled `Repertoire` repertoire
        """
        sample_name = repertoire.metadata['run']
        if new_usage_matrix is None:
            new_usage_matrix = old_usage_matrix

        old_run_frequencies = self.__preprocess_frequencies(old_usage_matrix, sample_name)
        new_run_frequencies = self.__preprocess_frequencies(new_usage_matrix, sample_name)
        for clone in repertoire:
            clone.clone_metadata['expected_freq'] = clone.cells / repertoire.number_of_reads
            for segment_type in old_run_frequencies.keys():
                segment_name = clone[segment_type.lower()].id
                clone.clone_metadata['expected_freq'] *= old_run_frequencies[segment_type][segment_name] / \
                                                         new_run_frequencies[segment_type][segment_name]
        sum_expected_freq = sum([x.clone_metadata['expected_freq'] for x in repertoire])
        for clone in repertoire:
            clone.clone_metadata['expected_freq'] /= sum_expected_freq
        repertoire.sort_by_clone_metadata(sort_by='expected_freq', reverse=True)

        repertoire_frequencies = [x.clone_metadata['expected_freq'] for x in repertoire]
        if n is None:
            n = repertoire.number_of_reads
        generated_counts_for_clones = self.__roulette_wheel_selection(repertoire_frequencies,
                                                                      desired_num_generated_reads=n)
        copy_clonotypes = []
        for new_count, clone in zip(generated_counts_for_clones, repertoire):
            clone.clone_metadata.pop('expected_freq')
            if new_count == 0:
                continue
            new_clone = copy(clone)
            new_clone.cells = new_count
            copy_clonotypes.append(new_clone)
        return Repertoire(copy_clonotypes, is_sorted=False, metadata=repertoire.metadata)

    @staticmethod
    def __preprocess_frequencies(usage_matrix, sample_name):
        """
        A function which transforms segment usages from integer numbers into frequencies

        :param usage_matrix: the `list[SegmentUsageTable]` object containing `SegmentUsageTable`s for each sample
        :param sample_name: the sample to perform preprocessing on
        :return: a dictionary mapping segment type to a dictionary of gene-usage items
        """
        run_frequencies = {usage_table.segment_type: usage_table[sample_name] for usage_table in usage_matrix}
        for k, v in run_frequencies.items():
            sum_freq = sum(list(run_frequencies[k].values()))
            for gene, freq in run_frequencies[k].items():
                run_frequencies[k][gene] = freq / sum_freq
        return run_frequencies

    def __roulette_wheel_selection(self, frequencies: list[int], desired_num_generated_reads: int):
        """
        A function which performs roulette wheel selection resampling. It makes segments of size equal to the\
        probability of the clone to be found in a sample. Afterwards, the random numbers are placed in the segments and\
        the new counts for each clonotype are identified.
        For more see https://en.wikipedia.org/wiki/Fitness_proportionate_selection

        :param frequencies: the desired frequencies for each clonotype
        :param desired_num_generated_reads: number of reads to generate
        :return: the list of counts for each clonotype
        """
        np.random.seed(self.random_seed)
        random_numbers = np.random.uniform(0, 1, desired_num_generated_reads).tolist()
        random_numbers.sort()

        for i in range(1, len(frequencies)):
            frequencies[i] += frequencies[i - 1]

        new_counts = [0 for _ in range(len(frequencies))]

        num_generated_values = 0
        freq_pointer = 0
        while num_generated_values < desired_num_generated_reads:
            if random_numbers[num_generated_values] <= frequencies[freq_pointer]:
                new_counts[freq_pointer] += 1
                num_generated_values += 1
            else:
                freq_pointer += 1

        assert sum(new_counts) == desired_num_generated_reads
        return new_counts
