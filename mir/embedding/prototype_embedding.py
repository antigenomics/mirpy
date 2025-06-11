from multiprocessing import Pool
from mir.common.repertoire import Repertoire
from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.distances.aligner import ClonotypeAligner
from mir.embedding.repertoire_embedding import Embedding
from enum import Enum
import os
import numpy as np
import tempfile


class Metrics(Enum):
    SIMILARITY = 'similarity'
    DISSIMILARITY = 'dissimilarity'


def worker_embed_clonotype_batch_to_file(args):
    clonotypes, prototype_repertoire, aligner, metrics, output_file, flatten = args

    n_clonotypes = len(clonotypes)
    n_prototypes = len(prototype_repertoire)

    if isinstance(clonotypes[0], ClonotypeAA):
        n_features_per_proto = 3
    elif isinstance(clonotypes[0], PairedChainClone):
        n_features_per_proto = 6
    else:
        raise ValueError(f"Unknown clonotype type: {type(clonotypes[0])}")

    n_features_total = n_prototypes * n_features_per_proto

    mmap = np.memmap(output_file, dtype='int16', mode='w+', shape=(n_clonotypes, n_features_total))

    for i, c in enumerate(clonotypes):
        if isinstance(c, ClonotypeAA):
            scores = [aligner.score_dist(anchor, c) if metrics == Metrics.DISSIMILARITY
                      else aligner.score(anchor, c) for anchor in prototype_repertoire]
        elif isinstance(c, PairedChainClone):
            scores = [aligner.score_dist_paired(anchor, c) if metrics == Metrics.DISSIMILARITY
                      else aligner.score_paired(anchor, c) for anchor in prototype_repertoire]
        else:
            raise ValueError(f"Unknown clonotype type: {type(c)}")

        flat = [s.get_flatten_score() for s in scores] if flatten else scores
        flat = [v for sub in flat for v in sub]
        mmap[i, :] = flat
        mmap.flush()
    return output_file, mmap.shape


class PrototypeEmbedding(Embedding):
    def __init__(self, prototype_repertoire: Repertoire, aligner: ClonotypeAligner = ClonotypeAligner.from_library(),
                 metrics=Metrics.SIMILARITY):
        super().__init__()
        self.prototype_repertoire = prototype_repertoire
        self.embedding_type = metrics
        self.aligner = aligner

    def embed_repertoire(self, repertoire: Repertoire, threads: int = 32, flatten_scores=True):
        chunks = self.__split_into_chunks(repertoire.clonotypes, threads)
        tmp_dir = tempfile.mkdtemp()

        args = []
        for i, chunk in enumerate(chunks):
            path = os.path.join(tmp_dir, f"emb_{i}.dat")
            args.append((chunk, self.prototype_repertoire, self.aligner,
                         self.embedding_type, path, flatten_scores))

        with Pool(threads) as pool:
            temp_files = pool.map(worker_embed_clonotype_batch_to_file, args)

        total_rows = sum(shape[0] for f, shape in temp_files)
        total_cols = temp_files[0][1][1]
        combined_path = os.path.join(tmp_dir, "combined.dat")
        combined = np.memmap(combined_path, dtype='int16', mode='w+', shape=(total_rows, total_cols))

        offset = 0
        for f, shape in temp_files:
            data = np.memmap(f, dtype='int16', mode='r', shape=shape)
            combined[offset:offset + data.shape[0]] = data
            offset += data.shape[0]
            os.remove(f)

        result = np.array(combined)
        del combined
        os.remove(combined_path)

        return result

    @staticmethod
    def __split_into_chunks(lst, k):
        n = len(lst)
        chunk_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        chunks = []
        start = 0
        for size in chunk_sizes:
            end = start + size
            chunks.append(lst[start:end])
            start = end
        return chunks
