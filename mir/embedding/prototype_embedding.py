import logging
from multiprocessing import Pool
import shutil

from mir.common.repertoire import Repertoire
from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.distances.aligner import ClonotypeAligner
from mir.embedding.repertoire_embedding import Embedding
from enum import Enum
import os
import tempfile
from pympler import asizeof


class Metrics(Enum):
    SIMILARITY = 'similarity'
    DISSIMILARITY = 'dissimilarity'


# переместить embed repertoire в абстрактный,

import pickle
import numpy as np
from mir.embedding.prototype_embedding import Metrics
from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.common.repertoire import Repertoire

def worker_embed_clonotype_batch_to_file(args):
    clonotypes_or_path, prototype_repertoire, aligner, metrics, output_file, flatten = args

    if isinstance(clonotypes_or_path, str):
        with open(clonotypes_or_path, 'rb') as f:
            repertoire = pickle.load(f)
    elif isinstance(clonotypes_or_path, Repertoire):
        repertoire = clonotypes_or_path
    else:
        raise ValueError("Expected Repertoire object or path to .pkl file")

    clonotypes = repertoire.clonotypes
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
        tmp_dir = tempfile.mkdtemp()
        chunks = repertoire.make_chunks(threads, tmp_dir)
        repertoire.clonotypes = None

        args = []
        for i, chunk in enumerate(chunks):
            path = os.path.join(tmp_dir, f"emb_{i}.dat")
            args.append((chunk, self.prototype_repertoire, self.aligner,
                         self.embedding_type, path, flatten_scores))

        with Pool(threads) as pool:
            temp_files = pool.map(worker_embed_clonotype_batch_to_file, args)

        total_rows = sum(shape[0] for _, shape in temp_files)
        total_cols = temp_files[0][1][1]
        combined_path = os.path.join(tmp_dir, "combined.dat")
        combined = np.memmap(combined_path, dtype='float16', mode='w+', shape=(total_rows, total_cols))

        offset = 0
        for f, shape in temp_files:
            data = np.memmap(f, dtype='int16', mode='r', shape=shape)
            combined[offset:offset + data.shape[0]] = data
            offset += data.shape[0]

        result = np.array(combined)
        del combined

        shutil.rmtree(tmp_dir)

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
