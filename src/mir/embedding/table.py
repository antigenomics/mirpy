"""RFF sample tables, bounded readers, and sample-PCA artifact IO."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
from vdjtools.signature.cohort import parallel_rows, resolve_sample
from vdjtools.signature.features import sanitise
from vdjtools.signature.corpus import TablePCA  # noqa: F401 — public compatibility export

from mir.repertoire import sample_embedding, _fit_repertoire_coordinates, _finish_repertoire_space


def embedding_space(models: dict, *, weight: str | None = None) -> str:
    """Fingerprint the ordered prototype panels and coordinate options, without publishing them."""
    panels = {loc: {"species": m.species, "mode": m.mode, "metric": m.metric,
                    "replicate": m.replicate, "gaps": m._gap_positions,
                    "shm_scale": m.shm_scale, "alignment": m._alignment,
                    "v": m._proto_v, "j": m._proto_j, "junction": m._proto_junction}
              for loc, m in sorted(models.items())}
    digest = hashlib.sha256(json.dumps({'panels': panels, 'weight': weight}, sort_keys=True).encode())
    for _, model in sorted(models.items()):
        base = getattr(model._germline, 'base', model._germline)
        for comp, data in sorted(base._components.items()):
            digest.update(json.dumps([comp, data.idx], sort_keys=True).encode())
            digest.update(data.ext.tobytes())
        if model._matrix is not None:
            alphabet = 'ARNDCQEGHILKMFPSTWYVBZX*'
            penalties = [[model._matrix.penalty(a, b) for b in alphabet] for a in alphabet]
            digest.update(json.dumps([model._matrix.scale(), penalties]).encode())
    return digest.hexdigest()


def _frames(sample):
    frames = resolve_sample(sample)
    if not isinstance(frames, dict):
        frames = {loc: sub for (loc,), sub in frames.group_by('locus')}
    return {loc: sanitise(df)[0] for loc, df in frames.items()}


def fit_sample_spaces(samples, models, *, pca_fit_cap=200_000, seed=0, **kwargs):
    """Fit the established shared clonotype basis without retaining the input cohort.

    Count rows, then read only the exact seeded selection used by
    fit_repertoire_space on the pooled cloud. Fit loci sequentially. Deferred
    readers must be repeatable; no computed intermediate is cached on disk.
    """
    items = list(samples)
    counts = {loc: [] for loc in models}
    for _, sample in items:
        frames = _frames(sample)
        for loc in models:
            counts[loc].append(frames[loc].height if loc in frames else 0)
        del frames
    spaces = {}
    for loc, model in models.items():
        total = sum(counts[loc])
        if not total:
            continue
        selected = (np.random.default_rng(seed).choice(total, pca_fit_cap, replace=False)
                    if total > pca_fit_cap else np.arange(total))
        # Sorting only the read schedule; restore RNG order before the existing fit.
        order = np.argsort(selected)
        sorted_rows = selected[order]
        from mir.density import _embed
        from vdjtools.cores import require_memory
        coordinate_space = kwargs.get('space', 'full')
        width = model.n_features if coordinate_space == 'full' else model.n_prototypes
        require_memory(8 * len(selected) * width + 16 * width**2, 'shared clonotype PCA fit')
        X = np.empty((len(selected), width), dtype=np.float32)
        chunk = max(1, (32 * 1024**2) // (12 * model.n_features))
        offset = 0
        for (_, sample), count in zip(items, counts[loc]):
            lo, hi = np.searchsorted(sorted_rows, [offset, offset + count])
            if hi > lo:
                frames = _frames(sample)
                df = frames[loc]
                if df.height != count:
                    raise ValueError('input changed between basis fitting passes')
                for start in range(lo, hi, chunk):
                    end = min(start + chunk, hi)
                    part = df[(sorted_rows[start:end] - offset).tolist()]
                    X[order[start:end]] = _embed(model, part, coordinate_space)
                    del part
                del frames, df
            offset += count
        fit_options = {key: kwargs[key] for key in ('n_components','space') if key in kwargs}
        clono = _fit_repertoire_coordinates(model, X, seed=seed, **fit_options)
        del X
        rff_options = {key: value for key,value in kwargs.items() if key not in fit_options}
        spaces[loc] = _finish_repertoire_space(clono, seed=seed, **rff_options)
    if not spaces:
        raise ValueError('no coding clonotypes to fit a shared RFF basis')
    return spaces


def _rff_space(spaces, weight):
    digest = hashlib.sha256(embedding_space(
        {loc: sp.clono.model for loc, sp in spaces.items()}, weight=weight).encode())
    for loc, sp in sorted(spaces.items()):
        digest.update(loc.encode())
        for array in (sp.clono.scaler.mean_, sp.clono.scaler.scale_,
                      sp.clono.pca.mean_, sp.clono.pca.components_, sp.rff.omega, sp.rff.b):
            digest.update(np.asarray(array).tobytes())
        if sp.rff2 is not None:
            digest.update(sp.rff2.omega.tobytes())
            digest.update(sp.rff2.b.tobytes())
    return digest.hexdigest()


def sample_embeddings(samples, spaces: dict, *, weight='log2p1',
                      blocks=('mean', 'diversity'), n_jobs=0) -> pl.DataFrame:
    """Read/reduce/release one sample per worker; retain RFF means and requested channels.

    Every sample uses the same fitted RepertoireSpace per locus. Diversity and
    second moments retain SampleEmbedding semantics; absent loci remain missing.
    ``n_jobs=0`` uses the allocated cores; ``1`` runs without a sample pool.
    """
    if not spaces or not blocks or set(blocks) - {'mean', 'diversity', 'second'}:
        raise ValueError('provide shared spaces and valid nonempty blocks')
    items = list(samples.items() if isinstance(samples, dict) else samples)
    ids = [sid for sid, _ in items]
    if not items or len(set(ids)) != len(ids):
        raise ValueError('sample ids must be nonempty and unique')
    from mir.distances.shm import MutatedGermlineDistances
    for sp in spaces.values():
        model = sp.clono.model
        if not isinstance(model._germline, MutatedGermlineDistances):
            model._germline = MutatedGermlineDistances(model._germline, scale=model.shm_scale)
        for comp in ('V', 'J') if model.mode == 'vjcdr3' else ('CDR1', 'CDR2'):
            model._proto_idx(comp)
    columns = []
    widths = {}
    for loc, sp in spaces.items():
        sizes = {'mean': sp.rff.dim, 'diversity': 4,
                 'second': (sp.meta.get('n_eigs') or sp.rff2.dim * (sp.rff2.dim + 1) // 2)
                           if sp.rff2 is not None else 0}
        cols = [f'{"phi" if block == "mean" else block}:{loc}:{i}'
                for block in ('mean', 'diversity', 'second') if block in blocks
                for i in range(sizes[block])]
        columns.extend(cols)
        widths[loc] = len(cols)

    def one(indexed):
        row, item = indexed
        frames = _frames(item[1])
        counts, effective = [], []
        offset = 0
        for loc, sp in spaces.items():
            df = frames.pop(loc, None)
            n = 0 if df is None else df.height
            counts.append(n)
            emb = sample_embedding(sp, df, weight=weight, blocks=blocks) if n else None
            matrix[row, offset:offset+widths[loc]] = emb.vector if emb else np.nan
            offset += widths[loc]
            effective.append(emb.n_eff if emb else np.nan)
            del df
        return counts, effective

    from vdjtools.cores import kernel_threads, require_memory
    require_memory(8 * len(items) * len(columns)
                   + min(len(items), kernel_threads(n_jobs)) * 32 * 1024**2,
                   "RFF sample reduction")
    # Workers write disjoint rows; column-contiguous storage passes to Polars without copying.
    matrix = np.empty((len(items), len(columns)), dtype=np.float64, order='F')
    compact = parallel_rows(enumerate(items), one, n_jobs)
    out = pl.DataFrame({'sample_id': ids,
                        'embedding_space': [_rff_space(spaces, weight)] * len(ids)})
    out = out.hstack(pl.from_numpy(matrix, schema=columns))
    for i, loc in enumerate(spaces):
        counts = [n[i] for n, _ in compact]
        out = out.with_columns(pl.Series(f'n_clonotypes:{loc}', counts),
                               pl.Series(f'present:{loc}', [int(n > 0) for n in counts]),
                               pl.Series(f'n_eff:{loc}', [n[i] for _, n in compact]))
    return out


def clonotype_embeddings(samples, models: dict, *, chunk: int = 4096):
    """Yield bounded raw clonotype tables with sample/clonotype IDs, without fitting.

    Samples resolve exactly once, one at a time. Missing clonotype IDs are assigned
    deterministic within-sample integers before batching; supplied IDs are retained.
    One native batch owns the models' CPU budget; there is no outer sample pool.
    """
    if chunk < 1:
        raise ValueError('chunk must be positive')
    space = embedding_space(models)
    for sid, sample in samples:
        frames = resolve_sample(sample)
        if not isinstance(frames, dict):
            frames = {loc: sub for (loc,), sub in frames.group_by('locus')}
        offset = 0
        for loc, df in frames.items():
            if loc not in models:
                continue
            df, _ = sanitise(df)
            if 'clonotype_id' not in df.columns:
                if 'sequence_id' in df.columns:
                    df = df.with_columns(pl.col('sequence_id').cast(pl.String).alias('clonotype_id'))
                else:
                    df = df.with_row_index('clonotype_id', offset=offset)
            df = df.with_columns(pl.col('clonotype_id').cast(pl.String))
            offset += df.height
            for start in range(0, df.height, chunk):
                part = df.slice(start, chunk)
                X = models[loc].embed(part)
                metadata = part.select([c for c in ('clonotype_id','junction_aa','v_call',
                                                    'j_call','duplicate_count') if c in part.columns])
                metadata = metadata.with_columns(pl.lit(sid).alias('sample_id'),
                                                  pl.lit(loc).alias('locus'),
                                                  pl.lit(space).alias('embedding_space'))
                yield metadata.hstack(pl.from_numpy(X, schema=[f'e{i}' for i in range(X.shape[1])]))
