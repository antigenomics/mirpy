"""Raw sample embedding tables and separately fitted, portable PCA rotations.

Sample embedding is a bounded map over independent samples. It fits nothing:
only fixed-prototype weighted means survive. PCA consumes that compact table.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import polars as pl
from vdjtools.signature.cohort import parallel_rows, resolve_sample
from vdjtools.signature.features import sanitise

from mir.signature.features import prototype_sum, weights


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


def sample_embeddings(samples, models: dict, *, weight: str = 'log2p1',
                      n_jobs: int = 1) -> pl.DataFrame:
    """One raw weighted-prototype row per sample, with identity and missing-locus masks.

    ``samples`` is an iterable of ``(sample_id, frame_or_deferred_reader)`` pairs.
    Each resolved sample is ``{locus: AIRR frame}``. Shared models must use automatic
    ``threads=0`` so the bounded sample workers own the compute budget. No scaler,
    PCA, random-feature fit or cohort of clonotype frames is materialized. Models
    define the same column set for training and future cohorts. Empty loci remain
    missing (NaN), with an explicit presence indicator; samples are never dropped.
    """
    from copy import copy
    from mir.distances.shm import MutatedGermlineDistances

    if not models or any(m.threads != 0 for m in models.values()):
        raise ValueError('sample workers require models with automatic threads=0')
    # Prepare immutable panel indices and the SHM adapter before sharing models across workers.
    models = {loc: copy(m) for loc, m in models.items()}
    for model in models.values():
        if not isinstance(model._germline, MutatedGermlineDistances):
            model._germline = MutatedGermlineDistances(model._germline, scale=model.shm_scale)
        for comp in ('V', 'J') if model.mode == 'vjcdr3' else ('CDR1', 'CDR2'):
            model._proto_idx(comp)
    items = list(samples.items() if isinstance(samples, dict) else samples)
    ids = [sid for sid, _ in items]
    if not items or len(set(ids)) != len(ids):
        raise ValueError('sample ids must be nonempty and unique')
    cols = [f'phi:{loc}:{i}' for loc, m in models.items() for i in range(m.n_features)]

    def one(item):
        frames = resolve_sample(item[1])
        if not isinstance(frames, dict):
            frames = {loc: sub for (loc,), sub in frames.group_by('locus')}
        values, counts = [], []
        for loc, model in models.items():
            df = frames.get(loc)
            if df is not None:
                df, _ = sanitise(df)
            n = 0 if df is None else df.height
            counts.append(n)
            if n:
                w = weights(df['duplicate_count'].to_numpy(), weight)
                phi, _ = prototype_sum(df, model, w)
            else:
                phi = np.full(model.n_features, np.nan)
            values.append(phi)
        return np.concatenate(values), counts

    compact = parallel_rows(items, one, n_jobs)
    matrix = np.vstack([row for row, _ in compact])
    out = pl.DataFrame({'sample_id': ids,
                        'embedding_space': [embedding_space(models, weight=weight)] * len(ids)})
    out = out.hstack(pl.from_numpy(matrix, schema=cols))
    for i, loc in enumerate(models):
        counts = [n[i] for _, n in compact]
        out = out.with_columns(pl.Series(f'n_clonotypes:{loc}', counts),
                               pl.Series(f'present:{loc}', [int(n > 0) for n in counts]))
    return out


def _space(table: pl.DataFrame) -> str:
    if 'embedding_space' not in table.columns or table['embedding_space'].null_count():
        raise ValueError('embedding space identity is missing')
    ids = table['embedding_space'].unique().to_list()
    if len(ids) != 1:
        raise ValueError('table mixes incompatible embedding spaces')
    return ids[0]


@dataclass
class TablePCA:
    """A fitted sample-table scaler and PCA, with column and prototype-space checks.

    Only the explicit fitted rotation is saved, never sample-derived intermediates.
    Missing coordinates use training medians (zero for wholly unobserved columns);
    explicit presence columns retain missing-locus information in the fit.
    """
    columns: list[str]
    space: str
    fill: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    center: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(cls, table: pl.DataFrame, *, n_components: int = 256, seed: int = 0):
        """Fit on the compact sample rows, retaining all samples and explicit masks."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        space = _space(table)
        cols = [c for c in table.columns if c.startswith(('phi:', 'present:'))]
        if not cols or table.height < 2 or n_components < 1:
            raise ValueError('PCA needs feature columns, at least two sample rows and a positive dimension')
        X = table.select(cols).to_numpy(order='c').copy()
        X[~np.isfinite(X)] = np.nan
        fill = np.zeros(len(cols))
        observed = np.isfinite(X).any(axis=0)
        fill[observed] = np.nanmedian(X[:, observed], axis=0)
        missing = np.isnan(X)
        np.copyto(X, fill, where=missing)
        del missing
        scaler = StandardScaler(copy=False).fit(X)
        scaler.transform(X, copy=False)
        pca = PCA(n_components=min(n_components, *X.shape), random_state=seed,
                  copy=False).fit(X)
        return cls(cols, space, fill, scaler.mean_, scaler.scale_, pca.mean_, pca.components_)

    def transform(self, table: pl.DataFrame) -> pl.DataFrame:
        """Apply this exact rotation; preserve sample IDs and metadata in input order."""
        if _space(table) != self.space:
            raise ValueError('embedding space differs from the fitted rotation')
        got = [c for c in table.columns if c.startswith(('phi:', 'present:'))]
        if set(got) != set(self.columns):
            raise ValueError('embedding columns differ from the fitted rotation')
        X = table.select(self.columns).to_numpy(order='c')
        X = np.where(np.isfinite(X), X, self.fill)
        X -= self.mean
        X /= self.scale
        X -= self.center
        scores = X @ self.components.T
        metadata = table.select([c for c in table.columns if c not in self.columns])
        return metadata.hstack(pl.from_numpy(scores, schema=[f'pc{i}' for i in range(scores.shape[1])]))

    def save(self, path) -> None:
        """Write an explicit NumPy artifact (no pickle or executable model payload)."""
        with open(path, 'wb') as f:
            np.savez_compressed(f, columns=np.array(self.columns), space=np.array(self.space),
                                fill=self.fill, mean=self.mean, scale=self.scale,
                                center=self.center, components=self.components)

    @classmethod
    def load(cls, path):
        """Load and validate a fitted rotation without executing arbitrary pickle data."""
        with np.load(path, allow_pickle=False) as z:
            if z['columns'].ndim != 1 or z['space'].ndim != 0:
                raise ValueError('invalid PCA artifact metadata')
            result = cls(z['columns'].tolist(), str(z['space']),
                         *(z[key].copy() for key in ('fill','mean','scale','center','components')))
        n = len(result.columns)
        if (not n or not result.space or len(set(result.columns)) != n
                or result.components.ndim != 2 or result.components.shape[0] < 1
                or result.components.shape[1] != n
                or any(a.shape != (n,) for a in (result.fill,result.mean,result.scale,result.center))
                or not all(np.isfinite(a).all() for a in
                           (result.fill,result.mean,result.scale,result.center,result.components))
                or (result.scale <= 0).any()):
            raise ValueError('invalid PCA artifact dimensions or numerical values')
        return result


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
