"""Fit separate statistical-feature and RFF reference rotations from fresh readers.

Call ``build_reference_rotations(readers, loci, output)`` with repeatable ``(sample_id, reader)`` pairs.
Readers return productive canonical locus frames. Input manifests and membership
stay private; only rotations, feature names and the required RFF bases are exported.
No signature corpus is fitted or modified, and no feature intermediates are saved.
"""
from pathlib import Path
import hashlib
import json
import shutil

import mir  # noqa: F401 — set runtime limits before numerical imports
import numpy as np
from vdjtools.cores import kernel_threads, serial_kernels
from vdjtools.signature import raw_columns, raw_features_cohort
from vdjtools.signature.features import gene_vocab
from vdjtools.signature.layout import LOCI, NO_LOCUS
from vdjtools.signature.corpus import TablePCA
from mir.embedding.table import fit_sample_spaces, sample_embeddings
from mir.embedding.tcremp import TCREmp
from mir.repertoire import RepertoireSpace


def build_reference_rotations(readers, loci, output, *, jobs=1, components=1024, n_rff=1024,
          fit_cap=200_000, coverage=0.8, seed=20261012, reuse_basis=None):
    """Build a new bundle; refuse to overwrite an existing output directory.

    Each bundle has its own shared clonotype basis. To apply its RFF rotation,
    embed new samples through the bases in that same bundle first. Statistical
    features require the same vocabulary, abundance weighting and coverage target.
    ``reuse_basis`` copies an existing basis directory byte-for-byte and skips
    clonotype fitting. Both sample-table PCAs are fitted from fresh reader outputs.
    """
    readers, loci = list(readers), tuple(loci)
    jobs = kernel_threads(jobs)
    ids = [sid for sid, _ in readers]
    if (len(readers) < 2 or len(set(ids)) != len(ids)
            or not loci or len(set(loci)) != len(loci)):
        raise ValueError('provide at least two unique samples and distinct loci')
    reused = Path(reuse_basis) if reuse_basis is not None else None
    spaces = None
    if reused is not None:
        spaces = {p.stem: RepertoireSpace.load(p) for p in sorted(reused.glob('*.pkl'))}
        if not spaces or not set(spaces).issubset(loci):
            raise ValueError('saved basis loci differ from requested reference')
        if any(sp.rff.dim != n_rff or sp.rff2 is not None for sp in spaces.values()):
            raise ValueError('saved basis RFF dimensions differ from requested reference')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    # Match `vdjtools features` even when the reference observes only TRA/TRB.
    vocab = {loc: gene_vocab(loc, 'human') for loc in LOCI}
    (out / 'feature-vocabulary.json').write_text(json.dumps(vocab, indent=2) + '\n')
    table = raw_features_cohort(readers, vocab=vocab, n_jobs=jobs, cstar_target=coverage)
    columns = [c for loc in (*loci, NO_LOCUS)
               for c in raw_columns('vsig', loc, vocab.get(loc))]
    with serial_kernels():
        rotation = TablePCA.fit(table, columns=columns, n_components=components, seed=seed)
        rotation.save(out / 'features-pca.npz')
        _check(rotation, table, out / 'features-pca.npz')
    variance = {'features': _variance(rotation)}
    (out / 'feature-columns.txt').write_text('\n'.join(columns) + '\n')
    del table, rotation

    # One native batch during basis fitting, then one serial kernel per sample worker.
    if spaces is None:
        models = {loc: TCREmp.from_defaults('human', loc, threads=jobs) for loc in loci}
        spaces = fit_sample_spaces(readers, models, pca_fit_cap=fit_cap,
                                   n_rff=n_rff, n_rff_second=0, seed=seed)
    for space in spaces.values():
        space.clono.model.threads = 1
    basis = out / 'basis'
    basis.mkdir()
    for loc, space in spaces.items():
        if reused is None:
            space.save(basis / f'{loc}.pkl')
        else:
            shutil.copyfile(reused / f'{loc}.pkl', basis / f'{loc}.pkl')
    table = sample_embeddings(readers, spaces, n_jobs=jobs)
    with serial_kernels():
        rotation = TablePCA.fit(table, n_components=components, seed=seed)
        rotation.save(out / 'rff-pca.npz')
        _check(rotation, table, out / 'rff-pca.npz')
    variance['rff'] = _variance(rotation)
    restored = {loc: RepertoireSpace.load(basis / f'{loc}.pkl') for loc in spaces}
    again = sample_embeddings(readers[:2], restored, n_jobs=1)
    if table['embedding_space'][0] != again['embedding_space'][0]:
        raise AssertionError('saved RFF bases changed coordinate identity')
    phi = [c for c in table.columns if c.startswith('phi:')]
    np.testing.assert_allclose(table.head(2).select(phi).to_numpy(),
                               again.select(phi).to_numpy(), rtol=1e-10, atol=1e-12,
                               equal_nan=True)
    import sklearn
    import vdjtools

    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'bundle.json').write_text(json.dumps({
        'format': 1, 'representations': ['vdjtools-features', 'mirpy-rff'],
        'species': 'human', 'loci': list(spaces), 'seed': seed,
        'weight': 'log2p1', 'coverage': coverage, 'n_rff': n_rff,
        'fit_cap': fit_cap, 'requested_components': components,
        'versions': {'mirpy': mir.__version__, 'vdjtools': vdjtools.__version__,
                     'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'variance': variance, 'files': hashes,
    }, indent=2) + '\n')


def _check(rotation, table, path):
    """Reload and project a small view without allocating a second cohort matrix."""
    rows = table.head(2)
    restored = TablePCA.load(path)
    assert restored.columns == rotation.columns and restored.space == rotation.space
    for name in ('fill', 'mean', 'scale', 'center', 'components', 'explained_variance_ratio'):
        np.testing.assert_array_equal(getattr(restored, name), getattr(rotation, name))
    expected = rotation.transform(rows)
    actual = restored.transform(rows)
    assert actual['sample_id'].to_list() == rows['sample_id'].to_list()
    pcs = [c for c in actual.columns if c.startswith('pc')]
    scores = expected.select(pcs).to_numpy()
    # Identical arrays can take different BLAS paths after NPZ changes their layout.
    # Parameters stay bit-exact; only projection roundoff has this float64 bound.
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(scores))))
    np.testing.assert_allclose(actual.select(pcs).to_numpy(), scores,
                               rtol=0, atol=tolerance)


def _variance(rotation):
    return {'components': rotation.components.shape[0],
            'cumulative': {str(k): float(rotation.explained_variance_ratio[:k].sum())
                           for k in (256, 512, 1024)}}
