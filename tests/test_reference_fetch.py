"""Reference bootstrap validates identity without loading models or using the network."""
import hashlib
import json

import pytest

from mir.embedding.reference import fetch_reference


def test_fetch_verifies_pinned_snapshot_and_offline_repeat(tmp_path, monkeypatch):
    import huggingface_hub
    root = tmp_path / 'rna-blood'
    root.mkdir()
    payload = root / 'feature-vocabulary.json'
    payload.write_text('{}')
    files = {payload.name: hashlib.sha256(b'{}').hexdigest()}
    for filename in ('features-pca.npz', 'rff-pca.npz', 'feature-columns.txt', 'basis/TRB.pkl'):
        (root / filename).parent.mkdir(exist_ok=True)
        (root / filename).write_bytes(b'fixture')
        files[filename] = hashlib.sha256(b'fixture').hexdigest()
    manifest = {'format': 1, 'loci': ['TRB'], 'files': files}
    (root / 'bundle.json').write_text(json.dumps(manifest))
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        if kwargs['force_download']:
            payload.write_text('{}')
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    for offline in (False, True):
        assert fetch_reference('rna-blood', revision='a'*40, cache_dir=tmp_path,
                               local_files_only=offline) == root
        assert calls[-1][1]['local_files_only'] == offline
        assert calls[-1][1]['revision'] == 'a'*40
        assert calls[-1][1]['allow_patterns'] == ['rna-blood/*']
    payload.write_text('corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch_reference('rna-blood', revision='a'*40)
    assert fetch_reference('rna-blood', revision='a'*40, force_download=True) == root
    assert calls[-1][1]['force_download']
    with pytest.raises(ValueError, match='online access'):
        fetch_reference('rna-blood', revision='a'*40, force_download=True, local_files_only=True)
    manifest['files']['../outside'] = '0'*64
    payload.write_text('{}')
    (root / 'bundle.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe payload path'):
        fetch_reference('rna-blood', revision='a'*40)
    with pytest.raises(ValueError, match='immutable'):
        fetch_reference('rna-blood', revision='main')
    with pytest.raises(ValueError, match='unknown reference'):
        fetch_reference('../other', revision='a'*40)


def test_reference_cli_passes_bootstrap_options(monkeypatch, capsys, tmp_path):
    from mir.cli import main
    import mir.embedding.reference as reference

    def fetch(name, **kwargs):
        assert name == 'rna-blood'
        assert kwargs == {'revision': 'a'*40, 'cache_dir': str(tmp_path),
                          'local_files_only': True, 'force_download': False}
        return tmp_path / name

    monkeypatch.setattr(reference, 'fetch_reference', fetch)
    main(['reference', 'rna-blood', '--revision', 'a'*40,
          '--cache-dir', str(tmp_path), '--offline'])
    assert capsys.readouterr().out.strip() == str(tmp_path / 'rna-blood')


def test_offline_miss_and_cli_errors_are_actionable(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from mir.cli import main

    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError('not staged')

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', missing)
    with pytest.raises(FileNotFoundError, match='mir reference rna-blood --revision'):
        fetch_reference('rna-blood', revision='a'*40, local_files_only=True)
    with pytest.raises(SystemExit, match='mir reference rna-blood --revision'):
        main(['reference', 'rna-blood', '--revision', 'a'*40, '--offline'])


def test_real_hub_offline_snapshot_without_mock_downloader(tmp_path):
    revision = 'b'*40
    root = tmp_path / 'datasets--isalgo--mirpy_models' / 'snapshots' / revision / 'rna-blood'
    root.mkdir(parents=True)
    files = {}
    for filename in ('features-pca.npz', 'rff-pca.npz', 'feature-columns.txt',
                     'feature-vocabulary.json', 'basis/TRB.pkl'):
        path = root / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'synthetic input fixture')
        files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'bundle.json').write_text(json.dumps({'format': 1, 'loci': ['TRB'], 'files': files}))
    assert fetch_reference('rna-blood', revision=revision, cache_dir=tmp_path,
                           local_files_only=True) == root
