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
                          'local_files_only': True}
        return tmp_path / name

    monkeypatch.setattr(reference, 'fetch_reference', fetch)
    main(['reference', 'rna-blood', '--revision', 'a'*40,
          '--cache-dir', str(tmp_path), '--offline'])
    assert capsys.readouterr().out.strip() == str(tmp_path / 'rna-blood')
