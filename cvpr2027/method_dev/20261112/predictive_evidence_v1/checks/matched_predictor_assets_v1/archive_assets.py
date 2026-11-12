"""Exclusive complete-asset SFS backup and compact evidence archive, no dataset access."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/cache/zihaohe/seer_matched_predictor_assets_20261112_v1')
DEST = Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/matched_predictor_assets_v1_final_v1')
SHA = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    interface = json.loads((ROOT/'evidence/synthetic_interface.json').read_text())
    assert interface['status'] == 'passed' and interface['states_unchanged']
    assert interface['asset_sha256'] == SHA
    frozen = json.loads((ROOT/'evidence/workflow_frozen_manifest.json').read_text())
    for relative, expected in frozen['files'].items():
        assert sha(ROOT/relative) == expected, relative
    assert not DEST.exists()
    paths = [path for path in ROOT.rglob('*') if path.is_file() and 'weights' not in path.parts
             and '__pycache__' not in path.parts and path.name != 'assets_compact_v1.tar.gz']
    paths += [ROOT/'weights/vitl.pt']
    DEST.mkdir(parents=True, exist_ok=False)
    rows = []
    for source in sorted(paths):
        relative = source.relative_to(ROOT)
        target = DEST/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        if source.name == 'vitl.pt':
            assert before == SHA
        shutil.copy2(source, target)
        assert sha(target) == before
        rows.append({'path': str(relative), 'bytes': source.stat().st_size, 'sha256': before})
    manifest = {'files': rows, 'nfiles': len(rows), 'bytes': sum(row['bytes'] for row in rows),
        'checkpoint_sha256': SHA, 'all_hashes_verified': True,
        'incomplete_transfer_parts_backed_up': False, 'dataset_pixels_read': 0,
        'scope': 'matched V-JEPA2-L asset and CPU synthetic API evidence only',
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (DEST/'backup_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    archive = ROOT/'assets_compact_v1.tar.gz'
    text = [path for path in DEST.rglob('*') if path.is_file() and path.suffix != '.pt']
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in sorted(text):
            bundle.add(path, arcname=str(path.relative_to(DEST)), recursive=False)
    receipt = {'destination': str(DEST), 'nfiles': len(rows), 'bytes': manifest['bytes'],
        'checkpoint_sha256': SHA, 'all_hashes_verified': True,
        'backup_manifest_sha256': sha(DEST/'backup_manifest.json'), 'text_archive': str(archive),
        'text_archive_sha256': sha(archive), 'text_archive_bytes': archive.stat().st_size,
        'text_files': len(text), 'pt_files_in_text_archive': 0}
    for target in [ROOT/'evidence/backup_receipt.json', DEST/'backup_receipt.json']:
        with target.open('x') as stream:
            json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
