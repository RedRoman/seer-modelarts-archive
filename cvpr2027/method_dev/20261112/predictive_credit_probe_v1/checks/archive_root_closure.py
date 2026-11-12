"""Exclusive verified SFS copy of final UTF-8 evidence, no pixels or weights."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('archive', 'backup', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    args = parser.parse_args()
    assert args.backup.is_absolute() and args.backup.resolve().is_relative_to(
        Path('/temp/zihaohe/seer_cvpr/method_dev_20261112'))
    assert args.receipt.is_absolute() and args.receipt.resolve().is_relative_to(Path('/cache'))
    assert not args.backup.exists() and not args.receipt.exists()
    archive_bytes = args.archive.read_bytes()
    assert sha(archive_bytes) == args.archive_sha256
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            assert member.name not in contents and not name.name.startswith('._')
            contents[member.name] = archive.extractfile(member).read()
            contents[member.name].decode('utf-8')
    manifest_name = 'root_closure_manifest.json'
    manifest = json.loads(contents[manifest_name])
    assert set(contents) == {row['path'] for row in manifest['files']} | {manifest_name}
    assert len(manifest['files']) == len(contents) - 1
    for row in manifest['files']:
        assert len(contents[row['path']]) == row['bytes']
        assert sha(contents[row['path']]) == row['sha256']
    args.backup.mkdir(parents=True, exist_ok=False)
    for name, data in sorted(contents.items()):
        target = args.backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(data)
        assert sha(target.read_bytes()) == sha(data)
    receipt = {'status': 'passed', 'scope': 'UTF-8 source/protocol/scalar JSON/log evidence only; no image/GT pixels or model weights',
               'backup_root': str(args.backup), 'payload_files': len(manifest['files']),
               'payload_bytes': sum(row['bytes'] for row in manifest['files']),
               'all_file_hashes_verified': True, 'archive_sha256': args.archive_sha256,
               'backup_manifest_sha256': sha(contents[manifest_name]),
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
