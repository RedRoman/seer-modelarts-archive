"""Exclusive SFS backup of canonical HVI bases and compact text evidence only."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


ROOT = Path('/cache/zihaohe/seer_color_contract_20261112')
DEST = Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/color_contract_rgb_hvi_source_v1_final_v1')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    report = json.loads((ROOT/'checks/export_outputs_verification.json').read_text())
    assert report['status'] == 'passed' and report['base_payloads_verified'] == 136
    assert (ROOT/'checks/export_npu_after.txt').exists(), 'Producer has not completed'
    assert not DEST.exists(), 'Backup directory is exclusive and must be new'
    DEST.mkdir(parents=True, exist_ok=False)
    paths = [p for p in (ROOT/'export_release_v1').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in (ROOT/'cache_rgb_hvi_v1').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'checks').glob('export_*') if p.is_file()]
    rows = []
    for source in sorted(set(paths)):
        relative = source.relative_to(ROOT)
        target = DEST/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, target)
        assert sha(target) == before == sha(source), relative
        rows.append({'path': str(relative), 'bytes': source.stat().st_size, 'sha256': before})
    manifest = {'purpose': 'canonical RGB HVI source136 exclusive SHA-verified backup',
        'files': rows, 'nfiles': len(rows), 'total_bytes': sum(row['bytes'] for row in rows),
        'base_payloads': sum(row['path'].startswith('cache_rgb_hvi_v1/') and row['path'].endswith('.pt') for row in rows),
        'no_original_gt_or_lq_copied': True, 'quality_scores_computed': False,
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    assert manifest['base_payloads'] == 136
    write_new(DEST/'export_backup_manifest.json', manifest)
    text_paths = [p for p in DEST.rglob('*') if p.is_file() and p.suffix != '.pt']
    archive = ROOT/'export_compact_text_v1.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in sorted(text_paths):
            bundle.add(path, arcname=str(path.relative_to(DEST)), recursive=False)
    receipt = {'destination': str(DEST), 'nfiles': len(rows), 'total_bytes': manifest['total_bytes'],
        'base_payloads': 136, 'all_hashes_verified': True,
        'backup_manifest_sha256': sha(DEST/'export_backup_manifest.json'),
        'text_archive': str(archive), 'text_archive_bytes': archive.stat().st_size,
        'text_archive_sha256': sha(archive), 'text_files': len(text_paths)}
    write_new(ROOT/'checks/export_backup_receipt.json', receipt)
    write_new(DEST/'export_backup_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
