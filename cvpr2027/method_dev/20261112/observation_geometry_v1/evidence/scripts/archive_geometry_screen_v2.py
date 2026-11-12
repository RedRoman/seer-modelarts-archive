"""Back up completed source-only geometry heads and evidence, excluding reused data caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    assert root == Path('/cache/zihaohe/seer_observation_geometry_20261112')
    complete = json.loads((root / 'screen_checks_v1/complete.json').read_text())
    assert complete['source_unchanged'] is True and complete['target_inference'] is False
    assert (root / 'launch_exit_v1.txt').read_text().strip() == '0'
    assert all(job['returncode'] == 0 for job in complete['jobs'])
    for mode in ('prior', 'pixel', 'shuffled_prior'):
        folder = root / 'runs_v1' / (mode + '_s0')
        metadata = json.loads((folder / 'complete.json').read_text())
        assert metadata['steps'] == 400 and metadata['frozen_node_unchanged']
        assert sha(folder / 'final.pt') == metadata['checkpoint_sha256']
        rows = [json.loads(line) for line in (folder / 'curve.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == list(range(1, 401))
    prepared = json.loads((root / 'prepared_v1/manifest.json').read_text())
    reference = Path(prepared['reference_head_path'])
    assert sha(reference) == prepared['reference_head_sha256']
    inputs = root / 'inputs'
    inputs.mkdir(exist_ok=False)
    anchor = inputs / 'frozen_raw_node_s0_final.pt'
    with reference.open('rb') as src, anchor.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    assert sha(anchor) == prepared['reference_head_sha256']
    write(inputs / 'dependencies.json', {
        'frozen_reference': {'original_path': str(reference), 'copied_path': anchor.name,
            'sha256': sha(anchor), 'bytes': anchor.stat().st_size},
        'source_cache_provenance_sha256': prepared['source_cache_provenance_sha256'],
        'memory_manifest_path': prepared['memory_manifest_path'],
        'memory_manifest_sha256': prepared['memory_manifest_sha256'],
        'reference_manifest_path': prepared['reference_manifest_path'],
        'reference_manifest_sha256': prepared['reference_manifest_sha256'],
        'existing_memory_backup': '/temp/zihaohe/seer_cvpr/method_dev_20261112/observation_memory_v1_final_v1',
        'copied_rgb_flow_address_cache': False})
    directories = ('release_v1', 'scripts', 'preflight', 'checks', 'screen_checks_v1',
                   'screen_logs_v1', 'runs_v1', 'results_v1', 'prepared_v1', 'inputs')
    sources = []
    for directory in directories:
        for source in sorted((root / directory).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.name.startswith('.'):
                continue
            if directory == 'prepared_v1':
                assert source.name == 'manifest.json', 'Do not archive any RGB cache'
            sources.append(source)
    for name in ('package_metadata_v1.json', 'delivery_check_v1.json', 'launch_v1.log', 'launch_exit_v1.txt'):
        sources.append(root / name)
    destination = Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/observation_geometry_v1_final_v1')
    total = sum(source.stat().st_size for source in sources)
    assert shutil.disk_usage(destination.parent).free > total + (1 << 30)
    destination.mkdir(exist_ok=False)
    inventory = {}
    for source in sources:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert sha(target) == digest and sha(source) == digest
        inventory[str(relative)] = {'sha256': digest, 'bytes': source.stat().st_size}
    manifest = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'source_root': str(root), 'backup_root': str(destination),
                'target_inference': False, 'copied_rgb_cache': False, 'files': inventory}
    write(destination / 'backup_manifest.json', manifest)
    archive = root / 'source_geometry_text_and_heads_v1.tar'
    with tarfile.open(archive, 'x') as tar:
        for source in sorted(destination.rglob('*')):
            if source.is_file():
                tar.add(source, arcname=str(source.relative_to(destination)), recursive=False)
    receipt = {'backup_root': str(destination), 'backed_up_files': len(inventory),
               'backed_up_bytes': total, 'backup_manifest_sha256': sha(destination / 'backup_manifest.json'),
               'local_bundle_sha256': sha(archive), 'local_bundle_bytes': archive.stat().st_size,
               'target_inference': False}
    write(root / 'backup_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
