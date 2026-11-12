"""Final identity verification reads source files as bytes, never as GT tensors."""
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path('/cache/zihaohe/seer_color_contract_20261112')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert (ROOT/'checks/export_npu_after.txt').exists(), 'Wait for producer completion'
    release = ROOT/'export_release_v1'
    manifest = json.loads((release/'export_release_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert sha(release/relative) == expected, relative
    protocol = json.loads((release/'source_export_protocol.json').read_text())
    source = Path(protocol['source_cache'])
    assert sha(source/'provenance.json') == protocol['source_provenance_sha256']
    assert sha(source/'cache_summary.json') == protocol['source_summary_sha256']
    input_rows = json.loads((release/'source_input_manifest.json').read_text())['per_sequence']
    for row in input_rows:
        assert sha(source/row['cache_path']) == row['cache_sha256'], row['name']
    old = Path('/cache/zihaohe/seer_hvi_source_anchor_20261112/release_v1')
    assert sha(old/'release_manifest.json') == protocol['legacy_release_manifest_sha256']
    old_manifest = json.loads((old/'release_manifest.json').read_text())
    for relative, expected in old_manifest['files'].items():
        assert sha(old/relative) == expected, relative
    active = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            argv = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, PermissionError):
            continue
        if len(argv) > 1 and Path(argv[0]).name.startswith('python') and argv[1].endswith('cache_hvi_rgb_source.py'):
            active.append({'pid': int(process.name), 'argv': argv})
    assert not active, 'Export process still active'
    cache_manifest = ROOT/'cache_rgb_hvi_v1/manifest.json'
    cache = json.loads(cache_manifest.read_text())
    assert cache['complete'] and cache['model_state_unchanged']
    report = {'status': 'passed', 'release_files_unchanged': len(manifest['files']),
        'old_release_files_unchanged': len(old_manifest['files']), 'source_payload_sha_rechecks': len(input_rows),
        'source_rechecks_are_raw_file_identity_only': True, 'source_tensors_deserialized': 0,
        'cache_manifest_sha256': sha(cache_manifest), 'active_export_processes': active,
        'weight_sha256': protocol['weight_sha256'], 'state_sha256': protocol['model_state_sha256'],
        'gt_storage_entries_read': 0, 'quality_scores_computed': False,
        'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (ROOT/'checks/export_final_identity_verification.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
