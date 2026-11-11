from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys

arm = sys.argv[1]
assert arm in ('pretrained', 'none')
root = Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1')
release = root / 'releases/v1_c65c0b4919cd'
shared = Path('/temp/zihaohe/seer_cvpr/method_dev_20261111')
runs = root / ('runs_cpu_v2' if arm == 'pretrained' else 'runs_none_cpu_v2')
result_name = 'bvi_val16_source_s0s1s2.json' if arm == 'pretrained' else 'bvi_val16_none_s0s1s2.json'
result = root / 'results' / result_name
data = json.loads(result.read_text())
assert data['n'] == 16 and len(data['per_sequence']) == 16 and len(data['selector_checkpoint_sha256']) == 18
assert all(data['state_unchanged'].values())
dest = shared / f'source_transfer_v1_{arm}_final'
dest.mkdir(exist_ok=False)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()

files = {}
def copy(source, relative):
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    digest = sha(source)
    assert sha(target) == digest
    files[str(relative)] = {'source': str(source), 'sha256': digest, 'bytes': source.stat().st_size}

for run in sorted(runs.iterdir()):
    assert (run / 'internal_holdout.json').is_file()
    for path in sorted(run.iterdir()):
        if path.is_file():
            copy(path, Path('runs') / run.name / path.name)
copy(result, Path('results') / result.name)
copy(release / 'source_manifest.json', Path('source_manifest.json'))
cache = root / 'cache_sdsd_p_s0' if arm == 'pretrained' else root / 'transfers/none_cache_payload_v1/cache'
for name in ('provenance.json', 'cache_summary.json'):
    copy(cache / name, Path('cache_metadata') / name)

if arm == 'pretrained':
    log_root = release / 'logs'
    check_root = release / 'checks'
    for path in sorted(log_root.iterdir()):
        if path.is_file() and ('none' not in path.name):
            copy(path, Path('logs') / path.name)
    for path in sorted(check_root.iterdir()):
        if path.suffix in ('.json', '.txt') and ('none' not in path.name):
            copy(path, Path('checks') / path.name)
    copy(root / 'transfers/failed_runtime_v1_evidence.tgz', Path('failed_runtime_v1_evidence.tgz'))
else:
    for subdir in ('logs', 'checks', 'tools'):
        for path in sorted((root / 'transfers/none_payload_v1' / subdir).glob('*')):
            if path.is_file():
                copy(path, Path('training_runtime') / subdir / path.name)
    for subdir in ('logs', 'checks', 'tools'):
        for path in sorted((root / 'transfers/none_cache_payload_v1' / subdir).glob('*')):
            if path.is_file():
                copy(path, Path('cache_runtime') / subdir / path.name)
    copy(release / 'logs/eval_none_18heads.log', Path('logs/eval_none_18heads.log'))
    for name in ('none_eval_launch_manifest.json', 'metric_runtime_none.json', 'npu_before_none_eval.txt', 'npu_after_none_eval.txt'):
        copy(release / 'checks' / name, Path('checks') / name)
    copy(root / 'transfers/none_received_manifest.json', Path('checks/none_received_manifest.json'))
for path in sorted((release / 'tools').glob('*.sh')):
    if ('none' in path.name) == (arm == 'none'):
        copy(path, Path('wrappers') / path.name)

base = shared / 'inputs' / f'e17_small_{arm}_s0_final_cpu.pt'
expected = '5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740' if arm == 'pretrained' else 'fa3c0c9e720a28b56f01069f7a2cd407bc5b4d752a2a1d4482f1bd63557b5516'
assert sha(base) == expected
manifest = {'arm': arm, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exclusive_directory': str(dest), 'files': files, 'files_verified': len(files), 'checkpoints': 18, 'source_archive_sha256': 'c65c0b4919cd8befdd850f9f96e5ce35565a45c3b3f5335ae2c50d00bebac14c', 'shared_base_checkpoint': {'path': str(base), 'sha256': expected, 'bytes': base.stat().st_size}, 'cache_tensor_files_copied': False}
text = json.dumps(manifest, indent=2) + '\n'
with (dest / 'backup_manifest.json').open('x') as stream:
    stream.write(text)
with (root / 'transfers' / f'backup_{arm}_manifest.json').open('x') as stream:
    stream.write(text)
print(json.dumps({'arm': arm, 'files': len(files), 'backup': str(dest), 'manifest_sha256': sha(dest / 'backup_manifest.json')}), flush=True)
