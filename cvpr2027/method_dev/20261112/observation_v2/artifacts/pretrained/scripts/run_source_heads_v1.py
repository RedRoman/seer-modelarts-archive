"""Operational launcher; frozen method source remains unchanged."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/seer_observation_dev_20261112')
RELEASE = ROOT / 'method_release_v1'
CODE = RELEASE / 'release/code_full'
ARM = sys.argv[1]
assert ARM in ('pretrained', 'none')
PREPARED = ROOT / 'prepared' / ARM
OLD = Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1')
OLD_RUNS = OLD / ('runs_cpu_v2' if ARM == 'pretrained' else 'runs_none_cpu_v2')
sys.path[:0] = [str(CODE), str(CODE / 'tools')]
import train_observation_v2 as trainer

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)

assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] == '0'
assert all(os.environ[k] == '4' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
assert json.loads((RELEASE / 'checks/cpu_result.json').read_text())['tests'] == 133
manifest = json.loads((RELEASE / 'release/source_manifest.json').read_text())
for name, sha in manifest['files'].items():
    assert digest(CODE / name) == sha, name
meta = json.loads((PREPARED / 'prepared_manifest.json').read_text())
assert meta['n'] == 136 and meta['oracle_computed'] is False and meta['target_frames_read'] is False
norm = meta['normalization']['raw_detail']
parity = []
for seed in (0, 1, 2):
    path = OLD_RUNS / f'raw_detail_s{seed}/provenance.json'
    old = json.loads(path.read_text())
    assert old['split'] == meta['split']
    assert old['cache_provenance']['base_checkpoint_sha256'] == meta['source_cache_provenance']['base_checkpoint_sha256']
    assert norm.keys() == old['normalization'].keys()
    diffs = {}
    for key, value in norm.items():
        expected = old['normalization'][key]
        if isinstance(value, (int, float, list)):
            a, b = np.asarray(value), np.asarray(expected)
            assert a.shape == b.shape
            error = float(np.max(np.abs(a - b)))
            assert np.allclose(a, b, atol=1e-12, rtol=1e-7), (key, error)
            diffs[key] = error
        else:
            assert value == expected
    parity.append({'seed': seed, 'old_provenance_sha256': digest(path), 'exact_equal': norm == old['normalization'], 'max_abs_differences': diffs})
write(RELEASE / f'checks/raw_normalization_{ARM}_parity.json', {
    'status': 'PASS', 'atol': 1e-12, 'rtol': 1e-7,
    'prepared_manifest_sha256': digest(PREPARED / 'prepared_manifest.json'), 'comparisons': parity})

torch.set_num_threads(4)
model = trainer.modules()['trajectory'].TrajectoryReconstruction(mode='conditioned')
assert all(p.device.type == 'cpu' for p in model.parameters())
param = torch.nn.Parameter(torch.ones(1))
optimizer = torch.optim.Adam([param])
param.square().sum().backward()
optimizer.step()
assert all(t.device.type == 'cpu' for state in optimizer.state.values() for t in state.values() if isinstance(t, torch.Tensor))
assert 'torch_npu' not in sys.modules
write(RELEASE / f'checks/cpu_optimizer_{ARM}.json', {
    'status': 'PASS', 'all_model_parameters_cpu': True, 'all_adam_state_cpu': True,
    'torch_npu_imported': False, 'torch': str(torch.__version__), 'numpy': str(np.__version__)})
del model, optimizer, param

RUNS = ROOT / 'runs' / ARM
RUNS.mkdir(parents=True, exist_ok=False)
LOGS = RELEASE / 'logs' / f'training_{ARM}'
LOGS.mkdir(exist_ok=False)
children = {}
records = []
def launch(variant, seed):
    name = f'{variant}_s{seed}'
    args = [sys.executable, '-B', str(CODE / 'tools/train_observation_v2.py'), 'train',
            '--prepared', str(PREPARED), '--variant', variant, '--seed', str(seed), '--out', str(RUNS / name)]
    handle = (LOGS / f'{name}.log').open('x')
    child = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT)
    children[name] = (child, handle)
    row = {'name': name, 'pid': child.pid, 'command': args, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    records.append(row)
    print(json.dumps({'launched': row}), flush=True)
    return name

first = [launch('raw_node', 0), launch('trajectory', 0)]
deadline = time.monotonic() + 1200
while True:
    for name in first:
        code = children[name][0].poll()
        if code is not None and code != 0:
            raise RuntimeError(f'{name} failed with code {code}; remaining heads not launched')
    ready = all((RUNS / name / 'curve.jsonl').exists() and (RUNS / name / 'curve.jsonl').stat().st_size > 0 for name in first)
    if ready:
        break
    if time.monotonic() > deadline:
        raise TimeoutError('First true training step pending after 20 minutes; remaining heads not launched')
    time.sleep(3)
first_rows = {}
for name in first:
    with (RUNS / name / 'curve.jsonl').open() as stream:
        row = json.loads(stream.readline())
    assert row['step'] == 1 and all(math.isfinite(row[k]) for k in ('loss', 'rgb_mse', 'base_mse'))
    first_rows[name] = row
p0, p1 = [json.loads((RUNS / name / 'provenance.json').read_text()) for name in first]
assert p0['sample_plan_sha256'] == p1['sample_plan_sha256']
assert p0['initial_state_sha256'] == p1['initial_state_sha256']
write(RELEASE / f'checks/first_steps_{ARM}.json', {
    'status': 'PASS', 'rows': first_rows, 'matching_plan_sha256': p0['sample_plan_sha256'],
    'matching_initial_state_sha256': p0['initial_state_sha256'], 'source_only': True,
    'steps_per_head_unchanged': 1000})
print(json.dumps({'first_steps': 'PASS', 'arm': ARM}), flush=True)
for seed in (0, 1, 2):
    for variant in trainer.VARIANTS:
        if f'{variant}_s{seed}' in children:
            continue
        launch(variant, seed)
        time.sleep(2)
write(RELEASE / f'checks/training_{ARM}_launch_manifest.json', {
    'arm': ARM, 'head_count': 21, 'source_archive_sha256': '6483532a2423d8427b07157e5eca463c378d43c6c51a5b70838f423a3692e996',
    'launcher_sha256': digest(__file__), 'prepared_manifest_sha256': digest(PREPARED / 'prepared_manifest.json'),
    'children': records, 'environment': {k: os.environ.get(k) for k in ('TORCH_DEVICE_BACKEND_AUTOLOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONPATH')}})
exit_codes = {}
for name, (child, handle) in children.items():
    exit_codes[name] = child.wait()
    handle.close()
write(RELEASE / f'checks/training_{ARM}_exit_codes.json', exit_codes)
assert all(code == 0 for code in exit_codes.values()), exit_codes
for name in children:
    complete = json.loads((RUNS / name / 'complete.json').read_text())
    assert complete['steps'] == 1000 and digest(RUNS / name / 'final.pt') == complete['checkpoint_sha256']
print(json.dumps({'all_heads_complete': 21, 'arm': ARM}), flush=True)
