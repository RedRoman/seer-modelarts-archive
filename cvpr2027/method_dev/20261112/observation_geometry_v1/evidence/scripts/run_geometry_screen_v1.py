"""CPU-only geometry screen framework. Final CLI spec is required before execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

MODES = ('prior', 'pixel', 'shuffled_prior')
PYTHON = '/cache/zihaohe/seer_cvpr/venv_npu/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def validate_spec(spec):
    assert spec['frozen'] is True
    assert spec['source_only'] is True and spec['target_inference'] is False
    assert spec['steps'] == 400 and spec['seed'] == 0
    assert spec['full_forward_resolution'] == 256
    assert spec['loss_crop'] == 128 and spec['supervised_frames'] == 8
    assert set(spec['train']) == set(MODES)
    for command in [spec['prepare'], *spec['train'].values(), spec['evaluate'], spec['analyze']]:
        assert isinstance(command, list) and command and all(isinstance(v, str) for v in command)
    assert spec['focused_cpu_tests']['failures'] == 0
    assert spec['focused_cpu_tests']['skips'] == 0
    assert spec['focused_cpu_tests']['tests_run'] > 0
    assert spec['actual_shape_smoke_passed'] is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True,
                        help='Frozen exact argv and expected output paths supplied after trainer CLI freeze')
    parser.add_argument('--execute', action='store_true', help='Without this flag only validate the spec')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    validate_spec(spec)
    if not args.execute:
        print(json.dumps({'spec_valid': True, 'executed': False, 'spec_sha256': sha(args.spec)}))
        return
    root, release = args.root, args.release
    code = release / 'code_full'
    manifest = json.loads((release / 'source_manifest.json').read_text())
    before = {name: sha(code / name) for name in manifest}
    assert before == manifest, 'Frozen release source changed'
    assert sha(release / 'PLAN.md') == spec['protocol_sha256']
    logs, checks = root / 'screen_logs_v1', root / 'screen_checks_v1'
    for path in (logs, checks, root / 'runs_v1', root / 'results_v1'):
        path.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               SEER_ROOT='/temp/zihaohe/seer_cvpr')
    env['PYTHONPATH'] = str(code) + ':' + str(code / 'tools') + ':' + env.get('PYTHONPATH', '')
    write(checks / 'launch.json', {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'runner_sha256': sha(__file__), 'spec_sha256': sha(args.spec),
          'protocol_sha256': spec['protocol_sha256'], 'source_sha256': before,
          'cpu_count': os.cpu_count(), 'loadavg': os.getloadavg(),
          'parallel_jobs': 3, 'threads_per_job': 4, 'target_inference': False})

    def run(label, argv):
        started = time.monotonic()
        log = logs / (label + '.log')
        with log.open('x') as stream:
            process = subprocess.run([PYTHON, *argv], cwd=code, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        result = {'label': label, 'argv': [PYTHON, *argv], 'returncode': process.returncode,
                  'elapsed_seconds': time.monotonic() - started, 'log_sha256': sha(log)}
        write(checks / (label + '.json'), result)
        print(json.dumps(result), flush=True)
        return result

    prepare = run('prepare', spec['prepare'])
    if prepare['returncode']:
        raise RuntimeError('Preparation failed; dependent training not started')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = list(pool.map(lambda mode: run(mode, spec['train'][mode]), MODES))
    if any(job['returncode'] for job in jobs):
        raise RuntimeError('Training incomplete; no holdout evaluation')
    for path in spec['training_complete_files']:
        completed = json.loads(Path(path).read_text())
        assert completed['steps'] == 400 and completed['frozen_node_unchanged'] is True
        assert sha(Path(path).parent / 'final.pt') == completed['checkpoint_sha256']
    evaluation = run('source_internal_evaluation', spec['evaluate'])
    if evaluation['returncode']:
        raise RuntimeError('Source internal evaluation failed')
    analysis = run('source_analysis', spec['analyze'])
    if analysis['returncode']:
        raise RuntimeError('Source analysis failed')
    after = {name: sha(code / name) for name in before}
    assert before == after, 'Frozen sources changed during execution'
    write(checks / 'complete.json', {'jobs': jobs, 'source_unchanged': True,
          'target_inference': False, 'raw_sha256': sha(spec['raw_result']),
          'analysis_sha256': sha(spec['analysis_result']),
          'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})


if __name__ == '__main__':
    main()
