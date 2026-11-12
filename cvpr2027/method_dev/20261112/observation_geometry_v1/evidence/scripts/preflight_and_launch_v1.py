"""Verify the frozen CPU release, run eight trainer tests, then the authorized source screen."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path('/cache/zihaohe/seer_observation_geometry_20261112')
RELEASE = ROOT / 'release_v1'
CODE = RELEASE / 'code_full'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    checks = ROOT / 'checks'
    checks.mkdir(exist_ok=False)
    manifest = json.loads((RELEASE / 'source_manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(CODE / name) == digest, name
        ast.parse((CODE / name).read_bytes(), filename=name)
    env_expected = {'TORCH_DEVICE_BACKEND_AUTOLOAD': '0', 'OMP_NUM_THREADS': '4',
                    'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    assert all(os.environ.get(k) == v for k, v in env_expected.items())
    sys.path[:0] = [str(CODE), str(CODE / 'tools')]
    tests = unittest.TestLoader().discover(str(CODE / 'tests'), pattern='test_train_observation_geometry.py')
    started = time.monotonic()
    with (checks / 'trainer_cpu_tests.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'skips': len(result.skipped),
              'elapsed_seconds': time.monotonic() - started,
              'log_sha256': sha(checks / 'trainer_cpu_tests.log'),
              'source_manifest_sha256': sha(RELEASE / 'source_manifest.json')}
    write(checks / 'trainer_cpu_tests.json', report)
    print(json.dumps(report), flush=True)
    if result.testsRun != 8 or not result.wasSuccessful() or result.skipped:
        raise RuntimeError('Focused trainer preflight failed; no dependent computation started')
    model_report = json.loads((ROOT / 'preflight/module/result.json').read_text())
    smoke = json.loads((ROOT / 'preflight/module/full256_smoke.json').read_text())
    assert model_report['passed'] is True and model_report['tests'] == 10
    assert model_report['failures'] == model_report['errors'] == model_report['skipped'] == 0
    for name, digest in model_report['source_sha256'].items():
        assert manifest[name] == digest
    assert smoke['model_sha256'] == manifest['models/observation_geometry.py']
    assert smoke['initial_anchor_exact'] and smoke['frozen_node_unchanged'] and smoke['finite_trainable_gradients']
    assert smoke['shape'] == [1, 3, 16, 256, 256] and smoke['hidden_dim'] == 16
    # The previously verified owner evidence is copied unchanged; do not rerun it.
    write(checks / 'combined_preflight.json', {'model_cpu_tests': model_report,
          'trainer_cpu_tests': report, 'model_full256_smoke': smoke,
          'actual_cpu_tests_total': 18, 'target_inference': False,
          'source_sha256_verified': True, 'python_files': len(manifest)})
    spec = json.loads((ROOT / 'scripts/screen_spec_template.json').read_text())
    spec['frozen'] = True
    spec['focused_cpu_tests'] = report
    spec['actual_shape_smoke_passed'] = True
    spec['protocol_sha256'] = sha(RELEASE / 'PLAN.md')
    write(ROOT / 'scripts/screen_spec_frozen.json', spec)
    argv = [sys.executable, str(ROOT / 'scripts/run_geometry_screen_v1.py'),
            '--root', str(ROOT), '--release', str(RELEASE),
            '--spec', str(ROOT / 'scripts/screen_spec_frozen.json'), '--execute']
    process = subprocess.run(argv, cwd=CODE)
    write(checks / 'runner_exit.json', {'returncode': process.returncode,
          'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
