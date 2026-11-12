"""Orchestration-only v2: separate FIT parity payload from its job receipt.

The original release and failed preflight records remain unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT=Path('/cache/zihaohe/seer_joint_search_20261112')
RELEASE=ROOT/'release_v1'
CODE=RELEASE/'code_full'
MODES=('joint_prior','independent_prior','constant_prior','joint_pixel','joint_shuffled')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    manifest=json.loads((RELEASE/'source_manifest.json').read_text())
    assert all(sha(CODE/n)==h for n,h in manifest.items()),'Frozen code changed'
    freeze=json.loads((RELEASE/'freeze.json').read_text())
    assert all(sha(RELEASE/n)==h for n,h in freeze['release_files'].items())
    assert all(os.environ[k]==v for k,v in {'TORCH_DEVICE_BACKEND_AUTOLOAD':'0',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items())
    for name in ('runs_v1','results_v1'):
        assert (ROOT/name).is_dir() and not any((ROOT/name).iterdir()),'No partial training/evaluation reuse'
    for name in ('checks_v2','screen_logs_v2'):(ROOT/name).mkdir(exist_ok=False)
    checks,logs=ROOT/'checks_v2',ROOT/'screen_logs_v2'
    write(checks/'launch.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'runner_sha256':sha(__file__),'freeze_sha256':sha(RELEASE/'freeze.json'),
        'source_manifest_sha256':sha(RELEASE/'source_manifest.json'),
        'parallel_jobs':5,'threads_per_job':4,'cpu_count':os.cpu_count(),'loadavg':os.getloadavg(),
        'source_only':True,'target_inference':False})
    sys.path[:0]=[str(CODE),str(CODE/'tools')]
    suite=unittest.TestSuite()
    for pattern in ('test_joint_observation_search.py','test_cache_joint_observation_search.py',
                    'test_train_joint_observation_search.py'):
        suite.addTests(unittest.TestLoader().discover(str(CODE/'tests'),pattern=pattern))
    start=time.monotonic()
    with (checks/'all_cpu_tests.log').open('x') as f:
        test=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    report={'tests_run':test.testsRun,'failures':len(test.failures),'errors':len(test.errors),
        'skips':len(test.skipped),'elapsed_seconds':time.monotonic()-start,
        'log_sha256':sha(checks/'all_cpu_tests.log')}
    write(checks/'all_cpu_tests.json',report);print(json.dumps(report),flush=True)
    assert test.testsRun==28 and test.wasSuccessful() and not test.skipped
    while not (ROOT/'prepare_logs/complete.json').exists():time.sleep(10)
    prepare=json.loads((ROOT/'prepare_logs/complete.json').read_text())
    assert prepare['returncode']==0,'No training after failed preparation'
    before={n:sha(CODE/n) for n in manifest}
    env=dict(os.environ,PYTHONPATH=str(CODE)+':'+str(CODE/'tools'),PYTHONUNBUFFERED='1')

    def run(label,argv):
        started=time.monotonic()
        with (logs/(label+'.log')).open('x') as f:
            process=subprocess.run([sys.executable,*argv],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
        result={'argv':[sys.executable,*argv],'returncode':process.returncode,
            'elapsed_seconds':time.monotonic()-started,'log_sha256':sha(logs/(label+'.log'))}
        write(checks/(label+'.json'),result);print(json.dumps({'job':label,**result}),flush=True)
        if process.returncode:raise RuntimeError('Job failed: '+label)
        return result

    trainer='tools/train_joint_observation_search.py'
    prepared=str(ROOT/'prepared_v1')
    run('fit_anchor_parity_job',[trainer,'preflight','--prepared',prepared,'--out',str(checks/'fit_anchor_parity.json')])
    parity=json.loads((checks/'fit_anchor_parity.json').read_text())
    assert parity['passed'] and parity['trainable_parameters']==4 and parity['frozen_parameters']==6626
    assert parity['holdout_metric_computed'] is False and parity['target_frames_read'] is False
    def train(mode):return run(mode,[trainer,'train','--prepared',prepared,'--out',
        str(ROOT/'runs_v1'/(mode+'_s0')),'--mode',mode,'--seed','0'])
    with ThreadPoolExecutor(max_workers=5) as pool:jobs=list(pool.map(train,MODES))
    for mode in MODES:
        folder=ROOT/'runs_v1'/(mode+'_s0');done=json.loads((folder/'complete.json').read_text())
        assert done['steps']==400 and done['frozen_node_unchanged'] is True
        assert sha(folder/'final.pt')==done['checkpoint_sha256']
        curves=[json.loads(line) for line in (folder/'curve.jsonl').read_text().splitlines()]
        assert [r['step'] for r in curves]==list(range(1,401))
    raw=ROOT/'results_v1/source_internal_raw.json';analysis=ROOT/'results_v1/source_internal_analysis.json'
    run('source_evaluation',[trainer,'evaluate','--prepared',prepared,'--runs',str(ROOT/'runs_v1'),'--out',str(raw)])
    run('analysis',[str(RELEASE/'analyze_screen.py'),'--input',str(raw),'--out',str(analysis)])
    run('separate_verification',[str(RELEASE/'separate_verify.py'),'--raw',str(raw),
        '--analysis',str(analysis),'--analysis-source',str(RELEASE/'analyze_screen.py'),
        '--out',str(ROOT/'results_v1/separate_verification.json')])
    run('reference_repeat',[str(RELEASE/'check_reference_repeat.py'),'--raw',str(raw),
        '--previous','/cache/zihaohe/seer_observation_geometry_20261112/results_v1/source_internal_raw.json',
        '--out',str(ROOT/'results_v1/reference_repeat_verification.json')])
    assert before=={n:sha(CODE/n) for n in manifest},'Frozen sources changed'
    write(checks/'complete.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_unchanged':True,'target_frames_read':False,'jobs':jobs,
        'raw_sha256':sha(raw),'analysis_sha256':sha(analysis),
        'separate_verification_sha256':sha(ROOT/'results_v1/separate_verification.json'),
        'reference_repeat_sha256':sha(ROOT/'results_v1/reference_repeat_verification.json')})


if __name__=='__main__':main()
