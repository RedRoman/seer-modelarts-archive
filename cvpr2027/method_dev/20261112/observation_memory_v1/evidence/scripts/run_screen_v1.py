"""Execute the frozen source-only five-mode screen; never target inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    release=root/'release_v1'
    code=release/'code_full'
    tool=code/'tools/train_observation_memory.py'
    python='/cache/zihaohe/seer_cvpr/venv_npu/bin/python'
    source=Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0')
    tokens=root/'tokens_source_v1';prepared=root/'prepared_v1';runs=root/'runs_v1'
    results=root/'results_v1';logs=root/'screen_logs_v1';checks=root/'screen_checks_v1'
    for p in (runs,results,logs,checks):p.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',SEER_ROOT='/temp/zihaohe/seer_cvpr')
    before={p:sha(code/p) for p in json.loads((release/'source_manifest.json').read_text())}
    assert before==json.loads((release/'source_manifest.json').read_text()),'Frozen package changed'
    write(checks/'launch.json',{'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'runner_sha256':sha(__file__),'cpu_count':os.cpu_count(),'loadavg':os.getloadavg(),
        'parallel_source_jobs':5,'threads_per_job':4,'target_inference':False,
        'code_sha256':before,'protocol_sha256':sha(release/'PLAN.md'),
        'analysis_sha256':sha(release/'analyze_screen.py')})

    def command(label,argv):
        start=time.monotonic()
        with (logs/(label+'.log')).open('x') as stream:
            proc=subprocess.run([python,*map(str,argv)],cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT)
        result={'label':label,'argv':[python,*map(str,argv)],'returncode':proc.returncode,
                'elapsed_seconds':time.monotonic()-start,'log_sha256':sha(logs/(label+'.log'))}
        write(checks/(label+'.json'),result)
        print(json.dumps(result),flush=True)
        return result

    prep=command('prepare',[tool,'prepare','--cache',source,'--tokens',tokens,
        '--splits','/temp/zihaohe/seer_cvpr/datasets/splits_cvpr.json','--out',prepared])
    if prep['returncode']:raise RuntimeError('Source preparation failed; preserve evidence')
    modes=['prior_cross','pixel_cross','shuffled_cross','prior_same','pixel_local']
    def train(mode):
        return command(mode,[tool,'train','--prepared',prepared,'--mode',mode,'--seed','0',
                             '--out',runs/(mode+'_s0')])
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs=list(pool.map(train,modes))
    if any(x['returncode'] for x in jobs):raise RuntimeError('Incomplete screen; no holdout evaluation')
    evaluation=command('source_internal_evaluation',[tool,'evaluate','--prepared',prepared,'--runs',runs,
        '--out',results/'source_internal_raw.json',
        '--reference-prepared','/cache/zihaohe/seer_observation_dev_20261112/prepared/pretrained',
        '--reference-head','/cache/zihaohe/seer_observation_dev_20261112/runs/pretrained/raw_node_s0/final.pt'])
    if evaluation['returncode']:raise RuntimeError('Source internal evaluation failed')
    analysis=command('source_analysis',[release/'analyze_screen.py','--input',results/'source_internal_raw.json',
                                        '--out',results/'source_internal_analysis.json'])
    if analysis['returncode']:raise RuntimeError('Source analysis failed')
    after={p:sha(code/p) for p in before}
    assert before==after,'Source changed while screen was running'
    write(checks/'complete.json',{'source_unchanged':True,'target_inference':False,'jobs':jobs,
        'raw_sha256':sha(results/'source_internal_raw.json'),
        'analysis_sha256':sha(results/'source_internal_analysis.json'),
        'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


if __name__=='__main__':main()
