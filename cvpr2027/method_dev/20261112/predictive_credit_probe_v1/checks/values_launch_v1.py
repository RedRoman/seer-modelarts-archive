"""Exclusive frozen CPU test and stage launcher; no target or train entrypoint."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112')
RELEASE=ROOT/'values_release_v1'
CHECKS=ROOT/'checks'
PYTHON=Path('/cache/zihaohe/seer_cvpr/venv_npu/bin/python')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
 'VALUES_LEGACY_READER':'/cache/zihaohe/seer_hvi_source_anchor_20261112/release_v1/cache_hvi_source.py'}
stage=sys.argv[1]
assert stage in ('observables','utilities')
CHECKS.mkdir(exist_ok=True)
if stage=='observables':
 testlog=CHECKS/'values_tests.log'
 with testlog.open('x') as f:
  result=subprocess.run([str(PYTHON),'-m','unittest','discover','-s',str(RELEASE),'-p','test_values_probe.py','-v'],env=env,stdout=f,stderr=subprocess.STDOUT)
 text=testlog.read_text()
 assert result.returncode==0 and 'Ran 10 tests' in text and '\nOK\n' in text and 'skipped=' not in text
 receipt={'status':'passed','returncode':result.returncode,'tests':10,'skips':0,'log_sha256':sha(testlog),
  'code_sha256':sha(RELEASE/'values_probe.py'),'test_sha256':sha(RELEASE/'test_values_probe.py'),
  'protocol_sha256':sha(RELEASE/'values_protocol.json'),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save(CHECKS/'values_test_receipt.json',receipt)
receipt=CHECKS/'values_test_receipt.json'
cmd=[str(PYTHON),str(RELEASE/'values_probe.py'),'--stage',stage,
 '--source-cache','/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0',
 '--hvi-cache','/cache/zihaohe/seer_color_contract_20261112/cache_rgb_hvi_v1',
 '--geometry-manifest','/cache/zihaohe/seer_observation_geometry_20261112/prepared_v1/manifest.json',
 '--subset',str(RELEASE/'subsets.json'),'--out',str(ROOT/('values_observables_v1' if stage=='observables' else 'values_utilities_v1')),
 '--rgb-runner','/cache/zihaohe/seer_color_contract_20261112/fit_release_v1/color_contract_v1/run_rgb_fit.py',
 '--color-source','/cache/zihaohe/seer_color_contract_20261112/fit_release_v1/color_contract_v1/color_contract.py',
 '--lq-reader',env['VALUES_LEGACY_READER'],
 '--math-source','/cache/zihaohe/seer_color_contract_20261112/fit_release_v1/hvi_temporal_fit_feasibility_v1/run_fit_feasibility.py',
 '--test-receipt',str(receipt),
 '--expected-hvi-manifest-sha256','5f25fdfcca5bc7d581c1484245fa760ef321da642f5e741077c128376616a94a',
 '--expected-subset-sha256','35292bbec0bc056481129857436950644448ca669d2261b72de350f841a2a92e',
 '--expected-test-receipt-sha256',sha(receipt),
 '--expected-rgb-runner-sha256','00b039e00a39812d2ae3e0d202c2436e6542ba6b964bf5c47cd78c78285d4fa2',
 '--expected-lq-reader-sha256','26c0c36656ad7b7d3b0ce04275578a45286c27931a476c9dcfb1ce1981e69db2']
if stage=='utilities':
 sidecar=ROOT/'values_observables_v1/observable_sidecar.json'
 frozen=json.loads((CHECKS/'values_observables_run_complete.json').read_text())
 assert frozen['returncode']==0 and frozen['source_immutable_after'] is True
 assert sha(sidecar)==frozen['observable_sidecar_sha256']
 cmd+=['--observable-sidecar',str(sidecar),'--expected-observable-sidecar-sha256',frozen['observable_sidecar_sha256']]
files={str(p):sha(p) for p in [RELEASE/n for n in ('values_probe.py','test_values_probe.py','values_protocol.json','subsets.json')]+[Path(__file__),receipt]}
manifest={'stage':stage,'files':files,'command':cmd,'env':{k:env[k] for k in ('TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
 'test_receipt_sha256':sha(receipt),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'target_inputs_read':0,'internal_inputs_read':0}
save(CHECKS/f'values_{stage}_launch_manifest.json',manifest)
log=CHECKS/f'values_{stage}_run.log'
with log.open('x') as f:
 run=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
record={'stage':stage,'returncode':run.returncode,'source_immutable_after':all(sha(p)==h for p,h in files.items()),
 'launcher_sha256':sha(__file__),'launch_manifest_sha256':sha(CHECKS/f'values_{stage}_launch_manifest.json'),
 'log_sha256':sha(log),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
sidecar=ROOT/'values_observables_v1/observable_sidecar.json'
if stage=='observables' and run.returncode==0:record['observable_sidecar_sha256']=sha(sidecar)
save(CHECKS/f'values_{stage}_run_complete.json',record)
print(json.dumps(record),flush=True)
raise SystemExit(run.returncode)
