"""Prescore frozen launch guards; exclusive diagnostic output and job receipt."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/cache/zihaohe/seer_color_contract_20261112')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
launch=json.loads((ROOT/'checks/fit_launch_manifest.json').read_text())
release=ROOT/'fit_release_v1'
assert all(sha(release/name)==expected for name,expected in launch['source_manifest']['files'].items())
checks=json.loads((release/'cpu_test_result.json').read_text())
assert checks['status']=='passed' and checks['returncode']==0 and checks['tests_expected']==10
assert sha(release/'cpu_tests.log')==checks['log_sha256']
assert checks['source_sha256']==launch['source_manifest']['files']
assert sha(ROOT/'checks/export_outputs_verification.json')==launch['expected_export_verification_sha256']
v=json.loads((ROOT/'checks/export_outputs_verification.json').read_text())
assert v['status']=='passed' and v['n']==v['base_payloads_verified']==136 and v['complete'] is True
assert v['manifest_sha256']==launch['expected_hvi_manifest_sha256']
assert sha(ROOT/'cache_rgb_hvi_v1/manifest.json')==launch['expected_hvi_manifest_sha256']
assert (ROOT/'checks/export_npu_after.txt').exists()
assert not (ROOT/'fit_results_v1').exists()
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (ROOT/'checks/fit_run.log').open('x') as log:
 p=subprocess.run(launch['command'],env={**os.environ,**launch['env']},stdout=log,stderr=subprocess.STDOUT)
record={'returncode':p.returncode,'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch_manifest_sha256':sha(ROOT/'checks/fit_launch_manifest.json'),'launcher_sha256':sha(__file__),'log_sha256':sha(ROOT/'checks/fit_run.log'),'source_immutable_after':all(sha(release/name)==expected for name,expected in launch['source_manifest']['files'].items())}
with (ROOT/'checks/fit_run_complete.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record),flush=True)
raise SystemExit(p.returncode)
