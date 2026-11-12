"""Exclusive SFS backup and compact download bundle for completed RGB FIT text evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/cache/zihaohe/seer_color_contract_20261112')
DEST = Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/color_contract_rgb_fit_v1_final_v1')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

run = json.loads((ROOT/'checks/fit_run_complete.json').read_text())
verify = json.loads((ROOT/'checks/fit_independent_verification_v2.json').read_text())
assert run['returncode'] == 0 and run['source_immutable_after'] is True
assert verify['passed'] is True
assert verify['result_sha256'] == sha(ROOT/'fit_results_v1/fit_results.json')
assert sha(ROOT/'checks/fit_run.log') == run['log_sha256']
manifest = json.loads((ROOT/'fit_release_v1/source_manifest.json').read_text())
files = [ROOT/'fit_release_v1'/name for name in manifest['files']]
files += [ROOT/'fit_release_v1'/name for name in ('source_manifest.json','cpu_tests.log','cpu_test_result.json')]
files += [ROOT/'checks'/name for name in ('fit_launch.py','fit_launch_manifest.json','fit_run.log',
    'fit_run_complete.json','verify_rgb_fit.py','fit_independent_verification.json','fit_verify.log',
    'verify_rgb_fit_v2.py','fit_independent_verification_v2.json','fit_verify_v2.log','fit_archive.py')]
files += [ROOT/'fit_results_v1'/name for name in ('execution_manifest.json','fit_results.json')]
files += [ROOT/'cache_rgb_hvi_v1/manifest.json',ROOT/'checks/export_outputs_verification.json']
assert all(path.suffix in {'.py','.json','.log'} for path in files)
DEST.mkdir(parents=True,exist_ok=False)
entries=[]
for source in files:
    name=source.relative_to(ROOT)
    target=DEST/name
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    source_sha=sha(source)
    assert sha(target)==source_sha
    entries.append({'path':str(name),'source':str(source),'backup':str(target),
                    'bytes':source.stat().st_size,'sha256':source_sha})
report={'status':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_root':str(ROOT),'backup_root':str(DEST),'files':entries,'n_files':len(entries),
    'total_bytes':sum(x['bytes'] for x in entries),'tensor_or_image_arrays_copied':False}
for target in (DEST/'backup_manifest.json',ROOT/'checks/fit_backup_manifest.json'):
    with target.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
bundle=ROOT/'checks/fit_text_evidence_v1.tgz'
with tarfile.open(bundle,'x:gz') as tar:
    for item in entries:tar.add(DEST/item['path'],arcname=item['path'])
    tar.add(DEST/'backup_manifest.json',arcname='backup_manifest.json')
receipt={'path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size,'files':len(entries)+1,
         'backup_manifest_sha256':sha(DEST/'backup_manifest.json')}
with (ROOT/'checks/fit_download_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
