"""Exclusive SFS archive of compact critic evidence; never copies model/video data."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112_v1')
BACKUP=Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/predictive_credit_probe_critic_v1_final_v1')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def save(path,payload):
    with path.open('x') as stream:json.dump(payload,stream,indent=2,allow_nan=False);stream.write('\n')
complete=json.loads((ROOT/'checks/critic_merge_complete_v2.json').read_text())
assert complete['status']=='passed' and complete['states_unchanged'] and complete['dataset_clip_forwards']==12
assert complete['npu_source_forwards']==0 and complete['gt_storage_entries_read']==0
manifest=json.loads((ROOT/'critic_cpu_outputs_v2/manifest.json').read_text())
assert manifest['count']==12 and manifest['device']=='cpu'
for row in manifest['clips']:assert sha(ROOT/'critic_cpu_outputs_v2'/row['path'])==row['sha256']
sidecar=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112/values_observables_v1/observable_sidecar.json')
assert sha(sidecar)=='580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23'
copy=ROOT/'checks/critic_observable_sidecar.json'
with sidecar.open('rb') as src,copy.open('xb') as dst:shutil.copyfileobj(src,dst)
assert sha(copy)==sha(sidecar)
files=sorted(path for path in ROOT.rglob('*') if path.is_file() and path.name!='critic_compact.tar.gz' and '__pycache__' not in path.parts)
assert not any(path.suffix in ['.pt','.pth','.npy','.npz','.pyc'] for path in files)
BACKUP.mkdir(parents=True,exist_ok=False)
rows=[]
for source in files:
    relative=source.relative_to(ROOT);target=BACKUP/relative;target.parent.mkdir(parents=True,exist_ok=True)
    expected=sha(source)
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    assert sha(source)==sha(target)==expected
    rows.append({'path':str(relative),'bytes':source.stat().st_size,'sha256':expected})
report={'status':'verified','backup':str(BACKUP),'count':len(rows),'bytes':sum(row['bytes'] for row in rows),
        'payloads':rows,'read_back_sha256_verified':True,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save(BACKUP/'critic_backup_manifest.json',report)
save(ROOT/'checks/critic_backup_manifest.json',report)
archive=ROOT/'critic_compact.tar.gz'
with tarfile.open(archive,'x:gz') as stream:
    for path in sorted(BACKUP.rglob('*')):
        if path.is_file():stream.add(path,arcname=str(path.relative_to(BACKUP)))
receipt={'status':'verified','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
         'backup_manifest_sha256':sha(BACKUP/'critic_backup_manifest.json'),'payload_count':len(rows)}
save(ROOT/'checks/critic_backup_receipt.json',receipt)
print(json.dumps(receipt,indent=2))
