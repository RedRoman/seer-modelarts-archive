"""Exclusive file-by-file verified SFS snapshot of compact source probe evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112')
BACKUP=Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/predictive_credit_probe_values_v1_final_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for stage in ('observables','utilities'):
 p=json.loads((ROOT/f'checks/values_{stage}_run_complete.json').read_text())
 assert p['returncode']==0 and p['source_immutable_after'] is True
files=[]
for folder in ('values_release_v1','values_observables_v1','values_utilities_v1'):
 files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [p for p in (ROOT/'checks').glob('values_*') if p.suffix in ('.py','.json','.log')]
assert len(files)==len(set(files)) and all(p.suffix in ('.py','.json','.log','.npz') for p in files)
BACKUP.mkdir(parents=True,exist_ok=False)
rows=[]
for p in sorted(files):
 rel=p.relative_to(ROOT);target=BACKUP/rel
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(p,target)
 assert sha(p)==sha(target)
 rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'scope':'Compact fixed source FIT12 probe evidence, no RGB/GT arrays','files':rows,'payload_count':len(rows),
 'bytes':sum(p['bytes'] for p in rows),'all_copy_hashes_verified':True,'source_root':str(ROOT),'backup_root':str(BACKUP),
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
save(BACKUP/'backup_manifest.json',manifest)
archive=ROOT/'checks/values_evidence_v1.tar'
with tarfile.open(archive,'x') as tar:
 for row in rows:tar.add(BACKUP/row['path'],arcname=row['path'],recursive=False)
 tar.add(BACKUP/'backup_manifest.json',arcname='backup_manifest.json',recursive=False)
receipt={'status':'passed','payload_count':len(rows),'payload_bytes':manifest['bytes'],'backup_root':str(BACKUP),
 'backup_manifest_sha256':sha(BACKUP/'backup_manifest.json'),'archive_path':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}
save(ROOT/'checks/values_backup_receipt.json',receipt)
print(json.dumps(receipt),flush=True)
