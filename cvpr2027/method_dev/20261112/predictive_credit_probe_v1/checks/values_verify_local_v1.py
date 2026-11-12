"""Verify downloaded immutable compact evidence and scalar utility identities."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import numpy as np

PHASE=Path(__file__).resolve().parents[1]
EXTERNAL=Path('/Users/hezihao/Documents/New project/seer_predictive_credit_probe_dev_20261112')
ARCHIVE=EXTERNAL/'values_evidence_v1.tar'
DEST=EXTERNAL/'values_evidence_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert sha(ARCHIVE)=='61733150d9eb0491bc55d92f9e5522ae000ea14b08f8d9fd0bef8dd69ac29c90'
DEST.mkdir(exist_ok=False)
with tarfile.open(ARCHIVE) as tar:
 for member in tar.getmembers():
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  with tar.extractfile(member) as source:
   p=DEST/member.name;p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as target:shutil.copyfileobj(source,target)
manifest=json.loads((DEST/'backup_manifest.json').read_text())
assert sha(DEST/'backup_manifest.json')=='837fd9994952010026134747549e0616575122c345c3730f743d0ffbe3effa2a'
assert len(manifest['files'])==manifest['payload_count']==23
assert all(sha(DEST/r['path'])==r['sha256'] and (DEST/r['path']).stat().st_size==r['bytes'] for r in manifest['files'])
for name in ('values_probe.py','test_values_probe.py','values_protocol.json','subsets.json'):
 assert sha(PHASE/name)==sha(DEST/'values_release_v1'/name)
report=json.loads((DEST/'values_utilities_v1/values_results.json').read_text())
npz=DEST/'values_utilities_v1/patch_statistics.npz'
assert sha(npz)==report['patch_statistics_sha256']=='4ccd7e9a807f43a3c64d6b102f3775a2aca9fed86d197643a4e1909c8d338a2c'
a=np.load(npz,allow_pickle=False)
assert all(a[k].shape==(3072,) and np.isfinite(a[k]).all() for k in a.files)
subset=json.loads((PHASE/'subsets.json').read_text())
assert len(report['clips'])==len(subset['entries'])==12
assert np.array_equal(a['clip_index'],np.repeat(np.arange(12),256))
assert np.array_equal(a['patch_y'],np.tile(np.repeat(np.arange(16),16),12))
assert np.array_equal(a['patch_x'],np.tile(np.arange(16),192))
recomputed={'u1':a['mse_base']-a['mse_single1'],'u2':a['mse_base']-a['mse_single2'],
 'u12':a['mse_base']-a['mse_pair'],'u1_given2':a['mse_single2']-a['mse_pair'],
 'u2_given1':a['mse_single1']-a['mse_pair']}
errors={k:float(np.abs(a[k]-value).max()) for k,value in recomputed.items()}
assert max(errors.values())==0
assert np.array_equal(a['mse_base'],a['mse_sameframe_zero'])
mean_errors={k:abs(float(a['mse_'+k].mean())-v) for k,v in report['mean_rgb_mse'].items()}
assert max(mean_errors.values())<=1e-15
sidecar=DEST/'values_observables_v1/observable_sidecar.json'
assert sha(sidecar)=='580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23'
obs=json.loads(sidecar.read_text());assert obs['gt_storage_entries_read']==0
for d in (1,2):
 for key in ('lq','hvi'):
  reference=np.array([clip['feature_'+key][d-1] for clip in obs['clips']]).reshape(-1)
  assert np.array_equal(reference,a[f'feature_{key}{d}'])
valid=(a['feature_valid1']>=.5)&(a['feature_valid2']>=.5)
def counts(values,mask):
 b=values[mask]
 return {'positive_above_1e-10':int((b>1e-10).sum()),'negative_below_minus_1e-10':int((b< -1e-10).sum()),
  'abs_at_most_1e-10':int((np.abs(b)<=1e-10).sum()),'total':len(b)}
per_scene=[]
for i,(clip,declared) in enumerate(zip(report['clips'],subset['entries'])):
 assert clip['name']==declared['name']==clip['scene']
 m=a['clip_index']==i
 baseline=clip['methods']['base']['psnr_rgb_from_central_two_frame_mse']
 gains={k:v['psnr_rgb_from_central_two_frame_mse']-baseline for k,v in clip['methods'].items()}
 per_scene.append({'name':clip['name'],'patches':256,'both_donors_valid_ge_half':int((m&valid).sum()),
  'central_two_frame_psnr_rgb_gain_vs_base':gains,
  'conditional_utility_label_counts_all_patches':{k:counts(a[k],m) for k in ('u1_given2','u2_given1')},
  'conditional_utility_label_counts_primary_eligible_support':{k:counts(a[k],m&valid) for k in ('u1_given2','u2_given1')},
  'max_abs_dc_sum_error':clip['constraints']['max_abs_patch_frame_channel_dc_sum_error']})
verification={'status':'passed','archive_sha256':sha(ARCHIVE),'backup_manifest_sha256':sha(DEST/'backup_manifest.json'),
 'verified_payloads':23,'verified_bytes':manifest['bytes'],'npz_path':str(npz),'npz_sha256':sha(npz),
 'values_results_sha256':sha(DEST/'values_utilities_v1/values_results.json'),
 'utility_identity_max_abs_error':errors,'mean_mse_max_abs_error':max(mean_errors.values()),
 'sameframe_mse_exact_base':True,'observable_features_bitwise_match_gt_free_stage':True,
 'max_abs_patch_frame_channel_dc_sum_error':max(c['max_abs_dc_sum_error'] for c in per_scene),
 'per_scene':per_scene,'gt_storage_entries_opened_by_this_verifier':0,'target_inputs_read':0}
for relative in ('values_results.json','execution_manifest.json'):
 target=PHASE/'results/values_v1'/relative;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(DEST/'values_utilities_v1'/relative,target)
text_evidence=PHASE/'checks/values_evidence_v1'
text_evidence.mkdir(exist_ok=False)
for row in manifest['files']:
 p=DEST/row['path']
 if p.suffix in ('.py','.json','.log'):
  p.read_text(encoding='utf-8')
  target=text_evidence/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,target)
shutil.copy2(DEST/'backup_manifest.json',text_evidence/'backup_manifest.json')
external_results=EXTERNAL/'results/values_v1';external_results.mkdir(parents=True,exist_ok=False)
for relative in ('values_results.json','execution_manifest.json','patch_statistics.npz'):
 shutil.copy2(DEST/'values_utilities_v1'/relative,external_results/relative)
verification['npz_path']=str(external_results/'patch_statistics.npz')
save(PHASE/'checks/values_local_verification.json',verification)
receipt={'status':'passed','payload_count':23,'payload_bytes':manifest['bytes'],'backup_root':manifest['backup_root'],
 'backup_manifest_sha256':sha(DEST/'backup_manifest.json'),'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
 'download_transport':'Background SCP after partial SFTP; final complete archive SHA and all23 payload SHAs verified',
 'external_results':str(external_results),'no_binary_arrays_copied_into_git_repository':True}
save(PHASE/'checks/values_backup_receipt.json',receipt)
print(json.dumps({'status':'passed','payloads':23,'rows':3072,'npz_path':verification['npz_path'],
 'verification_sha256':sha(PHASE/'checks/values_local_verification.json'),
 'max_dc':verification['max_abs_patch_frame_channel_dc_sum_error']}))
