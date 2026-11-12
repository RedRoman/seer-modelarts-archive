"""Independent stdlib-only fixture execution of the actual frozen merge script.

This tests merge orchestration and fail-closed guards, not model math/state-load.
No source/model/video/GT assets are read.
"""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types

PHASE=Path(__file__).resolve().parents[1]
MERGE=PHASE/'checks/critic_merge_shards_v2.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:json.dump(x,f)
def exclusive_write(p,x):
 with Path(p).open('x') as f:json.dump(x,f)
def fixture(root):
 release=root/'release';release.mkdir()
 write(release/'critic_release_manifest.json',{'fixture':True})
 entries=[{'name':f'fit{i}','scene':f'fit{i}','source_sha256':f'source{i}',
  'rgb_lq_tensor_sha256':f'rgb{i}','native_lq_tensor_sha256':f'native{i}'} for i in range(12)]
 write(release/'subsets.json',{'entries':entries})
 states={'encoder':'encoder_identical','trained':'trained_identical','random':'random_identical_seed'}
 for shard in range(4):
  checks=root/'checks'/f'critic_worker_v2_{shard}'
  write(checks/'critic_cpu_export_complete.json',{'status':'passed','device':'cpu','threads':4,
   'dataset_clip_forwards':3,'shard_index':shard,'clip_indices':list(range(3*shard,3*shard+3)),
   'gt_storage_entries_read':0,'npu_source_forwards':0,'target_inputs_read':0,'internal_holdout_inputs_read':0,
   'states_unchanged':True,'state_before':states,'state_after':states})
  write(checks/'critic_cpu16_synthetic.json',{'status':'passed','device':'cpu',
   'complete_four_coalitions_and_all_controls':True,'masked_withheld_target_and_donors_exact_context_invariance':True,
   'masked_withheld_target_and_donors_exact_prediction_invariance':True})
  rows=[];folder=root/'critic_cpu_outputs_v2'/f'worker_{shard}'
  for index in range(3*shard,3*shard+3):
   row=entries[index]
   payload={'name':row['name'],'scene':row['scene'],'clip_index':index,'split_role':'source_fit',
    'observable_sidecar_sha256':'580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23',
    'source_archive_sha256':row['source_sha256'],'rgb_lq_tensor_sha256':row['rgb_lq_tensor_sha256'],
    'lineage':{'lq_tensor_sha256':row['native_lq_tensor_sha256'],'archive_entries_read':['data.pkl','byteorder','lq_storage']},
    'coalition_order':['empty','donor1','donor2','both'],'patch_yx':[[p//16,p%16] for p in range(256)],
    'losses':{k:[[.25]*256 for c in range(4)] for k in ('trained','random','wrong_time')}}
   path=folder/f'clip_{index:03d}.json';write(path,payload)
   rows.append({'name':row['name'],'scene':row['scene'],'path':path.name,'sha256':sha(path),
    'rgb_lq_tensor_sha256':row['rgb_lq_tensor_sha256']})
  write(folder/'manifest.json',{'count':3,'shard_index':shard,'device':'cpu',
   'release_sha256':sha(release/'critic_release_manifest.json'),'clips':rows})
 return release

def modify_payload(root,key,value):
 folder=root/'critic_cpu_outputs_v2/worker_0'
 p=folder/'clip_000.json';data=json.loads(p.read_text());data[key]=value;write(p,data)
 m=folder/'manifest.json';manifest=json.loads(m.read_text());manifest['clips'][0]['sha256']=sha(p);write(m,manifest)

def run_case(name,mutation=None):
 with tempfile.TemporaryDirectory() as temporary:
  root=Path(temporary);release=fixture(root)
  if mutation:mutation(root)
  helper=types.ModuleType('critic_export')
  helper.check_release=lambda _: None
  helper.digest=sha;helper.write_new=exclusive_write
  original=sys.modules.get('critic_export');args=sys.argv
  sys.modules['critic_export']=helper;sys.argv=[str(MERGE),str(release)]
  try:
   try:
    runpy.run_path(str(MERGE),run_name='__main__')
    passed=True
   except (AssertionError,FileNotFoundError):passed=False
  finally:
   sys.argv=args
   if original is None:del sys.modules['critic_export']
   else:sys.modules['critic_export']=original
  if mutation:
   assert passed is False,name+' was accepted'
   assert not (root/'critic_cpu_outputs_v2/manifest.json').exists(),name+' published partial joint manifest'
  else:
   assert passed is True
   merged=json.loads((root/'critic_cpu_outputs_v2/manifest.json').read_text())
   assert merged['count']==12 and len({x['name'] for x in merged['clips']})==12
   assert all(x['path']==f'worker_{i//3}/clip_{i:03d}.json' for i,x in enumerate(merged['clips']))
   complete=json.loads((root/'checks/critic_merge_complete_v2.json').read_text())
   assert complete['finite_loss_checks']==12*3*4*256
   assert complete['manifest_sha256']==sha(root/'critic_cpu_outputs_v2/manifest.json')
 return {'case':name,'status':'passed'}

results=[run_case('complete_unique12_hash_and_relative_paths')]
results.append(run_case('reject_missing_worker_completion',lambda r:(r/'checks/critic_worker_v2_2/critic_cpu_export_complete.json').unlink()))
def mixed_state(root):
 p=root/'checks/critic_worker_v2_1/critic_cpu_export_complete.json';d=json.loads(p.read_text())
 d['state_before']=d['state_after']={'encoder':'wrong','trained':'wrong','random':'wrong'};write(p,d)
results.append(run_case('reject_mixed_worker_state',mixed_state))
results.append(run_case('reject_wrong_payload_name',lambda r:modify_payload(r,'name','wrong')))
results.append(run_case('reject_wrong_sidecar',lambda r:modify_payload(r,'observable_sidecar_sha256','wrong')))
results.append(run_case('reject_bad_source_lineage',lambda r:modify_payload(r,'source_archive_sha256','wrong')))
results.append(run_case('reject_bad_patch_order',lambda r:modify_payload(r,'patch_yx',[[0,0]]*256)))
results.append(run_case('reject_nonfinite_loss',lambda r:modify_payload(r,'losses',{k:[[float('nan')]*256 for c in range(4)] for k in ('trained','random','wrong_time')})))
print(json.dumps({'status':'passed','fixtures':len(results),'results':results,'merge_sha256':sha(MERGE),
 'dataset_inputs_read':0,'model_or_gt_assets_read':0,'limitations':'check_release stubbed to isolate actual merge program guards; this is not a model/encoder or runtime data test'}))
