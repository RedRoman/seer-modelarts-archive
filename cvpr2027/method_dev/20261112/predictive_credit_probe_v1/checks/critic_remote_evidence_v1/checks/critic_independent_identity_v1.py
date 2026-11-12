"""Independent closure of shared synthetic outputs, states and input lineage."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112_v1')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()
subsets=json.loads((ROOT/'cpu_sharded_release_v2/subsets.json').read_text())
manifest=json.loads((ROOT/'critic_cpu_outputs_v2/manifest.json').read_text())
merge=json.loads((ROOT/'checks/critic_merge_complete_v2.json').read_text())
assert merge['status']=='passed' and merge['manifest_sha256']==digest(ROOT/'critic_cpu_outputs_v2/manifest.json')
synthetic_hashes=[]
states=[]
for worker in range(4):
    prefix=ROOT/'checks'/('critic_worker_v2_'+str(worker))
    synthetic=json.loads((prefix/'critic_cpu16_synthetic.json').read_text())
    complete=json.loads((prefix/'critic_cpu_export_complete.json').read_text())
    assert synthetic['status']=='passed' and complete['status']=='passed'
    assert complete['state_before']==complete['state_after']
    states.append(complete['state_before'])
    synthetic_hashes.append(hashlib.sha256(json.dumps(synthetic['result'],sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest())
assert len(set(synthetic_hashes))==1,synthetic_hashes
assert all(state==states[0] for state in states)
checks=0
for expected,row in zip(subsets['entries'],manifest['clips']):
    path=ROOT/'critic_cpu_outputs_v2'/row['path']
    assert digest(path)==row['sha256']
    item=json.loads(path.read_text())
    assert item['name']==expected['name'] and item['scene']==expected['scene']
    assert item['source_archive_sha256']==expected['source_sha256']
    assert item['lineage']['lq_tensor_sha256']==expected['native_lq_tensor_sha256']
    assert item['rgb_lq_tensor_sha256']==expected['rgb_lq_tensor_sha256']
    assert item['observable_sidecar_sha256']=='580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23'
    assert len(item['lineage']['archive_entries_read'])==3
    assert item['pre_attention_first_block_shapes']==[[1,n,1024] for n in [256,2048,256,256,1280,1536,1536,1792,1280,1536,1536,1792]]
    checks+=9
assert len(manifest['clips'])==12
assert not list(ROOT.rglob('*.pt')) and not list(ROOT.rglob('*.pth'))
report={'status':'passed','canonical_synthetic_result_sha256':synthetic_hashes[0],
        'all_four_synthetic_results_exactly_identical':True,'all_four_states_exactly_identical':True,
        'shared_state':states[0],'clips':12,'lineage_shape_checks':checks,
        'manifest_sha256':digest(ROOT/'critic_cpu_outputs_v2/manifest.json'),
        'raw_video_or_model_binary_saved':False,'npu_source_forward':0,'gt_storage_entries_read':0}
with (ROOT/'checks/critic_independent_identity_v1.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(report,indent=2))
