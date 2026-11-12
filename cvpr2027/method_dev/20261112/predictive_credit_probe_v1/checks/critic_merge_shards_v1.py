"""No fitting or score selection: strict unique12 merge of fixed CPU shards."""
import datetime
import json
import math
from pathlib import Path
import sys

from critic_export import check_release, digest, write_new

release=Path(sys.argv[1]).resolve()
root=release.parent
check_release(release)
subsets=json.loads((release/'subsets.json').read_text())['entries']
assert len(subsets)==12
files=[]
states=None
checks=0
for shard in range(4):
    source=root/'critic_cpu_outputs_v1'/('worker_'+str(shard))
    evidence=root/'checks'/('critic_worker_'+str(shard))
    complete=json.loads((evidence/'critic_cpu_export_complete.json').read_text())
    synthetic=json.loads((evidence/'critic_cpu16_synthetic.json').read_text())
    assert complete['status']=='passed' and complete['device']=='cpu' and complete['threads']==4
    assert complete['dataset_clip_forwards']==3 and complete['shard_index']==shard
    assert complete['clip_indices']==list(range(3*shard,3*shard+3))
    assert complete['gt_storage_entries_read']==complete['npu_source_forwards']==0
    assert complete['target_inputs_read']==complete['internal_holdout_inputs_read']==0
    assert complete['states_unchanged'] and complete['state_before']==complete['state_after']
    if states is None:states=complete['state_before']
    assert states==complete['state_before']
    assert synthetic['status']=='passed' and synthetic['device']=='cpu'
    assert synthetic['complete_four_coalitions_and_all_controls']
    assert synthetic['masked_withheld_target_and_donors_exact_context_invariance']
    assert synthetic['masked_withheld_target_and_donors_exact_prediction_invariance']
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['count']==3 and manifest['shard_index']==shard and manifest['device']=='cpu'
    assert manifest['release_sha256']==digest(release/'critic_release_manifest.json')
    assert len(manifest['clips'])==3
    for row,index in zip(manifest['clips'],range(3*shard,3*shard+3)):
        expected=subsets[index]
        assert (row['name'],row['scene'])==(expected['name'],expected['scene'])
        assert Path(row['path']).name==row['path']
        path=source/row['path']
        assert digest(path)==row['sha256']
        payload=json.loads(path.read_text())
        assert payload['clip_index']==index and payload['split_role']=='source_fit'
        assert payload['source_archive_sha256']==expected['source_sha256']
        assert payload['rgb_lq_tensor_sha256']==expected['rgb_lq_tensor_sha256']==row['rgb_lq_tensor_sha256']
        assert payload['lineage']['lq_tensor_sha256']==expected['native_lq_tensor_sha256']
        assert len(payload['lineage']['archive_entries_read'])==3
        assert payload['coalition_order']==['empty','donor1','donor2','both']
        assert payload['patch_yx']==[[i//16,i%16] for i in range(256)]
        for key in ['trained','random','wrong_time']:
            assert len(payload['losses'][key])==4
            for coalition in payload['losses'][key]:
                assert len(coalition)==256
                assert all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=4.00001 for v in coalition)
                checks+=256
        files.append({**row,'path':'worker_'+str(shard)+'/'+row['path']})
assert len(files)==12 and len({row['name'] for row in files})==12
check_release(release)
manifest={'schema_version':1,'count':12,'clips':files,'device':'cpu',
          'gt_storage_entries_read':0,'quality_scores_computed':False,'raw_videos_saved':False,
          'release_sha256':digest(release/'critic_release_manifest.json'),
          'all_four_workers_passed':True,'all_worker_states_identical':True}
write_new(root/'critic_cpu_outputs_v1/manifest.json',manifest)
write_new(root/'checks/critic_merge_complete.json',{
    'status':'passed','device':'cpu','workers':4,'threads_each':4,'dataset_clip_forwards':12,
    'npu_source_forwards':0,'gt_storage_entries_read':0,'target_inputs_read':0,'internal_inputs_read':0,
    'all_four_synthetic_passed':True,'all_worker_states_identical':True,'states_unchanged':True,
    'unique_coverage12':True,'finite_loss_checks':checks,'shared_frozen_state_sha256':states,
    'manifest_sha256':digest(root/'critic_cpu_outputs_v1/manifest.json'),
    'release_sha256':digest(release/'critic_release_manifest.json'),
    'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps({'status':'passed','manifest_sha256':digest(root/'critic_cpu_outputs_v1/manifest.json'),
                  'count':12,'finite_loss_checks':checks}),flush=True)
