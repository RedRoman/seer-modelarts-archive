"""Fixed CPU-only sharding after an interrupted single-worker synthetic preflight."""
import argparse
import datetime
import json
from pathlib import Path
import time

import torch

from critic_export import (ASSET, ASSET_SHA, COLOR_RELEASE, SOURCE, COALITIONS,
    check_release, digest, evaluate_clip, ids_tensor, load_file, load_models,
    mask_plan, normalize_rgb, write_new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    release = args.release.resolve()
    check_release(release)
    protocol = json.loads((release/'critic_cpu_protocol.json').read_text())
    torch.set_num_threads(4)
    begin = time.monotonic()
    checks = release.parent/'checks'/('critic_worker_'+str(args.shard))
    checks.mkdir(exist_ok=False)
    encoder, trained, random, helper, before, omitted = load_models()
    print(json.dumps({'stage':'loaded','device':'cpu','threads':4,'elapsed_seconds':time.monotonic()-begin}),flush=True)
    plan = mask_plan()
    xy = [[16*(i%16)+7.5,16*(i//16)+7.5] for i in range(256)]
    synthetic = normalize_rgb(torch.rand((1,3,16,256,256), generator=torch.Generator().manual_seed(20261123)))
    synthetic_begin = time.monotonic()
    checks = release.parent/'checks'/('critic_worker_'+str(args.shard))
    checks.mkdir(exist_ok=False)
    synthetic_result = evaluate_clip(synthetic, encoder, trained, random, plan, {'patch_grid_xy':[xy,xy]})
    with torch.inference_mode():
        common = ids_tensor(plan['context_ids']['empty'],'cpu')
        target = ids_tensor(plan['target_ids'],'cpu')
        context = encoder(synthetic,masks=[common])
        changed = synthetic.clone()
        for tubelet in [3,4,5]:
            changed[:,:,2*tubelet:2*tubelet+2] += .12345
        changed_context = encoder(changed,masks=[common])
        prediction = trained(context,masks_x=[common],masks_y=[target],mask_index=0)
        changed_prediction = trained(changed_context,masks_x=[common],masks_y=[target],mask_index=0)
        assert torch.equal(context,changed_context) and torch.equal(prediction,changed_prediction)
    write_new(checks/'critic_cpu16_synthetic.json', {
        'status':'passed','device':'cpu','input_shape':[1,3,16,256,256],
        'normalized_loss_finite_and_range_0_4':True,'complete_four_coalitions_and_all_controls':True,
        'masked_withheld_target_and_donors_exact_context_invariance':True,
        'masked_withheld_target_and_donors_exact_prediction_invariance':True,
        'elapsed_seconds':time.monotonic()-synthetic_begin,'dataset_pixels_read':0,
        'gt_storage_entries_read':0,'result':synthetic_result})
    print(json.dumps({'stage':'cpu16_synthetic_passed','elapsed_seconds':time.monotonic()-synthetic_begin}),flush=True)
    # Only after the complete CPU data-shape contract passes may any selected LQ be read.
    subsets = json.loads((release/'subsets.json').read_text())
    selected = subsets['entries']
    assert len(selected) == 12 and len({item['scene'] for item in selected}) == 12
    assert all(row['name'] == row['scene'] and row['scene'] in subsets['split']['fit']
               and row['scene'] not in subsets['split']['internal_holdout'] for row in selected)
    sidecar_path = Path(protocol['observable_sidecar'])
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar['gt_storage_entries_read'] == 0 and sidecar['target_inputs_read'] == 0
    assert sidecar['subset_sha256'] == digest(release/'subsets.json')
    observable_by_name = {row['name']:row for row in sidecar['clips']}
    bgr = json.loads((COLOR_RELEASE/'proofs/source_color_receipt_closed.json').read_text())
    assert bgr['status'] == 'closed' and bgr['native_source_color'] == 'BGR'
    reader = load_file('cpu_frozen_selective_reader',COLOR_RELEASE/'dependencies/cache_hvi_source.py')
    color = load_file('cpu_frozen_explicit_color',COLOR_RELEASE/'dependencies/color_contract.py')
    if args.output.exists():
        raise FileExistsError('Refusing to reuse critic output directory')
    args.output.mkdir(parents=True)
    manifest=[]
    for clip_index in range(3*args.shard,3*args.shard+3):
        row=selected[clip_index]
        began=time.monotonic()
        path=SOURCE/row['source_file']
        assert digest(path)==row['source_sha256']
        name,scene,low,lineage=reader.read_lq_only(path)
        assert (name,scene)==(row['name'],row['scene'])
        assert lineage['lq_tensor_sha256']==row['native_lq_tensor_sha256']
        rgb=color.convert_video(low,source_order='BGR',target_order='RGB')
        rgb_sha=reader.tensor_digest(rgb)
        assert rgb_sha==row['rgb_lq_tensor_sha256']
        result=evaluate_clip(normalize_rgb(rgb),encoder,trained,random,plan,observable_by_name[name])
        result.update({'name':name,'scene':scene,'split_role':'source_fit','clip_index':clip_index,
            'cache_path':row['source_file'],'source_archive_sha256':row['source_sha256'],
            'rgb_lq_tensor_sha256':rgb_sha,'lineage':lineage,'mask_plan':plan,
            'coalition_order':list(COALITIONS),'patch_yx':[[i//16,i%16] for i in range(256)],
            'target_frames':[8,9],'donor_frames':[[6,7],[10,11]],'device':'cpu',
            'observable_sidecar_sha256':digest(sidecar_path),'elapsed_seconds':time.monotonic()-began})
        output=args.output/(Path(row['source_file']).stem+'.json')
        write_new(output,result)
        manifest.append({'path':output.name,'sha256':digest(output),'name':name,'scene':scene,
                         'rgb_lq_tensor_sha256':rgb_sha,'elapsed_seconds':result['elapsed_seconds']})
        print(json.dumps({'stage':'source_clip_complete',**manifest[-1]}),flush=True)
    write_new(args.output/'manifest.json',{'schema_version':1,'clips':manifest,'count':3,'shard_index':args.shard,
        'gt_storage_entries_read':0,'quality_scores_computed':False,'raw_videos_saved':False,
        'device':'cpu','release_sha256':digest(release/'critic_release_manifest.json')})
    after={name:helper.state_sha(model.state_dict()) for name,model in
           [('encoder',encoder),('trained',trained),('random',random)]}
    assert before==after
    check_release(release)
    assert digest(ASSET/'weights/vitl.pt')==ASSET_SHA
    write_new(checks/'critic_cpu_export_complete.json',{
        'status':'passed','device':'cpu','threads':4,'npu_source_forwards':0,
        'state_before':before,'state_after':after,'states_unchanged':True,
        'asset_sha256':ASSET_SHA,'known_static_position_omissions':omitted,
        'gt_storage_entries_read':0,'quality_scores_computed':False,
        'target_inputs_read':0,'internal_holdout_inputs_read':0,'dataset_clip_forwards':3,'shard_index':args.shard,'clip_indices':list(range(3*args.shard,3*args.shard+3)),
        'elapsed_seconds':time.monotonic()-begin,
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__=='__main__':
    main()
