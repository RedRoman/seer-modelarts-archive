"""Separate raw-statistics implementation; imports no primary screen analyzer."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','analysis','source-reference','code-root','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    raw=json.loads(a.raw.read_text());analysis=json.loads(a.analysis.read_text())
    reference=json.loads(a.source_reference.read_text())
    split=reference['split'];all_names=[r['name'] for r in reference['per_sequence']]
    expected={n for n in all_names if re.sub(r'_2$','',n) in split['internal_holdout']}
    assert len(expected)==34 and len(set(split['fit']))==51 and len(set(split['internal_holdout']))==17
    assert not set(split['fit'])&set(split['internal_holdout'])
    assert {r['name'] for r in raw['per_sequence']}==expected and len(raw['per_sequence'])==34
    assert raw['initial_states_and_plans_identical'] is True and raw['states_unchanged'] is True
    assert raw['target_frames_read'] is False and analysis['target_inference_allowed_by_this_result'] is False
    modes=('prior','pixel','shuffled_prior')
    assert set(raw['provenance'])==set(modes)
    for mode in modes:
        record=raw['provenance'][mode]
        assert record['mode']==mode and record['seed']==0 and record['steps']==400 and record['split']==split
        assert record['model_config']['address_mode']==mode and record['model_config']['hidden_dim']==16
        assert record['args']['forward_context']==256 and record['args']['loss_crop']==128
        assert record['reference_head_sha256']==reference['reference_head_sha256']
        for name,h in record['source_sha256'].items():assert sha(a.code_root/name)==h
    for key in ('initial_state_sha256','sample_plan_sha256','prepared_manifest_sha256'):
        assert len({r[key] for r in raw['provenance'].values()})==1
    assert analysis['raw_sha256']==sha(a.raw)
    methods=('base','prior','pixel','shuffled_prior','frozen_raw_node_1000')
    metrics=('rgb_mse_float','psnr_rgb_from_clip_mse')
    scenes=sorted(split['internal_holdout']);assert analysis['scene_order']==scenes
    matrix=np.empty((17,5,2),np.float64)
    for si,scene in enumerate(scenes):
        pair=[r for r in raw['per_sequence'] if r['scene']==scene]
        assert len(pair)==2 and {r['name'] for r in pair}=={scene,scene+'_2'}
        for j,method in enumerate(methods):
            values=np.array([[r['methods'][method][metric] for metric in metrics] for r in pair])
            assert np.isfinite(values).all() and (values[:,0]>0).all()
            np.testing.assert_allclose(-10*np.log10(values[:,0]),values[:,1],rtol=0,atol=1e-9)
            matrix[si,j]=values.sum(0)/2
    draws=np.random.default_rng(0).integers(17,size=(10000,17))
    counts=np.zeros((10000,17),np.int64)
    for i,draw in enumerate(draws):counts[i]=np.bincount(draw,minlength=17)
    errors=[];checks=0
    def equal(x,y):
        nonlocal checks
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        errors.append(error);np.testing.assert_allclose(x,y,rtol=0,atol=1e-12);checks+=np.size(x)
    for j,m in enumerate(methods):
        for k,metric in enumerate(metrics):
            equal(matrix[:,j,k].sum()/17,analysis['means'][m][metric])
            equal(matrix[:,j,k],[analysis['per_scene'][s][m][k] for s in scenes])
    contrasts={}
    for j,control in enumerate(methods):
        if control=='prior':continue
        delta=matrix[:,1]-matrix[:,j]
        point=delta.sum(0)/17
        ci=np.quantile(counts@delta/17,[.025,.975],axis=0)
        contrasts[control]=(point,ci)
        for k,metric in enumerate(metrics):
            record=analysis['contrasts']['prior_minus_'+control][metric]
            equal(point[k],record['mean']);equal(ci[:,k],record['ci95'])
            equal(delta[:,k],[record['per_scene'][s] for s in scenes])
    gate={
        'psnr_gain_over_frozen_anchor_at_least_005db':bool(contrasts['frozen_raw_node_1000'][0][1]>=.05),
        'psnr_gain_over_pixel_at_least_002db':bool(contrasts['pixel'][0][1]>=.02),
        'mse_ci_better_than_frozen_anchor':bool(contrasts['frozen_raw_node_1000'][1][1,0]<0),
        'mse_ci_better_than_pixel':bool(contrasts['pixel'][1][1,0]<0),
        'mean_mse_better_than_shuffled_prior':bool(contrasts['shuffled_prior'][0][0]<0)}
    assert gate==analysis['gate_checks'] and all(gate.values())==analysis['advance_to_replication']
    result={'passed':True,'implementation':'separate raw parsing and scene multiplicity matrix bootstrap; no main analyzer import',
        'source_name_scene_plan_and_provenance_checks':True,'scalar_statistic_checks':int(checks),
        'max_abs_error':max(errors),'gate_checks':gate,'raw_sha256':sha(a.raw),'analysis_file_sha256':sha(a.analysis),
        'reference_manifest_sha256':sha(a.source_reference),'verifier_sha256':sha(__file__),
        'scope':'Statistics and recorded provenance only; actual tensor-state invariance checked by remote evaluator'}
    with a.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
