"""Post-hoc SOURCE diagnosis of a rejected screen; no selection/target score."""
import hashlib
import json
from pathlib import Path
import sys

import torch

root=Path('/cache/zihaohe/seer_observation_memory_20261112')
code=root/'release_v2/code_full'
sys.path.insert(0,str(code/'tools'))
from train_correction_selector import local_module,state_digest
m=local_module('_memory_diagnostic',code/'models/observation_memory.py')
torch.set_num_threads(4)
meta=json.loads((root/'prepared_v1/manifest.json').read_text())
ck=torch.load(root/'runs_v1/prior_cross_s0/final.pt',map_location='cpu',weights_only=True)
model=m.ObservationMemory(**ck['provenance']['model_config']).eval()
model.load_state_dict(ck['model'],strict=True)
shuffled=m.ObservationMemory(address_mode='shuffled_prior',retrieval_mode='cross_frame').eval()
shuffled.load_state_dict(ck['model'],strict=True)
before=state_digest(model.state_dict());rows=[]
with torch.no_grad():
    for row in meta['per_sequence']:
        if row['scene'] not in meta['split']['internal_holdout']:continue
        s=torch.load(root/'prepared_v1'/row['cache_path'],map_location='cpu',weights_only=True,mmap=True)
        base,detail,prior,pixel=(s[k] for k in ('base','detail_values','prior_address','pixel_address'))
        real=model(base,detail,prior,pixel)
        zero=model(base,torch.zeros_like(detail),prior,pixel)
        wrong=shuffled(base,detail,prior,pixel)
        residual=real['pred_pixels']-base
        residual_energy=float(residual.square().mean())
        values={'base':base,'real_memory':real['pred_pixels'],'zero_memory_values':zero['pred_pixels'],
                'shuffled_addresses_frozen_head':wrong['pred_pixels']}
        record={'name':row['name'],'scene':row['scene'],**real['aux'],
            'raw_detail_rms':float(detail.square().mean().sqrt()),
            'memory_vs_zero_prediction_rms':float((values['real_memory']-values['zero_memory_values']).square().mean().sqrt()),
            'residual_rms':residual_energy**.5,
            'lowpass_residual_energy_ratio':float(m._lowpass(residual).square().mean())/max(residual_energy,1e-20),
            'methods':{name:float((v-s['gt']).square().double().mean()) for name,v in values.items()}}
        rows.append(record)
        print(json.dumps(record),flush=True)
assert len(rows)==34 and state_digest(model.state_dict())==before==state_digest(shuffled.state_dict())
result={'scope':'post-hoc source-only diagnosis after fixed gate failure; not a new gate or target result',
    'checkpoint_sha256':hashlib.sha256((root/'runs_v1/prior_cross_s0/final.pt').read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'states_unchanged':True,'target_frames_read':False,'per_sequence':rows}
with (root/'results_v1/posthoc_diagnosis.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
