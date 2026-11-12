import sys,json,time,hashlib
from pathlib import Path
import torch
import torch_npu
code=Path('/cache/zihaohe/seer_observation_memory_20261112/release_v2/code_full')
sys.path.insert(0,str(code/'tools'))
import cache_observation_memory as m
torch.set_num_threads(4)
start=time.monotonic()
wrapper=m.local_module('_cpu_vjepa',code/'models/vjepa_backbone.py')
repo=Path('/temp/zihaohe/seer_cvpr/modelzoo/vjepa2-main')
weights=Path('/temp/zihaohe/seer_cvpr/weights/vjepa2_1_vitl_dist_vitG_384.pt')
bb=wrapper.VJepaBackboneLocal21(local_repo=str(repo),local_ckpt='',feature_layers=(17,),image_size=256,frames_per_clip=16,patch_size=16,tubelet_size=2,freeze=True)
ck=torch.load(str(weights),map_location='cpu',weights_only=False,mmap=True)
state,ignored=m.clean_encoder_state(ck,bb.model.state_dict())
bb.model.load_state_dict(state,strict=True)
result={'strict_video_keys':len(state),'ignored_unused_image_keys':ignored,'parameters':sum(p.numel() for p in bb.parameters()),'all_parameters_cpu':all(p.device.type=='cpu' for p in bb.parameters()),'eval':not bb.training,'all_parameters_frozen':not any(p.requires_grad for p in bb.parameters()),'feature_layers':bb.feature_layers,'elapsed_seconds':time.monotonic()-start,'encoder_state_sha256':m.state_sha(bb),'upstream_source_sha256':m.upstream_sources(repo),'torch':str(torch.__version__)}
assert result['all_parameters_cpu'] and result['eval'] and result['all_parameters_frozen'] and result['feature_layers']==[17]
Path('/cache/zihaohe/seer_observation_memory_20261112/checks/cpu_real_npu_load_v2.json').write_text(json.dumps(result,indent=2)+'\n')
row=json.loads(Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0/cache_summary.json').read_text())['per_sequence'][0]
origin=json.loads(Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0/provenance.json').read_text())
low=m.load_lq(Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0')/row['cache_path'],row,origin['plan'][0])
assert low.device.type=='cpu'
print('torch_npu imported; real weights and cached LQ mmap loading PASS',flush=True)
print(json.dumps(result),flush=True)
