import sys,pathlib,json,hashlib,time
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112/tempretinex_reference_v1');sys.path[:0]=[str(r/'code_full'),str(r/'code_full/tools')]
import torch,torch_npu
from eval_tempretinex_reference import load_model,enhance_clip,state_digest
torch.set_num_threads(4);start=time.monotonic();model,p=load_model(r/'inputs',torch.device('npu:0'));before=state_digest(model.state_dict())
cpu=json.loads((r/'checks/cpu_smoke_frozen.json').read_text());assert before==cpu['provenance']['state_sha256']
low=torch.rand((2,3,256,256),generator=torch.Generator().manual_seed(2)).mul(.1).to('npu:0')
a=enhance_clip(model,low);b=enhance_clip(model,low);torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
assert a.shape==low.shape and torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
after=state_digest(model.state_dict());assert before==after
v={'status':'PASS','purpose':'Synthetic NPU actual crop256 two-frame forward/reset; no target data','shape':list(a.shape),'finite':True,'reset_max_abs':float((a-b).abs().max().cpu()),'state_before':before,'state_after':after,'cpu_state_exact':True,'elapsed_seconds':time.monotonic()-start,'physical_npu':1,'output_sha256':hashlib.sha256(a.cpu().numpy().tobytes()).hexdigest()}
with (r/'checks/npu_smoke_v1.json').open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v),flush=True)
