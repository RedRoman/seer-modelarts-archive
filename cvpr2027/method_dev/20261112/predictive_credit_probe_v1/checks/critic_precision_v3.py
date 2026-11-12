"""Third synthetic gate (NPU registration follows CPU loading): explicit FP32 precision, unchanged model/SDPA/weights."""
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from critic_export import load_models, normalize_rgb, normalized_loss, write_new

start = time.monotonic()
encoder, trained, random, helper, before, omitted = load_models()
import torch_npu
torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False
torch.set_num_threads(4)
torch.npu.set_device('npu:0')


video = normalize_rgb(torch.linspace(0, 1, 3*4*256*256, dtype=torch.float32).reshape(1,3,4,256,256))
context_ids = torch.arange(32).unsqueeze(0)
target_ids = torch.arange(256,272).unsqueeze(0)

def once(device):
    for model in (encoder, trained, random):
        model.to(device)
    clip, context, target = video.to(device), context_ids.to(device), target_ids.to(device)
    shapes = []
    hook = encoder.blocks[0].register_forward_pre_hook(lambda module,args:shapes.append(list(args[0].shape)))
    with torch.inference_mode():
        x = encoder(clip,masks=[context])
        y = encoder(clip,masks=[target])
        changed = clip.clone(); changed[:,:,2:] += .12345
        xx = encoder(changed,masks=[context])
        predictions = [model(x,masks_x=[context],masks_y=[target],mask_index=0) for model in (trained,random)]
        altered = trained(xx,masks_x=[context],masks_y=[target],mask_index=0)
        assert torch.equal(x,xx) and torch.equal(predictions[0],altered)
        values = {'context':x.cpu(),'target':y.cpu(),
                  'trained_normalized':F.normalize(predictions[0],dim=-1).cpu(),
                  'random_normalized':F.normalize(predictions[1],dim=-1).cpu(),
                  'trained_loss':normalized_loss(predictions[0],y).cpu(),
                  'random_loss':normalized_loss(predictions[1],y).cpu()}
    hook.remove()
    assert shapes == [[1,32,1024],[1,16,1024],[1,32,1024]]
    assert all(bool(torch.isfinite(value).all()) for value in values.values())
    return values,shapes

cpu, cpu_shapes = once('cpu')
npu, npu_shapes = once('npu:0')
differences = {}
for name in cpu:
    difference = (cpu[name]-npu[name]).abs()
    differences[name] = {'abs_max':float(difference.max()),'abs_mean':float(difference.mean())}
loss_diff = torch.cat([(cpu[key]-npu[key]).abs().flatten() for key in ['trained_loss','random_loss']])
passed = float(loss_diff.max()) <= .01 and float(loss_diff.mean()) <= .001
after = {name:helper.state_sha(model.state_dict()) for name,model in [('encoder',encoder),('trained',trained),('random',random)]}
assert before == after
report = {'status':'passed' if passed else 'failed','precision':'must_keep_origin_dtype',
          'conv_allow_hf32':bool(torch.npu.conv.allow_hf32),'matmul_allow_hf32':bool(torch.npu.matmul.allow_hf32),
          'attention_implementation':'unchanged official use_sdpa=True on bothCPU/NPU',
          'differences':differences,'cpu_losses':{key:cpu[key].tolist() for key in ['trained_loss','random_loss']},
          'npu_losses':{key:npu[key].tolist() for key in ['trained_loss','random_loss']},
          'normalized_loss_abs_max':float(loss_diff.max()),'normalized_loss_abs_mean':float(loss_diff.mean()),
          'predeclared_max_tolerance':.01,'predeclared_mean_tolerance':.001,
          'pre_attention_shapes':{'cpu':cpu_shapes,'npu':npu_shapes},
          'masked_context_exact_invariance':True,'masked_prediction_exact_invariance':True,
          'target_token_ids':target_ids.tolist(),'state_before':before,'state_after':after,'states_unchanged':True,
          'dataset_pixels_read':0,'gt_storage_entries_read':0,'quality_scores_computed':False,
          'elapsed_seconds':time.monotonic()-start}
write_new(Path(sys.argv[1]),report)
print(json.dumps(report,indent=2),flush=True)
assert passed, 'Unchanged parity thresholds failed; no data export allowed'
