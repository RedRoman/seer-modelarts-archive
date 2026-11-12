"""Verify the unchanged source base/anchor reproduce the preceding full source evaluation."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for k in ('raw','previous','out'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();new=json.loads(a.raw.read_text());old=json.loads(a.previous.read_text())
by_name={r['name']:r for r in old['per_sequence']}
assert len(by_name)==34 and set(by_name)=={r['name'] for r in new['per_sequence']}
errors=[]
for row in new['per_sequence']:
 previous=by_name[row['name']];assert row['scene']==previous['scene']
 for method in ('base','frozen_raw_node_1000'):
  for metric in ('rgb_mse_float','psnr_rgb_from_clip_mse'):
   error=abs(row['methods'][method][metric]-previous['methods'][method][metric]);errors.append(error)
   tolerance=1e-15 if metric=='rgb_mse_float' else 1e-10
   assert error<=tolerance,(row['name'],method,metric,error)
assert new['reference_checkpoint_sha256']==old['reference_checkpoint_sha256']
result={'passed':True,'scope':'Unchanged strong control and base across complete source34 clips; no target',
 'scalar_checks':len(errors),'max_abs_error':max(errors),'tolerances':{'rgb_mse_float':1e-15,'psnr_rgb_from_clip_mse':1e-10},'new_raw_sha256':sha(a.raw),
 'previous_raw_sha256':sha(a.previous),'verifier_sha256':sha(__file__)}
with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
