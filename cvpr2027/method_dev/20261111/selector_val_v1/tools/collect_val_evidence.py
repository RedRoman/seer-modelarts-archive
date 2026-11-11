import pathlib,json,hashlib,tarfile,math,datetime,subprocess
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_val_v1')
log=(root/'logs/eval_bvi_val16_s0.log').read_text()
assert '\nEND ' in log and 'Traceback' not in log
p=root/'results/bvi_val16_s0.json'
final=json.loads(p.read_text())
assert final['n']==16 and final['split']=='val' and final['domain']=='bvi'
rows=final['per_sequence'];names=[r['name'] for r in rows]
assert len(rows)==16 and len(set(names))==16
assert set(names)=={r['name'] for r in final['validation_pair_inventory']}
assert len({n.split('__')[0] for n in names})==8
methods={'base','fixed_average','initial_thirds','risk_full','direct_mse','direct_mse_normalized'}
for row in rows:
 assert set(row['methods'])==methods
 gt=[x['gt_tlp'] for x in row['methods'].values()]
 assert all(x==gt[0] for x in gt)
 assert all(math.isfinite(x) for scores in row['methods'].values() for x in scores.values())
for m in methods:
 for k,v in final['means'][m].items():
  assert abs(v-sum(r['methods'][m][k] for r in rows)/16)<1e-12
train_root=root.parent/'selector_v1/runs/s0'
for mode,expected in final['selector_checkpoint_sha256'].items():
 assert hashlib.sha256((train_root/f'selector_{mode}.pt').read_bytes()).hexdigest()==expected
manifest=json.loads((root/'checks/source_manifest.json').read_text())
for n,expected in manifest['files'].items():
 assert hashlib.sha256((root/'code_full'/n).read_bytes()).hexdigest()==expected,n
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
(root/'checks/npu_after.txt').write_text(smi)
assert 'No running processes found in NPU 7' in smi
paths=[]
for sub in ['results','logs','checks','tools','incoming']:
 paths.extend(p for p in (root/sub).rglob('*') if p.is_file())
paths.append(root/'launch_manifest.json')
inventory={str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
(root/'checks/final_artifact_inventory.json').write_text(json.dumps({'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n_sequences':16,'n_scenes':8,'files':inventory},indent=2)+'\n')
paths.append(root/'checks/final_artifact_inventory.json')
with tarfile.open(root/'selector_val_artifacts.tgz','w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(root)))
print(json.dumps({'bundle':str(root/'selector_val_artifacts.tgz'),'sha256':hashlib.sha256((root/'selector_val_artifacts.tgz').read_bytes()).hexdigest(),'size':(root/'selector_val_artifacts.tgz').stat().st_size,'means':final['means']},indent=2))
