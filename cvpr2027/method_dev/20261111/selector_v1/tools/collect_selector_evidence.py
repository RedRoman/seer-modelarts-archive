import pathlib,json,hashlib,tarfile,math,datetime
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_v1')
run=root/'runs/s0'
log=(root/'logs/train_selector_s0.log').read_text()
assert '\nEND ' in log and 'Traceback' not in log
modes=['risk_full','direct_mse','direct_mse_normalized']
prov=json.loads((run/'provenance.json').read_text())
plan=run/'sample_plan.json'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==prov['sample_plan_sha256']
assert len(json.loads(plan.read_text())['batches'])==1000
first=[]
for mode in modes:
 rows=[json.loads(x) for x in (run/f'curve_{mode}.jsonl').read_text().splitlines()]
 assert [x['step'] for x in rows]==list(range(1,1001))
 assert all(math.isfinite(x[k]) for x in rows for k in ['loss','batch_rgb_mse','batch_base_mse'])
 first.append(rows[0]['batch_rgb_mse'])
 side=json.loads((run/f'selector_{mode}.sha256.json').read_text())
 assert hashlib.sha256((run/f'selector_{mode}.pt').read_bytes()).hexdigest()==side['checkpoint_sha256']
 assert side['initial_state_sha256']==prov['initial_state_sha256']
 assert side['sample_plan_sha256']==prov['sample_plan_sha256']
assert first[0]==first[1]==first[2],first
final=json.loads((run/'internal_holdout.json').read_text())
assert final['clips']==12 and final['scenes']==6 and final['frames']==192
paths=[]
for sub in ['runs/s0','logs','checks','tools','incoming']:
 paths.extend(p for p in (root/sub).rglob('*') if p.is_file())
paths.append(root/'launch_manifest.json')
inventory={str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
(root/'checks/final_artifact_inventory.json').write_text(json.dumps({'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'first_batch_mse':first,'files':inventory},indent=2)+'\n')
paths.append(root/'checks/final_artifact_inventory.json')
with tarfile.open(root/'selector_artifacts.tgz','w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(root)))
print(json.dumps({'bundle':str(root/'selector_artifacts.tgz'),'sha256':hashlib.sha256((root/'selector_artifacts.tgz').read_bytes()).hexdigest(),'size':(root/'selector_artifacts.tgz').stat().st_size,'final_mse':{k:v['rgb_mse'] for k,v in final['methods'].items()}},indent=2))
