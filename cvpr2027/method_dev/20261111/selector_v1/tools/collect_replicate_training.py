import pathlib,json,hashlib,tarfile,math,datetime
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_v1')
reference=json.loads((root/'runs/s0/provenance.json').read_text())
modes=['risk_full','direct_mse','direct_mse_normalized']
paths=[];summaries={}
for seed in (1,2):
 run=root/f'runs/s{seed}'
 log=root/f'logs/train_selector_s{seed}.log'
 text=log.read_text();assert '\nEND ' in text and 'Traceback' not in text
 prov=json.loads((run/'provenance.json').read_text())
 assert prov['args']['seed']==seed and prov['args']['steps']==1000
 for key in ['split','normalization','features','cache_provenance_sha256','source_sha256']:
  assert prov[key]==reference[key],(seed,key)
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
  assert side['initial_state_sha256']==prov['initial_state_sha256'] and side['sample_plan_sha256']==prov['sample_plan_sha256']
 assert first[0]==first[1]==first[2]
 final=json.loads((run/'internal_holdout.json').read_text())
 assert final['clips']==12 and final['scenes']==6 and final['frames']==192
 summaries[seed]={'initial_state_sha256':prov['initial_state_sha256'],'sample_plan_sha256':prov['sample_plan_sha256'],'first_batch_mse':first,'internal_holdout':{k:v['rgb_mse'] for k,v in final['methods'].items()}}
 paths.extend(p for p in run.iterdir() if p.is_file())
 paths.extend([log,root/f'launch_manifest_s{seed}.json'])
paths.extend(root/p for p in ['tools/run_selector_replicate_35e7a7.sh','tools/collect_replicate_training.py','checks/source_manifest.json','logs/cpu_tests.log'])
assert len({reference['initial_state_sha256'],*[s['initial_state_sha256'] for s in summaries.values()]})==3
assert len({reference['sample_plan_sha256'],*[s['sample_plan_sha256'] for s in summaries.values()]})==3
inventory={str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
out=root/'checks/replicate_training_inventory.json'
out.write_text(json.dumps({'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summaries':summaries,'files':inventory},indent=2)+'\n');paths.append(out)
bundle=root/'selector_replicate_train_artifacts.tgz'
with tarfile.open(bundle,'w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(root)))
print(json.dumps({'bundle':str(bundle),'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'size':bundle.stat().st_size,'summaries':summaries},indent=2))
