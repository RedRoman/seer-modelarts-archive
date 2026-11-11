import pathlib,json,hashlib,tarfile,math,datetime,subprocess
root=pathlib.Path('/cache/zihaohe/seer_method_dev_20261111/selector_val_v1')
reference=json.loads((root/'results/bvi_val16_s0.json').read_text())
ref={r['name']:r for r in reference['per_sequence']}
methods={'base','fixed_average','initial_thirds','risk_full','direct_mse','direct_mse_normalized'}
paths=[];checks={}
for seed in (1,2):
 p=root/f'results/bvi_val16_s{seed}.json';log=root/f'logs/eval_bvi_val16_s{seed}.log'
 text=log.read_text();assert '\nEND ' in text and 'Traceback' not in text
 final=json.loads(p.read_text());rows=final['per_sequence'];names=[r['name'] for r in rows]
 assert final['n']==16 and final['split']=='val' and final['domain']=='bvi'
 assert len(rows)==16 and len(set(names))==16 and set(names)==set(ref)
 assert set(names)=={r['name'] for r in final['validation_pair_inventory']}
 assert len({n.split('__')[0] for n in names})==8
 assert final['selector_training']['args']['seed']==seed
 for key in ['source_sha256','base_cache_provenance','validation_pair_inventory']:
  assert final[key]==reference[key],(seed,key)
 maxbase={k:0. for k in final['means']['base']}
 drift={m:{k:0. for k in final['means'][m]} for m in ['fixed_average','initial_thirds']}
 for row in rows:
  assert set(row['methods'])==methods
  gt=[x['gt_tlp'] for x in row['methods'].values()];assert all(x==gt[0] for x in gt)
  assert all(math.isfinite(x) for scores in row['methods'].values() for x in scores.values())
  for k,v in row['methods']['base'].items():maxbase[k]=max(maxbase[k],abs(v-ref[row['name']]['methods']['base'][k]))
  for m in drift:
   for k,v in row['methods'][m].items():drift[m][k]=max(drift[m][k],abs(v-ref[row['name']]['methods'][m][k]))
 for m in methods:
  for k,v in final['means'][m].items():assert abs(v-sum(r['methods'][m][k] for r in rows)/16)<1e-12
 for mode,expected in final['selector_checkpoint_sha256'].items():
  assert hashlib.sha256((root.parent/f'selector_v1/runs/s{seed}/selector_{mode}.pt').read_bytes()).hexdigest()==expected
 checks[seed]={'base_max_abs_differences_vs_seed0':maxbase,'base_exactly_equal':all(v==0 for v in maxbase.values()),'fixed_correction_max_abs_differences_vs_seed0':drift,'means':final['means']}
 paths.extend([p,log,root/f'launch_manifest_s{seed}.json',root/f'checks/npu_before_s{seed}.txt'])
manifest=json.loads((root/'checks/source_manifest.json').read_text())
for n,expected in manifest['files'].items():assert hashlib.sha256((root/'code_full'/n).read_bytes()).hexdigest()==expected,n
smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
(root/'checks/npu_after_replicates.txt').write_text(smi)
assert 'No running processes found in NPU 7' in smi
paths.extend(root/p for p in ['checks/source_manifest.json','checks/npu_after_replicates.txt','tools/run_val_replicate_f2cf65.sh','tools/run_val_replicates_queue.sh','tools/collect_replicate_val.py','logs/eval_replicates_queue.log'])
inventory={str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
out=root/'checks/replicate_val_inventory.json'
out.write_text(json.dumps({'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'files':inventory},indent=2)+'\n');paths.append(out)
bundle=root/'selector_replicate_val_artifacts.tgz'
with tarfile.open(bundle,'w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(root)))
print(json.dumps({'bundle':str(bundle),'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'size':bundle.stat().st_size,'checks':checks},indent=2))
