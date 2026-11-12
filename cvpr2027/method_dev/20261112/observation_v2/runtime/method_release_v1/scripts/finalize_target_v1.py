import pathlib,json,hashlib,gzip,shutil,time,datetime,math,tarfile
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112');task=r/'method_release_v1'
shared=pathlib.Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/observation_v2_final_v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pending={'pretrained','none'};deadline=time.monotonic()+3600
while pending:
 for arm in sorted(tuple(pending)):
  log=task/'logs'/('eval_'+arm+'_31051_v1.log');txt=log.read_text()
  if 'Traceback (most recent call last)' in txt:raise RuntimeError(arm+' evaluation failed; frozen inputs unchanged')
  p=r/'results'/('bvi_val16_'+arm+'_observation_v2.json')
  if '\nEND ' not in txt or not p.exists():continue
  d=json.loads(p.read_text())
  assert d['n']==16 and len(d['per_sequence'])==16 and len({x['name'] for x in d['per_sequence']})==16
  assert len(d['training'])==21 and len(d['checkpoint_sha256'])==24 and all(d['state_unchanged'].values())
  assert all(len(row['methods'])==31 and all(math.isfinite(x) for m in row['methods'].values() for x in m.values()) for row in d['per_sequence'])
  gz=p.with_suffix('.json.gz')
  with gz.open('xb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(p.read_bytes())
  b=shared/'results'/arm;b.mkdir(parents=True,exist_ok=False)
  for source in [p,gz,task/'checks'/('eval_'+arm+'_31051_launch.json'),task/'scripts'/('run_eval_'+arm+'_31051_v1.sh'),log,task/'checks'/('npu_before_eval_'+arm+'_31051.txt'),task/'checks'/('npu_after_eval_'+arm+'_31051.txt')]:
   target=b/source.name
   with source.open('rb') as a,target.open('xb') as f:shutil.copyfileobj(a,f)
   assert sha(source)==sha(target)
  record={'arm':arm,'n_sequences':16,'n_methods':31,'n_new_heads':21,'state_unchanged':d['state_unchanged'],'raw_path':str(p),'raw_sha256':sha(p),'raw_bytes':p.stat().st_size,'gzip_path':str(gz),'gzip_sha256':sha(gz),'gzip_bytes':gz.stat().st_size,'backup':str(b),'metric_source_sha256':d['scoring_library']['sha256'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  dest=r/'results'/(arm+'_artifact_manifest.json')
  with dest.open('x') as f:json.dump(record,f,indent=2)
  with (b/'artifact_manifest.json').open('x') as f:json.dump(record,f,indent=2)
  print(json.dumps(record),flush=True);pending.remove(arm)
 if pending:
  if time.monotonic()>deadline:raise TimeoutError('Final target results not complete in 60 minutes')
  time.sleep(5)
bundle=r/'evidence/target_runtime_v1.tgz'
paths=[p for sub in ('checks','scripts','logs') for p in (task/sub).rglob('*') if p.is_file() and ('eval_' in p.name or 'backup_' in p.name or 'finalize_' in p.name)]
paths += [r/'results'/(arm+'_artifact_manifest.json') for arm in ('pretrained','none')]
with bundle.open('xb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w') as t:
   for p in sorted(set(paths)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
with (r/'evidence/target_runtime_v1.json').open('x') as f:json.dump({'path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size},f,indent=2)
print('ALL TARGET RESULTS FINALIZED',flush=True)
