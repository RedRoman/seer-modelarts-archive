import pathlib,tarfile,gzip,json,hashlib,sys,datetime
arm=sys.argv[1];r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112');task=r/'method_release_v1'
paths=[p for p in (r/'runs'/arm).rglob('*') if p.is_file()]
paths+=[r/'prepared'/arm/'prepared_manifest.json',task/'release/source_manifest.json',task/'release/PLAN.md']
for sub in ('checks','scripts','logs'):
 for p in (task/sub).rglob('*'):
  if p.is_file() and 'eval_' not in p.name and 'backup_' not in p.name:paths.append(p)
out=r/'evidence'/('training_'+arm+'_v1.tgz')
with out.open('xb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:
  with tarfile.open(fileobj=g,mode='w') as t:
   for p in sorted(set(paths)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
v={'arm':arm,'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'files':len(set(paths)),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (r/'evidence'/('training_'+arm+'_v1.json')).open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v),flush=True)
