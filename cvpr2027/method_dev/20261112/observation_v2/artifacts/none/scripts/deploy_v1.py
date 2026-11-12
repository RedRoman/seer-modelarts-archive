import pathlib,hashlib,json,tarfile,shutil,ast,sys,datetime
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112/method_release_v1')
s=pathlib.Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/transport/method_release_v1')
h='6483532a2423d8427b07157e5eca463c378d43c6c51a5b70838f423a3692e996'
if sys.argv[1]=='publish':
 s.mkdir(parents=True,exist_ok=False)
 with (r/'incoming/source_observation_v2.tgz').open('rb') as a,(s/'source_observation_v2.tgz').open('xb') as b:shutil.copyfileobj(a,b)
p=s/'source_observation_v2.tgz'
assert hashlib.sha256(p.read_bytes()).hexdigest()==h
for x in ('scripts','checks','logs'): (r/x).mkdir(parents=True,exist_ok=True)
d=r/'release';d.mkdir(exist_ok=False)
with tarfile.open(p) as t:t.extractall(d)
m=json.loads((d/'source_manifest.json').read_text())
assert hashlib.sha256((d/'source_manifest.json').read_bytes()).hexdigest()=='420f3cdb99c2c109d253f4354794941f95a0865cd860a7e28cc723bad67bfb5e'
assert hashlib.sha256((d/'PLAN.md').read_bytes()).hexdigest()=='a57534fa7cd37fb2a81175de89f929830044ae7dcb9f52e71f6c6cb83590a9f2'
for path,sha in m['files'].items():
 b=(d/'code_full'/path).read_bytes();assert hashlib.sha256(b).hexdigest()==sha,path;ast.parse(b,filename=path)
record={'status':'PASS','n_python':len(m['files']),'archive_sha256':h,'protocol_sha256':hashlib.sha256((d/'PLAN.md').read_bytes()).hexdigest(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'equivalent_code_commit':'9d10c9ae61c603aaee7c9a173fe8a9d675d58df5'}
with (r/'checks/deployment.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
