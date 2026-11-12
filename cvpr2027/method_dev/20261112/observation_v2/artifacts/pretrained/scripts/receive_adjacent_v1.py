import pathlib,json,hashlib,tarfile,sys,datetime
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112')
arm=sys.argv[1];s=pathlib.Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/transport/adjacent_flow_v1')/arm
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
m=json.loads((s/'transport_manifest.json').read_text());assert m['n_files']==138 and m['all_flow_sha_verified']
assert digest(s/'adjacent.tar')==m['archive_sha256']
d=r/'adjacent_flow'/arm;d.mkdir(parents=True,exist_ok=False)
with tarfile.open(s/'adjacent.tar') as t:t.extractall(d)
for name,v in m['files'].items():assert (d/name).stat().st_size==v['bytes'] and digest(d/name)==v['sha256'],name
rec={'arm':arm,'all_files_sha_verified':True,'archive_sha256':m['archive_sha256'],'n_files':138,'total_bytes':m['total_bytes'],'source':'30794','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'destination':str(d)}
with (r/'method_release_v1/checks'/('adjacent_'+arm+'_received.json')).open('x') as f:json.dump(rec,f,indent=2)
with (r/'method_release_v1/checks'/('adjacent_'+arm+'_transport_manifest.json')).open('x') as f:json.dump(m,f,indent=2)
print(json.dumps(rec),flush=True)
