import pathlib,hashlib,json,shutil,datetime
r=pathlib.Path('/cache/zihaohe/seer_observation_dev_20261112')
shared=pathlib.Path('/temp/zihaohe/seer_cvpr/method_dev_20261112')
out=shared/'observation_v2_final_v1';out.mkdir(exist_ok=False)
files={}
def copy(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)
 h=hashlib.sha256(source.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
 files[str(target.relative_to(out))]={'sha256':h,'bytes':target.stat().st_size,'source':str(source)}
for arm in ('pretrained','none'):
 source=shared/'transport/observation_heads_v1'/arm
 m=json.loads((source/'transport_manifest.json').read_text());assert hashlib.sha256((source/'heads.tar').read_bytes()).hexdigest()==m['archive_sha256']
 for name in ('heads.tar','transport_manifest.json'):copy(source/name,out/'checkpoints'/arm/name)
for source,name in [(shared/'transport/method_release_v1/source_observation_v2.tgz','source_observation_v2.tgz'),(r/'method_release_v1/release/source_manifest.json','source_manifest.json'),(r/'method_release_v1/release/PLAN.md','PLAN.md')]:copy(source,out/'source'/name)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'backup_root':str(out),'files':files,'head_count':42,'all_final_steps':1000,'includes_initial_state_and_sample_plan':True,'status':'all_copy_sha_verified','base_checkpoint_existing_shared':['/temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_pretrained_s0_final_cpu.pt','/temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_none_s0_final_cpu.pt']}
with (out/'checkpoint_backup_manifest.json').open('x') as f:json.dump(record,f,indent=2)
with (r/'method_release_v1/checks/checkpoint_backup_manifest.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
