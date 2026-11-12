from pathlib import Path
import datetime,json,subprocess,os,sys
root=Path('/cache/zihaohe/seer_hvi_temporal_fit_feasibility_20261112')
release=root/'release_v1'
python='/cache/zihaohe/seer_cvpr/venv_npu/bin/python'
env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
argv=[python,'-B',str(release/'run_fit_feasibility.py'),'--source-cache','/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0','--hvi-cache','/cache/zihaohe/seer_hvi_source_anchor_20261112/cache_v1','--geometry-manifest','/cache/zihaohe/seer_observation_geometry_20261112/prepared_v1/manifest.json','--out',str(root/'results_v1')]
with (root/'checks_v1/launch.json').open('x') as f:json.dump({'argv':argv,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_threads':4,'npu_use':False},f,indent=2)
with (root/'logs_v1/fit.log').open('x') as log:
 result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
with (root/'checks_v1/exit_status.json').open('x') as f:json.dump({'exit_code':result.returncode,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
print('FIT diagnostic exit',result.returncode,flush=True)
sys.exit(result.returncode)
