"""In-task closure after fixed workers complete; no inference, retries or selection."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/cache/zihaohe/seer_predictive_credit_probe_20261112_v1')
release=root/'cpu_sharded_release_v2'
deadline=time.monotonic()+3600
while True:
    receipts=[root/'checks'/('critic_worker_v2_'+str(i))/'critic_cpu_export_complete.json' for i in range(4)]
    if all(path.exists() for path in receipts):
        assert all(json.loads(path.read_text())['status']=='passed' for path in receipts)
        break
    if time.monotonic()>deadline:raise TimeoutError('Fixed workers did not complete; no merge or retry')
    time.sleep(45)
environment=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONDONTWRITEBYTECODE='1')
for script,arguments in [
    (release/'critic_merge_shards_v2.py',[str(release)]),
    (root/'checks/critic_independent_identity_v1.py',[]),
    (release/'critic_archive_v2.py',[])]:
    subprocess.run(['/cache/zihaohe/seer_cvpr/venv_npu/bin/python',str(script),*arguments],
                   cwd=release,env=environment,check=True)
print('CRITIC_FINALIZE_PASSED',flush=True)
