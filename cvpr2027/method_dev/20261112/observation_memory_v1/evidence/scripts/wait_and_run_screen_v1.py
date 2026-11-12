import json,os,time,sys
from pathlib import Path
r=Path('/cache/zihaohe/seer_observation_memory_20261112')
log=(r/'checks/cpu_tests.log').read_text()
assert 'Ran 24 tests' in log and '\nOK\n' in log and 'skipped=' not in log
start=time.monotonic()
print('Waiting for successful complete136 export; no target or NPU work in this waiter',flush=True)
while not (r/'checks/export_exit_code.txt').exists():
 if time.monotonic()-start>3600: raise TimeoutError('Export not complete within3600sec')
 time.sleep(5)
assert (r/'checks/export_exit_code.txt').read_text().strip()=='0','Export failed; no dependent training'
m=json.loads((r/'tokens_source_v1/manifest.json').read_text())
assert m['complete'] is True and m['n']==136 and len(m['per_sequence'])==136 and m['encoder_state_unchanged'] is True
assert m['target_input'] is False and m['gt_input'] is False
assert not (r/'runs_v1').exists(),'A source screen already exists; do not repeat'
print('Export complete and immutable; starting exactly one source CPU screen',flush=True)
py='/cache/zihaohe/seer_cvpr/venv_npu/bin/python'
os.execv(py,[py,str(r/'scripts/run_screen_v1.py'),'--root',str(r)])
