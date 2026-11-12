"""Prepare a byte-frozen synthetic-only release; not dataset execution."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

phase = Path(__file__).resolve().parents[1]
repo = phase.parents[4]
destination = Path('/tmp/seer_predictive_credit_smoke_release_v1')
destination.mkdir(exist_ok=False)
names = ['critic_export.py', 'test_critic_export.py', 'critic_protocol.json', 'subsets.json']
for name in names:
    shutil.copyfile(phase/name, destination/name)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
asset = phase.parent/'predictive_evidence_v1/checks/matched_predictor_assets_v1'
upstream = json.loads((asset/'evidence/upstream_manifest.json').read_text())
root = '/cache/zihaohe/seer_matched_predictor_assets_20261112_v1'
dependencies = {root+'/upstream/'+name: value for name, value in upstream['files'].items()}
dependencies.update({root+'/inspect_and_smoke.py': sha(asset/'inspect_and_smoke.py'),
    '/cache/zihaohe/seer_color_contract_20261112/export_release_v1/dependencies/cache_hvi_source.py':
        '26c0c36656ad7b7d3b0ce04275578a45286c27931a476c9dcfb1ce1981e69db2',
    '/cache/zihaohe/seer_color_contract_20261112/export_release_v1/dependencies/color_contract.py':
        '7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60',
    '/cache/zihaohe/seer_color_contract_20261112/export_release_v1/proofs/source_color_receipt_closed.json':
        'fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70'})
manifest = {'purpose': 'synthetic CPU/NPU gate only; dataset pixels0',
    'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files': {name: sha(destination/name) for name in names},
    'absolute_dependencies': dependencies}
(destination/'critic_release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(phase/'checks/critic_smoke_release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
with tarfile.open('/tmp/seer_predictive_credit_smoke_release_v1.tar', 'w') as archive:
    for path in sorted(destination.iterdir()):
        archive.add(path, arcname=path.name)
print(json.dumps({'release': str(destination), 'files': manifest['files'],
                  'tar_sha256': sha(Path('/tmp/seer_predictive_credit_smoke_release_v1.tar'))}, indent=2))
