"""Exclusive verified text-only SFS backup of the completed FIT diagnostic."""
from pathlib import Path
import hashlib
import json
import os
import tarfile

ROOT=Path('/cache/zihaohe/seer_hvi_temporal_fit_feasibility_20261112')
DEST=Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/hvi_temporal_fit_feasibility_v1_final_v1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    verify=json.loads((ROOT/'checks_v1/independent_verification.json').read_text())
    assert verify['passed'] and verify['fit_clips']==102 and verify['fit_scenes']==51
    assert verify['result_sha256']==sha(ROOT/'results_v1/fit_results.json')
    assert json.loads((ROOT/'checks_v1/exit_status.json').read_text())['exit_code']==0
    running=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal() or int(process.name)==os.getpid():continue
        try:argv=(process/'cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError,PermissionError):continue
        argv=[x.decode(errors='replace') for x in argv if x]
        if not argv or 'python' not in Path(argv[0]).name:continue
        if any(Path(x).name in {'run_fit_feasibility.py','launch_v1.py','verify_fit.py','verify_fit_v2.py'}
               and str(ROOT) in x for x in argv[1:]):
            running.append({'pid':int(process.name),'argv':argv})
    assert not running,running
    write(ROOT/'checks_v1/final_process_check.json',{'computation_processes':running,'count':0,'npu_used':False})
    DEST.mkdir(parents=True,exist_ok=False)
    entries=[]
    for source in sorted(ROOT.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts:continue
        assert source.suffix in {'.py','.md','.json','.log'},source
        source.read_text(encoding='utf-8')
        relative=source.relative_to(ROOT);target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as input_stream,target.open('xb') as output_stream:
            for block in iter(lambda:input_stream.read(1<<20),b''):output_stream.write(block)
        assert sha(source)==sha(target)
        entries.append({'path':str(relative),'sha256':sha(source),'bytes':source.stat().st_size})
    manifest={'files':entries,'files_verified':len(entries),'bytes_verified':sum(r['bytes'] for r in entries),
        'prediction_or_gt_arrays_included':False,'root':str(ROOT),'backup':str(DEST)}
    write(DEST/'backup_manifest.json',manifest)
    receipt={'backup':str(DEST),'backup_manifest_sha256':sha(DEST/'backup_manifest.json'),
        'files_verified':len(entries),'bytes_verified':manifest['bytes_verified'],'all_file_sha256_verified':True,
        'prediction_or_gt_arrays_included':False}
    write(ROOT/'checks_v1/backup_receipt.json',receipt)
    write(DEST/'backup_receipt.json',receipt)
    bundle=ROOT.parent/'seer_hvi_temporal_fit_evidence_v1.tar'
    with tarfile.open(bundle,'x') as tar:
        for source in sorted(ROOT.rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                tar.add(source,arcname=str(source.relative_to(ROOT)),recursive=False)
        tar.add(DEST/'backup_manifest.json',arcname='backup_manifest.json',recursive=False)
    artifact={'path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size}
    write(ROOT/'checks_v1/download_artifact.json',artifact)
    print(json.dumps({'receipt':receipt,'artifact':artifact},indent=2))


if __name__=='__main__':main()
