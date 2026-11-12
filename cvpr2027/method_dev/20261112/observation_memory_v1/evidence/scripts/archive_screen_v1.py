"""Exclusive, hash-verified backup after a complete source screen."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    done=json.loads((root/'screen_checks_v1/complete.json').read_text())
    assert done['source_unchanged'] is True and done['target_inference'] is False
    destination=Path('/temp/zihaohe/seer_cvpr/method_dev_20261112/observation_memory_v1_final_v1')
    destination.mkdir(parents=True,exist_ok=False)
    files=[]
    for folder in ['release_v1','release_v2','tokens_source_v1','runs_v1','results_v1','scripts','logs','checks','screen_logs_v1','screen_checks_v1']:
        directory=root/folder
        if directory.exists():files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.append(root/'prepared_v1/manifest.json')
    files=sorted(set(files));manifest={}
    for source in files:
        rel=source.relative_to(root);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        before=sha(source)
        with source.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b,1<<20)
        assert before==sha(target)==sha(source),str(rel)
        manifest[str(rel)]={'bytes':source.stat().st_size,'sha256':before}
    summary={'scope':'frozen source/code, all projected source addresses, five final heads/init/plans/curves, source scores, preflight and failed-runtime evidence; no original images or prepared RGB tensors',
        'backup_root':str(destination),'files':manifest,'all_sha256_verified':True,
        'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    backup_manifest=destination/'backup_manifest.json'
    with backup_manifest.open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
    local_manifest=root/'backup_manifest.json'
    with local_manifest.open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
    bundle=root/'source_screen_text_and_heads_v1.tar'
    with tarfile.open(bundle,'x') as archive:
        for source in files:
            rel=source.relative_to(root)
            if rel.parts[0]=='tokens_source_v1' and source.suffix=='.pt':continue
            archive.add(source,arcname=str(rel),recursive=False)
        archive.add(local_manifest,arcname='backup_manifest.json')
    receipt={'backup_manifest_sha256':sha(backup_manifest),'local_bundle_sha256':sha(bundle),
        'local_bundle_bytes':bundle.stat().st_size,'backed_up_files':len(files),
        'backed_up_bytes':sum(x['bytes'] for x in manifest.values())}
    with (root/'backup_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
