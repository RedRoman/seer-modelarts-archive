"""Two fixed FIT clips: GT-free source/archive/loader/channel-contract audit.

No model is constructed, no GT pixel is read, and no quality score is computed.
Original paired loaders receive zero GT placeholders solely for their shape
contract; actual LQ values are read by the unchanged historical utility.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def tensor_sha(value):return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    code=Path('/cache/zihaohe/seer_observation_geometry_20261112/release_v1/code_full')
    source=Path('/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0')
    geometry=Path('/cache/zihaohe/seer_observation_geometry_20261112/prepared_v1/manifest.json')
    hvi_root=Path('/cache/zihaohe/seer_hvi_source_anchor_20261112')
    root=Path('/temp/zihaohe/seer_cvpr');split_path=root/'datasets/splits_cvpr.json'
    metadata=json.loads(geometry.read_text());origin=json.loads((source/'provenance.json').read_text())
    assert sha(source/'provenance.json')=='0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0'
    assert metadata['split']['fit'][0]=='SDSD_outdoor/MVI_0975'
    selected=[(i,r) for i,r in enumerate(metadata['per_sequence']) if r['scene']==metadata['split']['fit'][0]]
    assert [i for i,_ in selected]==[108,109]
    for name,expected in origin['source_sha256'].items():assert sha(code/name)==expected
    sys.path[:0]=[str(code),str(code/'tools')]
    import data.common as common
    producer=load_module('_fixed_source_producer',code/'tools/cache_source_corrections.py')
    exporter=load_module('_fixed_hvi_lq_reader',hvi_root/'release_v1/cache_hvi_source.py')
    hvi_manifest=json.loads((hvi_root/'cache_v1/manifest.json').read_text())
    source_reader=common.read_image_01
    dataset=producer.SourceClips(split_path,root,seed=20261113,crop=256,clip_len=16)
    assert dataset.plan==origin['plan']
    # Execute only the original ListedSDSD class body, avoiding training imports.
    training_text=(code/'tools/seer_train_cvpr.py').read_text()
    klass=next(n for n in ast.parse(training_text).body if isinstance(n,ast.ClassDef) and n.name=='ListedSDSD')
    namespace={'Dataset':torch.utils.data.Dataset,'np':np,'os':os,
        **{name:getattr(common,name) for name in ('list_images','sample_clip_indices','clip_to_tensor',
            'make_eval_rng','paired_random_crop','paired_hflip','paired_vflip','paired_center_crop')}}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(code/'tools/seer_train_cvpr.py'),'exec'),namespace)
    urls={
      'retinexformer_data_util.py':'https://raw.githubusercontent.com/caiyuanhao1998/Retinexformer/1e9a0efce4b306b6701b824768370ff26066c32a/basicsr/data/util.py',
      'sdsd_data_util.py':'https://raw.githubusercontent.com/JIA-Lab-research/SDSD/main/data/util.py'}
    references={}
    for filename,url in urls.items():
        data=urllib.request.urlopen(url,timeout=30).read();target=args.out/filename
        target.write_bytes(data);references[filename]={'url':url,'sha256':sha(target)}
    official=load_module('_fixed_official_retinexformer_util',args.out/'retinexformer_data_util.py')
    provenance_root=root/'datasets/.sdsd_static_download_20261018'
    provenance=json.loads((provenance_root/'sdsd_static_provenance.json').read_text())
    receipt=json.loads((provenance_root/'download_receipt.json').read_text())
    archive=root/'datasets/sdsd_static_archives/outdoor_static_np.zip'
    sevenzip=root/'datasets/.sdsd_download_20261017/tools/7zz'
    arrays_read=[];gt_placeholders=[];records=[]
    torch.set_num_threads(4)
    for index,row in selected:
        assert row['scene'] in metadata['split']['fit'] and row['scene'] not in metadata['split']['internal_holdout']
        low_paths,_=dataset.items[index];seed=dataset.plan[index]['augmentation_seed']
        rng=np.random.default_rng(seed);indices=common.sample_clip_indices(len(low_paths),16,rng)
        paths=[Path(low_paths[i]) for i in indices];allowed={str(v.resolve()) for v in paths}
        last=[None]
        def guarded_reader(path):
            path=Path(path)
            if 'GT' in path.parts or 'output' in path.parts:
                assert last[0] is not None
                gt_placeholders.append(str(path));return np.zeros_like(last[0])
            assert str(path.resolve()) in allowed,('Unexpected LQ pixel request',str(path))
            value=source_reader(str(path));arrays_read.append(str(path));last[0]=value
            return value
        common.read_image_01=guarded_reader
        replay=dataset[index]['lq'][None]
        namespace['read_image_01']=guarded_reader
        namespace['make_train_rng']=lambda:np.random.default_rng(seed)
        domain,sequence=row['name'].split('/')
        folder=json.loads(split_path.read_text())[domain]['root']
        train=namespace['ListedSDSD'](folder,[sequence],16,256,True)[0]['lq'][None]
        cached_name,cached_scene,cached,export_lineage=exporter.read_lq_only(row['source_path'])
        assert sha(row['source_path'])==row['source_sha256']
        assert cached_name==row['name'] and cached_scene==row['scene']
        assert torch.equal(replay,cached) and torch.equal(train,cached)
        hm=hvi_manifest['per_sequence'][index]
        assert hm['name']==row['name'] and hm['lq_tensor_sha256']==tensor_sha(cached)==export_lineage['lq_tensor_sha256']
        # Reapply the identical RNG plan to the official sequence RGB reader.
        official_rgb=official.read_img_seq2([str(v) for v in paths]).permute(0,2,3,1).numpy()
        rgb,_=common.paired_random_crop(official_rgb,official_rgb,256,rng)
        rgb,_=common.paired_hflip(rgb,rgb,.5,rng);rgb,_=common.paired_vflip(rgb,rgb,.5,rng)
        rgb=common.clip_to_tensor(rgb)[None]
        swapped=cached[:,[2,1,0]]
        assert torch.equal(rgb,swapped)
        frame_metadata=[]
        for path in paths:
            raw=np.load(path,allow_pickle=False)
            assert raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[-1]==3
            member=str(path.relative_to(root/'datasets'))
            process=subprocess.Popen([str(sevenzip),'x','-so',str(archive),member],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            h=hashlib.sha256();byte_count=0
            for block in iter(lambda:process.stdout.read(1<<20),b''):
                h.update(block);byte_count+=len(block)
            error=process.stderr.read().decode(errors='replace');assert process.wait()==0,error
            local_sha=sha(path);assert h.hexdigest()==local_sha and byte_count==path.stat().st_size
            assert np.array_equal(source_reader(str(path)),raw.astype(np.float32)/np.float32(255))
            frame_metadata.append({'path':str(path),'sha256':local_sha,'archive_member':member,
                'archive_extracted_sha256':h.hexdigest(),'archive_bytes':byte_count,
                'shape':list(raw.shape),'dtype':str(raw.dtype),'raw_minimum':int(raw.min()),'raw_maximum':int(raw.max())})
        flattened=cached.permute(0,2,1,3,4).reshape(16,3,256,256)
        per_frame=torch.cat(list(cached.unbind(2)),dim=0)
        assert torch.equal(flattened,per_frame)
        diff=(cached-swapped).abs()
        records.append({'name':row['name'],'scene':row['scene'],'source_index':index,'augmentation_seed':seed,
            'frame_indices':indices,'shape':list(cached.shape),'dtype':str(cached.dtype),
            'producer_vs_cache_bitwise_equal':True,'training_loader_vs_cache_bitwise_equal':True,
            'hvi_export_input_vs_cache_bitwise_equal':True,'hvi_frame_flatten_vs_unbind_bitwise_equal':True,
            'canonical_lq_tensor_sha256':tensor_sha(cached),'official_rgb_tensor_sha256':tensor_sha(rgb),
            'official_rgb_equals_cached_channel_reversal':True,'official_rgb_equals_cached_without_swap':torch.equal(rgb,cached),
            'channel_reversal_max_abs_difference':float(diff.max()),'channel_reversal_mean_abs_difference':float(diff.mean()),
            'cached_minimum':float(cached.min()),'cached_maximum':float(cached.max()),
            'cached_channel_means':cached.double().mean((0,2,3,4)).tolist(),
            'archive_lq_files_exact_sha256_match':True,'frames':frame_metadata})
        common.read_image_01=source_reader
    assert len(records)==2 and len(set(arrays_read))==32
    report={'scope':'first fixed FIT scene only; two clips32 distinct original LQ frames; no model/GT/quality score',
        'selected_scene':metadata['split']['fit'][0],'records':records,'references':references,
        'gt_pixel_reads':0,'gt_placeholder_calls':len(gt_placeholders),'quality_scores_computed':False,
        'internal_holdout_pixel_reads':0,'target_pixel_reads':0,'new_model_predictions':0,
        'archive_byte_identity_checked':True,'original_png_counterpart_checked':False,
        'data_provenance':{'repository':provenance['repository'],'repository_commit':provenance['repository_commit'],
            'google_folder':receipt['google_folder'],'zip_crc_verified_at_download':receipt['zip_crc_verified'],
            'source_provenance_sha256':sha(provenance_root/'sdsd_static_provenance.json'),
            'download_receipt_sha256':sha(provenance_root/'download_receipt.json'),
            'archive_parts':[r for r in receipt['archives'] if r['name'].startswith('outdoor')],
            'sevenzip_path':str(sevenzip),'sevenzip_sha256':sha(sevenzip),
            'limitation':'local archive hashes, prior download receipt and exact32 member SHA; publisher cryptographic digest unavailable'},
        'source_sha256':{str(code/name):sha(code/name) for name in ('data/common.py','tools/cache_source_corrections.py','tools/seer_train_cvpr.py')},
        'hvi_exporter_sha256':sha(hvi_root/'release_v1/cache_hvi_source.py'),
        'hvi_manifest_sha256':sha(hvi_root/'cache_v1/manifest.json'),'geometry_manifest_sha256':sha(geometry),
        'audit_script_sha256':sha(__file__),'torch':str(torch.__version__),'numpy':str(np.__version__),
        'threads':4,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (args.out/'source_input_audit.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','data_provenance')},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('frames','frame_indices','cached_channel_means')} for r in records],indent=2))


if __name__=='__main__':main()
