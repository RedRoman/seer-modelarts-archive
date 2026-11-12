"""Independent FIT diagnostic arithmetic/coverage verification; no new scores."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

LAMBDAS=(.25,.5,1.)
METHODS=("hvi",*(f"temporal_mean_lambda_{v:g}" for v in LAMBDAS),
    *(f"sameframe_identity_lambda_{v:g}" for v in LAMBDAS))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def verify(root):
    import torch
    root=Path(root);torch.set_num_threads(4)
    execution=json.loads((root/"results_v1/execution_manifest.json").read_text())
    result=json.loads((root/"results_v1/fit_results.json").read_text())
    assert result["manifest_sha256"]==sha(root/"results_v1/execution_manifest.json")
    for name,value in execution["source_sha256"].items():assert sha(root/"release_v1"/name)==value
    lineage=execution["lineage"]
    assert sha(lineage["hvi_manifest_path"])==lineage["hvi_manifest_sha256"]
    assert sha(lineage["geometry_manifest_path"])==lineage["geometry_manifest_sha256"]
    geometry=json.loads(Path(lineage["geometry_manifest_path"]).read_text())
    hvi_path=Path(lineage["hvi_manifest_path"]).parent
    hvi=json.loads(Path(lineage["hvi_manifest_path"]).read_text())
    hvi_rows={r["name"]:r for r in hvi["per_sequence"]}
    split=geometry["split"];fit=set(split["fit"]);excluded=set(split["internal_holdout"])
    assert len(fit)==51 and len(excluded)==17 and not fit&excluded and execution["split"]==split
    expected=[r for r in geometry["per_sequence"] if r["scene"] in fit]
    raw=result["per_sequence"]
    assert len(expected)==len(raw)==result["n_clips"]==102
    assert result["n_scenes"]==51 and set(Counter(r["scene"] for r in expected).values())=={2}
    assert [r["name"] for r in expected]==[r["name"] for r in raw]==execution["fit_names"]
    assert all(r["scene"] in fit and r["scene"] not in excluded for r in raw)
    assert tuple(execution["fixed_methods"])==METHODS and tuple(execution["fixed_lambdas"])==LAMBDAS
    assert result["lambda_selection_performed"] is False and result["oracle_computed"] is False
    assert result["internal_holdout_scores_computed"] is False and result["target_scores_computed"] is False
    assert all(execution[k] is False for k in ("internal_holdout_labels_read","internal_holdout_scores_computed",
        "target_inputs_read","optimizer_or_fitting","prediction_or_gt_arrays_written"))
    comparisons=0;max_error=0.;coverage_error=0.;coverage_rows=[]
    def compare(a,b,tolerance=5e-12):
        nonlocal comparisons,max_error
        assert math.isfinite(float(a)) and math.isfinite(float(b))
        error=abs(float(a)-float(b));max_error=max(max_error,error);comparisons+=1
        assert error<=tolerance,(a,b,error)
    for original,scored in zip(expected,raw):
        assert scored["name"]==original["name"] and scored["scene"]==original["scene"]
        assert set(scored["methods"])==set(METHODS) and scored["sameframe_exact_anchor"] is True
        for method in METHODS:
            metric=scored["methods"][method];mse=metric["rgb_mse_float"]
            assert mse>0;compare(metric["psnr_rgb_from_clip_mse"],-10*math.log10(mse))
        for value in LAMBDAS:
            assert scored["methods"][f"sameframe_identity_lambda_{value:g}"]==scored["methods"]["hvi"]
        assert sha(original["source_path"])==original["source_sha256"]
        # Only FIT source flow/mask tensors are interpreted; GT is never indexed.
        source=torch.load(str(original["source_path"]),map_location="cpu",weights_only=True,mmap=True)
        assert source["name"]==original["name"] and source["scene"]==original["scene"]
        flow=source["tensors"]["flow"].numpy();mask=source["tensors"]["valid_mask"].numpy().astype(bool)
        assert flow.shape==(1,2,2,16,256,256) and mask.shape==(1,2,1,16,256,256)
        yy,xx=np.indices((256,256));times=np.arange(16)
        count=np.zeros((16,256,256),dtype=np.uint8)
        for j,delta in enumerate((-2,2)):
            x=xx[None]+flow[0,j,0];y=yy[None]+flow[0,j,1]
            valid=(mask[0,j,0] & (x>=0)&(x<=255)&(y>=0)&(y<=255)
                & ((times+delta>=0)&(times+delta<16))[:,None,None])
            count+=valid.astype(np.uint8)
        counts=[int(np.count_nonzero(count==n)) for n in (0,1,2)]
        assert sum(counts)==count.size
        for n,key in enumerate(("no_donor_fraction","one_donor_fraction","two_donor_fraction")):
            error=abs(counts[n]/count.size-scored["coverage"][key]);coverage_error=max(coverage_error,error)
            assert error<=5e-8
        a=hvi_rows[original["name"]];p=hvi_path/a["file"]
        assert sha(p)==a["file_sha256"]
        payload=torch.load(str(p),map_location="cpu",weights_only=True,mmap=True)
        assert payload["scene"] in fit and payload["name"]==original["name"]
        base=payload["base"].numpy()
        assert hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()==a["base_tensor_sha256"]
        duplicate=(base+base)*np.float32(.5)
        assert np.array_equal(duplicate.view(np.uint32),base.view(np.uint32))
        for value in LAMBDAS:
            pred=base+np.float32(value)*(duplicate-base)
            assert np.array_equal(pred.view(np.uint32),base.view(np.uint32))
        coverage_rows.append({"name":original["name"],"scene":original["scene"],
            "valid_neighbor_pixel_counts":counts,"total_frame_pixels":int(count.size),
            "sameframe_numpy_bitwise_identity":True})
    means={}
    for method in METHODS:
        means[method]={"mean_rgb_mse":math.fsum(r["methods"][method]["rgb_mse_float"] for r in raw)/102,
            "mean_clip_psnr_rgb":math.fsum(r["methods"][method]["psnr_rgb_from_clip_mse"] for r in raw)/102}
        for key,value in means[method].items():
            compare(value,result["mean"][method][key])
            compare(value-means["hvi"][key],result["mean_difference_vs_hvi"][method][key])
    scene_rows={r["scene"]:r for r in result["per_scene"]}
    assert set(scene_rows)==fit
    for scene in sorted(fit):
        rows=[r for r in raw if r["scene"]==scene];saved=scene_rows[scene]
        assert len(rows)==saved["clips"]==2
        values={m:{k:math.fsum(r["methods"][m][k] for r in rows)/2
                   for k in ("rgb_mse_float","psnr_rgb_from_clip_mse")} for m in METHODS}
        for method in METHODS:
            for key,value in values[method].items():
                compare(value,saved["methods"][method][key])
                compare(value-values["hvi"][key],saved["difference_vs_hvi"][method][key])
    return {"passed":True,"fit_clips":102,"fit_scenes":51,"methods":list(METHODS),
        "scalar_checks":comparisons,"max_scalar_abs_error":max_error,"max_coverage_fraction_abs_error":coverage_error,
        "sameframe_independently_bitwise_identical":True,"coverage_and_identity":coverage_rows,
        "mean_recomputed":means,"result_sha256":sha(root/"results_v1/fit_results.json"),
        "execution_manifest_sha256":sha(root/"results_v1/execution_manifest.json"),"verifier_sha256":sha(__file__),
        "independent_method":"math.fsum/log10 arithmetic; NumPy integer validity counts and bitwise identity; no predicted temporal frames recomputed",
        "gt_access_by_verifier":False,"internal_holdout_tensor_loads":0,"target_tensor_loads":0}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path);args=p.parse_args()
    report=verify(args.root)
    target=args.root/"checks_v1/independent_verification.json"
    with target.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k!="coverage_and_identity"},indent=2))
