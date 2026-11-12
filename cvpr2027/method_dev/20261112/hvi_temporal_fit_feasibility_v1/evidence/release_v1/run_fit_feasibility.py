"""Fixed HVI donor-mean diagnostic on the existing51 FIT scenes only.

All prediction helpers are GT-free. Label access is guarded by FIT membership
before loading either source or HVI tensor archives. No model or target loader
is imported, and only JSON scalar scores/provenance are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

SOURCE_PROVENANCE_SHA="0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0"
HVI_WEIGHT_SHA="6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27"
HVI_STATE_SHA="afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b"
LAMBDAS=(.25,.5,1.)
METHODS=("hvi",*(f"temporal_mean_lambda_{v:g}" for v in LAMBDAS),
         *(f"sameframe_identity_lambda_{v:g}" for v in LAMBDAS))


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_new(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def selected_fit_rows(rows,split):
    """Metadata-only selection; never load any tensor before this filter."""
    fit=set(split["fit"]);excluded=set(split["internal_holdout"])
    require(not fit&excluded,"FIT and excluded development scenes overlap")
    return [row for row in rows if row["scene"] in fit]


def bound_inputs(source_cache,hvi_cache,geometry_manifest):
    """Verify complete metadata/file identities without interpreting pixels."""
    source_cache,hvi_cache=Path(source_cache),Path(hvi_cache)
    require(digest(source_cache/"provenance.json")==SOURCE_PROVENANCE_SHA,"Wrong original source provenance")
    origin=json.loads((source_cache/"provenance.json").read_text())
    summary=json.loads((source_cache/"cache_summary.json").read_text())
    hvi=json.loads((hvi_cache/"manifest.json").read_text())
    geometry=json.loads(Path(geometry_manifest).read_text())
    require(geometry["source_cache_provenance"]==origin and geometry["n"]==136,
            "Frozen geometry does not bind this original source")
    require(hvi["n"]==136 and hvi["scenes"]==68 and hvi["model_state_unchanged"] is True
        and hvi["weight_sha256"]==HVI_WEIGHT_SHA and hvi["model_state_sha256"]==HVI_STATE_SHA
        and hvi["source_provenance_sha256"]==SOURCE_PROVENANCE_SHA
        and hvi["source_summary_sha256"]==digest(source_cache/"cache_summary.json")
        and hvi["gt_storage_entries_read"]==0 and hvi["target_inputs_read"]==0
        and hvi["quality_scores_computed"] is False,"Require completed unchanged GT-free HVI136 export")
    sr,hr,gr=summary["per_sequence"],hvi["per_sequence"],geometry["per_sequence"]
    names=origin["source_names"]
    require(len(sr)==len(hr)==len(gr)==len(names)==len(set(names))==136,"Incomplete source/HVI coverage")
    require([r["name"] for r in sr]==[r["name"] for r in hr]==[r["name"] for r in gr]==names,
            "Source/HVI sequence order differs")
    entries=[]
    for i,(s,a,g) in enumerate(zip(sr,hr,gr)):
        require(s["scene"]==a["scene"]==g["scene"] and s["cache_path"]==f"clip_{i:03d}.pt"
            and a["file"]==f"hvi_{i:03d}.pt" and a["source_cache_sha256"]==s["cache_sha256"]==g["source_sha256"]
            and a["shape"]==[1,3,16,256,256] and a["dtype"]=="float32" and a["finite_range_verified"] is True,
            "Source/HVI per-clip lineage differs")
        sp,hp=source_cache/s["cache_path"],hvi_cache/a["file"]
        require(sp.resolve()==Path(g["source_path"]).resolve() and digest(sp)==s["cache_sha256"]
                and digest(hp)==a["file_sha256"],"Immutable source/HVI file hash differs")
        entries.append({"name":s["name"],"scene":s["scene"],"source_path":str(sp.resolve()),
            "source_sha256":s["cache_sha256"],"hvi_path":str(hp.resolve()),"hvi_sha256":a["file_sha256"],
            "lq_tensor_sha256":a["lq_tensor_sha256"],"base_tensor_sha256":a["base_tensor_sha256"]})
    split=geometry["split"]
    require(split["seed"]==20261114 and len(split["fit"])==51 and len(split["internal_holdout"])==17,
            "Require the fixed51/17 source scene split")
    scenes=sorted(set(row["scene"] for row in entries))
    order=np.random.default_rng(20261114).permutation(len(scenes))
    require(split["fit"]==[scenes[int(i)] for i in order[:51]]
        and split["internal_holdout"]==[scenes[int(i)] for i in order[51:]],"Source scene split changed")
    fit=selected_fit_rows(entries,split)
    require(len(fit)==102 and set(Counter(r["scene"] for r in fit).values())=={2},"Require102 FIT clips/two per scene")
    return entries,split,{"source_provenance_sha256":SOURCE_PROVENANCE_SHA,
        "source_summary_sha256":digest(source_cache/"cache_summary.json"),
        "hvi_manifest_path":str((hvi_cache/"manifest.json").resolve()),"hvi_manifest_sha256":digest(hvi_cache/"manifest.json"),
        "geometry_manifest_path":str(Path(geometry_manifest).resolve()),"geometry_manifest_sha256":digest(geometry_manifest),
        "hvi_weight_sha256":HVI_WEIGHT_SHA,"hvi_state_sha256":HVI_STATE_SHA,
        "hvi_release_manifest_sha256":hvi["release_manifest_sha256"],"source_raft_state_sha256":origin["raft_state_sha256"],
        "flow_provenance":"frozen Seer restored-RGB RAFT±2 proposals; no GT flow"}


def load_fit_sample(row,fit_scenes):
    """Production FIT guard precedes all tensor loading and GT access."""
    import torch
    require(row["scene"] in set(fit_scenes),"Refusing to load any excluded scene tensors/labels")
    require(digest(row["source_path"])==row["source_sha256"] and digest(row["hvi_path"])==row["hvi_sha256"],
            "Source or HVI file changed after manifest validation")
    source=torch.load(str(row["source_path"]),map_location="cpu",weights_only=True,mmap=True)
    hvi=torch.load(str(row["hvi_path"]),map_location="cpu",weights_only=True,mmap=True)
    require(set(hvi)=={"name","scene","base","lineage"} and all(
        x["name"]==row["name"] and x["scene"]==row["scene"] for x in (source,hvi)),"Payload identity differs")
    t=source["tensors"];lineage=hvi["lineage"]
    require(lineage["source_cache_sha256"]==row["source_sha256"]
        and lineage["source_provenance_sha256"]==SOURCE_PROVENANCE_SHA
        and lineage["weight_sha256"]==HVI_WEIGHT_SHA and lineage["model_state_sha256"]==HVI_STATE_SHA
        and lineage["input_reencoded_to_uint8"] is False
        and tensor_digest(t["lq"])==row["lq_tensor_sha256"]==lineage["lq_tensor_sha256"]
        and tensor_digest(hvi["base"])==row["base_tensor_sha256"]==lineage["base_tensor_sha256"],
        "HVI observable input/output lineage differs")
    base,gt=hvi["base"],t["gt"]
    require(base.shape==gt.shape==(1,3,16,256,256) and base.dtype==gt.dtype==torch.float32
        and all(bool(torch.isfinite(x).all() & (x>=0).all() & (x<=1).all()) for x in (base,gt)),
        "Invalid FIT RGB/GT tensors")
    return (base,t["flow"],t["valid_mask"]),gt


def aligned_hvi_mean(base,flows,masks,offset=2):
    """GT-free actual HVI RGB gathering with conservative observed validity."""
    import torch
    from torch.nn import functional as F
    require(base.ndim==5 and base.shape[1]==3 and base.dtype==torch.float32,"Require float32 BCTHW HVI")
    b,_,t,h,w=base.shape
    require(t>offset and flows.shape==(b,2,2,t,h,w) and masks.shape==(b,2,1,t,h,w),"RGB/flow/mask shapes disagree")
    require(flows.dtype==torch.float32 and all(x.device.type=="cpu" for x in (base,flows,masks)),"CPU float32 flow required")
    require(all(bool(torch.isfinite(x).all()) for x in (base,flows,masks))
        and bool(((masks==0)|(masks==1)).all()),"Finite RGB/flow and binary mask required")
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij")
    times=torch.arange(t);values=[];validity=[]
    for j,delta in enumerate((-offset,offset)):
        st=times+delta
        x=xx[None,None]+flows[:,j,0];y=yy[None,None]+flows[:,j,1]
        valid=((x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)&((st>=0)&(st<t))[None,:,None,None])[:,None]&masks[:,j].bool()
        grid=torch.stack((2*(x+.5)/w-1,2*(y+.5)/h-1),-1).reshape(b*t,h,w,2)
        source=base.index_select(2,st.clamp(0,t-1)).permute(0,2,1,3,4).reshape(b*t,3,h,w)
        warped=F.grid_sample(source,grid,mode="bilinear",padding_mode="border",align_corners=False)
        values.append(warped.reshape(b,t,3,h,w).permute(0,2,1,3,4));validity.append(valid)
    support=torch.stack(validity,1);count=support.sum(1)
    mean=sum(value*valid.float() for value,valid in zip(values,validity))/count.clamp_min(1)
    correction=(mean-base)*(count>0)
    return correction,count


def fixed_predictions(base,flows,masks):
    """Exactly the seven prespecified predictions, with no labels accepted."""
    import torch
    correction,count=aligned_hvi_mean(base,flows,masks)
    result={"hvi":base}
    for value in LAMBDAS:
        result[f"temporal_mean_lambda_{value:g}"]=(base+value*correction).clamp(0,1)
        # Arithmetic identity only. Center copies are deliberately not warped.
        duplicate_mean=(base+base)*.5
        result[f"sameframe_identity_lambda_{value:g}"]=base+value*(duplicate_mean-base)
        require(torch.equal(result[f"sameframe_identity_lambda_{value:g}"],base),"Sameframe arithmetic identity failed")
    require(set(result)==set(METHODS),"Unexpected diagnostic method set")
    coverage={"no_donor_fraction":float((count==0).float().mean()),
        "one_donor_fraction":float((count==1).float().mean()),"two_donor_fraction":float((count==2).float().mean())}
    return result,coverage


def score_predictions(predictions,fit_labels):
    import torch
    scores={}
    for name in METHODS:
        value=predictions[name]
        require(value.shape==fit_labels.shape and value.dtype==fit_labels.dtype==torch.float32,"Score RGB/label shape differs")
        mse=float((value-fit_labels).square().double().mean())
        require(np.isfinite(mse) and mse>0,"Require finite positive full-RGB MSE")
        scores[name]={"rgb_mse_float":mse,"psnr_rgb_from_clip_mse":float(-10*np.log10(mse))}
    return scores


def evaluate_fit_rows(rows,split,loader):
    """Filter rows before invoking the production or synthetic tensor loader."""
    results=[]
    for row in selected_fit_rows(rows,split):
        inputs,labels=loader(row,split["fit"])
        predictions,coverage=fixed_predictions(*inputs)
        results.append({"name":row["name"],"scene":row["scene"],
            "methods":score_predictions(predictions,labels),"coverage":coverage,"sameframe_exact_anchor":True})
    return results


def aggregate(rows):
    scenes=sorted({r["scene"] for r in rows});per_scene=[]
    for scene in scenes:
        clips=[r for r in rows if r["scene"]==scene]
        methods={m:{k:float(np.mean([r["methods"][m][k] for r in clips]))
                    for k in ("rgb_mse_float","psnr_rgb_from_clip_mse")} for m in METHODS}
        differences={m:{k:methods[m][k]-methods["hvi"][k] for k in methods[m]} for m in METHODS}
        per_scene.append({"scene":scene,"clips":len(clips),"methods":methods,"difference_vs_hvi":differences})
    means={m:{"mean_rgb_mse":float(np.mean([r["methods"][m]["rgb_mse_float"] for r in rows])),
        "mean_clip_psnr_rgb":float(np.mean([r["methods"][m]["psnr_rgb_from_clip_mse"] for r in rows]))} for m in METHODS}
    return {"n_clips":len(rows),"n_scenes":len(scenes),"mean":means,"per_scene":per_scene,
        "mean_difference_vs_hvi":{m:{k:means[m][k]-means["hvi"][k] for k in means[m]} for m in METHODS},
        "lambda_selection_performed":False,"oracle_computed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("source-cache","hvi-cache","geometry-manifest","out"):
        parser.add_argument("--"+key,required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),"Preserve existing diagnostic output")
    rows,split,lineage=bound_inputs(args.source_cache,args.hvi_cache,args.geometry_manifest)
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(4)
    phase=Path(__file__).resolve().parent
    code_sha={name:digest(phase/name) for name in ("PLAN.md","run_fit_feasibility.py","test_fit_feasibility.py")}
    manifest={"schema_version":1,"scope":"SDSD source FIT51/102 only; diagnostic, not learned or generalization performance",
        "fixed_methods":list(METHODS),"fixed_lambdas":list(LAMBDAS),"split":split,"lineage":lineage,
        "fit_names":[r["name"] for r in selected_fit_rows(rows,split)],"source_sha256":code_sha,
        "prediction_helpers_have_no_gt_parameter":not ({"gt","labels","fit_labels"}&set(inspect.signature(fixed_predictions).parameters)),
        "internal_holdout_labels_read":False,"internal_holdout_scores_computed":False,"target_inputs_read":False,
        "optimizer_or_fitting":False,"prediction_or_gt_arrays_written":False,"threads":4,
        "runtime":{"torch":str(torch.__version__),"numpy":str(np.__version__),"device":"cpu float32"},
        "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(args.out/"execution_manifest.json",manifest)
    began=time.monotonic()
    with torch.no_grad():
        scores=evaluate_fit_rows(rows,split,load_fit_sample)
    require(len(scores)==102 and len({r["scene"] for r in scores})==51 and all(
        r["scene"] in split["fit"] and r["sameframe_exact_anchor"] for r in scores),"Incomplete or excluded source coverage")
    report={"schema_version":1,"scope":manifest["scope"],"manifest_sha256":digest(args.out/"execution_manifest.json"),
        "metric":"float32 squared RGB residual; float64 full16x256 mean; PSNR from each clip MSE",
        "per_sequence":scores,**aggregate(scores),"elapsed_seconds":time.monotonic()-began,
        "internal_holdout_scores_computed":False,"target_scores_computed":False,
        "interpretation":"fixed-lambda FIT feasibility only; no lambda/checkpoint selection or module/SOTA claim"}
    require(all(digest(phase/name)==sha for name,sha in code_sha.items()),"Diagnostic source changed during scoring")
    write_new(args.out/"fit_results.json",report)
    print(json.dumps({"completed_fit_clips":102,"fit_scenes":51,"results_sha256":digest(args.out/"fit_results.json")}),flush=True)


if __name__=="__main__":main()
