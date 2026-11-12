"""Fixed-budget source-only explicit joint-observation search screen."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/"tools"))
from train_correction_selector import digest,local_module,state_digest,write_json
from cache_source_corrections import source_hashes,check_source_hashes
from train_observation_memory import make_plan,new_directory
from train_observation_geometry import loss_view,reference_checkpoint,old_prediction,old_reference
from cache_joint_observation_search import checked_metadata

MODES={
    "joint_prior":("joint","prior","calibrated"),
    "independent_prior":("independent","prior","calibrated"),
    "constant_prior":("joint","prior","constant"),
    "joint_pixel":("joint","pixel","calibrated"),
    "joint_shuffled":("joint","shuffled_prior","calibrated"),
}
STEPS,LR=400,0.01
EXTRA=("models/joint_observation_search.py","tools/cache_joint_observation_search.py",
       "tools/train_joint_observation_search.py","tools/train_observation_geometry.py",
       "tools/train_observation_memory.py","tools/train_source_selector.py",
       "tools/cache_observation_memory.py","models/observation_geometry.py",
       "models/observation_candidates.py","models/correction_selector.py",
       "models/observation_memory.py","models/trajectory_reconstruction.py")


def search_module():
    return local_module("_joint_search_screen",CODE/"models/joint_observation_search.py")


def load_sample(prepared,row):
    import torch
    source=torch.load(str(row["source_path"]),map_location="cpu",weights_only=True,mmap=True)
    cached=torch.load(str(Path(prepared)/row["cache_path"]),map_location="cpu",weights_only=True,mmap=True)
    if (any(v["name"]!=row["name"] or v["scene"]!=row["scene"] for v in (source,cached))
            or cached["source_sha256"]!=row["source_sha256"]
            or cached["address_sha256"]!=row["address_sha256"]):
        raise ValueError("Source/cost payload identity or lineage differs")
    t=source["tensors"]
    # GT belongs only in the returned loss labels; costs were computed without it.
    return tuple(t[k] for k in ("lq","base","flow","valid_mask"))+(cached["costs"],),t["gt"]


def construct(meta,reference,mode):
    search,address,uncertainty=MODES[mode]
    return search_module().JointObservationSearch(meta["normalization"],reference["model"],
        search_mode=search,address_mode=address,uncertainty_mode=uncertainty)


def preflight(prepared,out):
    import torch
    torch.set_num_threads(4)
    if Path(out).exists():raise FileExistsError("Preserve previous preflight")
    meta=checked_metadata(prepared)
    reference=reference_checkpoint(meta["reference_head_path"])
    row=next(r for r in meta["per_sequence"] if r["scene"] in meta["split"]["fit"])
    inputs,_=load_sample(prepared,row)
    states,parity,configs=set(),{},{}
    with torch.no_grad():
        anchor,_=old_prediction(row,meta["normalization"],old_reference(reference))
        for mode in MODES:
            torch.manual_seed(0)
            model=construct(meta,reference,mode).eval()
            states.add(state_digest(model.state_dict()))
            result=model(*inputs)
            parity[mode]=float((result["pred_pixels"]-anchor).abs().max())
            if parity[mode]>1e-6 or bool(result["offsets"].any()):
                raise ValueError("Initial zero-search anchor parity failed")
            configs[mode]=model.get_config()
    if len(states)!=1:raise ValueError("Mode initial tensors differ")
    plan=make_plan([r["name"] for r in meta["per_sequence"]],meta["split"],0)
    if plan["steps"]!=STEPS:raise ValueError("Plan budget changed")
    write_json(out,{"passed":True,"fit_clip":row["name"],"prediction_max_abs":parity,
        "initial_states_identical":True,"initial_state_sha256":next(iter(states)),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters":sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "model_configs":configs,"source_sha256":source_hashes(*EXTRA),
        "prepared_manifest_sha256":digest(Path(prepared)/"manifest.json"),
        "holdout_metric_computed":False,"target_frames_read":False})
    print(json.dumps({"fit_zero_offset_parity":parity,"identical_initial_state":True}),flush=True)


def train(prepared,out,mode,seed):
    import torch
    torch.set_num_threads(4)
    meta=checked_metadata(prepared)
    out=new_directory(out)
    plan=make_plan([r["name"] for r in meta["per_sequence"]],meta["split"],seed)
    if plan["steps"]!=STEPS:raise ValueError("Plan budget changed")
    write_json(out/"sample_plan.json",plan)
    reference=reference_checkpoint(meta["reference_head_path"])
    torch.manual_seed(seed)
    model=construct(meta,reference,mode).train()
    initial=copy.deepcopy(model.state_dict())
    frozen_before=state_digest(model.node.state_dict())
    torch.save({"model":initial},out/"initial_state.pt")
    parameters=[p for p in model.parameters() if p.requires_grad]
    if len(parameters)!=4 or sum(p.numel() for p in parameters)!=4:
        raise ValueError("This screen permits exactly four trainable scalars")
    provenance={"schema_version":1,"mode":mode,"seed":seed,"steps":STEPS,
        "model_config":model.get_config(),"initial_state_sha256":state_digest(initial),
        "initial_file_sha256":digest(out/"initial_state.pt"),"sample_plan_sha256":digest(out/"sample_plan.json"),
        "prepared_manifest_sha256":digest(Path(prepared)/"manifest.json"),"source_sha256":source_hashes(*EXTRA),
        "split":meta["split"],"calibration_sha256":meta["calibration_sha256"],
        "reference_head_sha256":meta["reference_head_sha256"],"frozen_node_state_sha256":frozen_before,
        "fit_only_mse_normalizer":meta["fit_only_mse_normalizer"],
        "args":{"optimizer":"Adam defaults;no weight decay;post-step declared parameter projection",
                "lr":LR,"threads":4,"device":"cpu float32","forward_context":256,
                "loss_crop":128,"clip_len":16,"loss_frames":8,"clips_per_update":1},
        "trainable_parameters":4,"frozen_parameters":sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "selection":"Final400 only; all five modes before one complete source internal evaluation; no target",
        "runtime":{"torch":str(torch.__version__),"numpy":str(np.__version__)}}
    write_json(out/"provenance.json",provenance)
    optimizer=torch.optim.Adam(parameters,lr=LR)
    start=time.monotonic()
    with (out/"curve.jsonl").open("w") as stream:
        for step,spec in enumerate(plan["batches"],1):
            inputs,gt=load_sample(prepared,meta["per_sequence"][spec["clip_index"]])
            optimizer.zero_grad(set_to_none=True)
            result=model(*inputs)
            mse=(loss_view(result["pred_pixels"],spec)-loss_view(gt,spec)).square().mean()
            loss=mse/meta["fit_only_mse_normalizer"]
            if not bool(torch.isfinite(loss)):raise FloatingPointError("Nonfinite search loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Missing/nonfinite scalar search gradient")
            if any(p.grad is not None for p in model.node.parameters()):
                raise RuntimeError("Frozen selector received parameter gradients")
            optimizer.step();model.project_parameters_()
            entry={"step":step,"rgb_mse":float(mse.detach()),"loss":float(loss.detach()),
                "clip_index":spec["clip_index"],"elapsed_seconds":time.monotonic()-start,
                "prediction_timing":"before optimizer update",**result["aux"]}
            stream.write(json.dumps(entry,allow_nan=False)+"\n")
            if step==1 or step%25==0:
                stream.flush();print(json.dumps(entry),flush=True)
    if state_digest(model.node.state_dict())!=frozen_before:
        raise RuntimeError("Frozen selector changed")
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({"model":state,"model_state_sha256":state_digest(state),"steps":STEPS,
        "provenance":provenance,"optimizer":optimizer.state_dict()},out/"final.pt")
    write_json(out/"complete.json",{"steps":STEPS,"elapsed_seconds":time.monotonic()-start,
        "checkpoint_sha256":digest(out/"final.pt"),"model_state_sha256":state_digest(state),
        "frozen_node_unchanged":True,"frozen_node_state_sha256":frozen_before,
        "sample_plan_sha256":provenance["sample_plan_sha256"],"initial_state_sha256":provenance["initial_state_sha256"]})


def evaluate(prepared,runs,out):
    import torch
    torch.set_num_threads(4)
    if Path(out).exists():raise FileExistsError("Preserve previous source results")
    meta=checked_metadata(prepared)
    reference_ck=reference_checkpoint(meta["reference_head_path"])
    expected_plan=make_plan([r["name"] for r in meta["per_sequence"]],meta["split"],0)
    models,checkpoints,provenance,inits,plans={},{},{},set(),set()
    for mode,(search,address,uncertainty) in MODES.items():
        folder=Path(runs)/f"{mode}_s0"
        complete=json.loads((folder/"complete.json").read_text())
        ck=torch.load(str(folder/"final.pt"),map_location="cpu",weights_only=True)
        p=ck["provenance"]
        init=torch.load(str(folder/"initial_state.pt"),map_location="cpu",weights_only=True)["model"]
        torch.manual_seed(0)
        expected_model=construct(meta,reference_ck,mode).eval()
        if (ck["steps"]!=STEPS or complete["steps"]!=STEPS or p["mode"]!=mode or p["seed"]!=0
                or complete["frozen_node_unchanged"] is not True or p["trainable_parameters"]!=4
                or p["source_sha256"]!=source_hashes(*EXTRA)
                or p["prepared_manifest_sha256"]!=digest(Path(prepared)/"manifest.json")
                or p["calibration_sha256"]!=meta["calibration_sha256"]
                or p["reference_head_sha256"]!=meta["reference_head_sha256"]
                or complete["checkpoint_sha256"]!=digest(folder/"final.pt")
                or p["sample_plan_sha256"]!=digest(folder/"sample_plan.json")
                or p["initial_file_sha256"]!=digest(folder/"initial_state.pt")
                or state_digest(init)!=p["initial_state_sha256"]
                or state_digest(init)!=state_digest(expected_model.state_dict())
                or p["model_config"]!=expected_model.get_config()
                or json.loads((folder/"sample_plan.json").read_text())!=expected_plan
                or any(p["model_config"][k]!=v for k,v in (("search_mode",search),
                    ("address_mode",address),("uncertainty_mode",uncertainty)))):
            raise ValueError("Final search checkpoint lineage/configuration differs")
        model=search_module().JointObservationSearch(frozen_node_state=reference_ck["model"],**p["model_config"]).eval()
        model.load_state_dict(ck["model"],strict=True)
        if (state_digest(model.state_dict())!=ck["model_state_sha256"]
                or state_digest(model.node.state_dict())!=p["frozen_node_state_sha256"]):
            raise ValueError("Search or frozen selector state digest differs")
        if not 0<=float(model.amplitude_raw)<=4:raise ValueError("Unprojected amplitude")
        models[mode]=model;checkpoints[mode]=digest(folder/"final.pt");provenance[mode]=p
        inits.add(p["initial_state_sha256"]);plans.add(p["sample_plan_sha256"])
    if len(inits)!=1 or len(plans)!=1:raise ValueError("Initial tensors/training plans differ")
    reference=old_reference(reference_ck)
    before={m:state_digest(v.state_dict()) for m,v in models.items()}
    scores=[]
    with torch.no_grad():
        for row in meta["per_sequence"]:
            if row["scene"] not in meta["split"]["internal_holdout"]:continue
            inputs,gt=load_sample(prepared,row)
            anchor,old=old_prediction(row,meta["normalization"],reference)
            if not torch.equal(old["base"],inputs[1]) or not torch.equal(old["gt"],gt):
                raise ValueError("Reference base/labels differ")
            results={m:v(*inputs) for m,v in models.items()}
            predictions={"base":inputs[1],"frozen_raw_node_1000":anchor,
                         **{m:v["pred_pixels"] for m,v in results.items()}}
            values={}
            for name,value in predictions.items():
                mse=float((value-gt).square().double().mean())
                if not np.isfinite(mse) or mse<=0:raise ValueError("Invalid full-source MSE")
                values[name]={"rgb_mse_float":mse,"psnr_rgb_from_clip_mse":float(-10*np.log10(mse))}
            scores.append({"name":row["name"],"scene":row["scene"],"methods":values,
                "search":{m:r["aux"] for m,r in results.items()}})
            print(json.dumps({"evaluated_source_holdout":len(scores),"name":row["name"]}),flush=True)
    unchanged=(all(state_digest(v.state_dict())==before[m] for m,v in models.items())
               and state_digest(reference.state_dict())==reference_ck["model_state_sha256"])
    expected={r["name"] for r in meta["per_sequence"] if r["scene"] in meta["split"]["internal_holdout"]}
    if not unchanged or len(scores)!=34 or {r["name"] for r in scores}!=expected:
        raise ValueError("Incomplete source coverage or mutated state")
    write_json(out,{"schema_version":1,"scope":"source TRAIN module development; historical base saw all scenes; no target",
        "metric":"float32 RGB squared residual, float64 reduction over all16/full256; PSNR from clip MSE, equal scene means",
        "selection":"All five final400 scalar searches, one source development evaluation",
        "prepared_manifest_sha256":digest(Path(prepared)/"manifest.json"),"source_sha256":source_hashes(*EXTRA),
        "checkpoint_sha256":checkpoints,"provenance":provenance,
        "calibration_sha256":meta["calibration_sha256"],"reference_checkpoint_sha256":meta["reference_head_sha256"],
        "reference_state_sha256":reference_ck["model_state_sha256"],"states_unchanged":unchanged,
        "initial_states_and_plans_identical":True,"target_frames_read":False,"per_sequence":scores})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    for action in ("preflight","evaluate"):
        q=sub.add_parser(action);q.add_argument("--prepared",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
        if action=="evaluate":q.add_argument("--runs",type=Path,required=True)
    q=sub.add_parser("train");q.add_argument("--prepared",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.add_argument("--mode",choices=tuple(MODES),required=True);q.add_argument("--seed",type=int,choices=(0,),default=0)
    a=p.parse_args()
    if a.action=="preflight":preflight(a.prepared,a.out)
    elif a.action=="train":train(a.prepared,a.out,a.mode,a.seed)
    else:evaluate(a.prepared,a.runs,a.out)


if __name__=="__main__":main()
