"""Fixed source-only screen of bounded coordinate refinement, with frozen RGB selector.

Reuses immutable source RGB/flow and descriptor caches. The complete256 video
is sampled before the loss crop; GT is passed only to loss/metric code.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "tools"))
from train_correction_selector import digest, local_module, state_digest, write_json
from train_source_selector import cache_manifest, make_scene_split
from cache_source_corrections import source_hashes, check_source_hashes, source_scene
from train_observation_memory import checked_metadata as memory_metadata, make_plan, new_directory

MODES = ("prior", "pixel", "shuffled_prior")
STEPS, LR, HIDDEN = 400, 1e-3, 16
EXTRA = ("models/observation_geometry.py", "models/observation_candidates.py",
         "models/correction_selector.py", "models/trajectory_reconstruction.py",
         "tools/train_observation_geometry.py", "tools/train_observation_memory.py",
         "tools/train_source_selector.py", "models/observation_memory.py",
         "tools/cache_observation_memory.py")


def geometry_module():
    return local_module("_geometry_screen", CODE / "models/observation_geometry.py")


def reference_checkpoint(path):
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    p = ck["provenance"]
    check_source_hashes(p["source_sha256"])
    if (ck["steps"] != 1000 or p["variant"] != "raw_node" or p["seed"] != 0
            or state_digest(ck["model"]) != ck["model_state_sha256"]):
        raise ValueError("Require the unchanged frozen raw_node1000 seed0 state")
    return ck


def load_sample(row):
    import torch
    raw = torch.load(str(row["source_path"]), map_location="cpu", weights_only=True, mmap=True)
    mem = torch.load(str(row["address_path"]), map_location="cpu", weights_only=True, mmap=True)
    if any(s["name"] != row["name"] or s["scene"] != row["scene"] for s in (raw, mem)):
        raise ValueError("Source or descriptor payload identity changed")
    t = raw["tensors"]
    # Deliberate firewall: no GT enters the module argument tuple.
    inputs = tuple(t[k] for k in ("lq", "base", "flow", "valid_mask")) + (
        mem["prior_address"], mem["pixel_address"])
    return inputs, t["gt"]


def loss_view(video, spec):
    import torch
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1:] != (3,16,256,256) or video.shape[0] < 1:
        raise ValueError("Loss view requires complete [B,3,16,256,256] predictions/GT")
    top, left = spec["top"], spec["left"]
    frames = spec["frames"]
    if (any(type(v) is not int or v % 16 or not 0 <= v <= 128 for v in (top, left))
            or len(frames) != 8 or len(set(frames)) != 8
            or any(type(v) is not int or not 0 <= v < 16 for v in frames)):
        raise ValueError("Require aligned128 loss crop and eight distinct frames")
    return video[:, :, frames, top:top+128, left:left+128]


def old_prediction(row, norm, reference):
    import torch
    old = torch.load(str(row["reference_path"]), map_location="cpu", weights_only=True, mmap=True)
    x = ((old["features"]["raw_detail"] - torch.tensor(norm["feature_mean"])[None,None,:,None,None])
         / torch.tensor(norm["feature_std"])[None,None,:,None,None])
    result = reference(old["base"], old["corrections"]["raw_detail"], x,
        old["detail_observations"], old["adjacent_flow"], old["adjacent_mask"])["pred_pixels"]
    return result, old


def old_reference(ck):
    model = local_module("_geometry_old_reference", CODE / "models/trajectory_reconstruction.py").TrajectoryReconstruction(mode="node_only").eval()
    model.load_state_dict(ck["model"], strict=True)
    return model


def prepare(cache, memory, splits, reference_prepared, reference_head, out):
    import torch
    torch.set_num_threads(4)
    files, origin, rows = cache_manifest(cache, splits)
    if origin["base_saved_args"]["tag"] != "e17_small_pretrained_s0":
        raise ValueError("Use the fixed pretrained E17 seed0 base")
    mm = memory_metadata(memory)
    rp = Path(reference_prepared)
    rm = json.loads((rp / "prepared_manifest.json").read_text())
    check_source_hashes(rm["source_sha256"])
    rc = reference_checkpoint(reference_head)
    split = make_scene_split(origin["source_names"])
    if (mm["source_cache_provenance"] != origin or rm["source_cache_provenance"] != origin
            or mm["split"] != split or rm["split"] != split
            or rc["provenance"]["prepared_manifest_sha256"] != digest(rp/"prepared_manifest.json")
            or rc["provenance"]["normalization"] != rm["normalization"]["raw_detail"]
            or rc["provenance"]["base_checkpoint_sha256"] != origin["base_checkpoint_sha256"]
            or len(rm["per_sequence"]) != 136):
        raise ValueError("Frozen source/address/reference lineages differ")
    entries = []
    for i, (path, row, mr, rr) in enumerate(zip(files, rows, mm["per_sequence"], rm["per_sequence"])):
        if (any(r["name"] != row["name"] or r["scene"] != row["scene"] for r in (mr, rr))
                or mr["source_cache_sha256"] != row["cache_sha256"]
                or digest(rp/rr["cache_path"]) != rr["cache_sha256"]):
            raise ValueError("Per-clip immutable cache binding differs")
        entries.append({"name":row["name"], "scene":row["scene"],
            "source_path":str(path.resolve()), "source_sha256":row["cache_sha256"],
            "address_path":str((Path(memory)/mr["cache_path"]).resolve()), "address_sha256":mr["cache_sha256"],
            "reference_path":str((rp/rr["cache_path"]).resolve()), "reference_sha256":rr["cache_sha256"]})
    # One FIT clip only: implementation parity, no source-holdout metric selection.
    row = next(r for r in entries if r["scene"] in split["fit"])
    inputs, _ = load_sample(row)
    torch.manual_seed(0)
    model = geometry_module().ObservationGeometry(rm["normalization"]["raw_detail"], rc["model"], hidden_dim=HIDDEN).eval()
    with torch.no_grad():
        new = model(*inputs, return_aux=True)
        old, payload = old_prediction(row, rm["normalization"]["raw_detail"], old_reference(rc))
        error = float((new["pred_pixels"]-old).abs().max())
        feature_error = float((new["aux"]["features"]-payload["features"]["raw_detail"]).abs().max())
        correction_error = float((new["aux"]["corrections"]-payload["corrections"]["raw_detail"]).abs().max())
    if error > 1e-6 or feature_error > 1e-6 or correction_error > 1e-6 or bool(new["offsets"].any()):
        raise ValueError(f"Zero-offset reference parity failed: {error}, {feature_error}, {correction_error}")
    out = new_directory(out)
    write_json(out/"manifest.json", {"schema_version":1, "n":136, "split":split,
        "source_cache_provenance":origin, "source_cache_provenance_sha256":digest(Path(cache)/"provenance.json"),
        "memory_manifest_path":str((Path(memory)/"manifest.json").resolve()), "memory_manifest_sha256":digest(Path(memory)/"manifest.json"),
        "reference_manifest_path":str((rp/"prepared_manifest.json").resolve()), "reference_manifest_sha256":digest(rp/"prepared_manifest.json"),
        "reference_head_path":str(Path(reference_head).resolve()), "reference_head_sha256":digest(reference_head),
        "reference_model_state_sha256":rc["model_state_sha256"], "normalization":rm["normalization"]["raw_detail"],
        "fit_only_mse_normalizer":mm["fit_only_mse_normalizer"], "per_sequence":entries,
        "zero_offset_parity_fit_only":{"name":row["name"], "prediction_max_abs":error,
            "features_max_abs":feature_error,"corrections_max_abs":correction_error,"offsets_exactly_zero":True},
        "source_sha256":source_hashes(*EXTRA), "target_frames_read":False,
        "gt_input_to_module":False, "oracle_computed":False,
        "geometry_context":"Full256 RGB/flow/address context; only loss is cropped128/eight frames"})
    print(json.dumps({"prepared":136,"zero_offset_parity":error,"feature_parity":feature_error}),flush=True)


def checked_metadata(prepared, verify_files=True):
    meta = json.loads((Path(prepared)/"manifest.json").read_text())
    check_source_hashes(meta["source_sha256"])
    names = [r["name"] for r in meta["per_sequence"]]
    if (meta["n"] != 136 or len(names) != 136 or len(set(names)) != 136
            or names != meta["source_cache_provenance"]["source_names"]
            or meta["split"] != make_scene_split(names) or meta["source_sha256"] != source_hashes(*EXTRA)
            or any(meta[k] is not False for k in ("target_frames_read","gt_input_to_module","oracle_computed"))):
        raise ValueError("Invalid immutable source-only geometry manifest")
    for prefix in ("memory_manifest", "reference_manifest", "reference_head"):
        if digest(meta[prefix+"_path"]) != meta[prefix+"_sha256"]:
            raise ValueError("Dependency metadata or checkpoint changed")
    for row in meta["per_sequence"]:
        if row["scene"] != source_scene(row["name"]):
            raise ValueError("Source scene differs")
        if verify_files and any(digest(row[k+"_path"]) != row[k+"_sha256"] for k in ("source","address","reference")):
            raise ValueError("Source cached tensors changed")
    return meta


def train(prepared, out, mode, seed):
    import torch
    torch.set_num_threads(4)
    meta = checked_metadata(prepared)
    out = new_directory(out)
    plan = make_plan([r["name"] for r in meta["per_sequence"]], meta["split"], seed)
    if plan["steps"] != STEPS:
        raise ValueError("Plan budget changed")
    write_json(out/"sample_plan.json", plan)
    rc = reference_checkpoint(meta["reference_head_path"])
    torch.manual_seed(seed)
    model = geometry_module().ObservationGeometry(meta["normalization"], rc["model"], address_mode=mode, hidden_dim=HIDDEN).train()
    initial = copy.deepcopy(model.state_dict())
    frozen_before = state_digest(model.node.state_dict())
    torch.save({"model":initial}, out/"initial_state.pt")
    provenance = {"schema_version":1,"mode":mode,"seed":seed,"steps":STEPS,
        "model_config":model.get_config(),"initial_state_sha256":state_digest(initial),
        "initial_file_sha256":digest(out/"initial_state.pt"), "sample_plan_sha256":digest(out/"sample_plan.json"),
        "prepared_manifest_sha256":digest(Path(prepared)/"manifest.json"), "source_sha256":source_hashes(*EXTRA),
        "split":meta["split"],"reference_head_sha256":meta["reference_head_sha256"],
        "frozen_node_state_sha256":frozen_before,"fit_only_mse_normalizer":meta["fit_only_mse_normalizer"],
        "args":{"optimizer":"Adam defaults;no weight decay","lr":LR,"threads":4,"device":"cpu float32",
                "forward_context":256,"loss_crop":128,"clip_len":16,"loss_frames":8,"clips_per_update":1},
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters":sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "selection":"Final400 only; one complete source internal evaluation after all modes; no target",
        "runtime":{"torch":str(torch.__version__),"numpy":str(np.__version__)}}
    write_json(out/"provenance.json",provenance)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters,lr=LR)
    start = time.monotonic()
    with (out/"curve.jsonl").open("w") as stream:
        for step,spec in enumerate(plan["batches"],1):
            inputs,gt = load_sample(meta["per_sequence"][spec["clip_index"]])
            optimizer.zero_grad(set_to_none=True)
            result = model(*inputs)
            mse = (loss_view(result["pred_pixels"],spec)-loss_view(gt,spec)).square().mean()
            loss = mse/meta["fit_only_mse_normalizer"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite coordinate training loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Missing/nonfinite coordinate gradient")
            if any(p.grad is not None for p in model.node.parameters()):
                raise RuntimeError("Frozen RGB selector received parameter gradients")
            optimizer.step()
            entry={"step":step,"rgb_mse":float(mse.detach()),"loss":float(loss.detach()),
                "clip_index":spec["clip_index"],"elapsed_seconds":time.monotonic()-start,
                "prediction_timing":"before optimizer update",**result["aux"]}
            stream.write(json.dumps(entry,allow_nan=False)+"\n")
            if step==1 or step%25==0:
                stream.flush();print(json.dumps(entry),flush=True)
    if state_digest(model.node.state_dict()) != frozen_before:
        raise RuntimeError("Frozen RGB selector changed during training")
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({"model":state,"model_state_sha256":state_digest(state),"steps":STEPS,
        "provenance":provenance,"optimizer":optimizer.state_dict()},out/"final.pt")
    write_json(out/"complete.json",{"steps":STEPS,"elapsed_seconds":time.monotonic()-start,
        "checkpoint_sha256":digest(out/"final.pt"),"model_state_sha256":state_digest(state),
        "frozen_node_unchanged":True,"frozen_node_state_sha256":frozen_before,
        "sample_plan_sha256":provenance["sample_plan_sha256"],"initial_state_sha256":provenance["initial_state_sha256"]})


def evaluate(prepared, runs, out):
    import torch
    torch.set_num_threads(4)
    if Path(out).exists():
        raise FileExistsError("Preserve previous source results")
    meta=checked_metadata(prepared)
    rc=reference_checkpoint(meta["reference_head_path"])
    models,checkpoints,provenance,inits,plans={},{},{},set(),set()
    expected_plan=make_plan([r["name"] for r in meta["per_sequence"]],meta["split"],0)
    for mode in MODES:
        folder=Path(runs)/f"{mode}_s0"
        complete=json.loads((folder/"complete.json").read_text())
        ck=torch.load(str(folder/"final.pt"),map_location="cpu",weights_only=True)
        p=ck["provenance"]
        init=torch.load(str(folder/"initial_state.pt"),map_location="cpu",weights_only=True)["model"]
        if (ck["steps"]!=STEPS or complete["steps"]!=STEPS or p["mode"]!=mode or p["seed"]!=0
                or complete["frozen_node_unchanged"] is not True
                or p["source_sha256"]!=source_hashes(*EXTRA)
                or p["prepared_manifest_sha256"]!=digest(Path(prepared)/"manifest.json")
                or p["reference_head_sha256"]!=meta["reference_head_sha256"]
                or complete["checkpoint_sha256"]!=digest(folder/"final.pt")
                or p["sample_plan_sha256"]!=digest(folder/"sample_plan.json")
                or p["initial_file_sha256"]!=digest(folder/"initial_state.pt")
                or state_digest(init)!=p["initial_state_sha256"]
                or json.loads((folder/"sample_plan.json").read_text())!=expected_plan):
            raise ValueError("Geometry final checkpoint lineage differs")
        model=geometry_module().ObservationGeometry(frozen_node_state=rc["model"],**p["model_config"]).eval()
        model.load_state_dict(ck["model"],strict=True)
        if (state_digest(model.state_dict())!=ck["model_state_sha256"]
                or state_digest(model.node.state_dict())!=p["frozen_node_state_sha256"]):
            raise ValueError("Model state digest or frozen selector differs")
        models[mode]=model;checkpoints[mode]=digest(folder/"final.pt");provenance[mode]=p
        inits.add(p["initial_state_sha256"]);plans.add(p["sample_plan_sha256"])
    if len(inits)!=1 or len(plans)!=1:
        raise ValueError("Modes must share exact initial tensors and sample plan")
    reference=old_reference(rc)
    before={m:state_digest(v.state_dict()) for m,v in models.items()}
    scores=[]
    with torch.no_grad():
        for row in meta["per_sequence"]:
            if row["scene"] not in meta["split"]["internal_holdout"]:
                continue
            inputs,gt=load_sample(row)
            anchor,old=old_prediction(row,meta["normalization"],reference)
            if not torch.equal(old["base"],inputs[1]) or not torch.equal(old["gt"],gt):
                raise ValueError("Frozen reference base/GT differs")
            results={m:v(*inputs) for m,v in models.items()}
            predictions={"base":inputs[1],"frozen_raw_node_1000":anchor,
                         **{m:v["pred_pixels"] for m,v in results.items()}}
            values={}
            for name,value in predictions.items():
                mse=float((value-gt).square().double().mean())
                if not np.isfinite(mse) or mse<=0:
                    raise ValueError("Invalid full16/full256 source MSE")
                values[name]={"rgb_mse_float":mse,"psnr_rgb_from_clip_mse":float(-10*np.log10(mse))}
            scores.append({"name":row["name"],"scene":row["scene"],"methods":values,
                           "geometry":{m:r["aux"] for m,r in results.items()}})
            print(json.dumps({"evaluated_source_holdout":len(scores),"name":row["name"]}),flush=True)
    unchanged=(all(state_digest(v.state_dict())==before[m] for m,v in models.items())
               and state_digest(reference.state_dict())==rc["model_state_sha256"])
    expected={r["name"] for r in meta["per_sequence"] if r["scene"] in meta["split"]["internal_holdout"]}
    if not unchanged or {r["name"] for r in scores}!=expected or len(scores)!=34:
        raise ValueError("Source coverage incomplete or model mutated")
    write_json(out,{"schema_version":1,"scope":"source TRAIN internal module development; historical base saw all scenes; no target",
        "metric":"float32 RGB squared residual, float64 mean over all16/full256; PSNR from each clip MSE; equal scene mean",
        "selection":"Three final400 coordinate modules; frozen raw_node1000 anchor; one source evaluation",
        "prepared_manifest_sha256":digest(Path(prepared)/"manifest.json"),"source_sha256":source_hashes(*EXTRA),
        "checkpoint_sha256":checkpoints,"provenance":provenance,"reference_checkpoint_sha256":meta["reference_head_sha256"],
        "reference_state_sha256":rc["model_state_sha256"],"states_unchanged":unchanged,
        "initial_states_and_plans_identical":True,"target_frames_read":False,"per_sequence":scores})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("prepare")
    for k in ("cache","memory","splits","reference-prepared","reference-head","out"):
        q.add_argument("--"+k,type=Path,required=True)
    q=sub.add_parser("train")
    q.add_argument("--prepared",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.add_argument("--mode",choices=MODES,required=True);q.add_argument("--seed",type=int,choices=(0,),default=0)
    q=sub.add_parser("evaluate")
    for k in ("prepared","runs","out"):
        q.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    if a.action=="prepare":prepare(a.cache,a.memory,a.splits,a.reference_prepared,a.reference_head,a.out)
    elif a.action=="train":train(a.prepared,a.out,a.mode,a.seed)
    else:evaluate(a.prepared,a.runs,a.out)


if __name__=="__main__":main()
