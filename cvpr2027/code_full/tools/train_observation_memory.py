"""Source-only fixed-budget screen of video-addressed observation memory.

The source internal holdout is a development set, already seen by the historical
base. This tool never loads target data, encoders, RAFT, or a pixel base model.
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
sys.path.insert(0, str(CODE))
from train_correction_selector import digest, local_module, state_digest, write_json
from train_source_selector import cache_manifest, make_scene_split
from cache_source_corrections import source_hashes, check_source_hashes, source_scene

MODES = {
    "prior_cross": ("prior", "cross_frame"),
    "pixel_cross": ("pixel", "cross_frame"),
    "shuffled_cross": ("shuffled_prior", "cross_frame"),
    "prior_same": ("prior", "same_frame"),
    "pixel_local": ("pixel", "local"),
}
STEPS, CROP, PATCH, LOSS_FRAMES, LR = 400, 128, 16, 8, 1e-3
EXTRA = ("models/observation_memory.py", "tools/cache_observation_memory.py",
         "tools/train_observation_memory.py", "tools/train_source_selector.py")


def memory_module():
    return local_module("_observation_memory_screen", CODE / "models/observation_memory.py")


def new_directory(path):
    path = Path(path)
    if not path.resolve().is_relative_to(Path("/cache")):
        raise ValueError("Tensor outputs must be on host-local /cache")
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_plan(names, split, seed):
    if split != make_scene_split(names):
        raise ValueError("Unexpected source scene split")
    fit = [i for i, name in enumerate(names) if source_scene(name) in split["fit"]]
    if not fit:
        raise ValueError("Empty fit source scenes")
    rng = np.random.default_rng(seed)
    return {"seed": seed, "steps": STEPS, "crop": CROP, "patch": PATCH,
            "loss_frames": LOSS_FRAMES, "source_names": names,
            "batches": [{"clip_index": int(rng.choice(fit)),
                         "top": int(rng.integers(0, (256 - CROP) // PATCH + 1)) * PATCH,
                         "left": int(rng.integers(0, (256 - CROP) // PATCH + 1)) * PATCH,
                         "frames": rng.choice(16, LOSS_FRAMES, replace=False).tolist()}
                        for _ in range(STEPS)]}


def crop_inputs(sample, spec):
    top, left = spec["top"], spec["left"]
    if top % PATCH or left % PATCH or not (0 <= top <= 256-CROP and 0 <= left <= 256-CROP):
        raise ValueError("Crop must align to full cached patch grid")
    view = (..., slice(top, top+CROP), slice(left, left+CROP))
    fields = []
    for key in ("detail_values", "prior_address", "pixel_address"):
        x = sample[key]
        b, t, p, d = x.shape
        if p != 256:
            raise ValueError("Expected full256 patch16 source fields")
        x = x.reshape(b, t, 16, 16, d)
        fields.append(x[:, :, top//PATCH:(top+CROP)//PATCH,
                        left//PATCH:(left+CROP)//PATCH].reshape(b, t, -1, d).contiguous())
    return (sample["base"][view].contiguous(), *fields), sample["gt"][view].contiguous()


def prepare(cache, tokens, splits, out):
    import torch
    torch.set_num_threads(4)
    files, origin, rows = cache_manifest(cache, splits)
    if origin["base_saved_args"]["tag"] != "e17_small_pretrained_s0":
        raise ValueError("First screen uses the fixed pretrained E17 seed0 base")
    tokens, out = Path(tokens), new_directory(out)
    tm = json.loads((tokens / "manifest.json").read_text())
    module = memory_module()
    check_source_hashes(tm["source_sha256"])
    if (tm["n"] != 136 or len(tm["per_sequence"]) != 136
            or tm["source_cache_provenance_sha256"] != digest(Path(cache)/"provenance.json")
            or tm["source_cache_summary_sha256"] != digest(Path(cache)/"cache_summary.json")
            or tm["provenance_sha256"] != digest(tokens/"provenance.json")
            or tm["token_grid"] != [8,16,16] or tm["descriptor_dim"] != 64
            or tm["projection"] != module.memory_feature_config()
            or tm["source_names"] != origin["source_names"]
            or tm["encoder_weights_sha256"] != origin["encoder_weights_sha256"]
            or tm["encoder_state_before"] != tm["encoder_state_after"]
            or tm["encoder_state_unchanged"] is not True or tm["complete"] is not True
            or tm["encoder_eval"] is not True or tm["encoder_requires_grad"] is not False
            or tm["video_state_load_strict"] is not True or tm["feature_layer"] != 17
            or tm["tubelet_repeat"] != 2 or tm["projection_backend"] != "cpu_float32"
            or any(tm[k] is not False for k in ("gt_input","target_input","metric_computed",
                                                "oracle_computed","optimization_performed"))):
        raise ValueError("Frozen projected tokens do not match the original source cache")
    split = make_scene_split(origin["source_names"])
    result, fit_mses = [], []
    for i, (path, row, tr) in enumerate(zip(files, rows, tm["per_sequence"])):
        token_path = tokens / tr["cache_path"]
        if (tr["name"] != row["name"] or tr["scene"] != row["scene"]
                or tr["source_cache_sha256"] != row["cache_sha256"]
                or tr["cache_path"] != f"tokens_{i:03d}.pt"
                or digest(token_path) != tr["cache_sha256"]):
            raise ValueError("Token lineage or file SHA differs")
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        token = torch.load(token_path, map_location="cpu", weights_only=True)
        if (token["name"] != row["name"] or token["scene"] != row["scene"]
                or token["source_cache_sha256"] != row["cache_sha256"]):
            raise ValueError("Token payload identity differs")
        low, base, gt = (raw["tensors"][k] for k in ("lq", "base", "gt"))
        if any(x.shape != (1,3,16,256,256) or x.dtype != torch.float32
               or not bool(torch.isfinite(x).all()) or not bool(((x>=0)&(x<=1)).all())
               for x in (low, base, gt)):
            raise ValueError("Invalid cached source RGB tensor")
        with torch.no_grad():
            # Explicit input firewall: source GT is not passed to this helper.
            fields = module.prepare_observation_memory_inputs(low, base, token["tokens"],
                                                              prior_is_projected=True)
        payload = {"name": row["name"], "scene": row["scene"], "base": base.clone(),
                   "gt": gt.clone(), **{k: fields[k].contiguous() for k in
                       ("detail_values", "prior_address", "pixel_address")}}
        p = out / f"memory_{i:03d}.pt"
        torch.save(payload, p)
        result.append({"name": row["name"], "scene": row["scene"], "cache_path": p.name,
                       "cache_sha256": digest(p), "source_cache_sha256": row["cache_sha256"],
                       "tokens_cache_sha256": tr["cache_sha256"]})
        if row["scene"] in split["fit"]:
            fit_mses.append(float((gt-base).square().mean()))
        print(json.dumps({"prepared":i+1,"name":row["name"]}), flush=True)
    scale = float(np.mean(fit_mses))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fit-only base RGB MSE")
    write_json(out / "manifest.json", {"schema_version":1, "n":136, "split":split,
        "source_cache_provenance":origin, "tokens_manifest":tm,
        "tokens_manifest_sha256":digest(tokens/"manifest.json"),
        "source_cache_provenance_sha256":digest(Path(cache)/"provenance.json"),
        "source_cache_summary_sha256":digest(Path(cache)/"cache_summary.json"),
        "fit_only_mse_normalizer":scale, "per_sequence":result,
        "source_sha256":source_hashes(*EXTRA), "target_frames_read":False,
        "gt_input_to_module":False, "oracle_computed":False,
        "address_context":"Whole cached256 clip;128 crops slice matching16x16 patches without rerunning encoder"})


def checked_metadata(prepared, verify_files=True):
    meta = json.loads((Path(prepared)/"manifest.json").read_text())
    check_source_hashes(meta["source_sha256"])
    names = [r["name"] for r in meta["per_sequence"]]
    if (meta["n"] != 136 or len(names) != 136 or len(set(names)) != 136
            or meta["split"] != make_scene_split(names)
            or names != meta["source_cache_provenance"]["source_names"]
            or meta["source_sha256"] != source_hashes(*EXTRA)
            or meta["target_frames_read"] is not False or meta["gt_input_to_module"] is not False
            or meta["oracle_computed"] is not False):
        raise ValueError("Require the immutable complete source-only memory cache")
    for i, row in enumerate(meta["per_sequence"]):
        if (row["cache_path"] != f"memory_{i:03d}.pt" or row["scene"] != source_scene(row["name"])
                or (verify_files and digest(Path(prepared)/row["cache_path"]) != row["cache_sha256"])):
            raise ValueError("Prepared cache file identity differs")
    return meta


def train(prepared, out, mode, seed):
    import torch
    torch.set_num_threads(4)
    prepared, out = Path(prepared), new_directory(out)
    meta = checked_metadata(prepared)
    names = [r["name"] for r in meta["per_sequence"]]
    plan = make_plan(names, meta["split"], seed)
    write_json(out/"sample_plan.json", plan)
    torch.manual_seed(seed)
    model = memory_module().ObservationMemory(address_mode=MODES[mode][0], retrieval_mode=MODES[mode][1]).train()
    initial = copy.deepcopy(model.state_dict())
    torch.save({"model":initial}, out/"initial_state.pt")
    provenance = {"schema_version":1, "mode":mode,"seed":seed,"steps":STEPS,
        "model_config":model.get_config(),"initial_state_sha256":state_digest(initial),
        "initial_file_sha256":digest(out/"initial_state.pt"),
        "sample_plan_sha256":digest(out/"sample_plan.json"),
        "prepared_manifest_sha256":digest(prepared/"manifest.json"),
        "source_sha256":source_hashes(*EXTRA),"split":meta["split"],
        "fit_only_mse_normalizer":meta["fit_only_mse_normalizer"],
        "args":{"optimizer":"Adam defaults;no weight decay","lr":LR,"threads":4,
                "device":"cpu float32","crop":CROP,"patch":PATCH,"clip_len":16,
                "loss_frames":LOSS_FRAMES,"clips_per_update":1},
        "allocated_parameters":sum(p.numel() for p in model.parameters()),
        "selection":"Final400 only, followed by one complete source internal evaluation; no target",
        "runtime":{"torch":str(torch.__version__),"numpy":str(np.__version__)}}
    write_json(out/"provenance.json", provenance)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.monotonic()
    with (out/"curve.jsonl").open("w") as stream:
        for step, spec in enumerate(plan["batches"],1):
            row = meta["per_sequence"][spec["clip_index"]]
            sample = torch.load(prepared/row["cache_path"], map_location="cpu", weights_only=True, mmap=True)
            inputs, gt = crop_inputs(sample, spec)
            optimizer.zero_grad(set_to_none=True)
            result = model(*inputs)
            mse = (result["pred_pixels"][:,:,spec["frames"]]-gt[:,:,spec["frames"]]).square().mean()
            loss = mse/meta["fit_only_mse_normalizer"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite memory gradient")
            optimizer.step()
            entry = {"step":step,"rgb_mse":float(mse.detach()),"loss":float(loss.detach()),
                     "base_mse":float((inputs[0][:,:,spec["frames"]]-gt[:,:,spec["frames"]]).square().mean()),
                     "clip_index":spec["clip_index"],"elapsed_seconds":time.monotonic()-start,
                     "prediction_timing":"before optimizer update"}
            stream.write(json.dumps(entry,allow_nan=False)+"\n")
            if step==1 or step%50==0:
                stream.flush();print(json.dumps(entry),flush=True)
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({"model":state,"model_state_sha256":state_digest(state),"steps":STEPS,
                "provenance":provenance,"optimizer":optimizer.state_dict()},out/"final.pt")
    write_json(out/"complete.json",{"steps":STEPS,"elapsed_seconds":time.monotonic()-start,
        "checkpoint_sha256":digest(out/"final.pt"),"model_state_sha256":state_digest(state),
        "sample_plan_sha256":provenance["sample_plan_sha256"],"initial_state_sha256":provenance["initial_state_sha256"]})


def evaluate(prepared, runs, out, reference_prepared, reference_head):
    import torch
    torch.set_num_threads(4)
    prepared, runs = Path(prepared), Path(runs)
    if Path(out).exists():
        raise FileExistsError("Preserve previous source scores")
    meta = checked_metadata(prepared)
    module = memory_module()
    models, checkpoints, inits, plans, provenance = {}, {}, set(), set(), {}
    for mode, (address,retrieval) in MODES.items():
        folder = runs/f"{mode}_s0"
        complete = json.loads((folder/"complete.json").read_text())
        ck = torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
        p = ck["provenance"]
        check_source_hashes(p["source_sha256"])
        actual_initial = torch.load(folder/"initial_state.pt",map_location="cpu",weights_only=True)["model"]
        if (ck["steps"]!=STEPS or complete["steps"]!=STEPS or p["mode"]!=mode or p["seed"]!=0
                or p["prepared_manifest_sha256"]!=digest(prepared/"manifest.json")
                or complete["checkpoint_sha256"]!=digest(folder/"final.pt")
                or p["sample_plan_sha256"]!=digest(folder/"sample_plan.json")
                or p["initial_file_sha256"]!=digest(folder/"initial_state.pt")
                or state_digest(actual_initial)!=p["initial_state_sha256"]
                or json.loads((folder/"sample_plan.json").read_text())!=make_plan(
                    [r["name"] for r in meta["per_sequence"]],meta["split"],0)):
            raise ValueError("Final source checkpoint provenance differs")
        model = module.ObservationMemory(address_mode=address,retrieval_mode=retrieval).eval()
        model.load_state_dict(ck["model"],strict=True)
        if state_digest(model.state_dict())!=ck["model_state_sha256"]:
            raise ValueError("Memory state SHA mismatch")
        models[mode]=model;checkpoints[mode]=digest(folder/"final.pt");provenance[mode]=p
        inits.add(p["initial_state_sha256"]);plans.add(p["sample_plan_sha256"])
    if len(inits)!=1 or len(plans)!=1:
        raise ValueError("Screen modes must share initial parameters and exact training plan")
    rp = Path(reference_prepared)
    rm = json.loads((rp/"prepared_manifest.json").read_text())
    check_source_hashes(rm["source_sha256"])
    if rm["source_cache_provenance"]!=meta["source_cache_provenance"] or rm["split"]!=meta["split"]:
        raise ValueError("Strong frozen control comes from different source data/base")
    rc = torch.load(reference_head,map_location="cpu",weights_only=True)
    rcp = rc["provenance"]
    check_source_hashes(rcp["source_sha256"])
    if (rc["steps"]!=1000 or rcp["variant"]!="raw_node" or rcp["seed"]!=0
            or rcp["prepared_manifest_sha256"]!=digest(rp/"prepared_manifest.json")
            or rcp["base_checkpoint_sha256"]!=rm["source_cache_provenance"]["base_checkpoint_sha256"]
            or rcp["normalization"]!=rm["normalization"]["raw_detail"]):
        raise ValueError("Require the frozen observation-v2 raw_node seed0 reference")
    reference = local_module("_memory_v2_reference",CODE/"models/trajectory_reconstruction.py").TrajectoryReconstruction(mode="node_only").eval()
    reference.load_state_dict(rc["model"],strict=True)
    if state_digest(reference.state_dict())!=rc["model_state_sha256"]:
        raise ValueError("Frozen reference model digest mismatch")
    before={m:state_digest(v.state_dict()) for m,v in models.items()}
    reference_before=state_digest(reference.state_dict())
    scores=[]
    with torch.no_grad():
        for i,row in enumerate(meta["per_sequence"]):
            if row["scene"] not in meta["split"]["internal_holdout"]:
                continue
            s=torch.load(prepared/row["cache_path"],map_location="cpu",weights_only=True,mmap=True)
            inputs=(s["base"],s["detail_values"],s["prior_address"],s["pixel_address"])
            pred={"base":s["base"],**{m:v(*inputs)["pred_pixels"] for m,v in models.items()}}
            rr=rm["per_sequence"][i]
            if (rr["name"]!=row["name"] or digest(rp/rr["cache_path"])!=rr["cache_sha256"]):
                raise ValueError("Strong control cache identity differs")
            old=torch.load(rp/rr["cache_path"],map_location="cpu",weights_only=True,mmap=True)
            if not torch.equal(old["base"],s["base"]) or not torch.equal(old["gt"],s["gt"]):
                raise ValueError("Strong control base/labels differ")
            norm=rm["normalization"]["raw_detail"]
            x=(old["features"]["raw_detail"]-torch.tensor(norm["feature_mean"])[None,None,:,None,None])/torch.tensor(norm["feature_std"])[None,None,:,None,None]
            pred["frozen_raw_node_1000"]=reference(old["base"],old["corrections"]["raw_detail"],x,
                old["detail_observations"],old["adjacent_flow"],old["adjacent_mask"])["pred_pixels"]
            values={}
            for name,value in pred.items():
                mse=float((value-s["gt"]).square().double().mean())
                if not np.isfinite(mse) or mse<=0:
                    raise ValueError("Invalid full-clip source MSE")
                values[name]={"rgb_mse_float":mse,"psnr_rgb_from_clip_mse":float(-10*np.log10(mse))}
            scores.append({"name":row["name"],"scene":row["scene"],"methods":values})
            print(json.dumps({"evaluated_source_holdout":len(scores),"name":row["name"]}),flush=True)
    unchanged=all(state_digest(v.state_dict())==before[m] for m,v in models.items()) and state_digest(reference.state_dict())==reference_before
    if not unchanged or len(scores)!=34 or len({r["scene"] for r in scores})!=17:
        raise ValueError("Incomplete source internal evaluation or mutated model")
    write_json(out,{"schema_version":1,"scope":"source TRAIN internal module development; base has seen these scenes; no target",
        "metric":"float32 predictions/GT;float64 mean squared residual over every RGB pixel/all16 cached frames; PSNR from clip MSE",
        "selection":"one final400 screen across five modes; strong raw_node is frozen1000 reference with different training budget",
        "prepared_manifest_sha256":digest(prepared/"manifest.json"),"source_sha256":source_hashes(*EXTRA),
        "checkpoint_sha256":checkpoints,"provenance":provenance,"reference_checkpoint_sha256":digest(reference_head),
        "reference_state_sha256":reference_before,"states_unchanged":unchanged,
        "initial_states_and_plans_identical":True,"target_frames_read":False,"per_sequence":scores})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("prepare")
    for key in ("cache","tokens","splits","out"):
        q.add_argument("--"+key,type=Path,required=True)
    q=sub.add_parser("train")
    q.add_argument("--prepared",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.add_argument("--mode",choices=tuple(MODES),required=True);q.add_argument("--seed",type=int,choices=(0,),default=0)
    q=sub.add_parser("evaluate")
    for key in ("prepared","runs","out","reference-prepared","reference-head"):
        q.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    if a.action=="prepare":prepare(a.cache,a.tokens,a.splits,a.out)
    elif a.action=="train":train(a.prepared,a.out,a.mode,a.seed)
    else:evaluate(a.prepared,a.runs,a.out,a.reference_prepared,a.reference_head)


if __name__=="__main__":
    main()
