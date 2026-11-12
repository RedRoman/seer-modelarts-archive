"""Cache GT-free, phase-preserving joint donor costs; fit variance on FIT only.

The two nonnegative calibration coefficients describe effective observation
error against source FIT labels. They are not estimates of physical camera
noise and do not make donor errors independent. No backbone or flow is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from train_correction_selector import digest, local_module, write_json
from train_observation_geometry import checked_metadata as geometry_metadata
from train_observation_memory import new_directory
from cache_source_corrections import source_hashes, check_source_hashes

PATCH, OFFSET, SHIFTS, SHUFFLE_SEED = 16, 2, (-2, 0, 2), 2718
VARIANCE_FLOOR, MIN_SUPPORT = 1e-8, 0.5
SOURCE_FILES = ("tools/cache_joint_observation_search.py", "tools/train_observation_geometry.py",
    "tools/train_observation_memory.py", "tools/train_source_selector.py",
    "tools/train_correction_selector.py", "tools/cache_source_corrections.py",
    "tools/cache_observation_memory.py", "models/observation_candidates.py",
    "models/observation_memory.py", "models/observation_geometry.py", "models/joint_observation_search.py")


def observable_module():
    return local_module("_joint_cost_observations", CODE / "models/observation_candidates.py")


def calibrated_observation(low, base):
    """Observable q, gain and lowpass(low); no labels accepted."""
    m = observable_module()
    low_l, base_l = m._lowpass(low), m._lowpass(base)
    gain = (base_l / (low_l + m.GAIN_EPSILON)).clamp(m.GAIN_MIN, m.GAIN_MAX)
    return (base_l + gain * (low - low_l)).clamp(0, 1), gain, low_l


def _patches(video, patch=PATCH):
    """[B,C,T,H,W] -> [B*T,N,C*patch*patch], preserving within-patch phase."""
    import torch.nn.functional as F
    m = observable_module()
    return F.unfold(m._frames(video), kernel_size=patch, stride=patch).transpose(1, 2)


def calibration_statistics(low, base, fit_labels, patch_size=PATCH):
    """Sufficient statistics from explicitly supplied source FIT labels only."""
    import torch
    if fit_labels.shape != low.shape or not bool(torch.isfinite(fit_labels).all()):
        raise ValueError("FIT calibration labels must match finite RGB inputs")
    q, gain, low_l = calibrated_observation(low, base)
    m = observable_module()
    target = _patches(((q-m._lowpass(q))-(fit_labels-m._lowpass(fit_labels))).square(), patch_size).mean(-1)
    design = torch.stack((_patches(gain.square()*low_l, patch_size).mean(-1),
                          _patches(gain.square(), patch_size).mean(-1)), -1)
    x, y = design.reshape(-1, 2).double(), target.flatten().double()
    return {"xx": x.T@x, "xy": x.T@y, "yy": y@y, "count": len(y),"target_sum":y.sum()}


def fit_nonnegative_variance(statistics):
    """Exact two-variable NNLS from sufficient statistics, including boundaries."""
    import torch
    xx, xy, yy = statistics["xx"].double(), statistics["xy"].double(), statistics["yy"].double()
    if xx.shape != (2, 2) or xy.shape != (2,) or statistics["count"] <= 0:
        raise ValueError("Invalid effective-error calibration statistics")
    if not all(bool(torch.isfinite(x).all()) for x in (xx, xy, yy)):
        raise ValueError("Nonfinite calibration statistics")
    candidates = [torch.zeros(2, dtype=torch.float64)]
    for i in range(2):
        c = torch.zeros(2, dtype=torch.float64)
        if xx[i, i] > 0:
            c[i] = (xy[i]/xx[i, i]).clamp_min(0)
        candidates.append(c)
    solution = torch.linalg.lstsq(xx, xy).solution
    if bool((solution >= 0).all()):
        candidates.append(solution)
    scores = [yy-2*(c@xy)+c@xx@c for c in candidates]
    best = int(torch.stack(scores).argmin())
    return {"coefficients": candidates[best].tolist(), "patch_observations": statistics["count"],
        "global_effective_variance":float(statistics["target_sum"])/statistics["count"] if "target_sum" in statistics else None,
        "fit_squared_error": max(0., float(scores[best])),
        "design_gram": xx.tolist(), "design_target": xy.tolist(), "target_square_sum":float(yy),
        "formula":"a*gain^2*LP9(low)+b*gain^2; clamp_min(1e-8)",
        "target":"patch mean squared [HP9(q)-HP9(source FIT GT)]",
        "interpretation":"effective observation-error proxy; not physical camera noise"}


def fit_calibration(rows,fit_scenes,load_tensors):
    """Access labels only after checking explicit FIT scene membership."""
    total=None;fit_names=[]
    for row in rows:
        if row["scene"] not in fit_scenes:
            continue
        data=load_tensors(row)
        stat=calibration_statistics(data["lq"],data["base"],data["gt"])
        total=stat if total is None else {key:total[key]+stat[key] for key in total}
        fit_names.append(row["name"])
    if total is None:
        raise ValueError("At least one explicitly declared FIT scene is required")
    result=fit_nonnegative_variance(total)
    result.update({"fit_names":fit_names,"fit_scenes":list(fit_scenes),
        "internal_holdout_labels_read":False,"target_labels_read":False,"variance_floor":VARIANCE_FLOOR})
    if result["global_effective_variance"] is None or result["global_effective_variance"]<=0:
        raise ValueError("FIT effective observation error must be positive")
    return result


def _descriptor_cost(address, flows, shifts, offset, patch):
    import torch
    import torch.nn.functional as F
    b, t, n, d = address.shape
    _, _, _, _, h, w = flows.shape
    gh, gw = h//patch, w//patch
    if n != gh*gw:
        raise ValueError("Address grid differs from complete RGB context")
    values = address.reshape(b, t, gh, gw, d).permute(0, 1, 4, 2, 3).reshape(b*t,d,gh,gw)
    yy, xx = torch.meshgrid(torch.arange(gh,device=flows.device)*patch+(patch-1)/2,
        torch.arange(gw,device=flows.device)*patch+(patch-1)/2,indexing="ij")
    center_grid=torch.stack((2*(xx+.5)/w-1,2*(yy+.5)/h-1),-1)[None].expand(b*t,-1,-1,-1)
    target = F.normalize(values,dim=1)
    result=[]
    times=torch.arange(t,device=flows.device)
    for j,delta in enumerate((-offset,offset)):
        flow=flows[:,j].permute(0,2,1,3,4).reshape(b*t,2,h,w)
        sampled=F.grid_sample(flow,center_grid,mode="bilinear",padding_mode="border",align_corners=False)
        source=address[:,(times+delta).clamp(0,t-1)].reshape(b,t,gh,gw,d).permute(0,1,4,2,3).reshape(b*t,d,gh,gw)
        candidates=[]
        for dx,dy in shifts:
            grid=torch.stack((2*(xx[None]+sampled[:,0]+dx+.5)/w-1,
                2*(yy[None]+sampled[:,1]+dy+.5)/h-1),-1)
            donor=F.normalize(F.grid_sample(source,grid,mode="bilinear",padding_mode="border",align_corners=False),dim=1)
            candidates.append((1-(target*donor).sum(1)).clamp(0,2).reshape(b,t,n))
        result.append(torch.stack(candidates,-1))
    return torch.stack(result,1)


def shuffled_addresses(address, seed=SHUFFLE_SEED):
    """Private CPU generator; full-clip address permutation, no global RNG mutation."""
    import torch
    b,t,n,d=address.shape
    generator=torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([x.reshape(t*n,d).index_select(0,
        torch.randperm(t*n,generator=generator).to(address.device)).reshape(t,n,d) for x in address])


def _supported_ssd(a,b,va,vb,mask,patch,minimum_support,global_variance):
    """Mean variance-normalized SSD over common real supported RGB samples only."""
    fraction=mask.float().mean(-1)
    valid=fraction>=minimum_support
    # RGB channels are contiguous groups of patch^2 in F.unfold output.
    rgb_mask=mask.repeat(1,1,3)
    squared=(a-b).square()
    error=(squared/(va+vb).clamp_min(VARIANCE_FLOOR))*rgb_mask
    score=error.sum(-1)/rgb_mask.sum(-1).clamp_min(1)
    constant=(squared*rgb_mask).sum(-1)/rgb_mask.sum(-1).clamp_min(1)/(2*global_variance)
    return score.masked_fill(~valid,0),constant.masked_fill(~valid,0),valid,fraction


def build_joint_costs(low,base,flows,masks,prior_address,pixel_address,variance_coefficients,
                      *,patch_size=PATCH,offset=OFFSET,shifts=SHIFTS,
                      shuffle_seed=SHUFFLE_SEED,minimum_support=MIN_SUPPORT,global_variance):
    """Construct observable costs; this inference helper deliberately has no GT input.

    Returns row-major N layouts agreed with JointObservationSearch. All costs
    are nonnegative finite; invalid entries are zero and excluded by bool masks.
    Every donor is warped as calibrated RGB BEFORE the HP9 filter. Pairwise
    SSD is computed on common supported pixels, never on padded zeros.
    """
    import torch
    import torch.nn.functional as F
    m=observable_module();m._validate(low,base,flows,masks,offset)
    b,_,t,h,w=low.shape
    if h%patch_size or w%patch_size or not 0<minimum_support<=1:
        raise ValueError("Whole aligned patches and a positive support fraction are required")
    if len(variance_coefficients)!=2 or any(not 0<=float(c)<float("inf") for c in variance_coefficients):
        raise ValueError("Expected two finite nonnegative effective-error coefficients")
    if not 0<float(global_variance)<float("inf"):
        raise ValueError("Expected positive finite FIT-only global effective variance")
    n=(h//patch_size)*(w//patch_size)
    if any(a.shape!=(b,t,n,64) or a.device!=low.device or a.dtype!=low.dtype
           or not bool(torch.isfinite(a).all()) for a in (prior_address,pixel_address)):
        raise ValueError("Require finite matching64D addresses")
    shifts=tuple((dx,dy) for dy in shifts for dx in shifts)
    if len(shifts)!=9:
        raise ValueError("Expected the fixed3x3 candidate grid")
    q,gain,low_l=calibrated_observation(low,base)
    variance=(float(variance_coefficients[0])*gain.square()*low_l+
              float(variance_coefficients[1])*gain.square()).clamp_min(VARIANCE_FLOOR)
    target=_patches(q-m._lowpass(q),patch_size)
    target_var=_patches(variance,patch_size)
    yy,xx=torch.meshgrid(torch.arange(h,device=low.device),torch.arange(w,device=low.device),indexing="ij")
    times=torch.arange(t,device=low.device)
    all_values,all_variances,all_masks=[],[],[]
    unary,unary_constant,unary_valid,unary_fraction=[],[],[],[]
    for j,delta in enumerate((-offset,offset)):
        st=times+delta
        temporal=((st>=0)&(st<t))[None,:,None,None]
        source=m._frames(q.index_select(2,st.clamp(0,t-1)))
        source_var=m._frames(variance.index_select(2,st.clamp(0,t-1)))
        flow=m._frames(flows[:,j]);values,variances,supports=[],[],[]
        scores,constants,valids,fractions=[],[],[],[]
        for dx,dy in shifts:
            x,y=xx[None]+flow[:,0]+dx,yy[None]+flow[:,1]+dy
            grid=torch.stack((2*(x+.5)/w-1,2*(y+.5)/h-1),-1)
            geometric=((x>=0)&(x<=w-1)&(y>=0)&(y<=h-1)).reshape(b,t,h,w)&temporal
            support=m._support_mask(masks[:,j].bool()&geometric[:,None])
            warped=m._video(F.grid_sample(source,grid,mode="bilinear",padding_mode="border",align_corners=False),b,t)
            value=_patches(warped-m._lowpass(warped),patch_size)
            var=_patches(m._video(F.grid_sample(source_var,grid,mode="bilinear",padding_mode="border",align_corners=False),b,t),patch_size)
            mask=_patches(support.float(),patch_size).bool()
            score,constant,valid,fraction=_supported_ssd(target,value,target_var,var,mask,patch_size,minimum_support,global_variance)
            values.append(value);variances.append(var);supports.append(mask)
            scores.append(score.reshape(b,t,n));constants.append(constant.reshape(b,t,n));valids.append(valid.reshape(b,t,n));fractions.append(fraction.reshape(b,t,n))
        all_values.append(values);all_variances.append(variances);all_masks.append(supports)
        unary.append(torch.stack(scores,-1));unary_constant.append(torch.stack(constants,-1));unary_valid.append(torch.stack(valids,-1));unary_fraction.append(torch.stack(fractions,-1))
    pairs,pair_constant,pair_valid,pair_fraction=[],[],[],[]
    for km in range(9):
        row,constant_row,valid_row,fraction_row=[],[],[],[]
        for kp in range(9):
            mask=all_masks[0][km]&all_masks[1][kp]
            score,constant,valid,fraction=_supported_ssd(all_values[0][km],all_values[1][kp],
                all_variances[0][km],all_variances[1][kp],mask,patch_size,minimum_support,global_variance)
            valid=valid & unary_valid[0][...,km].reshape(b*t,n) & unary_valid[1][...,kp].reshape(b*t,n)
            row.append(score.masked_fill(~valid,0).reshape(b,t,n));constant_row.append(constant.masked_fill(~valid,0).reshape(b,t,n));valid_row.append(valid.reshape(b,t,n));fraction_row.append(fraction.reshape(b,t,n))
        pairs.append(torch.stack(row,-1));pair_constant.append(torch.stack(constant_row,-1));pair_valid.append(torch.stack(valid_row,-1));pair_fraction.append(torch.stack(fraction_row,-1))
    descriptor={"prior":_descriptor_cost(prior_address,flows,shifts,offset,patch_size),
        "pixel":_descriptor_cost(pixel_address,flows,shifts,offset,patch_size),
        "shuffled_prior":_descriptor_cost(shuffled_addresses(prior_address,shuffle_seed),flows,shifts,offset,patch_size)}
    valid=torch.stack(unary_valid,1)
    descriptor={key:value.masked_fill(~valid,0) for key,value in descriptor.items()}
    return {"unary_observation":torch.stack(unary,1),"pair_observation":torch.stack(pairs,-2),
        "unary_constant":torch.stack(unary_constant,1),"pair_constant":torch.stack(pair_constant,-2),
        "descriptor_costs":descriptor,"valid_unary":valid,"valid_pair":torch.stack(pair_valid,-2),
        "metadata":{"unary_support_fraction":torch.stack(unary_fraction,1),"pair_support_fraction":torch.stack(pair_fraction,-2)}}


def _cache_row(arguments):
    """Independent CPU worker; no labels are accessed during cost construction."""
    import torch
    torch.set_num_threads(4)
    i,row,out,coefficients,global_variance=arguments
    source=torch.load(str(row["source_path"]),map_location="cpu",weights_only=True,mmap=True)
    addresses=torch.load(str(row["address_path"]),map_location="cpu",weights_only=True,mmap=True)
    if any(v["name"]!=row["name"] or v["scene"]!=row["scene"] for v in (source,addresses)):
        raise ValueError("Immutable source/address payload identity differs")
    data=source["tensors"]
    with torch.no_grad():
        costs=build_joint_costs(data["lq"],data["base"],data["flow"],data["valid_mask"],
            addresses["prior_address"],addresses["pixel_address"],coefficients,global_variance=global_variance)
    tensors=[costs[k] for k in ("unary_observation","pair_observation","unary_constant","pair_constant")]
    tensors.extend(costs["descriptor_costs"].values())
    if any(not bool(torch.isfinite(v).all() & (v>=0).all()) for v in tensors):
        raise ValueError("Nonfinite or negative cached observable cost")
    filename=f"joint_{i:03d}.pt"
    with (Path(out)/filename).open("xb") as stream:
        torch.save({"name":row["name"],"scene":row["scene"],"costs":costs,
            "source_sha256":row["source_sha256"],"address_sha256":row["address_sha256"]},stream)
    return {**row,"cache_path":filename,"cache_sha256":digest(Path(out)/filename)}


def prepare(geometry_prepared,out,workers=4):
    import torch
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    torch.set_num_threads(4)
    if workers not in (1,4):
        raise ValueError("Use one or four deterministic CPU workers")
    meta=geometry_metadata(geometry_prepared)
    fit_rows=[r for r in meta["per_sequence"] if r["scene"] in meta["split"]["fit"]]
    if (len(meta["split"]["fit"])!=51 or len(fit_rows)!=102
            or any(sum(r["scene"]==s for r in fit_rows)!=2 for s in meta["split"]["fit"])):
        raise ValueError("Require exactly51 FIT scenes, two clips per scene")
    out=new_directory(out)
    calibration=fit_calibration(meta["per_sequence"],meta["split"]["fit"],lambda row:
        torch.load(str(row["source_path"]),map_location="cpu",weights_only=True,mmap=True)["tensors"])
    write_json(out/"calibration.json",calibration)
    arguments=[(i,row,str(out),calibration["coefficients"],calibration["global_effective_variance"])
               for i,row in enumerate(meta["per_sequence"])]
    entries=[]
    if workers==1:
        results=map(_cache_row,arguments)
        for entry in results:
            entries.append(entry)
            print(json.dumps({"cached":len(entries),"n":len(arguments),"name":entry["name"]}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            for entry in pool.map(_cache_row,arguments,chunksize=1):
                entries.append(entry)
                print(json.dumps({"cached":len(entries),"n":len(arguments),"name":entry["name"]}),flush=True)
    result={key:value for key,value in meta.items() if key not in ("per_sequence","source_sha256","zero_offset_parity_fit_only")}
    result.update({"schema_version":1,"kind":"joint_observation_search_costs","per_sequence":entries,
        "geometry_manifest_path":str((Path(geometry_prepared)/"manifest.json").resolve()),
        "geometry_manifest_sha256":digest(Path(geometry_prepared)/"manifest.json"),
        "calibration":calibration,"calibration_sha256":digest(out/"calibration.json"),
        "preparation_runtime":{"device":"CPU float32","workers":workers,"threads_per_worker":4,
            "manifest_order":"original source order regardless of worker completion"},
        "source_sha256":source_hashes(*SOURCE_FILES),"cost_gt_input":False,
        "cost_config":{"patch_size":PATCH,"offset":OFFSET,"shifts":list(SHIFTS),
            "candidate_order":"dy outer, dx inner; center4","shuffle_seed":SHUFFLE_SEED,
            "minimum_support_fraction":MIN_SUPPORT,"variance_floor":VARIANCE_FLOOR,
            "unary":"HP9(q_target) vs HP9(warp(q_source)), mean SSD / effective variance sum",
            "pair":"common valid pixel SSD of two warped donors / effective variance sum",
            "mask":"original valid intersect candidate geometric validity then HP9 support erosion",
            "descriptor":"cosine distance at patch center sampled using original flow plus candidate offset",
            "independence_assumption":False,"physical_camera_noise_claim":False}})
    write_json(out/"manifest.json",result)


def checked_metadata(prepared,verify_files=True):
    meta=json.loads((Path(prepared)/"manifest.json").read_text())
    check_source_hashes(meta["source_sha256"])
    upstream_path=Path(meta["geometry_manifest_path"])
    if digest(upstream_path)!=meta["geometry_manifest_sha256"]:
        raise ValueError("Frozen geometry manifest changed")
    upstream=geometry_metadata(upstream_path.parent,verify_files=verify_files)
    inherited={k:v for k,v in upstream.items() if k not in ("schema_version","per_sequence","source_sha256","zero_offset_parity_fit_only")}
    if (meta["kind"]!="joint_observation_search_costs" or meta["cost_gt_input"] is not False
            or meta["source_sha256"]!=source_hashes(*SOURCE_FILES)
            or meta["split"]!=upstream["split"] or meta["source_cache_provenance"]!=upstream["source_cache_provenance"]
            or meta["normalization"]!=upstream["normalization"] or meta["n"]!=136
            or len(meta["per_sequence"])!=136
            or digest(Path(prepared)/"calibration.json")!=meta["calibration_sha256"]
            or json.loads((Path(prepared)/"calibration.json").read_text())!=meta["calibration"]):
        raise ValueError("Invalid frozen joint cost manifest")
    if any(meta[key]!=value for key,value in inherited.items()):
        raise ValueError("Frozen source/reference metadata changed")
    fit_names=[r["name"] for r in upstream["per_sequence"] if r["scene"] in upstream["split"]["fit"]]
    if (meta["calibration"]["fit_names"]!=fit_names or meta["calibration"]["fit_scenes"]!=upstream["split"]["fit"]
            or meta["calibration"]["internal_holdout_labels_read"] is not False
            or meta["calibration"]["target_labels_read"] is not False):
        raise ValueError("Calibration split provenance differs")
    for row,original in zip(meta["per_sequence"],upstream["per_sequence"]):
        if any(row[key]!=value for key,value in original.items()):
            raise ValueError("Source/address/reference lineage changed")
        if verify_files and digest(Path(prepared)/row["cache_path"])!=row["cache_sha256"]:
            raise ValueError("Joint cost cache changed")
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--geometry-prepared",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--workers",type=int,choices=(1,4),default=4)
    args=parser.parse_args()
    prepare(args.geometry_prepared,args.out,args.workers)


if __name__=="__main__":
    main()
