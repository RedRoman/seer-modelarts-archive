"""Prespecified complete source-development joint-search gate; no model access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS=("base","frozen_raw_node_1000","joint_prior","independent_prior",
         "constant_prior","joint_pixel","joint_shuffled")
METRICS=("rgb_mse_float","psnr_rgb_from_clip_mse")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(raw):
    rows=raw["per_sequence"]
    scenes=sorted({r["scene"] for r in rows})
    if (len(rows)!=34 or len({r["name"] for r in rows})!=34 or len(scenes)!=17
            or raw["states_unchanged"] is not True or raw["target_frames_read"] is not False
            or raw["initial_states_and_plans_identical"] is not True):
        raise ValueError("Need complete34-clip/17-scene source results and immutable matched states")
    source={}
    for scene in scenes:
        pair=[r for r in rows if r["scene"]==scene]
        if len(pair)!=2:raise ValueError("Each scene needs two illumination conditions")
        source[scene]={}
        for method in METHODS:
            values=[]
            for row in pair:
                if set(row["methods"])!=set(METHODS):raise ValueError("Incomplete method coverage")
                mse,psnr=(row["methods"][method][m] for m in METRICS)
                if (not np.isfinite([mse,psnr]).all() or mse<=0
                        or abs(psnr+10*np.log10(mse))>1e-9):
                    raise ValueError("Invalid source RGB metrics")
                values.append([mse,psnr])
            source[scene][method]=np.mean(values,axis=0).tolist()
    draws=np.random.default_rng(0).integers(0,17,size=(10000,17))
    means={m:{metric:float(np.mean([source[s][m][j] for s in scenes]))
              for j,metric in enumerate(METRICS)} for m in METHODS}
    contrasts={}
    for control in METHODS:
        if control=="joint_prior":continue
        result={}
        for j,metric in enumerate(METRICS):
            delta=np.array([source[s]["joint_prior"][j]-source[s][control][j] for s in scenes])
            result[metric]={"mean":float(delta.mean()),
                "ci95":np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
                "per_scene":dict(zip(scenes,delta.tolist()))}
        contrasts[f"joint_prior_minus_{control}"]=result
    anchor=contrasts["joint_prior_minus_frozen_raw_node_1000"]
    unary=contrasts["joint_prior_minus_independent_prior"]
    checks={
        "psnr_gain_over_frozen_anchor_at_least_005db":anchor[METRICS[1]]["mean"]>=.05,
        "psnr_gain_over_unary_only_at_least_002db":unary[METRICS[1]]["mean"]>=.02,
        "mse_ci_better_than_frozen_anchor":anchor[METRICS[0]]["ci95"][1]<0,
        "mse_ci_better_than_unary_only":unary[METRICS[0]]["ci95"][1]<0,
    }
    secondary={
        "calibrated_mse_ci_better_than_constant":contrasts["joint_prior_minus_constant_prior"][METRICS[0]]["ci95"][1]<0,
        "prior_mse_ci_better_than_pixel":contrasts["joint_prior_minus_joint_pixel"][METRICS[0]]["ci95"][1]<0,
        "mean_mse_better_than_shuffled_prior":contrasts["joint_prior_minus_joint_shuffled"][METRICS[0]]["mean"]<0,
    }
    domains={}
    for domain in ("SDSD_indoor","SDSD_outdoor"):
        group=[s for s in scenes if s.startswith(domain+"/")]
        if not group:raise ValueError("Need both source domains")
        domains[domain]={"scenes":len(group),"means":{
            m:{metric:float(np.mean([source[s][m][j] for s in group]))
               for j,metric in enumerate(METRICS)} for m in METHODS}}
    return {"schema_version":1,"scope":raw["scope"],"scene_order":scenes,
        "metric_definition":raw["metric"],"bootstrap":{"unit":"whole source scene/two clips",
            "draws":10000,"rng":0,"ci":"unadjusted exploratory percentile95",
            "training_replication":"one new head seed and one historical base; no training uncertainty"},
        "means":means,"per_scene":source,"per_sequence":rows,"contrasts":contrasts,
        "domain_breakdown":domains,"gate_checks":checks,"secondary_component_checks":secondary,
        "advance_to_replication":all(checks.values()),"target_inference_allowed_by_this_result":False,
        "boundary":"Passing only prioritizes source replication; no target, cross-domain, novelty or SOTA claim"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();result=analyze(json.loads(a.input.read_text()))
    result["raw_sha256"]=sha(a.input);result["analysis_sha256"]=sha(__file__)
    with a.out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:result[k] for k in ("advance_to_replication","gate_checks",
        "secondary_component_checks","means")},indent=2))


if __name__=="__main__":main()
