"""Prespecified complete source-development gate; no model or target access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("base", "prior", "pixel", "shuffled_prior", "frozen_raw_node_1000")
METRICS = ("rgb_mse_float", "psnr_rgb_from_clip_mse")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(raw):
    rows = raw["per_sequence"]
    scenes = sorted({r["scene"] for r in rows})
    if (len(rows) != 34 or len({r["name"] for r in rows}) != 34 or len(scenes) != 17
            or raw["states_unchanged"] is not True or raw["target_frames_read"] is not False):
        raise ValueError("Need complete34-clip/17-scene source results, unchanged states and no target")
    source = {}
    for scene in scenes:
        pair = [r for r in rows if r["scene"] == scene]
        if len(pair) != 2:
            raise ValueError("Each source scene must have exactly two conditions")
        source[scene] = {}
        for method in METHODS:
            value = []
            for r in pair:
                if set(r["methods"]) != set(METHODS):
                    raise ValueError("Source method coverage differs")
                v = r["methods"][method]
                mse, psnr = (v[m] for m in METRICS)
                if (not np.isfinite([mse,psnr]).all() or mse <= 0
                        or abs(psnr + 10*np.log10(mse)) > 1e-9):
                    raise ValueError("Invalid float source MSE/clip-PSNR")
                value.append([mse, psnr])
            source[scene][method] = np.mean(value,axis=0).tolist()
    draws = np.random.default_rng(0).integers(0,17,size=(10000,17))
    means = {m: {metric:float(np.mean([source[s][m][j] for s in scenes]))
                 for j,metric in enumerate(METRICS)} for m in METHODS}
    contrasts = {}
    for control in METHODS:
        if control == "prior":
            continue
        result = {}
        for j,metric in enumerate(METRICS):
            delta=np.array([source[s]["prior"][j]-source[s][control][j] for s in scenes])
            interval=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist()
            result[metric]={"mean":float(delta.mean()),"ci95":interval,
                            "per_scene":dict(zip(scenes,delta.tolist()))}
        contrasts[f"prior_minus_{control}"]=result
    c=contrasts
    checks={
        "psnr_gain_over_frozen_anchor_at_least_005db":c["prior_minus_frozen_raw_node_1000"][METRICS[1]]["mean"]>=.05,
        "psnr_gain_over_pixel_at_least_002db":c["prior_minus_pixel"][METRICS[1]]["mean"]>=.02,
        "mse_ci_better_than_frozen_anchor":c["prior_minus_frozen_raw_node_1000"][METRICS[0]]["ci95"][1]<0,
        "mse_ci_better_than_pixel":c["prior_minus_pixel"][METRICS[0]]["ci95"][1]<0,
        "mean_mse_better_than_shuffled_prior":c["prior_minus_shuffled_prior"][METRICS[0]]["mean"]<0,
    }
    breakdown={}
    for domain in ("SDSD_indoor","SDSD_outdoor"):
        group=[s for s in scenes if s.startswith(domain+"/")]
        if not group:
            raise ValueError("Both source domains must be present")
        breakdown[domain]={"scenes":len(group),"means":{
            m:{metric:float(np.mean([source[s][m][j] for s in group]))
               for j,metric in enumerate(METRICS)} for m in METHODS}}
    return {"schema_version":1,"scope":raw["scope"],"scene_order":scenes,
        "metric_definition":raw["metric"],"bootstrap":{"unit":"whole source scene/two clips",
            "draws":10000,"rng":0,"ci":"unadjusted exploratory percentile95",
            "training_replication":"one new head seed and one historical base; no training uncertainty"},
        "means":means,"per_scene":source,"per_sequence":rows,"contrasts":contrasts,
        "domain_breakdown":breakdown,"gate_checks":checks,"advance_to_replication":all(checks.values()),
        "target_inference_allowed_by_this_result":False,
        "boundary":"Passing only prioritizes replication/stronger source controls; this is not a cross-domain or novelty result"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=analyze(json.loads(a.input.read_text()))
    result["raw_sha256"]=sha(a.input)
    result["analysis_sha256"]=sha(__file__)
    with a.out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"advance_to_replication":result["advance_to_replication"],
                      "gate_checks":result["gate_checks"],"means":result["means"]},indent=2))


if __name__=="__main__":main()
