"""Independent NumPy/multiplicity verification of real source-screen results.

Imports neither the primary analyzer nor any runtime/model code. No synthetic
scores are emitted. --raw and --analysis must be completed actual artifacts.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


METHODS = ("base", "frozen_raw_node_1000", "joint_prior", "independent_prior",
           "constant_prior", "joint_pixel", "joint_shuffled")
METRICS = ("rgb_mse_float", "psnr_rgb_from_clip_mse")
MODES = {"joint_prior": ("joint", "prior", "calibrated"),
         "independent_prior": ("independent", "prior", "calibrated"),
         "constant_prior": ("joint", "prior", "constant"),
         "joint_pixel": ("joint", "pixel", "calibrated"),
         "joint_shuffled": ("joint", "shuffled_prior", "calibrated")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_scene(name):
    require(isinstance(name, str) and re.fullmatch(r"SDSD_(indoor|outdoor)/[^/\\]+", name),
            "Source sequence name is invalid")
    require(name.split("/")[1] not in (".", ".."), "Sequence path is invalid")
    return re.sub(r"_2$", "", name)


def validate(raw):
    require(raw["states_unchanged"] is True and raw["initial_states_and_plans_identical"] is True
            and raw["target_frames_read"] is False, "Immutable source-only flags are missing")
    require(set(raw["provenance"]) == set(MODES) == set(raw["checkpoint_sha256"]),
            "All five actual modes must be present")
    require(all(valid_hash(x) for x in raw["checkpoint_sha256"].values()), "Invalid checkpoint hashes")
    origin = raw["source_sha256"]
    required_sources = {"models/joint_observation_search.py", "tools/train_joint_observation_search.py",
                        "tools/cache_joint_observation_search.py", "models/observation_geometry.py"}
    require(required_sources <= set(origin) and all(valid_hash(v) for v in origin.values()),
            "Required runtime source hashes are missing")
    for key in ("prepared_manifest_sha256", "calibration_sha256", "reference_checkpoint_sha256",
                "reference_state_sha256"):
        require(valid_hash(raw[key]), "Invalid raw lineage: " + key)
    first = raw["provenance"]["joint_prior"]
    split = first["split"]
    fit, held = split["fit"], split["internal_holdout"]
    require(len(fit) == len(set(fit)) == 51 and len(held) == len(set(held)) == 17
            and not set(fit) & set(held), "Expected 51/17 disjoint source scenes")
    scenes = sorted(fit + held)
    require(all(source_scene(s) == s for s in scenes), "Split entries must be scene names")
    require(Counter(s.split("/")[0] for s in scenes) == {"SDSD_indoor":23,"SDSD_outdoor":45},
            "Source TRAIN scene inventory differs")
    order = np.random.default_rng(20261114).permutation(68)
    require(split["seed"] == 20261114 and fit == [scenes[i] for i in order[:51]]
            and held == [scenes[i] for i in order[51:]], "Source split differs from fixed seed/order")
    common_fields = ("initial_state_sha256", "sample_plan_sha256", "prepared_manifest_sha256",
                     "calibration_sha256", "reference_head_sha256", "frozen_node_state_sha256",
                     "fit_only_mse_normalizer", "args", "split", "source_sha256")
    for mode, expected in MODES.items():
        p = raw["provenance"][mode]
        require(p["mode"] == mode and p["seed"] == 0 and p["steps"] == 400
                and p["trainable_parameters"] == 4 and p["frozen_parameters"] == 6626,
                "Incorrect fixed mode/seed/budget/parameter count")
        require(all(p[k] == first[k] for k in common_fields), "Matched provenance differs")
        require(p["source_sha256"] == origin
                and p["prepared_manifest_sha256"] == raw["prepared_manifest_sha256"]
                and p["calibration_sha256"] == raw["calibration_sha256"]
                and p["reference_head_sha256"] == raw["reference_checkpoint_sha256"],
                "Runtime/cache/reference lineage differs")
        require(all(valid_hash(p[k]) for k in ("initial_state_sha256", "sample_plan_sha256",
                "frozen_node_state_sha256", "initial_file_sha256")), "Invalid matched source hashes")
        conf = p["model_config"]
        require(tuple(conf[k] for k in ("search_mode", "address_mode", "uncertainty_mode")) == expected,
                "Model configuration is mislabeled")
        require(conf["normalization"] == first["model_config"]["normalization"]
                and conf["search_patch"] == 16 and conf["offset"] == 2 and conf["max_shift"] == 2,
                "Fixed reconstruction configuration differs")
        require(conf["amplitude_rule"] == "tanh_projected_nonnegative_zero_init"
                and conf["confidence_rule"] == "one_minus_marginal_entropy_over_log_valid_support",
                "Search rules differ")
        require(math.isfinite(p["fit_only_mse_normalizer"]) and p["fit_only_mse_normalizer"] > 0,
                "Invalid fit-only loss scale")
        require(p["args"]["lr"] == .01 and p["args"]["forward_context"] == 256
                and p["args"]["loss_crop"] == 128 and p["args"]["clip_len"] == 16
                and p["args"]["loss_frames"] == 8 and p["args"]["clips_per_update"] == 1,
                "Training budget/configuration differs")
    rows = raw["per_sequence"]
    expected_names = {s + suffix for s in held for suffix in ("", "_2")}
    require(len(rows) == 34 and {r["name"] for r in rows} == expected_names,
            "Need exactly both source passes from each of the 17 held scenes")
    for row in rows:
        require(row["scene"] == source_scene(row["name"]) and row["scene"] in held,
                "Source scene/name or FIT/holdout assignment differs")
        require(set(row["methods"]) == set(METHODS), "Incomplete result method coverage")
        for name in METHODS:
            mse, psnr = (row["methods"][name][key] for key in METRICS)
            require(math.isfinite(mse) and math.isfinite(psnr) and mse > 0
                    and abs(psnr + 10*math.log10(mse)) < 1e-9, "Invalid float-RGB score")
    return sorted(held)


def multiplicities(scenes=17, samples=10000):
    """Same declared RNG draws, independently evaluated as a count matrix."""
    indices = np.random.default_rng(0).integers(scenes, size=(samples, scenes))
    weights = np.zeros((samples, scenes), dtype=np.int64)
    np.add.at(weights, (np.repeat(np.arange(samples), scenes), indices.reshape(-1)), 1)
    return weights.astype(np.float64) / scenes


def recompute(raw, scenes):
    values = np.empty((len(scenes), len(METHODS), len(METRICS)), dtype=np.float64)
    for i, scene in enumerate(scenes):
        pair = sorted((r for r in raw["per_sequence"] if r["scene"] == scene), key=lambda r:r["name"])
        require(len(pair) == 2, "Scene does not have two passes")
        for j, method in enumerate(METHODS):
            for k, metric in enumerate(METRICS):
                values[i,j,k] = math.fsum(row["methods"][method][metric] for row in pair) / 2
    means = {m:{key:float(math.fsum(values[:,j,k])/len(scenes))
                for k,key in enumerate(METRICS)} for j,m in enumerate(METHODS)}
    per_scene = {s:{m:values[i,j].tolist() for j,m in enumerate(METHODS)} for i,s in enumerate(scenes)}
    weights = multiplicities()
    contrasts = {}
    prior_index = METHODS.index("joint_prior")
    for j, method in enumerate(METHODS):
        if method == "joint_prior":continue
        difference = values[:,prior_index,:] - values[:,j,:]
        boot = weights @ difference
        ci = np.quantile(boot, [.025,.975], axis=0, method="linear")
        contrasts["joint_prior_minus_"+method] = {
            metric:{"mean":float(math.fsum(difference[:,k])/len(scenes)),
                    "ci95":ci[:,k].tolist(),"per_scene":dict(zip(scenes,difference[:,k].tolist()))}
            for k,metric in enumerate(METRICS)}
    anchor = contrasts["joint_prior_minus_frozen_raw_node_1000"]
    independent = contrasts["joint_prior_minus_independent_prior"]
    gate = {"psnr_gain_over_frozen_anchor_at_least_005db":anchor[METRICS[1]]["mean"] >= .05,
            "psnr_gain_over_unary_only_at_least_002db":independent[METRICS[1]]["mean"] >= .02,
            "mse_ci_better_than_frozen_anchor":anchor[METRICS[0]]["ci95"][1] < 0,
            "mse_ci_better_than_unary_only":independent[METRICS[0]]["ci95"][1] < 0}
    secondary = {"calibrated_mse_ci_better_than_constant":contrasts["joint_prior_minus_constant_prior"][METRICS[0]]["ci95"][1] < 0,
                 "prior_mse_ci_better_than_pixel":contrasts["joint_prior_minus_joint_pixel"][METRICS[0]]["ci95"][1] < 0,
                 "mean_mse_better_than_shuffled_prior":contrasts["joint_prior_minus_joint_shuffled"][METRICS[0]]["mean"] < 0}
    domains = {}
    for domain in ("SDSD_indoor","SDSD_outdoor"):
        subset = [i for i,s in enumerate(scenes) if s.startswith(domain+"/")]
        require(bool(subset), "Both source domains must be present")
        domains[domain] = {"scenes":len(subset),"means":{
            m:{metric:float(math.fsum(values[subset,j,k])/len(subset))
               for k,metric in enumerate(METRICS)} for j,m in enumerate(METHODS)}}
    return {"scene_order":scenes,"means":means,"per_scene":per_scene,"contrasts":contrasts,
            "domain_breakdown":domains,"gate_checks":gate,"secondary_component_checks":secondary,
            "advance_to_replication":all(gate.values()),"target_inference_allowed_by_this_result":False}


def compare(expected, actual):
    errors, count = [], 0
    def walk(left, right, path):
        nonlocal count
        if isinstance(left, dict):
            require(isinstance(right, dict) and set(left) == set(right), "Key mismatch: "+path)
            for key in left:walk(left[key],right[key],path+"/"+key)
        elif isinstance(left, list):
            require(isinstance(right, list) and len(left) == len(right), "List mismatch: "+path)
            for i,(x,y) in enumerate(zip(left,right)):walk(x,y,path+"/"+str(i))
        elif isinstance(left, (float,int)) and not isinstance(left,bool):
            require(isinstance(right,(float,int)) and not isinstance(right,bool)
                    and math.isfinite(right), "Numeric type/value mismatch: "+path)
            difference=abs(left-right);errors.append((difference,path));count+=1
            require(math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12),
                    f"Independent mismatch {path}: {left!r} vs {right!r}")
        else:require(type(left) is type(right) and left == right, "Value mismatch: "+path)
    for key,value in expected.items():walk(value,actual[key],key)
    largest=max(errors,default=(0,None))
    return {"numeric_scalars_checked":count,"max_absolute_error":largest[0],"max_error_path":largest[1]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--analysis",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--analysis-source",type=Path,default=Path(__file__).resolve().parents[1]/"analyze_screen.py")
    args=p.parse_args()
    raw=json.loads(args.raw.read_text());analysis=json.loads(args.analysis.read_text())
    require(analysis["raw_sha256"] == digest(args.raw),"Analysis raw SHA differs")
    require(analysis["analysis_sha256"] == digest(args.analysis_source),"Primary analyzer SHA differs")
    scenes=validate(raw)
    require(analysis["per_sequence"] == raw["per_sequence"],"Analyzer changed raw sequence records")
    expected=recompute(raw,scenes)
    report=compare(expected,analysis)
    report.update({"passed":True,"raw_sha256":digest(args.raw),"analysis_result_sha256":digest(args.analysis),
                   "primary_analyzer_sha256":digest(args.analysis_source),"verifier_sha256":digest(__file__),
                   "method":"independent source pairing; fsum scene means; 10000x17 bootstrap multiplicity matrix",
                   "rng":0,"scene_count":17,"sequence_count":34,"ci_endpoints_checked":24,
                   "gate_checks":expected["gate_checks"],"secondary_component_checks":expected["secondary_component_checks"],
                   "advance_to_replication":expected["advance_to_replication"],
                   "target_inference_allowed":False,"primary_analyzer_imported":False})
    with args.out.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
