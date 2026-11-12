"""Independent result arithmetic/lineage check; reads no tensor or image pixels."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

METHODS=("hvi","temporal_mean_lambda_0.25","temporal_mean_lambda_0.5","temporal_mean_lambda_1",
    "sameframe_identity_lambda_0.25","sameframe_identity_lambda_0.5","sameframe_identity_lambda_1")
RECEIPT_SHA="fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70"
GEOMETRY_SHA="d7f90214ab2da0b30da9f3f5b9fba648bb4fcabfee548e34dd7788caa6610805"
OLD_RESULT_SHA="1984d5db1ded47f14b9bafc4d8fd8a9e3d8d4f7fc68d7b2b689fd31faf24b0e4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(result_directory,old_result,receipt):
    root=Path(result_directory)
    raw_path=root/"fit_results.json";execution_path=root/"execution_manifest.json"
    result=json.loads(raw_path.read_text());execution=json.loads(execution_path.read_text())
    assert result["manifest_sha256"]==digest(execution_path)
    for path,sha in execution["source_sha256"].items():assert digest(path)==sha
    lineage=execution["lineage"]
    assert digest(receipt)==lineage["bgr_evidence_sha256"]==RECEIPT_SHA
    closed=json.loads(Path(receipt).read_text())
    assert closed["verified"] is True and closed["native_source_color"]=="BGR"
    assert digest(lineage["geometry_manifest_path"])==lineage["geometry_manifest_sha256"]==GEOMETRY_SHA
    assert digest(lineage["hvi_manifest_path"])==lineage["hvi_manifest_sha256"]
    hvi=json.loads(Path(lineage["hvi_manifest_path"]).read_text())
    assert hvi["complete"] is True and hvi["output_color"]==hvi["input_color"]=="RGB"
    assert hvi["native_source_color"]=="BGR" and hvi["model_state_unchanged"] is True
    by_name={r["name"]:r for r in hvi["per_sequence"]}
    for r in closed["clip_checks"]:
        assert by_name[r["name"]]["rgb_lq_tensor_sha256"]==r["official_rgb_tensor_sha256"]
        # This legacy audit key names a stored-order tensor, not canonical RGB.
        assert by_name[r["name"]]["native_lq_tensor_sha256"]==r["canonical_lq_tensor_sha256"]
    assert tuple(execution["protocol"]["fixed_methods"])==METHODS
    assert execution["protocol"]["lambda_selection"] is False
    assert execution["internal_holdout_tensor_archives_opened"]==0
    assert all(execution[k] is False for k in ("internal_holdout_scores_computed","target_inputs_read",
        "optimizer_or_fitting","prediction_or_gt_arrays_written"))
    assert all(result[k] is False for k in ("internal_holdout_scores_computed","target_scores_computed",
        "lambda_selection_performed","oracle_computed"))
    rows=result["per_sequence"];fit=set(execution["split"]["fit"]);excluded=set(execution["split"]["internal_holdout"])
    assert len(rows)==result["n_clips"]==102 and len(fit)==result["n_scenes"]==51 and len(excluded)==17
    assert not fit&excluded and [r["name"] for r in rows]==execution["fit_names"]
    assert set(Counter(r["scene"] for r in rows).values())=={2}
    assert set(r["scene"] for r in rows)==fit
    assert digest(old_result)==OLD_RESULT_SHA
    old={r["name"]:r for r in json.loads(Path(old_result).read_text())["per_sequence"]}
    checks=0;max_error=0.
    def compare(a,b):
        nonlocal checks,max_error
        assert math.isfinite(a) and math.isfinite(b)
        error=abs(a-b);max_error=max(max_error,error);checks+=1
        assert error<5e-12,(a,b,error)
    for row in rows:
        assert row["gt_color"]==row["hvi_output_color"]=="RGB"
        assert set(row["methods"])==set(METHODS) and row["sameframe_exact_anchor"] is True
        assert row["coverage"]==old[row["name"]]["coverage"]
        for method in METHODS:
            score=row["methods"][method]
            compare(-10*math.log10(score["rgb_mse_float"]),score["psnr_rgb_from_clip_mse"])
        for method in METHODS[4:]:assert row["methods"][method]==row["methods"]["hvi"]
    means={};scene_rows={r["scene"]:r for r in result["per_scene"]}
    assert set(scene_rows)==fit
    for method in METHODS:
        means[method]={"mean_rgb_mse":math.fsum(r["methods"][method]["rgb_mse_float"] for r in rows)/102,
            "mean_clip_psnr_rgb":math.fsum(r["methods"][method]["psnr_rgb_from_clip_mse"] for r in rows)/102}
        for key,value in means[method].items():
            compare(value,result["mean"][method][key])
            compare(value-means["hvi"][key],result["mean_difference_vs_hvi"][method][key])
    improvement_counts={m:0 for m in METHODS}
    for scene in sorted(fit):
        pair=[r for r in rows if r["scene"]==scene];saved=scene_rows[scene]
        assert len(pair)==saved["clips"]==2
        values={m:{k:math.fsum(r["methods"][m][k] for r in pair)/2
            for k in ("rgb_mse_float","psnr_rgb_from_clip_mse")} for m in METHODS}
        for method in METHODS:
            improvement_counts[method]+=int(values[method]["rgb_mse_float"]<values["hvi"]["rgb_mse_float"])
            for key,value in values[method].items():
                compare(value,saved["methods"][method][key])
                compare(value-values["hvi"][key],saved["difference_vs_hvi"][method][key])
    return {"passed":True,"result_sha256":digest(raw_path),"execution_sha256":digest(execution_path),
        "verifier_sha256":digest(__file__),"scalar_checks":checks,"max_scalar_abs_error":max_error,
        "fit_clips":102,"fit_scenes":51,"methods":list(METHODS),"mean_recomputed":means,
        "scenes_with_lower_mean_rgb_mse":improvement_counts,"identity_controls_exact":True,
        "unchanged_geometry_coverage_exact_for_all_fit_clips":True,
        "two_original_fit_clip_rgb_input_hashes_equal_official_reader":True,
        "tensor_or_image_pixels_read":0,"new_quality_scores_computed":0,
        "limits":"Arithmetic and provenance only; full payload checks are in the export/FIT runtime receipts. Fixed FIT diagnostics do not establish learned-module innovation or cross-domain performance."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("result-directory","old-result","receipt","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();report=verify(args.result_directory,args.old_result,args.receipt)
    with args.out.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(report,indent=2))
