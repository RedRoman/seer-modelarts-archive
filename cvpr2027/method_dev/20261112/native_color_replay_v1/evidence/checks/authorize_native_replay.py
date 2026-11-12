"""Bind the fixed two-arm compatibility replay to four successful source checks.

This is an internal experiment gate under the user's existing authorization.
It is not a request for additional user permission and uses no target scores.
"""
import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL_SHA="3423db39aa7792557320f3836f5dbf31b39069db19294c9485bc040898e29be4"
RELEASE_SHA="a3e11205edc7fd55497f9f358849ee93ae4ad45f1dbf3ebaf8d55ee07214a676"
COLOR_SHA="fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70"
CHECKPOINTS={"pretrained":"5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740",
    "none":"fa3c0c9e720a28b56f01069f7a2cd407bc5b4d752a2a1d4482f1bd63557b5516"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize(protocol,release_manifest,source,color_receipt):
    assert digest(protocol)==PROTOCOL_SHA and digest(release_manifest)==RELEASE_SHA
    assert digest(color_receipt)==COLOR_SHA
    closed=json.loads(Path(color_receipt).read_text())
    expected={r["name"]:r["canonical_lq_tensor_sha256"] for r in closed["clip_checks"]}
    assert len(expected)==2
    parsed={}
    for arm,path in source.items():
        r=json.loads(Path(path).read_text());parsed[arm]=r
        assert r["passed"] is True and r["stage"]=="source" and r["arm"]==arm
        assert r["n"]==2 and r["state_unchanged"] is True
        assert r["target_pixel_reads"]==r["gt_storage_reads"]==0
        assert r["quality_scores_computed"] is False
        assert r["protocol_sha256"]==PROTOCOL_SHA and r["release_manifest_sha256"]==RELEASE_SHA
        assert r["checkpoint_sha256"]==CHECKPOINTS[arm] and r["color_receipt_sha256"]==COLOR_SHA
        assert {v["name"] for v in r["per_sequence"]}==set(expected)
        for v in r["per_sequence"]:
            assert v["wrapper_input_exact"] is True and v["wrapper_output_permutation_exact"] is True
            assert v["native_lq_sha256"]==expected[v["name"]]
            p=v["old_base_parity"]
            assert p["passed"] is True and p["max_abs_limit"]==1e-4 and p["mean_abs_limit"]==1e-5
            assert p["max_abs"]<=1e-4 and p["mean_abs"]<=1e-5
    assert set(parsed)==set(CHECKPOINTS)
    return {"authorized_stage":"target_bvi_val16","protocol_sha256":PROTOCOL_SHA,
        "release_manifest_sha256":RELEASE_SHA,"source_parity_sha256":{a:digest(p) for a,p in source.items()},
        "source_paths":{a:str(Path(p).resolve()) for a,p in source.items()},
        "source_color_receipt_sha256":COLOR_SHA,"authorization_script_sha256":digest(__file__),
        "source_compatibility_checks_passed":4,"cross_arm_lq_hashes_equal_closed_raw_audit":True,
        "arms":["pretrained","none"],"target":"BVI VAL16, center16/crop256, same RGB scorer",
        "fitting_or_color_selection":False,"target_scores_used_to_open_gate":False,
        "user_scope":"Continue project experiments; root reviewed this exact fixed protocol and both successful source receipts",
        "limits":"Compatibility repair only; neither module novelty nor an isolated causal estimate of prior pretraining. No TEST or further arms."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","release-manifest","pretrained","none","color-receipt","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    record=authorize(args.protocol,args.release_manifest,{"pretrained":args.pretrained,"none":args.none},args.color_receipt)
    with args.out.open("x") as stream:json.dump(record,stream,indent=2);stream.write("\n")
    print(json.dumps(record,indent=2))
