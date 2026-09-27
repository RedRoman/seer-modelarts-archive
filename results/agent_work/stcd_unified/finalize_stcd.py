import json
src = "/cache/zihaohe/stcd_unified/STCD_unified_eval.json"
dst = "/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/STCD_unified_eval.json"
d = json.load(open(src))
d["notes"] = [
    "USE L0_SDSD-indoor.clean6 for the Seer comparison (same iterator Seer uses; LQ/GT paired by sorted index).",
    "WARNING: eval_lib.iter_sdsd_indoor_stcd pairs LQ/GT by file basename. For pair50/52/58/60/68/70 the "
    "input/ and GT/ filenames are offset (e.g. pair60 LQ 0032.. vs GT 0054..) while content is aligned by "
    "sorted index (NCC of 512 centre crop, mid frame: index pairing 0.895/0.912/0.983/0.927/0.984/0.944 vs "
    "basename pairing 0.846/0.812/0.700/0.440/0.470/0.860). So STCD-12 numbers for those 6 seqs are scored "
    "against misaligned GT (this affects every baseline scored with iter_sdsd_indoor_stcd, e.g. all get ~8 dB "
    "on pair68). STCD-12_SDSD-indoor.clean6 is therefore NOT meaningful; STCD-12_indexpaired_derived replaces "
    "those 6 by their L0 (index-paired) entries.",
    "STCD is resolution/crop sensitive: pair50 first-16 index-paired PSNR 30.41 (512 crop) -> 26.24 (640) -> "
    "21.27 (full 1920x1080).",
]
d["sanity_vs_old"] = {
    "old_comprehensive_eval_json": {"pair13": 22.111, "pair50": 29.315},
    "unified_this_run": {"pair13": d["per_seq"]["STCD-12_SDSD-indoor"]["pair13"]["psnr_rgb"],
                         "pair50_STCD12_basename": d["per_seq"]["STCD-12_SDSD-indoor"]["pair50"]["psnr_rgb"],
                         "pair50_L0_index": d["per_seq"]["L0_SDSD-indoor"]["pair50"]["psnr_rgb"]},
    "old_stcd_repro_STCDexact_first30_960x512": {"pair13": 24.327, "pair50": 22.956},
    "reproduced_STCDexact_first30_960x512_basename": {"pair13": 24.327, "pair50": 22.956},
    "first16_fullres_1920x1080": {"pair13": 22.097, "pair50_basename": 19.527, "pair50_index": 21.269},
    "first16_crop512": {"pair13": 27.076, "pair50_basename": 28.343, "pair50_index": 30.407},
    "first16_tiled512_ov128_fullframe": {"pair13": 20.942, "pair50_basename": 17.068, "pair50_index": 17.780},
    "rgb_bgr": {"pair13_bgr_in(official)": 27.104, "pair13_rgb_in": 26.828,
                "pair50_bgr_in(official)": 24.040, "pair50_rgb_in": 23.848},
    "logs": ["/cache/zihaohe/stcd_unified/sanity.log", "/cache/zihaohe/stcd_unified/sanity2.log",
             "/cache/zihaohe/stcd_unified/sanity3.log"],
}
with open(dst, "w") as f:
    json.dump(d, f, indent=1)
print("wrote", dst)
