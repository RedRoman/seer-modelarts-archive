CP-transfer experiment (reviewer question: does Bayesian Chroma Preservation
help strong baselines as much as it helps Seer, or is it Seer-specific?)

Files:
  cp_module.py                      -- CP reimplementation from Eq.5-6 of the paper
                                        (03_method.tex sec:method:cp), from scratch
                                        (no original CP code survived on this host).
  run_cp_transfer_retinexformer.py  -- fits alpha* on 15 SDSD-indoor validation
                                        pairs using Retinexformer's own predictions,
                                        then applies CP to Retinexformer's full
                                        L2 BVI-RLV / L3 LOL-v1 / L4 LOLv2-Syn /
                                        L5 LOLv2-Real test-set predictions.
  run_cp_transfer_darkir.py         -- same, for DarkIR.
  seer_gpu_memory_v2.py             -- real torch.cuda.max_memory_allocated()
                                        measurement for Seer's 16-frame 512x512
                                        forward pass, using the actual
                                        v35_stage4/final.pt checkpoint (step 6000).

Results:
  retinexformer_cp_transfer.json, run_log.txt
  darkir_cp_transfer.json,        run_log_darkir.txt
  seer_gpu_memory.json

Checkpoints used (both match Table 1's Retinexformer/DarkIR rows exactly --
verified: raw PSNR/LPIPS numbers reproduced to 3 decimal places against
baselines/REPORT_v2.md):
  Retinexformer: /cache/zihaohe/baseline_ckpts/Retinexformer/SDSD_indoor.pth
                 (official weight, used cross-domain on L2-L5, same as Table 1)
  DarkIR:        /cache/zihaohe/baseline_ckpts/DarkIR/DarkIR_384.pt
                 (official LOL-Blur weight, only one reachable; used
                 cross-domain on all splits, same as Table 1)

Validation pool for alpha* fitting (N=15 SDSD-indoor sequences, held out from
the STCD-12 test split used for L0/Table-1, drawn from the 62 non-test SDSD
indoor sequences, index-position paired LQ/GT -- some "_2" variant sequences
have offset LQ/GT filenames, same convention as eval_lib.py's
iter_sdsd_outdoor):
  pair1, pair16_2, pair1_2, pair2, pair24, pair26, pair27_2, pair31_2,
  pair34, pair34_2, pair4_2, pair5, pair8, pair9, pair9_2
  (all frames of each sequence used, not just one frame per pair, for a
  richer empirical 2nd-moment estimate -- Seer's own alpha description does
  not specify whether it used all frames or single frames per pair; we
  used full sequences to be conservative/well-powered).

Bottom line: alpha* for both Retinexformer and DarkIR came out small-and-
NEGATIVE on all 3 channels (opposite sign from Seer's positive
(0.50,0.50,0.68)), and applying it moves PSNR/LPIPS by <0.07 dB / <0.005
LPIPS on every OOD split for both baselines -- i.e. CP's benefit does NOT
transfer to these two strong non-Seer baselines under a fair, independently-
fitted alpha. See full report in the conversation for interpretation.
