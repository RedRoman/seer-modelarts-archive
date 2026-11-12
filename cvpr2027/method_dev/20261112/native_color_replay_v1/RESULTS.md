# Native-color compatibility replay completed

Both fixed source arms passed before target execution: all four FIT predictions
matched their historical base caches exactly (max/mean absolute error zero),
and the whole-pipeline wrapper's input/output permutations were bitwise exact.
Pixel state/reconstruction matched original provenance; video-encoder loading
had zero missing keys and only the explicitly documented two unused image
patch-embedding keys. All states remained unchanged. Ten real CPU tests passed.

The root-owned exact-protocol authorization gate then passed. Both predeclared
BVI VAL16 runs completed, using canonical RGB PNG inputs/metrics and BGR around
the entire historical pixel+encoder pipeline, center16/crop256, BF16 forward,
float32 clamp and the original rounded-uint8 scorer. There was no fitting,
exposure alignment, checkpoint/arm selection, selector, RAFT computation, TEST
access or saved predicted RGB.

| Fixed base | PSNR-Y | PSNR-RGB | LPIPS | tLP gap |
|---|---:|---:|---:|---:|
| Pretrained V-JEPA | 19.330874 | 18.041200 | 0.265640 | 0.004970 |
| No tokens | 18.250729 | 17.247471 | 0.282623 | 0.006316 |

This closes the historical Seer interface ambiguity. It does not establish a
new module, SOTA performance or an isolated pretraining effect: the original
pixel initial states were not fully matched. The corrected pretrained base
remains below the previously replayed HVI reference (24.241232 PSNR-Y), so the
strong spatial anchor investigation remains necessary.

P minus N is +1.080144 PSNR-Y / +0.793729 PSNR-RGB, with the initialization
confound above. Against the historical unwrapped P base (19.284931 Y /
18.357351 RGB / 0.244829 LPIPS), corrected P changes Y by +0.045942 and RGB
by -0.316150; LPIPS increases by 0.020811. This is a contract repair, not an
overall quality improvement, and the small Y change is not a positive method
claim. All fixed arms and metrics are retained.

Independent receipt/scalar verification passed 188 checks with zero error in
reported means. Each arm's 32 uint8 THWC LQ/GT input hashes exactly matched all
three existing baseline runs: 192 comparisons (the original 96 checked once
for each arm). All GT tLP values matched exactly. The two arms also shared the
same float32 input/GT metric hashes. LPIPS sources and weights remained unchanged.
This verification used existing hashes/scalars and produced no new predictions.

Protocol SHA: `3423db39aa7792557320f3836f5dbf31b39069db19294c9485bc040898e29be4`.
Release manifest: `a3e11205edc7fd55497f9f358849ee93ae4ad45f1dbf3ebaf8d55ee07214a676`.
P result: `5ac59351405f0ce278124e224912a2b2e400cc61bfa17b46aa1a656cfaed14c8`.
N result: `c5d5af0fcc0c33e27dba5fa5c34d709f9a3b0313afeff9f4e9d40ae929241239`.
Execution root: `/cache/zihaohe/seer_native_color_replay_20261112` on 31051.

Preparation v1's missing N tensors was a metadata-location failure before any
prediction. Its original directory is preserved. V2 copied only the two fixed
original N cache files from producer 32193 and verified original SHA values;
the frozen inference algorithm and repair thresholds were not relaxed.

Final SFS evidence: `/temp/zihaohe/seer_cvpr/method_dev_20261112/native_color_replay_v1_final_v2`.
Backup manifest SHA: `3fc7da1fd312bc0e48ec438aa0850947eed578bfdcea60991f2a96b37c29629a`.
The downloaded 46-file UTF-8 tar is 768000 bytes, SHA
`266adf92bacff15b1bbd7be175f7f35e1fd74068a3d5a96848dee6421679a685`.
Nonexecuted AppleDouble metadata in the original source bundle is listed by
hash and omitted from the text backup. An initial backup attempt stopped at
its UTF-8 guard; the partial directory was retained, with scoring unchanged.
