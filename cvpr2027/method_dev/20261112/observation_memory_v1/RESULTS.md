# Observation-memory source screen: fixed gate failed

The five declared seed0 heads completed all400 updates. `prior_cross` reached
28.499223dB mean source-internal RGB PSNR, versus28.564868dB for the frozen
base and28.496305dB for matched `pixel_cross`. The advance gate failed; this
recipe was not evaluated on BVI or TEST. These are module-development scores
on34 cached source clips/17 scenes already seen by the historical base.

| Method | Mean clip RGB PSNR (dB) | Mean RGB MSE |
|---|---:|---:|
| Frozen base | 28.564868 | 0.002290923952 |
| prior_cross | 28.499223 | 0.002297622062 |
| pixel_cross | 28.496305 | 0.002298203904 |
| shuffled_cross | 28.496742 | 0.002299325952 |
| prior_same | 28.495484 | 0.002299055359 |
| pixel_local | 28.507021 | 0.002297623281 |
| Frozen raw_node reference (1000 updates) | 28.715434 | 0.002252018721 |

The prespecified paired17-scene bootstrap used10000 draws/RNG0. Prior versus
pixel was +0.002918dB [0.000315,0.005893], below the +0.05dB threshold;
the corresponding MSE interval crossed zero. Prior versus base was
−0.065646dB [−0.131453,−0.006947]. Prior versus the preceding raw_node
reference was −0.216211dB [−0.315934,−0.125259]. The latter is a practical
strong reference with a different architecture and1000-update budget,
not a matched400-update ablation. All modes, scenes and contrasts remain in
[the raw output](results/source_internal_raw.json) and
[the frozen analysis](results/source_internal_analysis.json).

The metric is float RGB MSE over all16 frames of each cached256×256 clip,
followed by clip PSNR and equal-scene averaging. It is not the uint8 target
benchmark. Intervals are exploratory, unadjusted and conditional on one
new head seed and one historical E17 base; they do not measure training-seed
uncertainty or cross-domain generalization. See [PLAN.md](PLAN.md).

## Execution and verification

All24 focused CPU tests passed. The initial NPU export stopped before NPU
parameter allocation or token output because this torch_npu build rejects a
`Path` argument to memory-mapped `torch.load`. The original release and failed
logs were preserved. A new immutable release changed only those filenames to
`str`; six exporter tests and a real strict checkpoint load with torch_npu
imported then passed. No model mathematics changed.

The successful source export produced all136 token fields from cached LQ only,
with no GT encoder input. Encoder state hashes matched before/after. The five
CPU heads used identical initial states and clip/crop/frame plans and completed
400 consecutive updates each; evaluation states and all frozen sources were
unchanged. The screen completed at2026-11-12T03:17:03Z. The final process scan
at03:28:31Z found zero remaining task processes. No target frames were read.

The deployed package was remote `release_v2`, local
`observation_memory_release_v3.tar`, SHA256
`766803077cfa7abe2e2180994bdb74b9e926c1d21238bfd2499c19164679a863`.
The counter difference records the packaging retry. Evidence includes both
release manifests, failed/successful export checks, all2000 training curve
rows, per-head plans/provenance/completion, runtime scripts and final checks.

## Post-hoc source diagnostic

After the fixed gate failed, the frozen prior head was examined on the same34
internal source clips with real memory values, zero memory values and shuffled
addresses. This is a diagnosis, not a new gate or selected target experiment.
Weights remained unchanged.

Mean memory RMS was0.00382196 versus raw-detail RMS0.04116788. Zeroing the
memory changed predictions by only0.00007272 RMS. The residual's9×9 low-pass
energy ratio was0.96634582. Mean MSE was0.002297622062 with real memory,
0.002297588084 with zero values and0.002297605512 with shuffled addresses.
This supports limited use of the transported values by this trained recipe;
it does not establish that observation memory in general cannot work.
The complete intervention outputs are in
[posthoc_diagnosis.json](results/posthoc_diagnosis.json), with the exact script
and logs under [evidence/posthoc](evidence/posthoc/).

## Artifact locations

This repository contains text only, verified byte-for-byte against the local
download. [The archive inventory](evidence/archived_files_sha256.json) records
every copied file and SHA. [Independent statistics verification](evidence/separate_verification.json)
and [local artifact verification](evidence/local_verification.json) are retained.
No checkpoint tensors, source images or prepared RGB tensors were added to Git.

The complete verified shared backup is
`/temp/zihaohe/seer_cvpr/method_dev_20261112/observation_memory_v1_final_v1`
(723 files,79019333 bytes before the separate post-hoc addition), with manifest
SHA256 `b3d2549eacb54d16f2e83ab9e7ec105389d6742a962529bb162e0a9d768a6fe9`.
The local bundle is under `seer_memory_dev_20261112/artifacts/`; its SHA256 is
`b23de7004a3cb7cba937f061416ab190202cc417381fe90e2364bc79f31bddd5`.
The backup includes the136 projected token files and all heads; the local
text-and-head bundle excludes token tensors. Post-hoc evidence is kept in a
separate exclusive backup subdirectory.
