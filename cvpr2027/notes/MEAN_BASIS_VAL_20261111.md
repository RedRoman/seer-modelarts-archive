# Exploratory mean × basis validation control — 2026-11-11

This is the completed post-hoc validation control specified in Amendment 21 of the protocol notes. It evaluates all 16 frozen SDSD validation sequences (eight scenes, two passes per scene) using the existing full-token-trained E17 pretrained checkpoints at seeds 0 and 1. No training, test-set evaluation, model selection, success threshold, or equivalence test is introduced here. The intervals are unadjusted exploratory intervals across multiple contrasts.

## Intervention and inputs

Each frozen encoder output is reused across all six conditions. For every clip and tap, the adaptive basis is computed once on CPU in float32 using the right singular vectors of the **clip-centered** token matrix. This definition remains the same when the reconstruction substitutes the training mean. The rank-4 reconstruction is `mean + (Z − mean) Vᵀ V`, where the rows of `V` are the four selected vectors. Mean only broadcasts the clip mean; identity retains the entire token field.

The training mean and training basis come from the same archived statistics file: 60 training clips, taps `[5, 11, 17, 23]`, 64 stored basis rows per tap, using the first four for this control. The experiment uses 16 frames and a 256-pixel crop per validation sequence. All choices and both seeds were fixed before this control ran.

## Absolute results

Values average sequences within each seed and then average the two seeds. PSNR is Y-channel dB; lower LPIPS is better.

| Condition | PSNR-Y | LPIPS |
|---|---:|---:|
| Identity (full token field) | 24.577 | 0.2414 |
| Clip mean only | 23.283 | 0.2518 |
| Clip mean + clip rank-4 basis | 24.698 | 0.2409 |
| Clip mean + training rank-4 basis | 22.740 | 0.2593 |
| Training mean + clip rank-4 basis | 17.735 | 0.3637 |
| Training mean + training rank-4 basis | 17.946 | 0.3517 |

## Paired contrasts

For each paired sequence, subtract the second condition from the first and average the two seed differences. The primary point estimate weights the 16 sequences equally. The sequence bootstrap samples 16 paired differences with replacement. The scene sensitivity samples eight scenes with replacement, retaining both passes of each selected scene, and recomputes the sequence-weighted sum divided by the resampled count. Seeds are held fixed; these intervals do not quantify uncertainty over training seeds.

| Contrast | Metric | Difference | Sequence 95% CI | Scene 95% CI | Seed 0 / seed 1 |
|---|---|---:|---|---|---|
| Clip rank-4 minus identity | PSNR-Y | +0.121 | [-0.152, +0.399] | [-0.089, +0.313] | +0.022 / +0.220 |
| Clip rank-4 minus identity | LPIPS | -0.0005 | [-0.0025, +0.0017] | [-0.0024, +0.0019] | +0.0001 / -0.0011 |
| Clip rank-4 minus mean only | PSNR-Y | +1.415 | [+0.554, +2.436] | [+0.351, +2.674] | +0.644 / +2.187 |
| Clip rank-4 minus mean only | LPIPS | -0.0109 | [-0.0235, -0.0017] | [-0.0259, -0.0005] | -0.0092 / -0.0126 |
| Clip minus training basis, clip mean fixed | PSNR-Y | +1.959 | [+0.443, +4.026] | [+0.354, +3.799] | +1.643 / +2.274 |
| Clip minus training basis, clip mean fixed | LPIPS | -0.0184 | [-0.0361, -0.0044] | [-0.0372, -0.0026] | -0.0175 / -0.0194 |
| Clip minus training mean, clip basis fixed | PSNR-Y | +6.964 | [+3.119, +11.151] | [+1.949, +12.803] | +6.867 / +7.060 |
| Clip minus training mean, clip basis fixed | LPIPS | -0.1229 | [-0.2267, -0.0409] | [-0.2676, -0.0194] | -0.1202 / -0.1256 |
| Clip minus training mean, training basis fixed | PSNR-Y | +4.794 | [+1.936, +7.942] | [+1.209, +8.924] | +5.044 / +4.544 |
| Clip minus training mean, training basis fixed | LPIPS | -0.0924 | [-0.1863, -0.0205] | [-0.2234, -0.0070] | -0.0912 / -0.0935 |

In this validation sample, retaining four clip-centered directions improves over the clip mean alone, and using the clip basis improves over the training basis when the clip mean is held fixed. Substituting the training mean is harmful with either basis held fixed. These matched contrasts support roles for both the clip-specific mean and clip-specific basis under this intervention. They do not establish a population-level mechanism from eight scenes or demonstrate that a smaller model can predict these quantities.

The rank-4-versus-identity intervals include zero. This is **not evidence of equivalence**: there is no equivalence margin or gate, and the LPIPS effects have opposite signs at seed 0 (+0.000116) and seed 1 (−0.001106). The adaptive basis still requires the full pretrained encoder output and an SVD, so this experiment does not remove the encoder or measure a cheaper deployment.

## Reproduction and integrity checks

From the repository root (Python with NumPy; no accelerator required):

```sh
python3 cvpr2027/tools/analyze_mean_basis_val.py --check
python3 cvpr2027/paper/tools/make_mean_basis_macros.py --check
```

Omit `--check` to regenerate the analysis JSON or macros. Defaults resolve relative to each script, so the same commands also work using absolute script paths from another working directory. The analysis accepts `--input-dir`, `--out`, `--basis`, `--splits`, `--manifest`, and `--source-root`; `--upstream-source` optionally supplies a local copy of the upstream transformer source for byte verification.

The deterministic bootstrap uses 10,000 draws and **one global NumPy `default_rng(0)` stream across the entire analysis**. It preserves the original condition/contrast order, then PSNR-Y before LPIPS, and consumes all 10,000 sequence draws before the 10,000 scene draws for each metric. It does not reset the RNG for each contrast. Sorted sequence names define scene insertion order; stripping only the final `_2` preserves the SDSD domain prefix. This dataset has no frame-rate suffixes.

The portable script reproduces the existing analysis JSON byte for byte: SHA256 `6713feeaf3b16e9b36171aeedefb873209fb310cb38accc490dd6016e736b898`. Independent review also reproduced every mean, per-seed effect, and interval using a separately vectorized resampling implementation.

The script checks all six condition labels and ranks, finite metrics, reported aggregate/row agreement, 16 unique identical names against the archived validation split, seeds 0/1, saved full-token reconstruction settings, and identical cross-seed token/mean/basis fingerprints. It hashes the six local runtime source files, the split file, and the training basis file against the raw reports. Checkpoint hashes are compared with the archived checkpoint manifest; checkpoint tensors are not loaded by this statistics script. The upstream transformer source is not vendored: absent `--upstream-source`, its digest is checked against recorded provenance only. These checks verify the available record and do not replay inference or verify unrecorded dependency bytes.

- Training basis SHA256: `30176997369915d607cfdd9f0684e92b1be424daff9c286ff1e8c8246cee6b11`.
- Validation split SHA256: `fb6be5ea78ef73f5cc9db90bca75813bf597275f7609fe39d82f5b2f883e9ad0`.
- Mean-control runtime SHA256: `781bc248107bae2cd072b6faf13515b1ef2eb22f836c34085167dca347a0959c`.
- Inputs and complete contrasts: `results/protocol_audit_20261111/mean_basis_val/`.
- Paper macros: `paper/tables/result_macros_mean_basis.tex` (`meanval@` prefix, six absolute conditions and five matched contrasts, sequence/scene intervals and seed signs).
