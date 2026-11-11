# Correction selection on frozen BVI VAL16

Direct MSE correction selection improves PSNR-Y by **0.45885 dB** over the frozen E12 base, averaged over selector seeds 0/1/2. Its scene-paired 95% interval is **[0.13644, 0.94816] dB**. Dividing the direct MSE loss by the fit-set base MSE gives a similar **0.45809 dB** gain, with interval **[0.14056, 0.94198]**. This is positive exploratory validation evidence for learned selection of aligned corrections; it does not establish a novel risk objective or a SOTA result.

All runs use the same E12 BVI-only **base training seed 0**, final 1,000-step selectors, and frozen VAL16: eight unseen scenes with two illumination conditions per scene, one center 16-frame clip and center 256-pixel crop per sequence. The three selector seeds vary initialization and sampling. They are not three independent base trainings. No TEST frames or VAL oracle are used. The base had seen the 24 TRAIN scenes; the earlier six-scene selector holdout was internal to those TRAIN scenes. The fixed 18/6 selector split, fit-only normalization, input features, network capacity, per-seed initial state, sample plan, optimizer and training budget are shared across the three losses.

| Method | PSNR-RGB mean ↑ | PSNR-Y mean ↑ | PSNR-Y range across selector seeds | LPIPS mean ↓ | LPIPS range |
|---|---:|---:|---|---:|---|
| Frozen E12 base | 28.105014 | 29.127795 | [29.127795, 29.127795] | 0.15323244 | [0.15323244, 0.15323244] |
| Direct MSE | **28.554749** | **29.586642** | [29.578138, 29.593887] | 0.14493868 | [0.14475512, 0.14503824] |
| Direct MSE / fit base MSE | 28.555442 | 29.585884 | [29.577998, 29.593137] | 0.14490669 | [0.14473429, 0.14520149] |
| Full risk supervision | 28.511765 | 29.530988 | [29.523008, 29.537995] | 0.14772360 | [0.14674371, 0.14827944] |
| Fixed average of two donors | 28.108517 | 29.036993 | [29.036974, 29.037028] | 0.16379365 | [0.16379341, 0.16379399] |
| Initial equal thirds | 28.233557 | 29.186145 | [29.186129, 29.186168] | 0.15628344 | [0.15628222, 0.15628407] |

The direct MSE PSNR-Y gains for selector seeds 0/1/2 are **+0.45034 / +0.46609 / +0.46011 dB**; normalized direct gains are **+0.45020 / +0.45872 / +0.46534 dB**. This comparison uses all predeclared final checkpoints, with no choice of the best seed or VAL checkpoint.

| Contrast | ΔPSNR-RGB [scene 95% CI] | ΔPSNR-Y [scene 95% CI] | ΔLPIPS [scene 95% CI] | ΔtLP gap [scene 95% CI] |
|---|---|---|---|---|
| Direct − base | +0.44973 [0.16021, 0.90153] | +0.45885 [0.13644, 0.94816] | −0.008294 [−0.025285, 0.002585] | −0.000623 [−0.002706, 0.001362] |
| Normalized direct − base | +0.45043 [0.16233, 0.90333] | +0.45809 [0.14056, 0.94198] | −0.008326 [−0.025469, 0.002638] | −0.000695 [−0.002846, 0.001324] |
| Risk − base | +0.40675 [0.12816, 0.83613] | +0.40319 [0.09488, 0.86065] | −0.005509 [−0.020927, 0.004455] | −0.000050 [−0.001826, 0.001668] |
| Risk − direct | −0.04298 [−0.07450, −0.01821] | −0.05565 [−0.09080, −0.02919] | +0.002785 [0.001353, 0.004615] | +0.000573 [−0.000032, 0.001445] |
| Risk − normalized direct | −0.04368 [−0.07662, −0.01836] | −0.05490 [−0.08835, −0.02945] | +0.002817 [0.001321, 0.004782] | +0.000644 [−0.000010, 0.001635] |

PSNR increases are favorable; LPIPS and tLP-gap decreases are favorable. **LPIPS and temporal-gap intervals against the base cross zero**, so these data do not establish stable perceptual or temporal improvement. Risk supervision trails both direct controls in PSNR and LPIPS for all three selector seeds. The close direct/normalized means do not constitute an equivalence test. The tLP gap is the per-sequence `abs(tLP(pred) − tLP(GT))`; a small gap alone cannot rule out oversmoothing.

For each contrast, the analysis first averages the paired effect across the three selectors for each sequence, then resamples eight whole scenes with both conditions together (10,000 draws, NumPy RNG seed 0). The estimator is total paired sums divided by total sequence counts. Intervals are unadjusted and conditional on these three fitted selectors, the one base, and the actual candidate computations. **The 48 seed-by-sequence observations are not independent units.** No base-seed uncertainty, validation-selection correction, or training-seed bootstrap is implied.

## Repeatability and evidence

All 16 base sequences and all six base metrics are exactly identical across runs; every method uses exactly the same GT temporal reference. The three runs record identical base/cache provenance, RAFT weight-file SHA, evaluator/cache/selector source SHAs, paired-frame inventory and split. Each run computes one candidate pair and shares it among all three heads.

Fixed controls are not bitwise identical across runs. Against seed 0, fixed-average maximum/mean absolute per-sequence PSNR-Y drift is **0.00032138 / 0.00009013 dB**; initial-thirds drift is **0.00022065 / 0.00006391 dB**. Their maximum drifts are respectively **0.07004% / 0.04809%** of the mean direct-vs-base gain. The full metric and per-sequence drift records are retained in the analysis. Runtime verification additionally records every pair of runs. The source of this small numerical drift was not isolated, so cross-run variation cannot all be attributed to selector initialization. The RAFT weight file was SHA-checked and loaded strictly; **no independent RAFT state fingerprint was recorded**.

- [Full three-seed statistics](results/analysis_selector_seeds_012.json) and [complete tables](results/analysis_selector_seeds_012.md) retain all six methods, six contrasts, four analysis metrics, seed ranges, paired effects and drift.
- Raw VAL JSONs and single-seed analyses are in [s0/results](s0/results), [s1/results](s1/results), and [s2/results](s2/results). Their corresponding launch manifests and logs retain the actual commands.
- [Independent local check](checks/root_independent_summary.json) recomputed the raw-row statistics without importing either analyzer; all 12 checked effects and 24 interval endpoints agree within 1e-14. This is a **local derived check**, not an item from the remote deployment manifest.
- [Runtime verification](checks/three_seed_runtime_verification.json), [replicate inventory](checks/replicate_val_inventory.json), [source manifest](checks/source_manifest.json), and [SFS backup manifests](checks/backup_replicates_manifest.json) preserve provenance. The original seed-0 backup is in [backup_manifest.json](checks/backup_manifest.json).
- Selector training evidence for all three seeds is in [../selector_v1/runs](../selector_v1/runs). No checkpoint tensors or duplicate source tarballs are stored here; [source_package.json](source_package.json) identifies the external/remote frozen package. The training package SHA begins `35e7a7`; the separate VAL package SHA is `f2cf65cfdb0a79e9cfc071127d2275b003f8d3a0e369800987625f0a5aa4ec1e`.

The standalone analyzers read JSON and the archived split, require the exact 16 names/eight scenes/two conditions, recompute reported means and temporal gaps, cross-check recorded hashes and matched training settings, and reject duplicate seeds. They do not load models or video frames. Runtime inventory and independent checkpoint-byte verification supply the separate artifact checks.

Recompute from the repository root with Python 3 and NumPy, using a new output path (the scripts refuse to overwrite evidence):

```bash
python3 cvpr2027/method_dev/20261111/analyze_correction_seeds.py \
  --inputs \
    cvpr2027/method_dev/20261111/selector_val_v1/s0/results/bvi_val16_s0.json \
    cvpr2027/method_dev/20261111/selector_val_v1/s1/results/bvi_val16_s1.json \
    cvpr2027/method_dev/20261111/selector_val_v1/s2/results/bvi_val16_s2.json \
  --out /tmp/seer_selector_seeds_012_recomputed.json
```

For an individual run, use `analyze_correction_val.py --input <raw-val.json> --out <new-analysis.json>`. Input paths in newly generated JSON may differ, but numeric statistics should reproduce. This result supports continued method development on validation; it is not a final held-out test or comparison against an external strong baseline.
