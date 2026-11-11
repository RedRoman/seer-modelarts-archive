# Scene-bootstrap sensitivity audit — 2026-11-11

This is a **post-hoc sensitivity analysis**, not a new preregistered test or a rerun of inference. It uses archived per-sequence JSONs and leaves their values unchanged.

Within each sequence, average the matched seeds first, then subtract pretrained minus none. The primary point estimate remains sequence weighted. Sample whole scene clusters with replacement (10,000 replicates, NumPy default_rng seed 0); each replicate is total sampled sequence differences divided by total sampled sequence count. Strip the 10/20-fps suffix and terminal `_2`, keeping the dataset/domain prefix, matching the split builder's scene groups on these rows. Paired passes/fps variants always travel together.

| Comparison | Sequences / scenes | Metric | Effect | Sequence CI | Scene-cluster CI | Seed signs | Verdict |
|---|---:|---|---:|---|---|---|---|
| indomain_sdsd | 39 / 21 | psnr_y | +1.290 | [+0.270, +2.449] | [+0.314, +2.346] | +/+/+ | benefit |
| indomain_sdsd | 39 / 21 | lpips | -0.0240 | [-0.0321, -0.0168] | [-0.0321, -0.0164] | −/−/− | benefit |
| h5_sdsd | 39 / 21 | psnr_y | +0.292 | [-1.875, +2.519] | [-2.577, +3.209] | −/+ | inconclusive |
| h5_sdsd | 39 / 21 | lpips | -0.0257 | [-0.0471, -0.0059] | [-0.0529, -0.0008] | −/− | benefit |
| h5_bvi | 16 / 8 | psnr_y | +0.298 | [-1.593, +1.965] | [-1.035, +1.717] | +/+ | inconclusive |
| h5_bvi | 16 / 8 | lpips | -0.0229 | [-0.0435, -0.0017] | [-0.0415, -0.0075] | −/− | benefit |

The CI/sign verdict is unchanged for every listed comparison. In-domain PSNR and LPIPS benefits persist. Transfer PSNR remains inconclusive for SDSD and BVI; LPIPS remains favorable under scene resampling, with the SDSD upper bound close to zero (−0.000761).

Equal-scene weighting is a separate estimand, not a replacement for the primary result:

| Comparison | Metric | Equal-scene effect | Scene-bootstrap CI |
|---|---|---:|---|
| indomain_sdsd | psnr_y | +1.124 | [+0.166, +2.162] |
| indomain_sdsd | lpips | -0.0222 | [-0.0305, -0.0145] |
| h5_sdsd | psnr_y | +0.397 | [-2.402, +3.267] |
| h5_sdsd | lpips | -0.0280 | [-0.0552, -0.0025] |
| h5_bvi | psnr_y | +0.298 | [-1.035, +1.717] |
| h5_bvi | lpips | -0.0229 | [-0.0415, -0.0075] |

SDSD contains 21 scenes: 18 with two sequences and 3 with one; BVI contains 8 scenes with two fps versions each. The intervals condition on the available trained seeds; they do not estimate uncertainty over a population of training seeds, repair any split leakage, or address unrecorded dependence beyond the scene grouping. The small BVI scene count limits precision.

Source paths, SHA-256 hashes, per-seed effects and scene membership are in `results/protocol_audit_20261111/scene_cluster_bootstrap.json`. Reproduce from the repository root:

```sh
python3 cvpr2027/tools/scene_robustness.py --out /tmp/scene_audit.json --note /tmp/scene_audit.md
```
