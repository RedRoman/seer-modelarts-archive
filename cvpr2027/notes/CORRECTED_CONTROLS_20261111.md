# Corrected controls — 2026-11-11

The source JSONs in `results/cvpr_converged/` are unchanged. New JSONs,
reconstruction metadata and file hashes are under `results/protocol_audit_20261111/`.
`paper/tools/make_audit_macros.py` pairs sorted, identical sequence-name sets,
matches the available seed numbers, and performs 10,000 paired sequence
bootstrap replicates with NumPy RNG 0. Intervals condition on these models.

## Random encoder, seed 0

Training used encoder initialization seed 1234; the archived evaluator did not
reuse it. Corrected random PSNR-Y is 24.084147, versus historical 19.809233.
The corresponding pretrained checkpoint scores 24.688059.

Pretrained minus corrected random: PSNR-Y +0.603912, CI [-0.481890,+1.870245];
LPIPS -0.019357, CI [-0.0308,-0.0092] (exact endpoints are generated in
`corrected_comparisons.json`).
The former is unresolved, the latter favors pretrained conditioning for this
single trained random-control seed. The old 4.88 dB content-attribution claim
must not be retained. Sequence resampling cannot demonstrate multi-seed stability.

## Fixed rank 64 trained, seeds 0 and 1

The generic evaluator did not restore the saved training projection. The
corrected evaluator applies that same basis at inference. The mean PSNR-Y is
24.491158; matched full-token pretrained seeds 0/1 average 24.552087.
Difference: -0.060929, CI [-0.571,+0.457]. LPIPS difference: about -0.000216,
CI [-0.0041,+0.0040]. Exact values and per-seed effects are generated in the
comparison JSON. Both metric effects reverse sign between seeds.

The old comparison used three reference seeds; retaining that reference after
the inference fix gives PSNR -0.097919, CI [-0.619659,+0.440330], also unresolved.
Thus the verdict change is not explained by matching reference seeds.
Neither version proves equivalence or a cheaper encoder. The earlier
significant training-time penalty is withdrawn.
