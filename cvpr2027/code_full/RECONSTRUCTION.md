# Checkpoint reconstruction

`tools/eval_e4.py` now defaults to `--reconstruction saved`. It restores the
checkpoint's branch, attention variant, key/value mode, layer combination, taps
and token projection. Random V-JEPA encoders use the training scripts' fixed CPU
initialization seed, 1234. Pretrained encoders still load the same external
checkpoint. Image preprocessing, metric definitions and dataset splits are
unchanged. Training augmentation RNG behavior is unchanged.

Use `--token_basis /relocated/token_basis_train.json` when a `global_rank`
checkpoint's saved basis path no longer exists. Known `/temp/.../seer_cvpr` and
`/cache/.../seer_cvpr` roots can also be remapped through `SEER_ROOT`. Basis taps,
rank and tensor shapes are checked; the evaluation JSON records its SHA-256.

The explicit `--reconstruction legacy` mode retains the archived evaluator's
default per-tubelet/single-layer mapping, default taps, unprojected tokens and
unseeded random encoder. This is a diagnostic mode, not a claim of exact historical
reproduction: the old random encoder state and runtime were not recorded.
Every new result records reconstruction version, mode and effective settings.
Keep corrected results in a separate directory from archived JSON files.

## Archived-checkout paths

The original server layout keeps `baselines/` beside the model code. In this
archive it is at repository-root `code/baselines/`, not beside `code_full/`.
For a full checkout, explicitly export its read-only helpers before evaluation:

```sh
export SEER_REPO=/absolute/path/to/seer-modelarts-archive
export SEER_ROOT=/absolute/path/to/data-and-public-weights
export PYTHONPATH="$SEER_REPO/code/baselines:$SEER_REPO/code/baselines/common:${PYTHONPATH:-}"
python "$SEER_REPO/cvpr2027/code_full/tools/eval_e4.py" --help
```

`SEER_ROOT` must contain `datasets/` (including frozen split JSONs),
`modelzoo/vjepa2-main/` and `weights/`. The checkpoint may be elsewhere.
The NPU runners additionally source CANN and use the existing NPU environment;
the historical CUDA `env/` freeze is not the current Ascend runtime manifest.
The observed Ascend package versions are in
[`runtime_31051_20261111.json`](../env/runtime_31051_20261111.json).
This is an environment inventory, not a complete dependency lockfile.

For an actual CUDA evaluation, use the exports above, a recovered checkpoint,
and a new result path. This example reproduces the corrected random control's
split and crop; CUDA numerical results need not match the recorded Ascend run:

```sh
export RAFT_WEIGHTS=/absolute/path/to/raft_small_C_T_V2-01064c6d.pth
export SEER_CHECKPOINT=/absolute/path/to/runs_e17/e17_small_random_s0/final_cpu.pt
export SEER_RESULT=/absolute/path/to/new-results/e17_small_random_s0_saved.json
test ! -e "$SEER_RESULT" && \
python "$SEER_REPO/cvpr2027/code_full/tools/eval_e4.py" \
  --ckpt "$SEER_CHECKPOINT" --heldout sdsd --split test --crop 256 \
  --device cuda --reconstruction saved --out "$SEER_RESULT"
```

Use the host-specific launch records and staging instructions in
[`runners/takeover_20261111/README.md`](../runners/takeover_20261111/README.md)
to reproduce the original Ascend setup. Neither example performs training.
The frozen evaluator itself permits overwriting `--out`; retain the explicit
new-path guard when invoking it directly. Missing or failed RAFT computation
can omit `warp_err`, so inspect the output metrics rather than assuming it ran.

## Historical result interpretation

The archived E26 runner trains with global rank 64 but invokes the generic
evaluator, whose archived source does not apply projection. Its archived test
PSNR-Y values are 23.809427 (seed 0) and 24.048665 (seed 1). The available source
therefore describes **train-lowrank / eval-full**, conditional on that source
being the actual evaluation version. The JSON files contain no code hash or
conditioning metadata, so they do not establish train-and-eval-lowrank results.
Do not relabel them as corrected results without a new evaluation.

The same reconstruction audit applies to E2/deep-tap checkpoints, E15/E18 hybrid
and all-layer checkpoints, random controls and forced-random-token runs. The
spatial and exposure-head options were already restored by the archived evaluator.
The gated module existed in the archive but was not wired into the branch factory;
that connection is now present. None of these source fixes retroactively verifies
or invalidates a historical metric.

## Checks

From this directory:

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
```

The complete suite has 20 checks: 11 reconstruction, four mean/basis, and five
aggregation checks. All 20 passed without skips in the recorded environment;
the log is at
[`final_cpu_tests.log`](../results/protocol_audit_20261111/checks/final_cpu_tests.log).
Configuration/path checks use only the standard library. The full suite needs
PyTorch, einops, NumPy and Pillow; tensor tests report explicit skips if PyTorch
or einops is unavailable. The suite uses tiny CPU modules and temporary
checkpoints, with no datasets, pretrained weights, GPU/NPU or training.
