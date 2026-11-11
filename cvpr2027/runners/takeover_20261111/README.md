# Replaying the 2026-11-11 audit

These files preserve the launch scripts used on Ascend host 31051. They contain
that host's absolute paths, CANN installation and NPU assignment. They do not
resolve paths relative to this checkout. Keep these recorded scripts unchanged
when checking their provenance hashes. For a new location or CUDA evaluation,
use the explicit command in [RECONSTRUCTION.md](../../code_full/RECONSTRUCTION.md)
and select a new output path instead of assuming these scripts are portable.

## Inputs and environment

The observed Python/package versions and CANN version are recorded in
[runtime_31051_20261111.json](../../env/runtime_31051_20261111.json). This records
key packages, not a complete lockfile. The interpreter was
`/cache/zihaohe/seer_cvpr/venv_npu/bin/python`, with Python 3.12, Torch 2.7.1,
Torch-NPU 2.7.1.post2 and CANN 8.5.2. The September CUDA environment at the
repository-root `env/` is a different runtime.

The archived scripts expect these inputs before running:

| Historical location | Required contents / source |
|---|---|
| `/temp/zihaohe/seer_cvpr/datasets/` | Public datasets arranged according to the frozen split JSONs; copy the archived `results/splits_cvpr.json` here. The datasets are not in Git. |
| `/temp/zihaohe/seer_cvpr/modelzoo/vjepa2-main/` | External V-JEPA 2.1 source tree; not vendored. The recorded upstream transformer SHA256 is `d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9`; this alone does not pin every upstream dependency. |
| `/temp/zihaohe/seer_cvpr/weights/vjepa2_1_vitl_dist_vitG_384.pt` | Public pretrained V-JEPA checkpoint; omitted from Git and the recovery archive. |
| `/temp/zihaohe/seer_cvpr/code/baselines/` | Repository-root `code/baselines/`, including `common/eval_lib.py`; a sparse checkout must explicitly include it. |
| `/temp/zihaohe/seer_cvpr/code/ext_methods/Retinexformer/` | Repository-root `code/ext_methods/Retinexformer/`, needed for the qualitative baseline. |
| `/temp/zihaohe/seer_cvpr/weights_small/` | Repository-root `weights_small/`: includes RAFT-small and the retrained Retinexformer checkpoint at the exact paths used by the scripts. Include these in a sparse checkout when needed. |
| `/cache/zihaohe/seer_cvpr/runs_e17/`, `runs_e18/`, `runs_e26/` | The recovered final checkpoints; see [CHECKPOINTS.md](../../../CHECKPOINTS.md) for the verified archive and its limits. These CVPR weights are not published in the September GitHub Release. |
| `/cache/zihaohe/seer_cvpr/results/token_basis_train.json` | Copy of `results/cvpr_converged/probes/token_basis_train.json`, needed for E26 and mean/basis controls. SHA256 `30176997369915d607cfdd9f0684e92b1be424daff9c286ff1e8c8246cee6b11`. |
| `/cache/zihaohe/seer_cvpr/qual_regen/eval_baseline.py` | Exact `eval_baseline.py` archived in this directory (also included in the checkpoint recovery archive). The qualitative generator imports this absolute path. |

LPIPS also needs pyiqa's pretrained metric weights and its underlying torchvision
weights in the runtime cache or available to its downloader. These metric caches
are not part of the CVPR checkpoint recovery archive. RAFT failure is handled by
the evaluator by omitting warp metrics; check the resulting JSON for availability.
The scripts expose physical NPU 7 and use logical device 0; they do not allocate
or reserve that device through a scheduler.

## Stage the original layout on a restoration host

First restore the dependencies above. The block below stages only the frozen
model code, launch records and small inputs; it does not install dependencies,
restore datasets, or start an accelerator process. Use it on a restoration host
where `/cache/zihaohe/seer_takeover_20261111` does not already exist. The original
audited workspace already has this directory and completed results.

```sh
(
  set -eu
  SEER_REPO=/absolute/path/to/seer-modelarts-archive
  SEER_RUNNERS="$SEER_REPO/cvpr2027/runners/takeover_20261111"
  SEER_TASK_ROOT=/cache/zihaohe/seer_takeover_20261111
  SEER_MODEL_ROOT=/cache/zihaohe/seer_cvpr
  test ! -e "$SEER_TASK_ROOT"
  mkdir -p "$SEER_TASK_ROOT/incoming" "$SEER_TASK_ROOT/results" "$SEER_TASK_ROOT/logs"
  cp -R "$SEER_REPO/cvpr2027/code_full" "$SEER_TASK_ROOT/code_full"
  cp "$SEER_RUNNERS/"* "$SEER_TASK_ROOT/incoming/"

  # The generator deliberately imports the historical loader path.
  mkdir -p "$SEER_MODEL_ROOT/qual_regen" "$SEER_MODEL_ROOT/results"
  if [ ! -e "$SEER_MODEL_ROOT/qual_regen/eval_baseline.py" ]; then
    cp "$SEER_RUNNERS/eval_baseline.py" "$SEER_MODEL_ROOT/qual_regen/eval_baseline.py"
  fi
  cmp "$SEER_RUNNERS/eval_baseline.py" "$SEER_MODEL_ROOT/qual_regen/eval_baseline.py"
  if [ ! -e "$SEER_MODEL_ROOT/results/token_basis_train.json" ]; then
    cp "$SEER_REPO/cvpr2027/results/cvpr_converged/probes/token_basis_train.json" \
      "$SEER_MODEL_ROOT/results/token_basis_train.json"
  fi
  cmp "$SEER_REPO/cvpr2027/results/cvpr_converged/probes/token_basis_train.json" \
    "$SEER_MODEL_ROOT/results/token_basis_train.json"

  cd "$SEER_TASK_ROOT/incoming"
  printf '%s  %s\n' \
    235d80c719e69607b5ccddaed1cdf2f7df5affcaf68f78713a9b9cf009a58dd7 \
    qualitative_selection_margins.json | sha256sum -c -
)
```

`qualitative_selection_margins.json` is the original 1,749-byte file, copied
without reserialization. `qualitative_selection_provenance.json` records the
three-seed selection rule, source hashes and selected percentile examples.
Both must be staged beside the generator in `incoming/`, because its provenance
writer reads the selection record from the margins file's directory.

The generator's baseline loader also refers to the original
`/temp/zihaohe/seer_cvpr/code/vjepa_llve` path. It imports the frozen `eval_e4`
already loaded by the generator, with an assertion verifying that module's
origin; the original model-code directory is not substituted for the staged
`code_full` model. Its `baselines` and `ext_methods` siblings are still required.

## CPU verification and recorded launches

From the staged code directory, run the full 20-check CPU suite before a model
replay. This needs Torch, einops, NumPy and Pillow, but no model weights, datasets
or accelerator allocation:

```sh
cd /cache/zihaohe/seer_takeover_20261111/code_full
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /cache/zihaohe/seer_cvpr/venv_npu/bin/python -m unittest discover -s tests -v
```

The original run passed all 20 without skips; see the
[archived log](../../results/protocol_audit_20261111/checks/final_cpu_tests.log).
The following are separate inference jobs, shown in their original order. Run
them sequentially when the specified NPU is assigned to this replay. Each script
checks that its primary output paths are absent. Use the fresh staged directory
above to keep all outputs and provenance separate from the completed audit.

```sh
bash /cache/zihaohe/seer_takeover_20261111/incoming/run_random_saved.sh
bash /cache/zihaohe/seer_takeover_20261111/incoming/run_e26_saved.sh
bash /cache/zihaohe/seer_takeover_20261111/incoming/run_qualitative_verified.sh
bash /cache/zihaohe/seer_takeover_20261111/incoming/run_mean_basis_val.sh
bash /cache/zihaohe/seer_takeover_20261111/incoming/run_e18_saved.sh
```

Random, E26 and E18 JSONs go into `TASK_ROOT/results/`; the qualitative images
and provenance go into `TASK_ROOT/qualitative_verified/`. The mean/basis job
also writes to `TASK_ROOT/results/`, and is strictly the exploratory SDSD VAL16
control. None of these scripts starts training. E18's `layers_all_npu_s0`
checkpoint has saved seed **2** and is written as `e18_layers_all_s2_saved.json`.

For statistics and paper macros, the repository copies of raw JSONs and their
analyses are in `results/protocol_audit_20261111/`. See the corresponding audit
notes, especially [mean/basis reproduction](../../notes/MEAN_BASIS_VAL_20261111.md),
for commands that only read the archived outputs and require no accelerator.
