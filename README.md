# Seer / V-JEPA-LLVE — ModelArts archive (2026-09-27)

Backup made before the ModelArts hosts (30313, 31279, 31866, 30674, 31511, 30594, 30555, 30138)
were reclaimed. Private. Nothing here was modified relative to the servers except removing
`.git`, `__pycache__`, optimizer states and duplicate files.

## CVPR 2027 round (`cvpr2027/`)

Work after the ModelArts hosts were reclaimed. Start with `cvpr2027/STATUS.md`: it records the
method change (three of the original paper's four contributions measured as ineffective and
removed), the new exposure-decoupled component and its probe results, the frozen data splits,
and what is still running. `cvpr2027/notes/CVPR_PREREG.md` is the pre-registration: hypotheses,
decision rules and the amendment log, all written before the corresponding numbers existed.

## Layout
| path | what | server origin |
|---|---|---|
| `code/vjepa_llve/` | Seer model/training/eval code **as of 2026-09-26**, incl. the ICLR-round tools (`tools/v35_stage4_ctrl.py`, `seer_eval_unified.py`, `seer_eval_gtmean.py`, `seer_variants.py`, `temporal_metrics.py`, runners, `dump_qualitative.py`) | `/temp/zihaohe/workspace/vjepa_llve` |
| `code/quick_ablations/`, `code/HMP-Net/` | AAAI-round analysis scripts (invariance, predictive coherence, CP transfer) | `/temp/zihaohe/workspace/` |
| `code/baselines/` | baseline eval harness (`common/eval_lib*.py`, `run_*`, fixed-pairing / temporal / CP / GT-mean scripts) + baseline repos (no `.git`) | `/temp/zihaohe/workspace/baselines` |
| `code/ext_methods/` | Retinexformer (incl. the Seer-split option files), STCD, SCI, Zero-DCE, MIRNet code | `/temp/zihaohe/cache/ext_methods` |
| `results/server_shared/` | every result JSON and training log: `seer_iclr_evals/` (ICLR round, 09-25..27), `encswap_evals/` (July encoder swap, reference ckpt 6-split eval), all `*_eval_v2.json`, `v35_*/train_*.log`, stage logs | `/temp/zihaohe/cache/checkpoint/runs/` |
| `results/controlled_runs/host<port>/` | per-run local files of the ICLR-round controlled runs (train logs, eval JSONs/logs, gtmean, adaptation), Retinexformer Seer-split retraining logs | `/cache/zihaohe/runs/seer_iclr/` on each host |
| `results/agent_work/` | STCD unified eval, CP-for-baselines shards, leakage diagnostics | `/cache/zihaohe/{stcd_unified,cp_baselines}` (30594), `diag` (30555) |
| `weights_small/` | all baseline weights used (Retinexformer, URCSA, STCD, DarkIR, HVI-CIDNet, SCI, Zero-DCE), RAFT-small, Retinexformer retrained on Seer's split (union / indoor-only, 8k..24k) and on the official split (160k) | various |
| `paper/Seer_ICLR27/` | ICLR draft workspace (LaTeX, generated tables, figures, notes incl. `STATUS.md`, `CVPR_PREREG.md`), draft PDF | local Mac |
| `env/` | `pip freeze` of the `vjepa2-pip` env (torch 2.3.1+cu121, py3.11, A100 driver 535), `setup_host.sh` | |

## Large checkpoints (GitHub Releases, split into < 2 GB parts)
See `CHECKPOINTS.md` for the list, sizes and sha256. Restore with
`cat NAME.part-* > NAME && sha256sum -c`.

## Not archived (too large / publicly available)
- Datasets: SDSD indoor/outdoor (70 GB), BVI-RLV (426 GB), LOL (1.3 GB), REDS — public releases.
- V-JEPA 2.1 ViT-L `vjepa2_1_vitl_dist_vitG_384.pt` and V-JEPA 2 `vitl.pt` (5 GB each) — public
  (`dl.fbaipublicfiles.com/vjepa2/`).
- conda env tarball (74 GB) — rebuild from `env/`.
- Unrelated projects on the same disk (4klsdb, csig_xfer, vsr_xfer, zihao_old, wanvj runs).

## Known caveats (see paper/Seer_ICLR27/notes/STATUS.md)
- `code/vjepa_llve/models/pixel_branch_v10.py` + `tools/v35_stage4_train.py` are a 07-20
  re-implementation; loading `v35_stage3/final.pt` gives zero-iter audit 24.03 dB vs 27.08 with
  the original (lost) code. All ICLR-round controlled runs use this re-implementation.
- `eval_lib.iter_sdsd_indoor_stcd` pairs LQ/GT by file name, which is wrong for pair50/52/58/60/68/70;
  use `run_fixedpairing_all.iter_sdsd_indoor_stcd_idx` or `seer_eval_unified.iter_sdsd_indoor_seer`.
- The AAAI-27 manuscript numbers were not measured; the measured values are in `results/`.

## CVPR 2027 round (converged, 2026-11-11)

Start at `cvpr2027/STATUS.md` (current verdicts, what is unfinished, decisions
pending) and `cvpr2027/EXPERIMENTS.md` (every experiment's flags, seeds, host
port, server run and result paths, and archived location). Host 32677 has been
unreachable since 2026-11-05; what was recovered from it and what was not is
listed in both files.
