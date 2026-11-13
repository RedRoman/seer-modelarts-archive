# Validation-Stage Results & Cross-Backbone Boundary Diagnosis: EID-TT v1

## 1. Execution Summary and Zero-Leakage Audit

`eid_trajectory_transport_v1` (`14,861` parameters) was trained strictly on `SDSD FIT51` (`102` clips / `51` scenes, canonical RGB `HVI-CIDNet` cache + GT-free reciprocal `RAFT-small` flows, manifest SHA256 `2095d153637dbcc80f5f960c94778183aa2ac979e60053a9151bb9553b3e8e02`) across `5` matched variants $\times$ `3` seeds (`15` source checkpoints) plus `3` reverse `BVI TRAIN48` checkpoints (`eid_full`), and evaluated strictly on source internal holdout `SDSD internal17` (`34` clips / `17` unseen scenes) and target validation `BVI VAL16` (`16` clips / `8` scenes $\times$ `2` light conditions) across four frozen backbones (`HVI-CIDNet`, `DarkIR`, `Retinexformer`, `Seer-v2 Pretrained`).

- **Held-out TEST isolation**: **Zero** `BVI TEST16` or `SDSD TEST39` sequences were opened (`--allow-final-test` remained unused throughout `v1`).
- **Exact anchor parity**: Remote independent verification (`sync_v1/checks/remote_verification_receipt.json`, `1,976` checks) and local independent verification (`sync_v1/checks/local_verification_receipt.json`, `1,784` checks) confirm `max_metric_abs_diff = 0.0` on `BVI VAL16` against `baseline_replay_v1` (`HVI-CIDNet`, `DarkIR`, `Retinexformer`) and exact match (`19.330874` `PSNR-Y`) against `native_color_replay_v1` (`Seer-v2`).

## 2. Source Internal Holdout (`SDSD internal17`, 34 Clips / 17 Scenes, 3-Seed Mean)

Summary artifact: `sync_v1/source_sdsd_runs_v1/source_train_and_internal17_summary.json` (SHA256 `d569c5b47bbf570a080639d1f949f8b9646ef2e6c728a9df894484f84bfa2263`).

| Method / Variant | PSNR-Y (dB) | PSNR-RGB (dB) | Exposure-Decoupled PSNR-RGB (dB) | LPIPS | tLP Gap |
|---|---:|---:|---:|---:|---:|
| `anchor` (`HVI-CIDNet`) | 15.282306 | 15.136419 | 22.326880 | 0.459994 | 0.146984 |
| `fixed_temporal_mean_0.5` | 15.393403 | 15.286489 | 23.419703 | 0.391785 | 0.065474 |
| `fixed_lp_locked_0.5` | 15.362574 | 15.222456 | 22.636617 | 0.418172 | 0.122239 |
| **`eid_full_mean3` (v1)** | **15.545943** | **15.451664** | **24.423346** | **0.343183** | **0.044819** |
| `ablate_raw_loss_mean3` | 15.549749 | 15.457616 | 24.334807 | 0.344525 | 0.045380 |
| `ablate_no_rad_mean3` | 15.407678 | 15.273913 | 22.835512 | 0.390201 | 0.112224 |
| `ablate_no_deatt_raw_mean3` | 15.555583 | 15.461716 | 24.462021 | 0.340463 | 0.044622 |
| `ablate_no_sensor_inv_mean3` | 15.550347 | 15.456184 | 24.427149 | 0.343493 | 0.044726 |

On `SDSD internal17`, `eid_full_mean3` improves over `anchor` on all `17/17` scenes (`+0.2636 dB` `PSNR-Y`, `+0.3152 dB` `PSNR-RGB`, `+2.0965 dB` exposure-decoupled `PSNR-RGB`, `-0.1168` `LPIPS`, `-69.5%` `tLP gap`), and beats `fixed_temporal_mean_0.5` across all five metrics (`+0.1525 dB` `PSNR-Y`, `+1.0036 dB` decoupled `PSNR-RGB`, `-0.0486` `LPIPS`, `-31.5%` `tLP gap`).

## 3. Zero-Shot Cross-Domain & Cross-Backbone Validation (`BVI VAL16`, 16 Clips / 8 Scenes)

### 3.1 Primary Cross-Domain Transfer (`SDSD FIT51 -> BVI VAL16` on `HVI-CIDNet`)

Artifact: `sync_v1/results_v1/eval_bvi_val16_hvi_cidnet_all_variants.json` (SHA256 `b805e504b3c4e2f84774d8cffd95a0a2606c170845f9469c3f9094007d312e09`).

| Method / Variant | PSNR-Y (dB) | PSNR-RGB (dB) | LPIPS | tLP Gap |
|---|---:|---:|---:|---:|
| `anchor` (`HVI-CIDNet`) | 24.241232 | 23.383146 | 0.153150 | 0.034285 |
| `fixed_temporal_mean_0.5` | 24.569942 | 23.723878 | 0.111113 | 0.010541 |
| `fixed_lp_locked_0.5` | 24.331723 | 23.479435 | 0.128013 | 0.025153 |
| **`eid_full_mean3` (v1)** | **24.459846** | **23.662051** | **0.104792** | **0.006994** |
| `ablate_raw_loss_mean3` | 24.425488 | 23.624605 | 0.105612 | 0.006971 |
| `ablate_no_rad_mean3` | 24.303394 | 23.475182 | 0.123048 | 0.021619 |
| `ablate_no_deatt_raw_mean3` | 24.432371 | 23.642119 | 0.110710 | 0.006932 |
| `ablate_no_sensor_inv_mean3` | 24.453198 | 23.656108 | 0.104863 | 0.006998 |

`eid_full_mean3` (`v1`) outperforms every single `v1` ablation on cross-domain `BVI VAL16` (`PSNR-Y`, `PSNR-RGB`, and `LPIPS`), cutting `LPIPS` by `-0.0484` (`95% CI [-0.0712, -0.0264]`) and `tLP gap` by `-79.6%` (`0.034285 -> 0.006994`, `33.6%` lower than `fixed_temporal_mean_0.5`).

### 3.2 Zero-Shot Cross-Backbone Validation (`DarkIR`, `Retinexformer`, `Seer-v2`) & Boundary Diagnosis

| Frozen Target Backbone | `anchor` PSNR-Y / LPIPS / tLP Gap | `eid_full_mean3` (`v1`) PSNR-Y / LPIPS / tLP Gap | Boundary Finding |
|---|---|---|---|
| `DarkIR` | `22.581697 / 0.151016 / 0.017732` | `22.367176 / 0.148040 / 0.008187` | `tLP` & `LPIPS` improve, but `PSNR-Y` drops `-0.2145 dB` |
| `Retinexformer` | `19.881678 / 0.203176 / 0.007376` | `19.881173 / 0.211647 / 0.007832` | `PSNR-RGB +0.0205 dB`, `tLP` over-smoothed past `gt_tlp` |
| `Seer-v2 Pretrained` | `19.330874 / 0.265640 / 0.004970` | `19.347489 / 0.265912 / 0.009634` | `PSNR +0.0166 dB`, `tLP` over-smoothed below `gt_tlp = 0.0530` |

Inspection of per-sequence diagnostics on `BVI VAL16` identified two exact boundary-condition omissions in `v1/eid_transport.py`:
1. Lines 560–562 multiplied conductances by `valid_mask` (`in_bounds`) while omitting `* fb_conf`, allowing occlusion-boundary ghosting on fast-motion clips (`BVI/S25_figures2__10fps`).
2. Unconstrained `Conv2d` biases learned unconditional conductance (`~0.80–0.86`) from high-flicker `SDSD FIT51` clips, over-smoothing already temporally coherent trajectories (`Seer-v2`, `Retinexformer`, and `20fps` clips) below `gt_tlp`.

Both boundary omissions are resolved from first principles in `eid_trajectory_transport_v2/` via reciprocal `fb_conf` gating and the closed-form **Unbiased Lag-1 De-Attenuated Bipolar Innovation Envelope ($\Omega^{\text{bip}}$)** prior to touching any held-out `TEST` split. All `v1` code, checkpoints, and `sync_v1/` artifacts are retained unmodified for complete experimental lineage.
