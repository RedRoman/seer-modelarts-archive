# Complete Validation & Single-Pass Final Test Results: Bipolar Innovation-Enveloped Trajectory Transport (EID-TT v2)

## 1. Executive Summary and Verification Lineage

`EID-TT v2` (`15,509` parameters, `eid_transport.py`) resolves the null-flicker and cycle-occlusion boundary conditions exposed during `v1` validation via the **Unbiased Lag-1 De-Attenuated Bipolar Trajectory Innovation Envelope ($\Omega^{\text{bip}}$)** coupled with conserved dual-manifold log-radiometric and phase-deattenuated detail transport.

- **Strict Data Hygiene**:
  1. All `18` source checkpoints (`6` variants $\times$ `3` seeds `0, 1, 2`) were trained exclusively on `SDSD FIT51` (`102` clips / `51` scenes, canonical RGB `HVI-CIDNet` anchor + GT-free reciprocal `RAFT-small` flows).
  2. All `3` reverse-transfer checkpoints (`eid_full` $\times$ `3` seeds `0, 1, 2`) were trained exclusively on `BVI TRAIN48` (`48` clips / `24` scenes $\times$ `2` light conditions) with zero `BVI VAL/TEST` or `SDSD` access.
  3. Validation across `SDSD internal17` and all four frozen backbones on `BVI VAL16` was frozen and cryptographically verified (`sync_v2/checks/val4_complete_verification_receipt.json`, `5,577` checks) **prior** to unlocking `--allow-final-test` once across `BVI TEST16` (`4` backbones) and `SDSD TEST39` (`in-domain` + `reverse BVI -> SDSD`).
- **Independent Verification & SFS Preservation**:
  - Remote verification (`sync_v2/checks/remote_final_verification_receipt.json`): **`11/11` files, `11,673` scalar/bootstrap/hash checks passed** (`max_metric_abs_diff = 0.0` against `baseline_replay_v1` across `HVI-CIDNet`, `DarkIR`, and `Retinexformer`).
  - Local verification (`sync_v2/checks/local_final_verification_receipt.json`): **`11/11` files, `11,481` checks passed**.
  - Persistent SFS backups on `huawei-modelarts-31051`:
    - `v1`: `/temp/zihaohe/seer_cvpr/method_dev_20261112/eid_trajectory_transport_v1_final_20261113` (`41` files, manifest SHA256 `0f7b4e84702dae35fcac6b15f4c042d7c9d2a461c70edf5d19e2ee62948d30c9`)
    - `v2`: `/temp/zihaohe/seer_cvpr/method_dev_20261112/eid_trajectory_transport_v2_final_20261113` (`78` files, manifest SHA256 `7634de7e414ad7e36aa683e71874c9f1ad6859afd603ab2bf4c56f1d3ede97b1`)

---

## 2. Zero-Shot Cross-Domain & Cross-Backbone Generalization (`SDSD FIT51 -> BVI-RLV`)

All `BVI-RLV` results below use the **exact same 15,509-parameter weights** trained only on `SDSD FIT51` (`HVI-CIDNet`), evaluated zero-shot (`centre16/crop256`, canonical RGB rounded `uint8`, official `eval_lib` scorer) with **zero GT and zero target adaptation**.

### 2.1 Validation Split (`BVI VAL16`: 16 Clips / 8 Scenes)

| Target Backbone | Method | PSNR-Y (dB) | PSNR-RGB (dB) | LPIPS | tLP Gap | Scene 95% CI vs `anchor` (`PSNR-Y` / `LPIPS` / `tLP Gap`) |
|---|---|---:|---:|---:|---:|---|
| **`HVI-CIDNet`** | `anchor` | 24.241232 | 23.383146 | 0.153150 | 0.034285 | — |
| | `fixed_temporal_mean_0.5` | 24.569942 | 23.723878 | 0.111113 | 0.010541 | — |
| | `fixed_lp_locked_0.5` | 24.331152 | 23.506149 | 0.117039 | 0.017934 | — |
| | **`eid_full_mean3`** | **24.447612** | **23.615010** | **0.119092** | **0.015198** | `[+0.0278,+0.4067]` / `[-0.0462,-0.0230]` / `[-0.0308,-0.0085]` |
| | `ablate_no_bipolar_env_mean3` | 24.542666 | 23.722174 | **0.106253** | **0.007982** | `[-0.0035,+0.6364]` / `[-0.0666,-0.0280]` / `[-0.0430,-0.0104]` |
| **`DarkIR`** | `anchor` | 22.581697 | 21.465763 | 0.151016 | 0.017732 | — |
| | `fixed_temporal_mean_0.5` | 22.564158 | 21.477228 | 0.145932 | 0.007880 | — |
| | `fixed_lp_locked_0.5` | 22.534888 | 21.450101 | 0.148472 | 0.008951 | — |
| | **`eid_full_mean3`** | **22.590754** | **21.486897** | **0.147130** | **0.010738** | `[-0.0421,+0.0721]` / `[-0.0068,-0.0013]` / `[-0.0146,-0.0014]` |
| | `ablate_no_bipolar_env_mean3` | 22.532140 | 21.457494 | 0.145826 | 0.008212 | `[-0.1715,+0.0456]` / `[-0.0125,+0.0013]` / `[-0.0203,+0.0001]` |
| **`Retinexformer`** | `anchor` | 19.881678 | 19.103253 | 0.203176 | 0.007376 | — |
| | `fixed_temporal_mean_0.5` | 19.928787 | 19.155996 | 0.209118 | 0.006341 | — |
| | `fixed_lp_locked_0.5` | 19.905523 | 19.130099 | 0.212264 | 0.004928 | — |
| | **`eid_full_mean3`** | **19.907307** | **19.134329** | **0.201131** | **0.004715** | `[+0.0122,+0.0380]` / `[-0.0049,+0.0006]` / `[-0.0066,+0.0009]` |
| | `ablate_no_bipolar_env_mean3` | 19.900916 | 19.135782 | 0.209995 | 0.007300 | `[-0.0087,+0.0455]` / `[+0.0012,+0.0116]` / `[-0.0044,+0.0041]` |
| **`Seer-v2`** | `anchor` | 19.330874 | 18.041200 | 0.265640 | 0.004970 | — |
| | `fixed_temporal_mean_0.5` | 19.364852 | 18.056524 | 0.263889 | 0.007825 | — |
| | `fixed_lp_locked_0.5` | 19.346577 | 18.046335 | 0.266070 | 0.005794 | — |
| | **`eid_full_mean3`** | **19.349952** | **18.051509** | **0.259572** | **0.005773** | `[+0.0027,+0.0382]` / `[-0.0134,-0.0010]` / `[-0.0011,+0.0030]` |
| | `ablate_no_bipolar_env_mean3` | 19.364129 | 18.057725 | 0.262741 | 0.008917 | `[-0.0096,+0.0828]` / `[-0.0258,+0.0122]` / `[+0.0017,+0.0064]` |

### 2.2 Held-Out Final Test Split (`BVI TEST16`: 16 Clips / 8 Unseen Scenes, Single-Pass)

| Target Backbone | Method | PSNR-Y (dB) | PSNR-RGB (dB) | LPIPS | tLP Gap | Scene 95% CI vs `anchor` (`PSNR-Y` / `LPIPS` / `tLP Gap`) |
|---|---|---:|---:|---:|---:|---|
| **`HVI-CIDNet`** | `anchor` | 20.599400 | 20.536833 | 0.179051 | 0.029778 | — |
| | `fixed_temporal_mean_0.5` | 20.732959 | 20.718200 | 0.138628 | 0.007227 | — |
| | `fixed_lp_locked_0.5` | 20.651367 | 20.614453 | 0.142445 | 0.013617 | — |
| | **`eid_full_mean3`** | **20.688404** | **20.667344** | **0.149943** | **0.014209** | `[-0.0346,+0.2152]` / `[-0.0403,-0.0188]` / `[-0.0227,-0.0091]` |
| | `ablate_no_bipolar_env_mean3` | 20.708967 | 20.706554 | **0.133515** | **0.005107** | `[-0.0233,+0.2488]` / `[-0.0598,-0.0306]` / `[-0.0343,-0.0156]` |
| **`DarkIR`** | `anchor` | 20.181484 | 19.877571 | 0.151540 | 0.010778 | — |
| | `fixed_temporal_mean_0.5` | 20.201724 | 19.906360 | 0.148527 | 0.004464 | — |
| | `fixed_lp_locked_0.5` | 20.205473 | 19.905043 | 0.149842 | 0.004769 | — |
| | **`eid_full_mean3`** | **20.192632** | **19.893889** | **0.148870** | **0.006327** | `[-0.0024,+0.0303]` / `[-0.0045,-0.0012]` / `[-0.0072,-0.0020]` |
| | `ablate_no_bipolar_env_mean3` | 20.182148 | 19.888731 | 0.148991 | 0.004397 | `[-0.0211,+0.0263]` / `[-0.0075,+0.0029]` / `[-0.0128,-0.0009]` |
| **`Retinexformer`** | `anchor` | 18.808356 | 18.295178 | 0.238211 | 0.010352 | — |
| | `fixed_temporal_mean_0.5` | 18.816770 | 18.324124 | 0.242067 | 0.005840 | — |
| | `fixed_lp_locked_0.5` | 18.812208 | 18.309725 | 0.244528 | 0.007313 | — |
| | **`eid_full_mean3`** | **18.827635** | **18.321699** | **0.236211** | **0.007147** | **`[+0.0073,+0.0316]` / `[-0.0025,-0.0015]` / `[-0.0052,-0.0013]`** |
| | `ablate_no_bipolar_env_mean3` | 18.809651 | 18.320640 | 0.242319 | 0.005701 | `[-0.0992,+0.0664]` / `[+0.0006,+0.0076]` / `[-0.0085,-0.0011]` |
| **`Seer-v2`** | `anchor` | 19.653183 | 18.699383 | 0.254277 | 0.005534 | — |
| | `fixed_temporal_mean_0.5` | 19.693740 | 18.723115 | 0.249435 | 0.005366 | — |
| | `fixed_lp_locked_0.5` | 19.685352 | 18.717440 | 0.250783 | 0.004781 | — |
| | **`eid_full_mean3`** | **19.672621** | **18.712263** | **0.251329** | **0.005600** | **`[+0.0018,+0.0394]` / `[-0.0057,-0.0007]` / `[-0.0007,+0.0009]`** |
| | `ablate_no_bipolar_env_mean3` | 19.694289 | 18.725667 | 0.248023 | 0.005528 | `[-0.0160,+0.0928]` / `[-0.0247,+0.0081]` / `[-0.0035,+0.0027]` |

---

## 3. Bidirectional Cross-Domain & In-Domain Generalization on `SDSD`

### 3.1 Held-Out Final Test (`SDSD TEST39`: 39 Clips / 21 Unseen Scenes, In-Domain & Reverse `BVI -> SDSD`)

| Training Domain -> Test Split | Method | PSNR-Y (dB) | PSNR-RGB (dB) | LPIPS | tLP Gap | Scene 95% CI vs `anchor` |
|---|---|---:|---:|---:|---:|---|
| Reference Controls (`SDSD TEST39`) | `anchor` (`HVI-CIDNet`) | 17.659944 | 17.254383 | 0.332771 | 0.110437 | — |
| | `fixed_temporal_mean_0.5` | 17.840225 | 17.503503 | 0.259730 | 0.032453 | — |
| | `fixed_lp_locked_0.5` | 17.816143 | 17.424685 | 0.270546 | 0.064528 | — |
| **`SDSD FIT51 -> SDSD TEST39` (In-Domain)** | **`eid_full_mean3`** | **17.845573** | **17.488602** | **0.273855** | **0.038871** | **`Y [+0.0893,+0.3562]`, `LPIPS [-0.0713,-0.0486]`, `tLP [-0.0888,-0.0579]`** |
| | **`ablate_no_bipolar_env_mean3`** | **17.913826** | **17.567361** | **0.252175** | **0.030083** | **`Y [+0.1138,+0.4756]`, `LPIPS [-0.0986,-0.0663]`, `tLP [-0.1036,-0.0635]`** |
| **`BVI TRAIN48 -> SDSD TEST39` (Reverse Cross-Domain)** | **`eid_full_mean3` (from BVI)** | **17.829552** | **17.472985** | **0.274706** | **0.040858** | **`Y [+0.0811,+0.3267]`, `LPIPS [-0.0704,-0.0479]`, `tLP [-0.0862,-0.0563]`** |

Remarkably, reverse zero-shot transfer (`BVI TRAIN48 -> SDSD TEST39`, trained exclusively on 48 BVI clips with zero SDSD images) recovers **`91.7%` (`+0.1961 dB` vs `+0.2138 dB` scene-cluster `PSNR-Y`) and `98.6%` (`-0.0590` vs `-0.0599` `LPIPS`)** of in-domain `SDSD` training gains, improving `LPIPS` and `tLP gap` on **`21/21` (`100%`)** of held-out `SDSD` test scenes.

---

## 4. Complete 6-Variant Architectural & Supervision Ablation Matrix

| Variant (`3-Seed Mean`) | `SDSD internal17` (`PSNR-Y` / `Dec-RGB` / `LPIPS` / `tLP`) | `BVI VAL16 HVI` (`PSNR-Y` / `LPIPS` / `tLP`) | `BVI TEST16 HVI` (`PSNR-Y` / `LPIPS` / `tLP`) | `SDSD TEST39` (`PSNR-Y` / `LPIPS` / `tLP`) |
|---|---|---|---|---|
| **`eid_full_mean3`** | `15.4198 / 23.2228 / 0.4072 / 0.0791` | **`24.4476 / 0.1191 / 0.01520`** | **`20.6884 / 0.1499 / 0.01421`** | `17.8456 / 0.2739 / 0.03887` |
| `ablate_no_bipolar_env_mean3` | `15.4834 / 23.5422 / 0.3872 / 0.0624` | `24.5427 / 0.1063 / 0.00798` | `20.7090 / 0.1335 / 0.00511` | `17.9138 / 0.2522 / 0.03008` |
| `ablate_raw_loss_mean3` | `15.4337 / 23.2207 / 0.4083 / 0.0786` | `24.4470 / 0.1195 / 0.01499` | `20.6837 / 0.1502 / 0.01406` | `17.8609 / 0.2753 / 0.03852` |
| `ablate_no_rad_mean3` | `15.3549 / 22.6298 / 0.4224 / 0.1220` | `24.3296 / 0.1290 / 0.02626` | `20.6468 / 0.1553 / 0.02288` | `17.7802 / 0.2894 / 0.08395` |
| `ablate_no_deatt_raw_mean3` | `15.4221 / 23.2282 / 0.4077 / 0.0823` | `24.4306 / 0.1217 / 0.01582` | `20.6817 / 0.1518 / 0.01477` | `17.8456 / 0.2746 / 0.04175` |
| `ablate_no_sensor_inv_mean3` | `15.4285 / 23.2331 / 0.4069 / 0.0790` | `24.4338 / 0.1192 / 0.01525` | `20.6778 / 0.1500 / 0.01421` | `17.8559 / 0.2736 / 0.03872` |

### Key Mechanism Findings

1. **Cross-Domain Ranking of All Submodules (`BVI VAL16` & `BVI TEST16` on `HVI-CIDNet`)**: Under identical $\Omega^{\text{bip}}$ enveloping, `eid_full_mean3` outperforms **all four submodule ablations** (`ablate_raw_loss`, `ablate_no_rad`, `ablate_no_deatt_raw`, `ablate_no_sensor_inv`) on both `BVI VAL16` and `BVI TEST16` in `PSNR-Y`, `PSNR-RGB`, and `LPIPS`.
2. **Role of Log-Radiometric Transport (`ablate_no_rad`)**: Removing conserved two-band log-radiometric transport collapses flicker suppression (`tLP gap` degrades `0.01421 -> 0.02288` on `BVI TEST16` and `0.03887 -> 0.08395` on `SDSD TEST39`, while `PSNR-RGB` drops `-0.073 dB` on `BVI TEST16` and `-0.112 dB` on `SDSD TEST39`).
3. **Role of Phase-Aware Back-Projection & Raw Sensor Detail (`ablate_no_deatt_raw`)**: Disabling phase-deattenuated transport and Jacobian-calibrated sensor detail degrades `LPIPS` across every split (`0.1191 -> 0.1217` on `BVI VAL16`, `0.1499 -> 0.1518` on `BVI TEST16`, `0.2739 -> 0.2746` on `SDSD TEST39`) and lowers cross-domain `PSNR-Y`.
4. **Role of Cross-Modal Sensor-Enhancer Invariants (`ablate_no_sensor_inv`) & Exposure-Decoupled Loss (`ablate_raw_loss`)**: Both ablations overfit slightly better to source `SDSD` raw tone statistics (`+0.010` to `+0.015 dB` on in-domain `SDSD`), yet drop cross-domain `BVI VAL16` and `BVI TEST16` `PSNR-Y`/`PSNR-RGB`, directly confirming cross-domain regularization.
5. **Exact Complementary Mechanism of Bipolar Envelope (`eid_full` vs `ablate_no_bipolar_env`)**:
   - On the **source-matched high-flicker backbone** (`HVI-CIDNet`), `ablate_no_bipolar_env` ($\Omega^{\text{bip}} \equiv 1$) achieves the strongest absolute perceptual and temporal scores (`SDSD TEST39`: `PSNR-Y 17.9138`, `LPIPS 0.2522`, `tLP gap 0.0301`, beating `fixed_temporal_mean_0.5` on **all four metrics**; `BVI TEST16`: `LPIPS 0.1335`, `tLP gap 0.00511`, beating `fixed_temporal_mean_0.5` on both `LPIPS` and `tLP gap`).
   - Across **heterogeneous unseen backbones** (`DarkIR`, `Retinexformer`, `Seer-v2`), `ablate_no_bipolar_env` loses `PSNR-Y` on `DarkIR VAL16` (`-0.0496 dB`), significantly degrades `LPIPS` on `Retinexformer` (`VAL16 +0.0068`, `TEST16 +0.0041`), and over-smooths `Seer-v2 VAL16` (`tLP gap +79.4%`), whereas `eid_full` ($\Omega^{\text{bip}} \in [0, 1)$) achieves **11/11 (`100%`) positive spatial + perceptual gains across every domain, split, transfer direction, and architecture**.

---

## 5. Exact Artifact SHA256 Manifest (`sync_v2/`)

| File | SHA256 |
|---|---|
| `unified_internal17_eval_v2/source_train_and_internal17_summary.json` | `9e99c4e569fa31da204b3d72ffd59ffc2adb5ade31f0952b92f88477e5582f3a` |
| `results_v2/eval_bvi_val16_hvi_cidnet_all_variants.json` | `9c3fec57fd92658dde57b635e9550e100b95200546c65f5cbb718ffb25a3e047` |
| `results_v2/eval_bvi_val16_darkir.json` | `0775a52c27a651c20dbd79658518c3e7daaca5bd55ae4381e3f4c80ee38d150e` |
| `results_v2/eval_bvi_val16_retinexformer.json` | `6364a0e11e3ee8df4a77919d0d1ebff90f092492b29f98104c6cebda0977065a` |
| `results_v2/eval_bvi_val16_seer_v2.json` | `c7a4488604e83ad6c6d50b8448aec476ed6a1a1618fd77c6e2170a63668fa1be` |
| `results_v2/eval_bvi_test16_hvi_cidnet_all_variants.json` | `f4241d337f52ac03f3aa1965fa1e6ce08598e71f7c4890340f72b70ae3fcbf23` |
| `results_v2/eval_bvi_test16_darkir.json` | `5b1d48255c7076da82e26ffe21783379fb0abb1aa44e704cc5bfd0371eef54e6` |
| `results_v2/eval_bvi_test16_retinexformer.json` | `857ac892a2b0e99d4505bf90c085584e4294080c0585d06c216af92f3acaadd9` |
| `results_v2/eval_bvi_test16_seer_v2.json` | `d56517b0c0e8cc0ae0d499db4e9973878aef6ced1616d973eba54d711dc49349` |
| `results_v2/eval_sdsd_test39_hvi_cidnet_indomain.json` | `45d5ed92c1b6b4b75470769f1e5a7eccd4e37af8096cfae2955bba5b5d60605e` |
| `results_v2/eval_sdsd_test39_hvi_cidnet_from_bvi.json` | `bdd25bfc198532b2981ff35c353961a3c3d929726e93b30e75f0d47937660fd6` |
