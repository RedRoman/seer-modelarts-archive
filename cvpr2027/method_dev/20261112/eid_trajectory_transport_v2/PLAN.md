# Protocol: Bipolar Innovation-Enveloped Photometric-Invariant Trajectory Transport (EID-TT v2)

## 1. Objective and Validation-Stage Boundary Discovery

Framewise low-light enhancers (`HVI-CIDNet`, `DarkIR`, `Retinexformer`) achieve high spatial fidelity on unseen target domains (`BVI-RLV`) compared with historical `Seer-v2`, yet suffer substantial inter-frame radiometric and structural flicker (`tLP gap` up to `0.0343`).

In `eid_trajectory_transport_v1` (preserved intact without modification and with zero `TEST` split access), source-only training on `SDSD FIT51` (`HVI-CIDNet`) validated positive cross-domain gains on `HVI-CIDNet` (`BVI VAL16`: `+0.219 dB` `PSNR-Y`, `+0.279 dB` `PSNR-RGB`, `-0.0484` `LPIPS`, `-79.6%` `tLP gap`), while cross-backbone `BVI VAL16` diagnostics exposed two exact mathematical boundary-condition omissions:

1. **Missing Geometric Cycle-Consistency Multiplication**: Effective conductances multiplied `valid_mask` without `fb_conf`, permitting non-zero donor blending across internal motion-occlusion boundaries.
2. **Missing Null-Flicker Boundary Envelope**: Because every `SDSD FIT51` (`HVI-CIDNet`) training clip exhibits heavy flicker (`tLP gap = 0.147`), unconstrained `Conv2d` biases learned unconditional diffusion conductances (`~0.80`), over-smoothing `tLP` below `gt_tlp` when transferred zero-shot to already temporally smooth backbones (`Seer-v2`, `Retinexformer`) or clean `20fps` clips (`DarkIR`).

`EID-TT v2` resolves both boundary conditions from first principles while maintaining zero `GT` access and zero target-domain adaptation at inference.

## 2. Module Innovation: Bipolar Innovation-Enveloped Dual-Manifold Transport (`eid_transport.py`)

Given observable canonical-RGB low-light input $X \in [0, 1]^{3\times T\times H\times W}$, frozen enhancer prediction $B \in [0, 1]^{3\times T\times H\times W}$, and GT-free reciprocal RAFT-small flows $\{f_{t\to t+\delta}\}_{\delta \in \{\pm 1, \pm 2\}}$:

1. **Pillar 1 — Unbiased Lag-1 De-Attenuated Bipolar Innovation Envelope ($\Omega^{\text{bip}}$)**:
   - For each scale-normalized band $Z \in \{u^{\text{coarse}}_B, u^{\text{mid}}_B, \tilde{H}_B\}$, compute symmetric lag-1 phase-deattenuated innovations $d_{t,\pm 1}(Z) = \mathcal{W}^{\text{deatt}}_{t\leftarrow t\pm 1}(Z_{t\pm 1}) - Z_t$ weighted by reciprocal cycle confidence $r_{t,\pm 1} = r^{\text{fb}}_{t,\pm 1} m_{t,\pm 1}$.
   - By trajectory kinematics, monotonic scene motion or illumination velocity $v$ yields opposite-sign innovations $\langle d_{t,-1}, d_{t,+1} \rangle = -\|v\|^2 \le 0$, whereas uncorrelated frame-wise enhancer flicker $\eta_t$ yields positive covariance $\mathbb{E}[\langle d_{t,-1}, d_{t,+1} \rangle] = +\sigma_{\eta,t}^2 > 0$.
   - Spatiotemporally pool the unclamped inner product $\overline{C}(Z)$ and symmetric energy $\overline{E}(Z)$ prior to `ReLU` (eliminating Jensen positivity bias) to form the closed-form Minimum-MSE Wiener envelope:
     $$\Omega^{\text{bip}}(Z) = \frac{2\,\text{ReLU}(\overline{C}(Z))}{\overline{E}(Z) + \text{ReLU}(\overline{C}(Z)) + \tau_Z^2} \in [0, 1)$$
     which vanishes identically ($\Omega^{\text{bip}} = 0$) on flicker-free trajectories while recovering the optimal symmetric shrinkage weight on flickering clips, and is strictly invariant under $(X, B) \mapsto (\beta X, \alpha B)$.
2. **Pillar 2 — Doubly-Enveloped Log-Radiometric & Phase-Aware De-Attenuated Detail Transport**:
   - Feed 19 strictly dimensionless features (16 cross-modal Retinex, structural SNR, and flow-phase features plus 3 bandwise $\Omega^{\text{bip}}$ maps) into a compact donor-consensus network (`15,509` parameters).
   - Gate predicted anchored-simplex conductances multiplicatively by both $\Omega^{\text{bip}}$ and $r^{\text{fb}}_{t,\delta} m_{t,\delta}$, executing conserved log-radiometric transport (`coarse` Gaussian + `mid` phase-deattenuated Laplacian) followed by phase-deattenuated high-frequency detail transport and Jacobian-calibrated raw sensor detail injection.
3. **Pillar 3 — Sequence-Static Exposure-Decoupled Source Supervision (Train-Time Loss Only)**:
   - Supervise on `SDSD FIT51` (`102` clips) against sequence-static exposure-aligned `GT` (`StopGrad` $33\times 33$ smooth temporal-mean ratio) with spatial Charbonnier fidelity, multi-scale edge preservation, Eulerian temporal innovation matching, and Lagrangian trajectory innovation matching.

## 3. Experimental Matrix and Locked Final Test Protocol

- **Source Training (`SDSD FIT51`, `102` clips) & Internal Holdout (`SDSD internal17`, `34` clips)**:
  6 matched variants $\times$ 3 seeds (`0, 1, 2` = `18` source checkpoints) + 3 non-learned controls (`anchor`, `fixed_temporal_mean_0.5`, `fixed_lp_locked_0.5`):
  1. `eid_full`
  2. `ablate_no_bipolar_env` (sets $\Omega^{\text{bip}} \equiv 1$, isolating Pillar 1 cross-backbone self-calibration)
  3. `ablate_raw_loss` (unaligned raw `GT` loss, isolating Pillar 3)
  4. `ablate_no_rad` (disables log-radiometric transport)
  5. `ablate_no_deatt_raw` (disables phase-aware back-projection and raw sensor detail injection, isolating Pillar 2)
  6. `ablate_no_sensor_inv` (removes cross-modal sensor-enhancer Retinex discrepancy & sensor SNR correlation)
- **Reverse Cross-Domain Training (`BVI TRAIN48`, `48` clips)**: 3 seeds (`0, 1, 2`) of `eid_full` trained strictly on `BVI TRAIN48` with zero `VAL/TEST` or `SDSD` access.
- **Cross-Domain & Cross-Backbone Validation (`BVI VAL16`)**: Zero-shot evaluation across `HVI-CIDNet`, `DarkIR`, `Retinexformer`, and `Seer-v2 Pretrained`.
- **Single-Pass Locked Final Test Benchmarks (`--allow-final-test`)**: Executed strictly once after all checkpoints and `VAL16`/`internal17` tables freeze:
  - `BVI TEST16` across all 4 frozen backbones (`HVI-CIDNet` full ablation + `DarkIR`, `Retinexformer`, `Seer-v2`).
  - `SDSD TEST39` in-domain (`SDSD FIT51 -> SDSD TEST39`) and reverse cross-domain (`BVI TRAIN48 -> SDSD TEST39`).
