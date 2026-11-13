# Pre-Registered Protocol: Photometric-Invariant Dual-Manifold Trajectory Transport (EID-TT v1)

## 1. Objective and Problem Diagnosis

Framewise low-light image restoration backbones (`HVI-CIDNet`, `DarkIR`, `Retinexformer`) achieve strong single-frame spatial restoration on unseen video domains (`BVI-RLV`), substantially exceeding legacy `Seer-v2` in `PSNR-Y` and `LPIPS`, yet exhibit severe inter-frame radiometric and structural flicker (`tLP gap` up to `0.0343`). Prior trajectory modules failed to resolve this trade-off for two structural reasons:

1. **Source-Domain Static Exposure Confound**: Frozen enhancers exhibit a clip-constant exposure/tone-curve offset relative to `SDSD` ground truth (`+6.79 dB` gap between raw and sequence-static exposure-decoupled `PSNR-RGB`). Supervising a fusion module with raw source `MSE` forces it to learn a source-specific tone shift that degrades cross-domain `PSNR`, while locking low-pass `LP(B)` leaves all low- and mid-frequency exposure flicker untouched.
2. **Subpixel Bilinear Attenuation & Scale-Sensitive Features**: Undirected high-frequency Laplacian smoothing damps high-frequency spatial contrast under fractional pixel motion (degrading `LPIPS`), and raw RGB moment features shift across domains and backbones.

## 2. Module Innovation: EID-TT (`eid_transport.py`, 14,861 Parameters)

Given observable canonical-RGB low-light input $X \in [0, 1]^{3\times T\times H\times W}$, frozen enhancer prediction $B \in [0, 1]^{3\times T\times H\times W}$, and GT-free reciprocal RAFT-small flows $\{f_{t\to t+\delta}\}_{\delta \in \{\pm 1, \pm 2\}}$:

1. **Pillar 1 — Cross-Modal Retinex Invariant & Two-Band Markov Log-Radiometric Equilibrium**:
   - Decompose smooth log-radiometry into two orthogonal spatial bands (`coarse` $25\times 25, \sigma=6.0$ and `mid` band-pass $7\times 7, \sigma=1.6$ minus `coarse`) with clip-adaptive relative floors $\epsilon_{\text{rel}} \mu_{\text{clip}}$.
   - Construct 16 strictly dimensionless observable donor features, led by the **Cross-Modal Retinex Flicker Invariant** $\mathcal{I}^{(s)}_{t,\delta} = |\Delta u^{(B,s)}_{t,\delta} - \Delta u^{(X,s)}_{t,\delta}|$, which cancels physical scene illumination/shadow changes shared by $X$ and $B$ while isolating enhancer-induced radiometric drift.
   - Predict anchored-simplex conductances $\sum_{\delta} c^{(s)}_{t,\delta} < 1$ and execute $M = 2$ Markov trajectory relaxation steps on $u^{\text{coarse}}, u^{\text{mid}}$, updating $B^{\text{rad}}_t = B_t \odot \exp(\delta u^{\text{coarse}}_t + \delta u^{\text{mid}}_t)$. By the Discrete Maximum Principle, constant exposure shifts cancel identically (zero static domain bias) and fine spatial edges of $B_t$ undergo zero resampling blur.
2. **Pillar 2 — Phase-Aware De-Attenuated Detail & Jacobian-Calibrated Sensor Transport**:
   - Compensate subpixel bilinear transfer attenuation via cycle-consistent forward-backward back-projection $H^{\text{deatt}}_{t\leftarrow s} = \mathcal{W}_{t\leftarrow s}(H_s + \beta_{\text{bp}} r^{\text{fb}}_{s,-\delta}(H_s - \mathcal{W}_{s\leftarrow t}\mathcal{W}_{t\leftarrow s} H_s))$.
   - Inject trajectory-aggregated raw sensor high-frequency detail calibrated by the dimensionless local RMS Jacobian ratio $\tilde{\kappa}_t$, satisfying exact two-parameter scaling equivariance $\mathcal{T}_\theta(\beta X, \alpha B) = \alpha \mathcal{T}_\theta(X, B)$ for all $\alpha, \beta > 0$ prior to display clamping.
3. **Pillar 3 — Sequence-Static Exposure-Decoupled Source Supervision (Train-Time Loss Only)**:
   - Align source `GT` $G_t$ to $B_t$ via a strictly time-invariant (`mean` over $T$, $33\times 33$ two-pass box-smoothed) clip exposure ratio $\rho_{\text{clip}}$ (`StopGrad`), supervised with spatial Charbonnier fidelity, multi-scale spatial gradient preservation, multi-scale Eulerian temporal difference matching, and Lagrangian trajectory innovation matching. Zero `GT` or target statistics enter inference.

## 3. Pre-Registered Experimental Matrix and Data Hygiene

- **Source Training Split (`SDSD FIT51`)**: `51` scenes (`102` clips) from `SDSD` train split (`seed = 20261114`).
- **Source Internal Holdout (`SDSD internal17`)**: `17` unseen `SDSD` scenes (`34` clips), evaluated strictly post-training without checkpoint cherry-picking (`final_checkpoint.pt` at epoch 12).
- **Matched Seeds & Variants (`3 seeds: 0, 1, 2`)**:
  1. `eid_full`: Full EID-TT + Exposure-Decoupled Source Loss.
  2. `ablate_raw_loss`: Full EID-TT + unaligned raw canonical-RGB GT loss (isolates Pillar 3).
  3. `ablate_no_rad`: Disable log-radiometric Markov transport (isolates Pillar 1).
  4. `ablate_no_deatt_raw`: Disable phase-aware back-projection and raw sensor detail injection (isolates Pillar 2).
  5. `ablate_no_sensor_inv`: Remove cross-modal sensor-enhancer log-ratio invariant and sensor SNR correlation (isolates cross-modal physical verification).
  6. Non-learned controls on identical flow geometry: `anchor`, `fixed_temporal_mean_0.5`, `fixed_lp_locked_0.5`.
- **Zero-Shot Cross-Domain & Cross-Backbone Generalization**:
  - Primary Cross-Domain Validation (`BVI VAL16`, `8` scenes $\times$ `2` conditions, `centre16/crop256`, rounded `uint8` `eval_lib` scorer): Evaluate the exact `SDSD + HVI-CIDNet`-trained 3-seed weights zero-shot across **four frozen backbones**: `HVI-CIDNet`, `DarkIR`, `Retinexformer`, and `Seer-v2 Pretrained`.
  - Locked Single-Pass Final Test Benchmarks (opened once after all training and validation checks freeze):
    - `BVI TEST16` (`8` held-out scenes $\times$ `2` conditions): Zero-shot cross-domain & cross-backbone test evaluation.
    - `SDSD TEST39` (`39` held-out test clips in canonical RGB): In-domain & reverse cross-domain (`BVI TRAIN -> SDSD TEST39`) test evaluation.
- **Statistical Verification**: 10,000-resample scene-cluster bootstrap 95% confidence intervals (`seed = 20261112`) pairing conditions within each physical scene.
