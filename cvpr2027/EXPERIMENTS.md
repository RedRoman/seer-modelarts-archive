# Seer -> CVPR 2027: experiment configuration and locations (2026-11-11)

Every converged experiment of the CVPR round: what it changes, where it ran,
where its checkpoints and evaluation JSONs live on the server, and where the
archived copy is in this repository. Verdicts are in `STATUS.md`; the
pre-registered rule each one was judged by is in `notes/CVPR_PREREG.md`
(20 amendments, each written before the result it governs).

## 1. Hosts

All hosts are `ma-user@dev-modelarts-cnnorth9.huaweicloud.com`, key
`KeyPair-liusonghua.pem`; `~/.ssh/config` has an alias `huawei-modelarts-<port>`
for each.

| port | hardware | role | state on 2026-11-11 |
|---|---|---|---|
| **31051** | Ascend 910B3 x8, 64 GB each | E17, E18 (spatial / hybrid / seed-2 edc, layers_all), E20, E26, all probes on seeds 0-1 | reachable |
| **32677** | A100-80GB x4, shared with other tenants | E18 edc seeds 0-1, E19, E21, E22, E24, E25, E2c, E28, Retinexformer retrain, td probes, qualitative figure | **unreachable since 2026-11-05** (`kex_exchange_identification: Connection closed by remote host`) |

`/cache` is a local disk on each port, **not** shared across ports (checked
2026-11-05: 30141's `/cache/zihaohe/seer_cvpr/` contains only `weights/`).
`/temp` on 31051 is SFS Turbo. Only 31051 has a complete CANN operator package;
the other Ascend hosts fail with 561103.

### 31051 (Ascend) paths

| what | path |
|---|---|
| code | `/temp/zihaohe/seer_cvpr/code/vjepa_llve` (archived as `code_full/`) |
| python | `/cache/zihaohe/seer_cvpr/venv_npu/bin/python` |
| runs | `/cache/zihaohe/seer_cvpr/runs_{e4,e12,e16,e17,e18,e20,e26}/<tag>/final.pt` (e17/e18/e4/e12/e16 as `final_cpu.pt`) |
| results | `/cache/zihaohe/seer_cvpr/results/` |
| datasets | `/temp/zihaohe/seer_cvpr/datasets/` (`indoor_static_np`, `outdoor_static_np`, `BVI-Lowlight-videos`, `lol`) |
| frozen splits | `/temp/zihaohe/seer_cvpr/datasets/splits_cvpr.json`, `splits_lol.json` |
| V-JEPA 2.1 ViT-L | `/temp/zihaohe/seer_cvpr/weights/vjepa2_1_vitl_dist_vitG_384.pt`, repo `/temp/zihaohe/seer_cvpr/modelzoo/vjepa2-main` |
| RAFT (warp error) | `/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth` |
| runner scripts | `/cache/zihaohe/*.sh` (archived as `runners/host31051_ascend/`) |
| env for eval | `SEER_ROOT=/temp/zihaohe/seer_cvpr RAFT_WEIGHTS=<RAFT path above> ASCEND_RT_VISIBLE_DEVICES=<n>` |

### 32677 (A100) paths

| what | path |
|---|---|
| code | `/cache/zihaohe/seer_cvpr/code/vjepa_llve` (the CUDA tree; it led the Ascend tree by 127 lines on 11-01 and was synced into it, so `code_full/` contains every change that ran here) |
| python | `/cache/zihaohe/seer_cvpr/venv_cu/bin/python` |
| runs | `/cache/zihaohe/seer_cvpr/runs_e19/<tag>/` for E19, E21, E22, E24, E25, E2c, E28; `runs_e18/` for edc seeds 0-1 |
| results | `/cache/zihaohe/seer_cvpr/results/{e17_eval_test,e18_eval_test,e21_eval_test,h17_eval_test,wave_eval_test,td_probe,qualitative_cvpr,test_indomain_sdsd}` |
| Retinexformer, retrained 160k it on our split | `/cache/zihaohe/seer_cvpr/weights_small/checkpoint/runs/retinexformer_seer_protocol/experiments/RetinexFormer_SDSD_union_seer_protocol/models/net_g_160000.pth` |
| runner scripts | `/cache/zihaohe/*.sh` (not reachable; recovered verbatim from the session transcript into `runners/host32677_a100/`) |
| env | `SEER_ROOT=/cache/zihaohe/seer_cvpr`, `SCRIPT=tools/seer_train_cvpr.py` must be exported before `e19.sh` |

## 2. Shared protocol

| item | setting |
|---|---|
| in-domain training | `tools/seer_train_cvpr.py`, trains on SDSD train (136 sequences), validates on SDSD val (16) every 1000 it, **not** used for selection |
| cross-domain training | `tools/seer_train_e4.py --heldout <d>`, leave-one-domain-out, the two remaining domains sampled with p = 0.5 each |
| schedule | 20000 iterations, gradient accumulation 8, `final.pt` scored (no checkpoint selection) |
| architecture | `--arch v2 --branch small`: 5-scale Restormer-style U-Net, dims (20,40,80,160,320), blocks (2,4,4,6,8), heads (2,4,4,8,8), 22.4M trainable; frozen V-JEPA 2.1 ViT-L; zero-init residual cross-attention at scales 1-4; taps (5,11,17,23); `--xattn global --kv per_tubelet` |
| clip | 16 frames, 256 crop, tubelet 2 -> token grid 8 x 16 x 16 = 2048 tokens x 1024 dims per tap |
| evaluation | `tools/eval_e4.py --split test --crop 256`, centre 16-frame clip; PSNR-Y on uint8, LPIPS (pyiqa AlexNet), GT-mean, tLP, warp error (RAFT); the test split is read once, on final checkpoints |
| statistics | paired over sequences, seed-averaged, 10k bootstrap 95% CI, and every seed must agree in sign (ground rule 4); `tools/aggregate.py`, paper numbers via `paper/tools/make_macros_cvpr.py` |
| splits | SDSD indoor 46/6/14, outdoor 90/10/25 (3 sequences dropped for scene leakage), BVI-RLV 48/16/16 by scene (24/8/8), LOL v1 / v2-real / v2-syn 436/49/15, 620/69/100, 810/90/100 |

## 3. Training experiments (all 20000 it, 22.4M branch unless stated)

Baseline for every in-domain comparison is the three-seed pretrained arm
(E17 s0, s1 + E19 s2). Archived JSONs are under `results/cvpr_converged/`.

| id | hypothesis | change vs baseline | seeds | host | server run dir | server eval JSON | archived |
|---|---|---|---|---|---|---|---|
| E17 | prior vs none vs random at convergence | `--vjepa {pretrained,none,random}` | p 0,1; n 0,1; r 0 | 31051 | `runs_e17/e17_small_<arm>_s<S>` | `results/e17_eval_test/` | `indomain/` |
| E19 | third seed for E17 | `--vjepa {pretrained,none} --seed 2` | 2 | 32677 | `runs_e19/e19_seed2_<arm>` | `results/e17_eval_test/e19_small_<arm>_s2.json` | `indomain/` |
| E18 | injection variants | `--xattn spatial` / `--kv hybrid` / EDC on (weight 1.0) / `--layers all` | spatial 0,1; hybrid 0,1; edc 0,1 (32677) + 2 (31051); layers_all 2 | 31051, 32677 | `runs_e18/e18_<variant>_s<S>`, `e18_edc_npu_s0` = seed 2, `e18_layers_all_npu_s0` = seed 2 | `results/e18_eval_test/` | `injection/` |
| E20 | H5, cross-domain | `seer_train_e4.py --heldout {sdsd,bvi,lol} --vjepa {pretrained,none}` | 0,1 | 31051 | `runs_e20/e20_<dom>_<arm>_s<S>` | `results/e20_eval_test/` | `xdomain/` |
| E21 | H16, trust gate | `--xattn gated` | 0,1 | 32677 | `runs_e19/e21_gated_s<S>` | `results/e21_eval_test/` (+ `_randtok`, `gate_audit`) | `injection/` |
| E22 | H17, latent temporal loss (relative form) | `--tdelta 0.3` | 0,1 | 32677 | `runs_e19/e22_tdelta_s<S>` | `results/h17_eval_test/` | `h17/` |
| E24 | H17, latent temporal loss (cosine form) | `--tdelta 0.3 --tdelta_form cosine` | 0,1 | 32677 | `runs_e19/e24_tdcos_s<S>` | `results/h17_eval_test/` (+ `_tdprobe`, `_order`) | `h17/` |
| E25 | H16', gate trained with corrupted tokens | `--xattn gated --token_corrupt 0.3` (previous clip's tokens; realised 30.0%) | 0,1 | 32677 | `runs_e19/e25_cgate_s<S>` | `results/wave_eval_test/` | `wave/` |
| E2c | tap depth at convergence | `--taps 23,23,23,23` | 0,1 | 32677 | `runs_e19/e2c_deep_s<S>` | `results/wave_eval_test/` | `wave/` |
| E26 | H18, fixed low-rank tokens trained | `--token_proj global_rank --token_proj_k 64 --token_basis results/token_basis_train.json` | 0,1 | 31051 NPU 0,1 | `runs_e26/e26_grank64_s<S>` | `results/e26_eval_test/` | `e26/` |
| E27 | H19, distil the code into a small predictor | -- | -- | -- | -- | -- | **cancelled** by its pre-registered gate (Amendment 19) |
| E28 | H20, per-clip rank-4 tokens trained | `--token_proj clip_rank --token_proj_k 4` | 0,1 | 32677 GPU 0,1 | `runs_e19/e28_cliprank4_s<S>` | **none** -- 13300/20000 at 11-03 15:25, host lost 11-05, no eval was queued | -- |

Launch pattern on 32677, for reproduction:
```
export SCRIPT=tools/seer_train_cvpr.py
./e19.sh <GPU> "<tag>:--arch+v2+--branch+small+--vjepa+pretrained+<flags joined by +>+--seed+<S>"
```
On 31051 the equivalent runners are `e17_runner.sh`, `e18_npu.sh`, `e20.sh`, `e26.sh`.

## 4. Inference-only probes

| probe | what it measures | tool | host / server path | archived |
|---|---|---|---|---|
| token structure | across-tubelet distance, frozen-input control, shuffle change per tapped block | `tools/token_probe.py` | 31051 `results/token_probe_sdsd.json` | `probes/` |
| frame-order intervention | intact / shuffled / frozen, prior vs none | `tools/temporal_ablate.py` | 31051 `results/e14_converged_sdsd.json` | `probes/` |
| latent delta alignment | td, mag, cos of the prediction's tubelet deltas vs GT | `tools/td_probe.py` | 32677 `results/td_probe/` | final-checkpoint probes in `h17/`; baseline, step-5k/10k and ceiling JSONs only on 32677 (numbers in Amendments 14 and 17) |
| delta-field ceiling | same, for GT under gamma / noise / 1-px shift | `tools/td_ref.py` | 32677 `results/td_probe/reference_ceiling.json` | not archived (numbers in Amendment 14) |
| gate audit | mean gate value under real vs random tokens | `tools/gate_audit.py` | 32677 `results/{e21_eval_test,wave_eval_test}/` | `injection/`, `wave/` |
| per-clip sufficiency | per-clip rank k, spatial mean, temporal mean | `tools/token_sufficiency.py` | s0, s1 31051 `results/`; s2 32677 | `sufficiency/` (all three seeds) |
| fixed basis | train-split PCA of the tokens, 60 clips, rank 64 | `tools/global_basis.py` | 31051 `results/token_basis_train.json` | `probes/` |
| fixed vs per-clip projection | both projections in one pass | `tools/global_sufficiency.py` | 31051 `results/global_sufficiency_s{0,1}.json` | `sufficiency/` (s2 version only on 32677) |
| qualitative figure | 4 sequences at margin percentiles 10/35/60/85 vs retrained Retinexformer | `tools/dump_qual_cvpr.py` | 32677 `results/qualitative_cvpr/` (20 PNG) | **not archived**; reproducible on 31051 from E17 checkpoints, rule unchanged |

## 5. What is not in this repository

- Checkpoints (90 MB each for the 22.4M branch): on the hosts only. The 31051
  ones are intact; the 32677 ones (E19, E21, E22, E24, E25, E2c, E28, edc s0-1,
  Retinexformer retrain) are inaccessible while the host is down.
- The 32677-side JSONs listed as "not archived" above. Every number from them
  that the paper uses is recorded in `notes/CVPR_PREREG.md`.
- Datasets and V-JEPA weights (public).
