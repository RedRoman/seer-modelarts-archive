# V-JEPA-LLVE Phase 2 — Project Progress

**Last update:** 2026-05-24 17:30 (v14-v16 mechanism search reviewed, next target re-parameterization)
**Target:** CVPR 2027 (deadline ~Nov 2026)

---

## TL;DR

**当前 PSNR SOTA: v10-seed0 = 21.578** (final ckpt 11-seq, NO AdaIN)
**当前 perception SOTA: v12a-seed42 = CLIPIQA 0.2528** (LPIPS 0.2333, DISTS 0.2164)
**V-JEPA TruePAC 单变量贡献: +1.71 dB** (v9 21.523 vs wanlora_novjepa 19.81)

**2026-05-24 update:** v14-v16 已验证多条后续机制。v14 spatial 指标最高但主观闪烁/轮廓变差；v15/v15b temporal transport 有局部序列信号但会伤害 pair31 等保守序列；v16/v16a side-condition 未带来稳定增益。当前结论是：继续在 conditioning/adapter 上做增量补丁不够，下一步应改训练目标/输出参数化，优先验证 residual enhancement diffusion。

主架构: Wan2.1-T2V-1.3B (frozen) + LoRA r64 + V-JEPA 2.1 ViT-L (frozen) + TruePAC predictor.

---

## 1. 完整 11-seq IQA 主表 (single-seed — except multi-seed below)

| Run | iters | PSNR | SSIM | LPIPS↓ | DISTS↓ | CLIPIQA↑ | ckpt |
|---|---|---|---|---|---|---|---|
| Phase 1 (LoRA only, no V-JEPA-2.1) | 25k | 18.42 | 0.726 | 0.320 | 0.253 | 0.225 | best |
| NAFNet++ (prior SOTA) | 30k | 20.71 | 0.742 | 0.261 | 0.241 | 0.242 | — |
| wanlora_novjepa-seed42 | 12k | 19.81 | 0.765 | 0.270 | 0.242 | 0.215 | final |
| **v9-seed42 (TruePAC) ★ pre-v10 SOTA** | 12k | **21.523** | 0.7621 | 0.2565 | 0.2443 | 0.2137 | best (iter 7k) |
| v9-seed42 (TruePAC) | 12k | TBD | TBD | TBD | TBD | TBD | final (NOT measured) |
| v10-seed42 (multilayer+LPIPS+4-step) | 12k | 21.053 | 0.7644 | 0.2431 | 0.2350 | 0.2279 | final |
| v11a-12k (logit-normal A1) | 12k | 20.769 | — | — | — | — | final |
| v11a-12k best.pt (iter 4k) | 4k | 20.544 | 0.7459 | 0.2929 | 0.2714 | 0.2148 | best |
| **v12a-seed42 (AdaLN-Zero) ★ Perception-SOTA** | 12k | 21.065 | **0.7686** | **0.2333** | **0.2164** | **0.2528** | final |
| v12b (MeanFlow) | 12k | 21.344 | 0.7647 | 0.2526 | 0.2371 | 0.2162 | final |
| v11 (A1+A2 bundled) | 9k (killed) | — | — | — | — | — | failed |
| v11a-3k (A1 only) | 3k | 20.845 | 0.7542 | 0.2911 | 0.2725 | 0.2339 | final |
| v11b-3k (A2 only) | 3k | 20.166 | 0.7486 | 0.3052 | 0.2750 | 0.2075 | final |

## 2. Multi-seed 11-seq (final ckpt @ iter 12000)

| Method | seed | PSNR | SSIM | LPIPS↓ | DISTS↓ | CLIPIQA↑ |
|---|---|---|---|---|---|---|
| v9 | 0 | 20.590 | 0.7608 | 0.2712 | 0.2530 | 0.2195 |
| v9 | 100 | 20.789 | 0.7592 | 0.2612 | 0.2463 | 0.2094 |
| v9 | mean (0,100) | 20.690 ± 0.10 | — | — | — | — |
| v12a | 0 | 20.586 | 0.7587 | 0.2636 | 0.2457 | 0.2267 |
| v12a | 100 | **21.396** | 0.7574 | 0.2537 | 0.2391 | 0.2206 |
| v12a | mean (0,100) | 20.991 ± 0.40 | — | — | — | — |
| **v10** | **0** | **21.578** ★ | 0.7664 | 0.2422 | 0.2343 | 0.2396 |
| wanlora_novjepa | 0 | 20.714 | 0.7641 | 0.2620 | 0.2332 | 0.2395 |

**惊人发现**: v10-seed0 final = 21.578, **比 v9-seed42 best.pt (21.523) 还高**. v10 (multilayer + LPIPS + 4-step) 在多 seed 下证实是最强 PSNR 路径.

## 3. 单变量 ablation 结论

| 改动 | 结果 | 评估 |
|---|---|---|
| **TruePAC** (v9) | +1.71 dB over wanlora_novjepa | ✓ keep, 主新意 |
| **multi-layer V-JEPA + LPIPS + 4-step** (v10) | PSNR 21.58 SOTA, perception 居中 | ✓ keep, current 最强 PSNR |
| **AdaLN-Zero** (v12a) | -0.46 dB PSNR / +0.04 CLIPIQA | ✓ keep, perception 路径 |
| Logit-normal t (v11a) | -0.98 dB PSNR | ✗ reject (1-NFE 不匹配) |
| lq_cond ResBlock3D (v11b) | -1.36 dB PSNR | ✗ reject (干扰 frozen DiT 输入) |
| MeanFlow (v12b) | ≈ v9 (neutral) | ✗ reject (straight rect-flow 上冗余) |
| DoRA (v6) | -0.35 dB | ✗ reject |
| Q-Former (v7) | -1.0 dB | ✗ reject (cross-attn 冗余) |
| PAC broken (v8) | -0.33 dB | ✗ reject (实现 bug) |
| EMA (v5+EMA) | -2 dB | ✗ reject (decay 太低) |

## 4. Pareto frontier (paper 主 figure)

```
PSNR ↑
21.58 ● v10  (multi-layer + LPIPS + 4-step, perception 居中)
21.52 ● v9-seed42 best.pt  (TruePAC, distortion-optimal — but best.pt iter mismatch)
21.40 ● v12a-seed100  (AdaLN-Zero, seed 方差大)
21.07 ● v12a-seed42  (perception-optimal, CLIPIQA 0.253 ★)
20.71 ● NAFNet++  (prior SOTA)
20.69 ● v9-mean (multi-seed)
19.81 ● wanlora_novjepa  (无 V-JEPA baseline)
18.42 ● Phase 1
                                       
                                       CLIPIQA ↑
       0.214  0.221  0.228  0.240  0.253
       v9     v12a-100 v10  wanlora  v12a-42
```

## 5. 已废弃 / 失败案例 (诚实记录)

- **v3/v3.1/v3.2 (ControlNet)**: 多 loss 同时 collapse, 训坏. Code cleanup 已删 ControlNet.
- **v4 (norm unfreeze)**: -0.61 dB vs v5 (freeze norm).
- **v6 (DoRA)**: -0.35 dB, DoRA 在我们 setup 中无 LoRA 增益.
- **v7 (Q-Former)**: -1.0 dB, Q-Former 与 DiT cross-attn 冗余.
- **v8 (PAC broken)**: 实现 bug — RoPE 关闭、mask_tokens 未用. 修正版 v9 (TruePAC) 才正确.
- **v11 (A1+A2 bundled)**: A1+A2 互相干扰, trajectory 18-21 oscillate, killed @ iter 9k.
- **v11a (A1 logit-normal)**: train 时 t 集中在中间, inference t=0 欠学 → -1 dB PSNR.
- **v11b (A2 lq_cond residual)**: 给 frozen DiT 加 input offset, 训练不稳定 → -1.36 dB.
- **v12b (MeanFlow)**: rectified flow trajectory 是直线, consistency loss 冗余 → 无增益.
- **v13a (SpatialAdaLN) v1**: shape unpack bug, 跑了 0 iter 就 crash. 现在已 fix.
- **Wan-Full (full 1.42B FT)**: 启 5 min 后用户判断与 LoRA 不公平 → 改成 wanlora_novjepa baseline.

## 6. Pending / In-progress (2026-05-17 snapshot)

| 任务 | 状态 |
|---|---|
| v13a SpatialVJepaAdaLN (修复后重启) | 待 launch (~3h) |
| **best.pt 11-seq IQA batch eval** (6 multi-seed runs) | 待 batch (~30 min) |
| **v9-seed42 final ckpt 11-seq** (补) | 待 |
| **v9-seed42, v12a-seed42 best.pt 11-seq** (补完 seed 一致性) | 待 |
| 多数据集 eval (SMID, LRR, SDSD-indoor) | 待 (5-7 天) |
| MGLD-VSR / LLVE-STCD 直接对比 | 待 (3-5 天, gdrive 阻塞) |
| TruePAC attention map visualization | 待 (2 天) |
| User study (5-10 人 × 12 clips) | 选做 |

## 7. 下一步规划 (2026-05-17 snapshot; superseded by §10.7)

### 立即 (今天 22:00-明天 12:00)

1. **修 v13a bug + relaunch** (3h)
2. **写 best.pt batch eval 脚本 + 跑 6 runs** (30 min)
3. **补跑 v9-seed42 final ckpt 11-seq + v12a-seed42 best.pt 11-seq** (10 min each)
4. **整理 paper Pareto frontier 主表** (mean ± std)

### 短期 (本周 5/18-5/24)

5. **数据扩充: SMID + LRR 数据集 pipeline**
6. **MGLD-VSR / LLVE-STCD ckpt 重新获取 + 直接对比**
7. **TruePAC analysis: predictor attention map figure** (paper 解释力)

### 中期 (5/25-6/15)

8. paper 一稿 (focus v9 + v10 + v12a Pareto)
9. 第二个 v13 idea (待 v13a 结果决定)
10. multi-seed 扩到 5+ seeds (paper rebuttal 必备)

### 长期 (6 月底前)

11. user study
12. paper 二稿 → 投稿

## 8. 资源 / 环境

- **Host**: 30138 (huawei modelarts)
- **GPU**: A100 80GB single, cgroup RAM 125 GB
- **Train script**: `tools/v11_wanvj_phase2_v3_train.py` (clean, ControlNet 已删)
- **Wan path**: `/cache/zihaohe/checkpoint/wan21_t2v_1_3b`
- **V-JEPA 2.1 ckpt**: `/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt`
- **BVI-RLV**: `/cache/zihaohe/datasets/BVI_RLV/extracted/...` (143 GB)
- **SDSD**: 123 train + 11 held-out (pair13/15/21/23/31/33/50/52/58/60/70, pair68 excluded)
- **Inference**: 1-NFE rectified flow (default), 4-step Euler optional (v10)

## 9. 关键代码 modules (v3 trainer)

- `TruePACProjector` — 完整 12-block V-JEPA predictor frozen + text_proj
- `MultiLayerReducer` — channel-concat 多层 V-JEPA fusion
- `VJepaAdaLN` — pooled V-JEPA AdaLN-Zero (v12a)
- `SpatialVJepaAdaLN` — per-token spatial V-JEPA AdaLN-Zero (v13a)
- `LQCondResBlock3D` — v11b 已 reject
- inject_lora_on_dit — LoRA r64 on q/k/v/o + ffn.0/2
- `_attach_adaln_hooks` — forward hook 把 (γ, β) 应用到 DiT block 输出
- (REMOVED) WanLQControlNet, _attach_controlnet_hooks, INJECT_POSITIONS

---

## 10. 2026-05-24 接手后进展与发现

### 10.1 事实校正

- **21.578 不是 v9-seed42 的结果，而是 v10-seed0 final ckpt 的 11-seq 结果。**
- v9-seed42 best.pt 是 21.523；v10-seed0 final 是当前更稳的 PSNR SOTA。
- 后续分析均以 v10-seed0 final 作为主 baseline，而不是以 v9-seed42 best.pt 作为唯一 baseline。

### 10.2 v14-v16 主实验结果

| Run | 核心机制 | PSNR | SSIM | LPIPS↓ | DISTS↓ | CLIPIQA↑ | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| **v10-seed0** | multilayer V-JEPA + TruePAC + LPIPS + 4-step | **21.578** | 0.7664 | 0.2422 | 0.2343 | 0.2396 | 当前稳定 PSNR baseline |
| **v14 spatial** | clean token adapter + SpatialAdaLN | **21.874** | **0.7694** | **0.2259** | 0.2302 | 0.2394 | 指标最高，但主观闪烁/细节轮廓不佳 |
| v14b clean-only | clean token only | 20.904 | 0.7584 | 0.2572 | 0.2395 | 0.2274 | 可视化较干净，但指标不足 |
| v15 temporal transport | temporal neighborhood consensus 改写 V-JEPA tokens | 21.103 | 0.7663 | 0.2432 | 0.2315 | 0.2162 | pair15 明显增益，但 pair31 大幅受损 |
| v15b adaptive transport | per-token signed router 改写 V-JEPA tokens | 20.979 | 0.7643 | 0.2534 | 0.2410 | 0.2267 | router 未学出有效选择性 |
| v16 side | temporal side residual, from scratch | 20.562 | 0.7679 | 0.2532 | 0.2401 | 0.2170 | 从头训练不稳，机制未成立 |
| v16a side-v10init | v10 init + freeze base + side only | 21.487 | 0.7650 | 0.2430 | 0.2341 | 0.2385 | 基本贴近 v10，但没有实质增益 |
| v16a best.pt | same as above, best checkpoint eval | 21.523 | 0.7661 | 0.2424 | 0.2339 | 0.2387 | 仍略低于 v10 |

### 10.3 关键 per-seq 发现

| Run | pair15 | pair23 | pair31 | pair33 | pair52 | pair60 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| v10-seed0 | 18.288 | 24.641 | **25.333** | 24.563 | 15.147 | **21.820** | baseline |
| v14 spatial | 15.434 | **30.613** | 21.965 | **28.165** | 15.017 | 21.966 | pair23/33 拉高均值，但主观不稳 |
| v14b clean-only | 15.108 | 25.225 | 24.310 | **28.242** | 15.079 | 21.234 | pair33 强，整体不足 |
| v15 transport | **22.024** | 24.749 | 19.236 | 23.194 | **15.786** | 20.812 | pair15 +3.736，但 pair31 -6.097 |
| v15b adaptive | 19.093 | 26.525 | 20.240 | 25.750 | 15.620 | 18.576 | 局部增益，代价仍大 |
| v16a best.pt | 18.013 | 24.669 | 25.117 | 24.610 | 15.149 | 21.821 | 近似回到 v10，无新能力 |

结论：v14/v15 的机制确实能在部分序列上产生大幅变化，但不是稳定能力提升。v14 主要靠 pair23/pair33 拉高平均值，主观上出现局部闪烁和轮廓/细节退化；v15 在 pair15 上有强信号，但同时破坏 pair31 这种本来应保护的低运动、低频稳定序列。

### 10.4 可视化与主观质量结论

- v14 虽然 11-seq PSNR/SSIM/LPIPS 指标最好，但可视化中存在局部区域持续闪烁，细节与轮廓甚至不如 v9/v10 baseline。
- 这说明当前指标可能被少数序列或少数帧拉高，不能单独作为 paper 主结论。
- v15 在部分低频/结构序列上有主观积极信号，但指标显示它对 pair31 的保守结构破坏过大。
- v16a 可视化更接近 baseline，但这是因为 side 分支基本没有形成有效新能力，属于保守贴近而不是突破。

### 10.5 机制诊断

#### v14: SpatialAdaLN / clean adapter

- 正信号：能大幅提升 pair23/pair33，说明空间 V-JEPA 调制对 illumination/low-frequency gap 有作用。
- 负信号：局部闪烁和轮廓变差，说明调制可能直接扰动 DiT 主干特征，导致局部重绘。
- 判断：不能作为最终主线，最多保留为分析或辅助 ablation。

#### v15/v15b: temporal token transport

- 正信号：v15 在 pair15 上从 18.288 提到 22.024，说明 temporal/structure evidence 确实能改变模型行为。
- 负信号：v15 pair31 从 25.333 掉到 19.236；v15b 虽然尝试 per-token router，但 router 诊断显示 alpha/gate 几乎全局打开，没有学出真正 content-aware 的选择性。
- 判断：**pre-TruePAC token rewrite 风险太高**。它不是给模型补充信息，而是在进入 TruePAC 前改写语义 token，容易污染原本应保留的结构。

#### v16/v16a: side-condition

- 正信号：v16a 从 v10 init + freeze base 能维持接近 v10 的质量，说明 side residual 没有完全破坏主路径。
- 负信号：v16a final 的 side gate 接近饱和，best.pt 的 side residual 又偏弱；整体没有超过 v10。
- 判断：side branch 容易落入两个极端：要么被忽略，要么变成全局扰动。它仍然没有改变模型真正学习的目标。

### 10.6 当前总判断

目前已经可以排除三类方向作为下一阶段主线：

1. **继续调 loss 权重或训练策略**：不符合当前目标，也难以解释为新的组件/机制。
2. **继续扩大 adapter/conditioning**：v14-v16 已显示会出现主观质量、闪烁、结构保护之间的冲突。
3. **继续 pre-TruePAC rewrite**：会污染 V-JEPA token，pair31 collapse 是强警告。

更关键的问题不是“条件信息不够”，而是 **训练目标本身仍在要求模型重建整帧/整 latent**。对 LLVE 来说，这会鼓励模型重绘画面，而不是专注增强低照度残差，容易造成细节漂移和闪烁。

### 10.7 下一步建议: v17 residual enhancement diffusion

下一步建议转向 **Residual Enhancement Diffusion / 残差增强目标**，作为真正的机制级改动：

```text
pixel version:
residual = GT - LQ
output = LQ + predicted_residual

latent version:
r0 = z_gt - z_lq
model learns diffusion/flow on r0
z_out = z_lq + r_hat
```

这不是在 v10 上增量训练，也不是继续加权/加 adapter，而是改变模型要学习的变量。预期收益：

- 把模型能力集中在 illumination correction、detail recovery、noise residual，而不是重新生成整帧。
- 天然保护 pair31 这种低运动、低频稳定序列，因为模型只需要预测小残差。
- 对 pair15/pair23/pair33 这种存在明显残差结构的序列，应能释放更大增益。
- 从机制上解释并缓解 v14 的主观闪烁：不要让模型重绘局部结构，而是学习增强残差。

建议 v17 采用：

- 从原始 Wan/V-JEPA 初始化训练，不从 v10 checkpoint 做增量 fine-tune。
- 训练配置尽量沿用 v10 的稳定部分：TruePAC、multi-layer V-JEPA、LoRA r64、freeze norm、LPIPS 0.1、4-step eval。
- 只改 output parameterization / target construction，保证单变量可解释。
- first milestone 跑 9000-12000 iters，并同步保存可视化。

成功标准：

- full 11-seq PSNR 接近或超过 v10-seed0 21.578。
- pair31 不低于 v10 - 0.2 dB，不能再出现 v15 式 collapse。
- pair15/pair23/pair33 至少一个明显超过 v10，且不是靠牺牲主观轮廓换来。
- 可视化上局部闪烁减少，细节/轮廓不弱于 v10 baseline。

### 10.8 代码与环境操作记录

- 新增/使用过的实验脚本：
  - `tools/v14_wanvj_clean_token_train.py`
  - `tools/v15_wanvj_temporal_transport_train.py`
  - `tools/v15b_wanvj_adaptive_transport_train.py`
  - `tools/v16_wanvj_transport_side_train.py`
- 实验输出主要位于：
  - `/cache/zihaohe/checkpoint/v11_probes`
  - `/cache/zihaohe/visualizations`
- 30138 机器曾出现 workspace/overlay 空间紧张，已排查并清理。
- `novel_to_audiobook/CosyVoice` 已迁移到 cache 盘并用软链接连回 workspace，避免继续占用 overlay。
