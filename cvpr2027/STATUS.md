# Seer / CVPR 2027 — 接管与协议审计（2026-11-11）

当前入口是本文件。接管时发现通用评测器没有恢复部分训练配置，
因此此前“已定稿”的若干归因与消融结论需要撤回或重评。
原状态保存在 `notes/STATUS_BEFORE_PROTOCOL_AUDIT_20261111.md`；
原始逐序列 JSON 未修改。新结果位于 `results/protocol_audit_20261111/`。

## 已核验的结果

全部主模型为 22.4M 像素分支、冻结 V-JEPA 2.1 ViT-L、20000 步，
SDSD test 39 序列。CI 是给定这些训练模型后的逐序列配对 bootstrap，
不是训练种子总体的不确定性。

| 比较 | PSNR-Y 差及 95% CI | LPIPS 差及 95% CI | 限制 |
|---|---|---|---|
| pretrained − none，3 seeds | +1.290 [+0.270,+2.449] | −0.0240 [−0.0321,−0.0168] | 各 seed 同号，幅度变化大；同名 seed 不代表像素分支共享初值 |
| pretrained − 正确重建的 random，seed 0 | +0.604 [−0.482,+1.870] | −0.0194 [−0.0308,−0.0092] | PSNR 未定；LPIPS 有利；只有一个 random 训练 seed |
| 固定 rank-64 训练+推理 − full-token 训练，匹配 seeds 0/1 | −0.061 [−0.571,+0.457] | −0.0002 [−0.0041,+0.0040] | 两 seed 效果方向相反；未检出差异不等于等效 |
| hybrid KV − per-tubelet KV，匹配 seeds 0/1 | +0.049 [−0.425,+0.492] | −0.0048 [−0.0081,−0.0016] | LPIPS 两 seed 均改善；PSNR 未定 |
| all-layers − single-layer，seed 2 | +0.476 [−0.276,+1.245] | −0.0045 [−0.0094,+0.0001] | 两指标未定；单 seed 且训练硬件不同，不能归因于架构 |

新宏由 `paper/tools/make_audit_macros.py` 直接从新 JSON 和对应原始基线生成；
精确点估计、输入 SHA 和逐 seed 差在 `corrected_comparisons.json`。
E18 的场景敏感性与时间指标见 `notes/E18_CORRECTED_20261111.md` 及
`results/protocol_audit_20261111/e18_comparisons.json`。
旧 random 的 4.88 dB 归因差距、旧 E26 的显著训练期惩罚均不再使用。
若沿用原 E26 的三种子参考均值，修正后 PSNR 差为 −0.098，也不显著；
结论变化并非由参考 seed 匹配造成。

H5 的留一域训练结论维持：SDSD、BVI 的 LPIPS 与 LOL 的 PSNR 分别通过
原预注册规则，另一指标未显著变差。不能表述为三域所有指标都改善。
补做的 **post-hoc** 场景 cluster bootstrap 没有翻转域内与两视频域的判定；
SDSD 跨域 LPIPS 的场景 CI 上界接近零。详见
`notes/SCENE_BOOTSTRAP_AUDIT_20261111.md`。

与同划分重训 Retinexformer 比，PSNR 均值 +1.968，但中位数 −0.307，
只赢 19/39；不能据此声称全面领先或直接对照已发表的其他划分表格。

## 机制与评测修正

- 每 clip 的 rank-4 操作保留 **完整 1024 维 clip 均值**、自适应方向和 token
  坐标，只压低中心化变化的秩。它不是四标量代码，也不证明编码器可被移除。
- 原自适应/固定 PCA 对比同时改变均值与基底，不能把差距全部归因于基底。
  Amendment 21 在结果出现前规定的 VAL16 六条件 mean×basis 控制已经完成：
  两 seed、16 序列、8 场景。clip 均值有贡献；保留 clip 均值时，自适应
  方向优于固定方向（PSNR +1.959，场景 CI [+0.354,+3.799]）。rank-4
  相对完整 token 的差异仍未定，不能声称等效。详见
  `notes/MEAN_BASIS_VAL_20261111.md`；属于探索性验证，不是新的 test 结论。
- tLP 统一为 `abs(seed_mean(tLP) − seed_mean(GT_tLP))`。E22 的恶化为
  +0.3555 [+0.3391,+0.3724]；E24 为 +0.0066 [+0.0044,+0.0088]，
  两者的两个 seed 均变差。旧宏使用有符号差，现已修正。
- 空间定位未检出收益，EDC 的负面结果仍可按原配置解释。
- hybrid 两 seed 和 all-layers 实际 seed 2 已按配置重评；all-layers 旧 21.568 dB
  掉分撤回，正确结果为 25.139 dB。它的 temporal gap 存在变差，不能只选有利指标。
- E2/E2c 的 taps、门控及随机替换 token 的历史结果，
  需要按实际配置重评；当前论文不以这些旧差值支撑架构结论。

`code_full/tools/eval_e4.py` 默认 `--reconstruction saved`，恢复 attention、
KV、layers、taps、投影和 random seed=1234；`legacy` 仅供诊断。
支持明确重定位 basis，并记录其 SHA。门控模块已经接入构造器。
统计工具也修正了时间指标逐 seed 符号和缺失指标错配问题。
CPU 检查在有 Torch 的远端运行，配置/重建 11 项、mean×basis 4 项均通过；
本地统计回归 5 项通过。

## 主机与材料

用户提供的 12 个端口中，10 个可建立 SSH；32141、31514 本轮连接失败。
旧 32677 仍失败。可识别硬件均为 Ascend 910B3×8，没有找回旧 A100 本地盘。
30778 的 SSH 可达，但项目目录访问阻塞，不能把超时当成文件不存在。
多台主机的 `/temp` 是同一 SFS；`/cache` 是各主机本地盘。

31051 的 NPU 7 经进程检查后用于本轮顺序评测，其他任务未停止。
完整明细见 `notes/TAKEOVER_20261111.md` 与 `EXPERIMENTS.md`。

25 个最终 checkpoint、日志、结果和恢复图，共 135 文件，已打包到
`/temp/zihaohe/seer_cvpr/recovery_20261111/seer_converged_31051_20261111.tar`。
共享盘副本 SHA-256 校验通过；大小 2,257,909,760 bytes，SHA 为
`8397ef128529888a5e151c0ba22dbb8b58d17046ad04c469777025f523f8dfa6`。
逐文件 SHA 已归档到 `results/protocol_audit_20261111/checkpoint_manifest_31051.json`。
这批 CVPR 权重尚未发布到 GitHub Release。说明见根目录 `CHECKPOINTS.md`。

## 当前执行与交付

- random seed 0、E26 seeds 0/1、hybrid seeds 0/1 与 all-layers seed 2 的
  正确配置重评共六项已完成，原结果保留；本轮 NPU 队列已结束。
- 定性图按原 10/35/60/85 百分位规则重新生成，补 checkpoint/source/图片 SHA。
- VAL16 的 mean×basis 两 seed 六条件控制及独立统计复算已完成并归档。
- 论文所有章节已改写为当前 Seer-v2 与审计后的证据，旧 M1/M2/M3/CP
  叙述已移除；只载入当前 CVPR、审计与 mean×basis 结果宏，缺宏会编译失败。
  审阅稿为 `paper/main_takeover_20261111.pdf`，正文 7 页、含参考文献与附录
  共 11 页。已修复当前 LaTeX 的双栏行号覆盖并重渲染检查；尚未投稿。
- E28/H20 没有已核验的最终结果。32677 不可达；不把未知结果当作阴性结果。
  混合重复 taps 的原生排序去重行为、更多 random 训练 seeds 仍需后续处理。

当前优先级是审阅修正后的论文与已归档的本轮控制，
然后依据这些结果决定新增训练。没有把旧阴性结果当作取消新方向的充分依据。
