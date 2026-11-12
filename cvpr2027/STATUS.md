# Seer / CVPR 2027 — 方法研发（2026-11-12）

> 当前研究目标已按用户要求调整：暂停论文写作，优先研发有创新、能在强基线下
> 验证正向收益的方法，目标是 SOTA 方法论文。首轮方法实验见
> `method_dev/20261111/PLAN.md`；下文保留上一轮评测与材料恢复记录。

当前下一轮 **观测对照与联合时序重建实验** 已完成42个head的源域训练：
两种E17基座、七个预定变体、三个head seed，全部跑满1000步。
同帧细节、错配邻帧、逐帧融合与四种图重建变体使用相同采样计划；
新增真实±1光流约束，源域crop128，目标域crop256。
两台训练机各133项CPU测试通过，无跳过；权重、初始化、计划和曲线
已校验并保全。两臂各31个方法的完整BVI VAL16评分已完成，
结果见 `method_dev/20261112/observation_v2/RESULTS.md`。

本轮更明确的正面证据是正确邻帧观测：pretrained臂相对同帧细节对照，
PSNR-Y提高0.02276 dB、LPIPS降低0.01769、时序gap降低0.00261，
三项场景区间均支持改善。none臂LPIPS、gap也改善，PSNR-Y区间跨零。
联合重建相对各自base提高PSNR-Y0.07365/0.22078 dB，LPIPS降低
0.01633/0.03683；none臂的时序gap改善也有场景区间支持，
pretrained臂的时序改善仍不确定。相对旧v1，联合重建降低gap但牺牲
部分画质。学习条件边权未超过固定边权，跨观测条件特征不作为创新点。
372个原始均值、408个CI端点及全部效应已由独立脚本核验。

新增TempRetinex作者标注DID的公开权重参考已完成同协议VAL16评分，
排除了GT直方图匹配；它的训练数据与精确训练配置不匹配本项目，单独报告。
协议、严格加载证据与原始结果在同目录的 `REFERENCE_PROTOCOL.md`
和 `references/tempretinex/`。该参考的空间画质更强、时序gap更大；
本轮没有建立SOTA。保留观测分支研发，论文写作继续暂停。

上一轮 **SDSD→BVI 源域训练实验已完成**：pretrained/none 两种 E17
基础模型各配三类候选、两种 loss、三个选择器 seed，共36个最终head。
原始暗帧细节候选加直接MSE选择器，相对各自base的PSNR-Y分别提升
**+0.0976 / +0.2525 dB**，场景95% CI分别为[+0.0420,+0.1737]与
[+0.1041,+0.4346]；LPIPS分别降低0.04227与0.06006。
pretrained臂的PSNR-Y、LPIPS在8/8场景均改善。源域训练、归一化和
选择器拟合不使用BVI；目标GT只供最终评分，未读取TEST图像。

这是实际正向跨数据集证据，仍有时序取舍：pretrained臂的时序gap
明显增加，学习选择器也尚未在PSNR/LPIPS上全面超过固定equal-thirds。
none臂的纠正收益更大，不能将新增收益归因为视频先验独有作用。
两臂各只有一个基础训练seed，历史像素分支初值不完全匹配；三个seed
仅复现选择器。BVI VAL曾参与此前项目开发，不声称整个研究target-blind。
频率分离、增益校准本身不是新原理，也尚未建立SOTA。
36个head、全部固定控制、STCD参考、独立统计及备份记录见
`method_dev/20261111/source_transfer_v1/RESULTS.md`。
两臂与STCD均使用31051同一评分环境，GT时间指标逐序列完全相同；
全部小权重与记录已哈希核验并保全，本轮实验进程均已结束。

此前的域内正向对照已完成：冻结 BVI 域内 E12 基础模型，使用预测驱动的
RAFT 邻帧对齐及 6,626 参数的局部纠正选择器。独立 BVI VAL16
（8 场景、两光照条件、center16/crop256）上，直接 MSE 选择器的三个
初始化/采样 seed 分别提升 PSNR-Y **0.4503 / 0.4661 / 0.4601 dB**，
均值 **+0.4588 dB**；PSNR-RGB 均值 +0.4497 dB。
按 seed 平均后的逐序列效应、再按整场景 bootstrap，PSNR-Y 的
95% 区间为 [+0.1364,+0.9482]。LPIPS 均值由 0.15323 降至 0.14494，
但 LPIPS 与时序 gap 的场景区间仍跨零，不声称这两项稳定改善。

这是三个选择器 seed，基础模型始终是同一个 E12 seed0；不是三次独立
基础模型训练。固定 loss 单位的同容量 MSE 对照也得到 +0.4581 dB。
风险监督约 +0.4032 dB，未超过直接 MSE，且其凸融合目标已有直接先例。
当前将直接 MSE 选择器固定为后续创新机制必须超过的强对照，
不把风险目标包装为新原理。完整训练记录、独立验证、场景分析与保全记录见
`method_dev/20261111/selector_v1/` 和 `selector_val_v1/`。
选择器小权重已下载并逐一核验；完整 E12 基础权重与最终选择器已保全至共享盘。

此前的匹配特征融合筛选完整保留于 `pilot_v1/`、`pilot_v2/`；
TRAIN48 的 GT 辅助 oracle 只用于候选可行性诊断，不计作学得性能。
本轮未打开 TEST 图像。当前验证协议与公开全分辨率榜单不同，尚未建立
SOTA 结论。下一阶段区分跨帧证据与一般细节增强，研发保留观测细节且
控制闪烁的重建机制，并推进官方源域划分、独立基础训练seed及AMNet等
强视频方法的同协议比较；论文写作继续暂停。

## 已封存的接管审计

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

当前优先级是新方法的实现、验证集筛选与公开协议下的强视频基线对照。
上一轮论文草稿暂停推进；评测修复供新实验复用。
