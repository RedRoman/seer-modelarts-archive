# Seer 技术交接：2026-11-12

研究与实验快照截至 `163122a93a2ad7a6518f7abc1a70f7050c67d4fd`。工作分支为 `codex/seer-method-dev-20261111`；本次交接文档提交在该实验提交之后，不重算或改写已冻结结果。

## 1. 接手时先知道的结论

用户目标是有实质模块创新、正向实验与强方法对照的 SOTA 方法论文。论文写作暂停；当前尚未达到该目标。

- 已闭合 SDSD 原始数据为 **BGR** 的证据、RGB HVI 缓存重算，以及旧 Seer 整条推理链的颜色兼容复现。这些是正确性修复，不是方法创新。
- 固定时域融合有可复现的源域正向结果，必须保留为强基线。
- 最新“masked predictive credit → 观测运输预算”候选已完成固定 FIT12 机制筛选，**未通过主门槛，停止该版本的训练与扩跑**。不要沿用早期规格中的“下一步实现”作为当前任务。
- 尚无独立成立的 JEPA 模块贡献、SOTA 或新的 cross-domain 优势证据。接下来需要新的核心模块假设；不能靠重新命名 Shapley、caps、水位分配或普通融合补足创新。

阅读顺序：本文件 → [当前进度](STATUS.md) → [最新实验结果](method_dev/20261112/predictive_credit_probe_v1/RESULTS.md) → 对应 `PLAN.md`、JSON 与执行收据。`code_full/PROJECT_HANDOFF.md` 和 `code_full/PROGRESS.md` 是五月历史快照，不作为当前状态依据。

## 2. 当前进度与指标口径

| 阶段 | 已核验事实 | 当前用途 |
|---|---|---|
| [RGB 修复与 FIT102](method_dev/20261112/color_contract_v1/RESULTS.md) | 51 scenes / 102 clips，完整16帧，float RGB；HVI 15.322981，固定 temporal mean λ=.5 为15.695540，**+0.372559 dB**；51/51 scenes 的 mean MSE 降低 | 固定基线；三个预定 λ 都报告，未选最优 λ 部署 |
| [最新 FIT12 固定动作](method_dev/20261112/predictive_credit_probe_v1/RESULTS.md) | 12 scenes / 12 clips，仅中央帧8,9；HVI 19.008205，DC pair 19.260180，**+0.251975 dB，12/12 scenes 正向** | 固定细节运输基线，不是新预测模块收益 |
| 同一 FIT12 无 DC 平均 | 19.512881，**+0.504676 dB**，但4/12 scenes 下降 | 必须保留的较强均值对照；不能只报告 DC pair |
| 最新预测信用 | scene-macro AUROC **0.456270**，LQ 一致性 **0.562261**；候选减对照为−0.105992，95% scene-bootstrap CI [−0.166133,−0.044401] | 五项主比较全部未过；停止该候选 |
| [旧 Seer native-color replay](method_dev/20261112/native_color_replay_v1/RESULTS.md) | BVI VAL16：P 为19.330874 Y /18.041200 RGB，N 为18.250729 Y /17.247471 RGB | 兼容复现；P/N 历史像素初值不完全匹配，不能隔离预训练贡献 |

FIT102 与 FIT12 的帧范围、样本与聚合方式不同，不能比较其绝对 PSNR 或合并成一张统一性能表。FIT12 是每个 scene 中央两帧 MSE 转 PSNR 后的 scene 均值；不是完整16帧或官方全视频成绩。

前序 [memory](method_dev/20261112/observation_memory_v1/RESULTS.md)、[geometry](method_dev/20261112/observation_geometry_v1/RESULTS.md)、[joint search](method_dev/20261112/joint_observation_search_v1/RESULTS.md) 均未通过各自 source gate；没有据此启动 target 评测。joint 相对其强 anchor 约−0.000005359 dB，不能当成模块提升。

更早的 [source transfer](method_dev/20261111/source_transfer_v1/RESULTS.md) 和 [observation v2](method_dev/20261112/observation_v2/RESULTS.md) 保留其当时协议下的正向与取舍记录；解释旧结果时必须同时阅读后续颜色合同修复，不能直接替代当前 canonical RGB 基线。

## 3. 对比方法与 cross-domain 边界

[同协议 image-reference replay](method_dev/20261112/baseline_replay_v1/RESULTS.md) 的 BVI VAL16 结果如下。它们使用 center16/crop256、rounded-uint8 scorer；训练来源和预算不同，是参考对照，不是公平同训练条件排行榜。

| 固定参考 | PSNR-Y | PSNR-RGB | LPIPS |
|---|---:|---:|---:|
| HVI-CIDNet / generalization | 24.241232 | 23.383146 | .153150 |
| DarkIR / DarkIR_384 | 22.581697 | 21.465763 | .151016 |
| Retinexformer / same-SDSD-split160k | 19.881678 | 19.103253 | .203176 |

HVI 的固定历史权重对应外部 LOLv2-Syn 预训练，不能标成 SDSD-only；DarkIR 的精确训练划分未独立核验。旧 Seer P 的19.330874 Y 仍明显低于 HVI 的24.241232 Y。颜色修复后 P 相比旧未包装 P 的 Y 仅+.045942，而 RGB −.316150、LPIPS 增加.020811；不能把修复称为整体质量提升。

LLVE-STCD 与 TempRetinex 已有先前同 clip 协议参考，分别见 [source transfer](method_dev/20261111/source_transfer_v1/RESULTS.md) 和 [observation v2](method_dev/20261112/observation_v2/RESULTS.md)。TempRetinex 使用作者标注 DID 的公开权重，训练配置不匹配本项目；排除了 GT histogram matching。现代视频基线 AMNet 的代码/接口已审，但指定 finetuned 权重尚未取得，**没有推理结果或分数**；见 [AMNET_READINESS.md](method_dev/20261112/baseline_replay_v1/AMNET_READINESS.md)。

源域 FIT 已多次用于开发。旧 base 见过全部68个 source scenes，现有 internal17 不能称为全新盲测；BVI VAL 也参与过此前开发。最新预测筛选未读 internal17 / target 输入，但这不代表整个项目 target-blind。当前没有新的 TEST 结果或可成立的 cross-domain SOTA claim。

## 4. 技术合同与代码入口

| 工作 | 入口 |
|---|---|
| 显式颜色边界 | [`color_contract_v1/color_contract.py`](method_dev/20261112/color_contract_v1/color_contract.py) |
| RGB HVI 重算与 FIT102 诊断 | [`color_contract_v1/`](method_dev/20261112/color_contract_v1/RESULTS.md) |
| 历史 Seer 全链包装/复现 | [`native_color_replay_v1/`](method_dev/20261112/native_color_replay_v1/RESULTS.md) |
| 预测 critic / masks / 模型加载 | [`predictive_credit_probe_v1/critic_export.py`](method_dev/20261112/predictive_credit_probe_v1/critic_export.py) |
| 实际 CPU 分片与严格合并 | 该目录 [`checks/critic_cpu_sharded_v2.py`](method_dev/20261112/predictive_credit_probe_v1/checks/critic_cpu_sharded_v2.py)、[`checks/critic_merge_shards_v2.py`](method_dev/20261112/predictive_credit_probe_v1/checks/critic_merge_shards_v2.py) |
| GT-free 动作与独立标签评分 | [`values_probe.py`](method_dev/20261112/predictive_credit_probe_v1/values_probe.py)；`prepare_values` 不接收 GT，`score_utilities` 单独生成标签 |
| 冻结主分析 / 独立复核 | [`analyze_probe.py`](method_dev/20261112/predictive_credit_probe_v1/analyze_probe.py)、[`checks/independent_verify.py`](method_dev/20261112/predictive_credit_probe_v1/checks/independent_verify.py) |

必须保留的语义：

1. SDSD 原始 NPY / 旧 cache 为 BGR。HVI 需要输入先转 RGB 并**重新前向**；不能只交换旧输出通道冒充修复。旧通用 `code_full/data/common.py` 不能直接当 canonical RGB loader 使用。
2. 旧 SDSD Seer 模型在整条 pixel+encoder pipeline 外做 RGB→BGR→RGB 包装，才能复现原状态；这没有重新训练并修复其内部 RGB 预训练 encoder 历史上接受 BGR 的问题。不要给所有其他 RGB 模型套同一个包装。
3. 新 HVI cache 不改旧 Seer-native flow / validity，也未重算 RGB HVI flow；方法结果只适用于这一固定提案配方。
4. 旧 V-JEPA2.1 distilled encoder1024 与 predictor1664 不匹配，不能直接作 native prediction residual。新 critic 使用完整匹配的 V-JEPA2-L EMA encoder1024 / predictor384→1024，严格加载，按官方 hub 推理配对；不把它称为训练时 online/EMA 配对。
5. critic 实际在第一层 attention 前按原位置编号 mask，不能从已混入邻帧信息的缓存 token 中删行代替。目标固定 tubelet4（帧8,9），donor tubelets3/5（帧6,7与10,11）；四 coalition 共用同一 target-local LQ EMA reference。
6. 当前保留已有256 crop，只加 ImageNet RGB mean/std；没有完整官方 resize292/center256。target-local reference 也偏离官方 full-context teacher 分布。损失是单位向量的 squared L2，不能称校准的 RGB 风险或恢复因果效应。
7. 固定 RGB 动作使用真实 warped HVI donor、γ=.5、patch16内逐帧逐通道有效支持 DC 去除、pair valid-count normalization、box/fixed-sum projection。utility 只描述此 singleton-versus-averaged-pair 配方，不是一般物理交互真值。

最新筛选每 scene256 patches，总3072；两 donor 条件效用共6144标签，其中5160正、984负，均满足双 donor valid fraction≥.5。所有五个对照固定：random predictor、wrong-time predictor、flow-aligned EMA cosine、负 LQ MSE、负 HVI MSE。主门槛为共同完整 scene≥8、candidate macro AUC≥.55，且20,000次完整 scene paired bootstrap 对全部五个对照的提升下界>0；seed20261123。实际12 scenes全部有效但 gate失败。

`v12=L0−L12`，`phi1=(v1+v12−v2)/2`，`phi2` 对称。τ=1时 `B=clip(v12,0,1)`、`c=clip(phi,0,1)`。本批 v12 全小于1，分配在100% patches上唯一，softmax proposal没有额外作用；51.5625%零预算，平均 anchor weight .98331054。详见 [代数记录](method_dev/20261112/predictive_credit_probe_v1/checks/analysis_budget_algebra_v1.md)。

## 5. 环境、缓存和运行状态

以下是31051的既有路径；`/cache` 为主机本地盘，`/temp` 为共享 SFS。端口/实例可能回收，恢复时先核验文件与收据，不把路径存在当作字节正确。

| 资源 | 路径 |
|---|---|
| Python | `/cache/zihaohe/seer_cvpr/venv_npu/bin/python` |
| runtime 初始化 | `/temp/zihaohe/seer_cvpr/method_dev_20261111/method_runtime_env.sh`；`SEER_CANN_ROOT=/usr/local/Ascend/cann-8.5.2` |
| native BGR source cache | `/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0` |
| 新 RGB HVI cache | `/cache/zihaohe/seer_color_contract_20261112/cache_rgb_hvi_v1` |
| geometry manifest | `/cache/zihaohe/seer_observation_geometry_20261112/prepared_v1/manifest.json` |
| 匹配 V-JEPA2-L 权重与上游源码 | `/cache/zihaohe/seer_matched_predictor_assets_20261112_v1/{weights/vitl.pt,upstream/}` |
| values / analysis 根 | `/cache/zihaohe/seer_predictive_credit_probe_20261112` |
| critic 根，注意多一个后缀 | `/cache/zihaohe/seer_predictive_credit_probe_20261112_v1` |

正式环境为 Torch2.7.1+cpu / torch_npu2.7.1.post2 / CANN8.5.2。CPU 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，OMP/MKL/OPENBLAS线程均为4。Mac 本地没有 Torch，但 NumPy 标量分析可运行。

**预测 critic 的 NPU parity 未通过**：阈值固定 max .01 / mean .001，未放宽；正式数据全部 CPU。四 worker各固定3 clips、4线程，真实16帧 synthetic全部通过且结果/状态一致。完整 probe 含全部对照与静态接口，平均228.4 CPU秒/clip，不能当核心模块 latency。此 NPU 失败结论仅针对该 critic 路径，不代表所有项目模型都无法用 NPU。

末次实验收口在2026-11-12 07:57 UTC（15:57 SGT）记录 owned processes=0、NPU6/7空闲；**这是当时快照，不保证接手时仍空闲**。重启计算前重新检查占用，不能停止其他任务。当前没有待续训的预测信用 job、自动监控或自动扩跑。

## 6. 关键 SHA256 与恢复位置

| 文件 | SHA256 |
|---|---|
| RGB HVI cache `manifest.json` | `5f25fdfcca5bc7d581c1484245fa760ef321da642f5e741077c128376616a94a` |
| 匹配 `vitl.pt`（5,127,726,842 bytes） | `5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012` |
| GT-free `observable_sidecar.json` | `580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23` |
| scalar `patch_statistics.npz` | `4ccd7e9a807f43a3c64d6b102f3775a2aca9fed86d197643a4e1909c8d338a2c` |
| critic `critic_cpu_outputs_v2/manifest.json` | `beb8748f72c7512da68b4f6cbcfe9a910b3bd137a16f6a42757488bc6844070b` |
| `analysis_frozen_manifest_v4.json` | `3151df089d6617772ba4ef069e984a5adbc1ad8c09c62e8fb897a0dac950737b` |
| 正式 `source_fit_probe_analysis.json` | `f7edb2fb98af3977a05cba7baa999a4cf8f30694e632e4a0ddf0677ff25a44f4` |

SFS 基目录为 `/temp/zihaohe/seer_cvpr/method_dev_20261112/`：

| 目录后缀 | 内容 |
|---|---|
| `color_contract_rgb_hvi_source_v1_final_v1` | 全136 RGB HVI outputs及导出证据 |
| `color_contract_rgb_fit_v1_final_v1` | FIT102固定融合诊断 |
| `matched_predictor_assets_v1_final_v1` | 完整匹配权重、冻结上游源码与strict-load证据 |
| `native_color_replay_v1_final_v2` | 旧 Seer native-color复现 |
| `predictive_credit_probe_values_v1_final_v1` | 23 payloads；含 `values_utilities_v1/patch_statistics.npz` |
| `predictive_credit_probe_critic_v1_final_v1` | 103 payloads；含 `critic_cpu_outputs_v2/manifest.json`及四worker结果 |
| `predictive_credit_probe_analysis_v1_final_v1` | 36 payloads；分析、冻结文件与执行收据 |
| `predictive_credit_probe_root_closure_v1_final_v1` | 90 root文本payloads；引用各component快照并列出完整文本inventory |

最新 root closure manifest SHA为 `f9bb1a4d40cd008b880556f0cc7cc1dd588c81ae27b25203b2b9e46452d73c47`。详见 [root closure manifest](method_dev/20261112/predictive_credit_probe_v1/checks/root_closure_manifest.json) 与 [receipt](method_dev/20261112/predictive_credit_probe_v1/checks/root_final_backup_receipt.json)。不要把root小包当成所有大型payload的完整副本；它明确引用单独校验的component备份。

Git内保存代码、协议、标量JSON及文本日志；没有原始视频、GT pixels、模型权重或utility NPZ。本机 NPZ 目录可从 [values receipt](method_dev/20261112/predictive_credit_probe_v1/checks/values_backup_receipt.json) 的 `external_results` 查到；另一台机器可从上述 SFS values 快照取回并先核SHA。更早收敛基础权重的恢复入口见根目录 [CHECKPOINTS.md](../CHECKPOINTS.md)。

上述快照不代表原始 SDSD/BVI dataset 的完整备份。若原始source cache丢失，需核查原数据与缓存生成计划的恢复情况；HVI输出和标量统计不能重建GT或原flow。下面的标量复现不需要恢复原始视频。

## 7. 无需重新前向的复现入口

在repo根目录执行；需要 Python3 / NumPy。`SEER_UTILITIES` 必须指向已核SHA的外部标量NPZ；下面给出本次Mac上的现成位置，换机器时修改这一项。

```bash
SEER_PROBE=cvpr2027/method_dev/20261112/predictive_credit_probe_v1
SEER_UTILITIES="/Users/hezihao/Documents/New project/seer_predictive_credit_probe_dev_20261112/results/values_v1/patch_statistics.npz"
SEER_REPLAY_DIR=$(mktemp -d)

python3 -m unittest discover -s "$SEER_PROBE" -p test_analyze_probe.py -v

python3 "$SEER_PROBE/analyze_probe.py" analyze \
  --subset "$SEER_PROBE/subsets.json" \
  --protocol "$SEER_PROBE/PLAN.md" \
  --frozen-analysis-manifest "$SEER_PROBE/checks/analysis_frozen_manifest_v4.json" \
  --critic "$SEER_PROBE/checks/critic_remote_evidence_v1/critic_cpu_outputs_v2/manifest.json" \
  --values "$SEER_UTILITIES" \
  --observable-sidecar "$SEER_PROBE/checks/values_evidence_v1/values_observables_v1/observable_sidecar.json" \
  --expected-observable-sha256 580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23 \
  --output "$SEER_REPLAY_DIR/analysis.json"

python3 "$SEER_PROBE/checks/independent_verify.py" \
  --subset "$SEER_PROBE/subsets.json" \
  --critic "$SEER_PROBE/checks/critic_remote_evidence_v1/critic_cpu_outputs_v2/manifest.json" \
  --values "$SEER_UTILITIES" \
  --analysis "$SEER_PROBE/results/analysis_v1/source_fit_probe_analysis.json" \
  --output "$SEER_REPLAY_DIR/independent.json"
```

`analyze` 是子命令，不是 `--mode analyze`。冻结v4的18个source paths为相对路径，保持目录布局即可本地验证，无需重新冻结。正式远端 [launch manifest](method_dev/20261112/predictive_credit_probe_v1/results/analysis_v1/analysis_launch_manifest.json) 的 argv 则含原机器绝对路径。重新分析时 provenance路径变化会改变整个输出JSON的SHA，应比较 `primary_gate`、`primary_macro_auc`、`per_scene` 等结果字段；归档正式JSON的SHA不能被新文件覆盖。

统计测试应为27项；独立pairwise-AUC复核应为326项通过。critic/value合同另有8/10项真实CPU测试、8项merge fixtures、四个完整输入尺寸实模型测试及36,864项finite-loss检查。这些不等于53个真实数据质量实验。此前本地与远端独立复核receipt字节一致。

## 8. 下一位执行者的工作边界

1. 先核最新结果、颜色边界和大文件hash；不要重跑已闭合的预测导出或训练失败候选。其旧 [module_decision_v1.md](method_dev/20261112/predictive_evidence_v1/module_decision_v1.md) 只是当时规格，已被后续probe停止决策覆盖。
2. 新模块应先写清区别于普通gate/已有融合的结构假设，冻结数据、动作与匹配对照，再做小规模source筛选。近邻工作笔记见 [nearest_work.md](method_dev/20261112/predictive_evidence_v1/nearest_work.md)。固定temporal mean、DC pair、普通可靠性/相似度对照都需保留；不能用更弱anchor或同帧恒等冒充完整机制对照。
3. 需要主张跨域或SOTA时，先解决强空间基线差距、训练初值/预算匹配和独立评测问题；不能把已开发使用的internal/VAL重新命名为盲测。AMNet仍是待取得指定权重的对比项。
4. 已冻结代码、协议、日志、结果和失败记录保持原样，新实验用新目录/新版本；不按结果翻转score、选scene、选τ或改主门槛。`before_actual_scores_read` 的含义是未读actual critic-loss/patch-utility数组，并非从未见aggregate信息：六个固定动作宏PSNR曾在分析最终冻结前经消息出现，已在 [disclosure](method_dev/20261112/predictive_credit_probe_v1/checks/analysis_freeze_disclosure_v1.json) 如实说明。
5. 大文件留在已校验SFS/cache，Git只推可审查文本与代码；不要提交PT/PTH、视频、raw GT、tar、pyc或AppleDouble元数据。不要将未完成下载或只有路径、未验SHA的材料标为恢复成功。

当前没有可直接交给完整训练的已验证新模块，也没有新论文写作任务。上述待解决问题是后续研究工作，不应将现有筛选失败改写为负面结果论文目标。
