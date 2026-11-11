# SDSD→BVI 源域训练实验：正向空间质量收益与时序取舍

本轮完成 36 个最终选择器（两种基础模型 × 三类候选 × 两种 loss × 三个选择器 seed），没有按目标域分数挑选 checkpoint。论文写作继续暂停。这是方法研发结果，尚未建立 SOTA 或新原理。

## 协议

- 基础模型为历史 E17 SDSD-only small final20k，pretrained/none 各一个训练 seed0。源域缓存包含 SDSD TRAIN136、68 场景，每序列一个固定 crop256/clip16；选择器按场景固定为51 fit/17 internal holdout。后者曾供基础模型训练，不是整个模型的新留出集。
- 所有选择器采用相同19维可观测特征、6,626参数CNN、Adam1e-3、batch8、1,000步。每个seed的初值和采样计划在候选、loss及两基础模型间匹配；统计与归一化仅使用源域fit场景。
- 目标为 BVI VAL16、8场景/两光照、center16/crop256。源域训练与归一化不读取目标数据；目标GT仅用于预测后的评分。此前项目开发已经使用BVI VAL，不能宣称整个研究过程target-blind。本轮没有打开TEST图像。
- 每种基础模型的全部18heads在同一次base/RAFT推理下评分，附6个固定控制。两臂与STCD均在31051、torch2.7.1 / torch-npu2.7.1.post2 / pyiqa0.1.13运行；实际评分库SHA一致，逐序列GT tLP差为0。
- 三个seed只复现选择器，基础模型每臂仍只有一个训练seed。历史两基础模型的像素分支初值并非完全相同，不把跨臂差值解释为孤立的预训练因果效应。

## 全部主要臂的绝对指标

学习臂是三个选择器seed的均值。PSNR越高越好；LPIPS与`abs(tLP−GT_tLP)`越低越好。完整固定控制、逐seed及24组配对比较见各臂机器生成的分析文件。

| 基础模型 | 候选 / 训练或固定权重 | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |
|---|---|---:|---:|---:|---:|
| pretrained | base | 18.357351 | 19.284931 | 0.244829 | 0.005237 |
| pretrained | restored/direct_mse | 18.291503 | 19.197356 | 0.247441 | 0.004285 |
| pretrained | restored/direct_mse_normalized | 18.293257 | 19.199601 | 0.247401 | 0.004262 |
| pretrained | restored_detail/direct_mse | 18.338855 | 19.261550 | 0.247747 | 0.003842 |
| pretrained | restored_detail/direct_mse_normalized | 18.339641 | 19.262591 | 0.247609 | 0.003887 |
| pretrained | raw_detail/direct_mse | 18.433029 | 19.382555 | 0.202555 | 0.006852 |
| pretrained | raw_detail/direct_mse_normalized | 18.432525 | 19.381355 | 0.203204 | 0.006501 |
| pretrained | raw_detail/initial_thirds | 18.426742 | 19.386914 | 0.187835 | 0.013272 |
| none | base | 17.365602 | 18.546477 | 0.264879 | 0.005863 |
| none | restored/direct_mse | 17.305764 | 18.452967 | 0.272922 | 0.005088 |
| none | restored/direct_mse_normalized | 17.306708 | 18.453222 | 0.272787 | 0.005140 |
| none | restored_detail/direct_mse | 17.351348 | 18.519566 | 0.272173 | 0.004433 |
| none | restored_detail/direct_mse_normalized | 17.352334 | 18.520800 | 0.272097 | 0.004432 |
| none | raw_detail/direct_mse | 17.520055 | 18.799002 | 0.204816 | 0.008566 |
| none | raw_detail/direct_mse_normalized | 17.520128 | 18.799141 | 0.204980 | 0.008561 |
| none | raw_detail/initial_thirds | 17.506198 | 18.809854 | 0.204367 | 0.016531 |

## 原始观测细节候选的配对收益

这里完整展示预声明的直接MSE控制；normalized loss结果同向，全部保留在原始分析中。先在每序列内平均三个选择器seed的效应，再对8个完整场景做10,000次bootstrap（RNG0）。区间未做多重比较调整，是给定当前训练模型后的探索性场景不确定性。

| 基础模型 | raw_detail/direct_mse 对比 | 指标 | 均值差 | 场景95% CI |
|---|---|---|---:|---|
| pretrained | − base | psnr_rgb | +0.075678 | [+0.037388,+0.140729] |
| pretrained | − base | psnr_y | +0.097623 | [+0.042007,+0.173686] |
| pretrained | − base | lpips | -0.042274 | [-0.048873,-0.035205] |
| pretrained | − base | tlp_gap | +0.001615 | [+0.000528,+0.002702] |
| pretrained | − restored | psnr_rgb | +0.141526 | [+0.075143,+0.227337] |
| pretrained | − restored | psnr_y | +0.185199 | [+0.110117,+0.277584] |
| pretrained | − restored | lpips | -0.044886 | [-0.065149,-0.020604] |
| pretrained | − restored | tlp_gap | +0.002567 | [-0.000469,+0.005420] |
| pretrained | − restored_detail | psnr_rgb | +0.094173 | [+0.027337,+0.185261] |
| pretrained | − restored_detail | psnr_y | +0.121005 | [+0.044345,+0.220520] |
| pretrained | − restored_detail | lpips | -0.045192 | [-0.066733,-0.019192] |
| pretrained | − restored_detail | tlp_gap | +0.003010 | [+0.000335,+0.005456] |
| pretrained | − initial_thirds | psnr_rgb | +0.006287 | [-0.016810,+0.026551] |
| pretrained | − initial_thirds | psnr_y | -0.004360 | [-0.032604,+0.023344] |
| pretrained | − initial_thirds | lpips | +0.014720 | [+0.002760,+0.028050] |
| pretrained | − initial_thirds | tlp_gap | -0.006420 | [-0.008964,-0.004197] |
| none | − base | psnr_rgb | +0.154453 | [+0.071938,+0.284853] |
| none | − base | psnr_y | +0.252524 | [+0.104120,+0.434560] |
| none | − base | lpips | -0.060063 | [-0.070989,-0.049559] |
| none | − base | tlp_gap | +0.002704 | [-0.000060,+0.005408] |
| none | − restored | psnr_rgb | +0.214291 | [+0.090866,+0.371618] |
| none | − restored | psnr_y | +0.346035 | [+0.168393,+0.555601] |
| none | − restored | lpips | -0.068106 | [-0.088459,-0.045513] |
| none | − restored | tlp_gap | +0.003479 | [-0.000857,+0.007673] |
| none | − restored_detail | psnr_rgb | +0.168707 | [+0.055487,+0.323084] |
| none | − restored_detail | psnr_y | +0.279436 | [+0.108785,+0.484383] |
| none | − restored_detail | lpips | -0.067357 | [-0.089319,-0.042400] |
| none | − restored_detail | tlp_gap | +0.004133 | [-0.000020,+0.008208] |
| none | − initial_thirds | psnr_rgb | +0.013857 | [-0.022039,+0.040846] |
| none | − initial_thirds | psnr_y | -0.010852 | [-0.067475,+0.037490] |
| none | − initial_thirds | lpips | +0.000450 | [-0.009058,+0.011510] |
| none | − initial_thirds | tlp_gap | -0.007965 | [-0.011574,-0.004555] |

原始暗帧的可观测细节在两种基础模型上均改善了PSNR/LPIPS，相对仅使用恢复帧细节的匹配控制也有正向收益。pretrained臂相对base的PSNR-Y与LPIPS在8/8场景均改善。这提供了SDSD→BVI的实际源域训练正向证据。

仍需解决两个问题。第一，pretrained臂相对base的时序gap增加，区间完全为正；none臂的均值也增加、区间跨零。第二，学习选择器对固定equal-thirds的PSNR-Y没有优势；pretrained臂LPIPS反而更差，但两臂的时序gap都明显降低。现有学习器主要在空间质量与时序gap之间取舍，不能宣称全面超过固定融合。

纠正收益在none臂更大，不能将新增收益包装成视频先验特有的优势。联合分析完整报告两臂的绝对差和相对base的增益差；历史初始化混杂与单基础seed限制始终适用。

## STCD 参考与数据进展

官方SDSD-indoor STCD发布权重完成零BVI微调参考：PSNR-RGB 16.202430、PSNR-Y 16.607015、LPIPS 0.315305、tLP gap 0.004884。使用官方两帧BGR接口、输出[2]；末帧以自身作为第二帧（不读取clip外帧）。权重strict-load、两处构造期CUDA移动仅在内存移除，原仓库源码未修改。STCD仅indoor源训练；Seer使用indoor+outdoor，pretrained另有基础模型预训练。当前crop/clip协议也不同于公开全分辨率表格，因此这些数值不是同数据重训或公开SOTA判定。

DID作者的公开目录可匿名列出`DID_1080.zip`（32,795,182,798 bytes）与`DID_2k.zip`；官方train238/test87列表已取得。匿名文件页缺少正常下载所需的sign/timestamp，下载guard停止，未取得压缩包或读取视频。不能以网页可达代替数据已就绪，也不据此断言必须账号。

## 方法研发的下一步

1. 先加入同帧原始细节、打乱邻帧的匹配控制，区分真正的跨帧证据与一般细节增强。
2. 研发能保留原始观测细节、同时约束时序不一致的重建机制；同时超过restored直接MSE与固定equal-thirds，作为进入下一阶段的门槛。频率分离、增益校准和凸融合本身已有直接先例。
3. 在官方源域划分上以共享初值/数据计划重训基础模型，增加独立基础训练seed，并在第三视频域和统一原生分辨率协议下对比STCD/AMNet等强方法。

## 可复核材料

- [冻结实验协议](PLAN.md)、[分析冻结记录](analysis_frozen_v1.json)、[源码包清单](source_package.json)。
- [pretrained原始25-method结果](results/pretrained_raw.json)、[全部分析](results/pretrained_analysis_v1.md)。
- [none原始25-method结果](results/none_raw.json)、[全部分析](results/none_analysis_v1.md)。
- [联合分析与STCD配对参考](results/joint_analysis_v1.json)、[STCD原始结果](references/stcd/results/stcd_indoor_release_bvi_val16.json)。
- 训练曲线、固定采样计划、归一化和权重SHA在`artifacts/`；小权重保存在本地外部实验目录及独占共享盘副本，未把`.pt`或11GB观测缓存加入Git。
- CPU训练初次启动因torch_npu自动补丁调用设备而在step1前失败；旧输出保留。新wrapper设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，两主机CPU Adam smoke与74项CPU检查通过，冻结模型源码未改；全部36heads使用新wrapper完成，pretrained失败任务在新目录重跑。
- 独立统计复算与最终备份清单在`checks/`。本轮没有编辑论文源文件。

共享盘保全根为`/temp/zihaohe/seer_cvpr/method_dev_20261111/`，使用独占目录：

| 目录 | 文件数（不含清单） | 清单SHA-256 |
|---|---:|---|
| `source_transfer_v1_pretrained_final/` | 175 | `ce6bdac9849042575b06bd337aceaf3ba03fca083f090e0ed444fc602856f1ea` |
| `source_transfer_v1_none_final/` | 144 | `f0507d729833e18773a49bcba5fe6f2b03bd20b343704dfc25e205b4faf1a753` |
| `source_transfer_v1_runtime_final/` | 29 | `9f3de3ff80255f1931ba5197460d7b9e95b65f1ef9e57f8a18c884dc1f801005` |

独立统计复算核验全部600个CI端点，最大数值误差`7.11e-15`。逐文件证据见[归档清单](checks/archived_evidence_files.json)。本任务的缓存、训练、评测、备份和数据获取进程均已结束；其他使用者的进程未受影响。
