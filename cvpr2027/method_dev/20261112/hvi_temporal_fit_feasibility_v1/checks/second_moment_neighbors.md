# 条件误差二阶矩 + simplex 融合：近邻与创新性审计

2026-11-12，只读 primary 文献核查；没有修改实验、训练、缓存或计算实图分数。

**判断：目前这个算子不能作为新的核心模块原理。** 预测三个候选的未中心化
误差二阶矩，再解非负、和为1的最小二次风险融合，与 CsNet 的核心公式直接
重合。这个判断否定的是当前的**新颖性主张**，并不否定它可能有效；它适合作为
未来必须击败的强融合基线。局部/时序条件化和 V-JEPA 特征可能构成新的实证
问题，但当前没有足以支撑强方法论文的新算子或已验证机制。

## 最直接的先例

| Primary 来源 | 已有机制与本提案的关系 | 必须区分的地方 |
|---|---|---|
| [Choi et al., Optimal Combination of Image Denoisers / CsNet, TIP2019](https://arxiv.org/pdf/1711.06712)，PDF p3，P1、Eq4、footnote1；Section III-B | 定义 `Σ=E[(Zhat−Z)^T(Zhat−Z)]`，解 `min w^TΣw, w>=0, sum(w)=1`；脚注明确其不属于通常的中心化 covariance，因为 denoiser 可以有 bias。通过学习 MSE、结合可观测候选差异构造非对角项。 | 本提案的矩阵、约束和风险解释已经覆盖。原文的 MSE 网络也以局部64×64 patch为输入，再聚合到全图；把融合系数局部化是差异，但不能仅凭“用了patch”声称新原理。可移除其可选 booster，保留已有核心融合。 |
| [Mildenhall et al., Burst Denoising with Kernel Prediction Networks, CVPR2018](https://arxiv.org/pdf/1712.02327)，Section4、Eq2 | 预测空间变化的时空核，并直接作用于输入 burst；网络不必自由生成输出RGB。还讨论训练容易忽略其他帧，并用退火辅助目标处理。 | **原版 KPN 不使用 softmax，允许负权重**；不能误写成与 simplex完全一致。固定对齐、仅3个候选、非负系数是受约束的融合变体，不是“首次观测值而非RGB decoder”。 |
| [Back et al., Deep Combiner for Independent and Correlated Pixel Estimates, TOG2020，作者项目页](https://cglab.gist.ac.kr/publications/2020_Deep_Combiner_for_Independent_and_Correlated_Pixel_Estimates.html)；[作者代码](https://github.com/CGLab-GIST/deep-combiner) | 将既有处理器当作黑箱，用学习的组合核减少相关像素估计的残余误差与系统误差，并扩展到多buffer。 | 应用是 Monte Carlo rendering，不是本LLVE协议，也不是同一个显式QP；但“融合相关且有系统偏差的现成恢复结果”已经有明确先例。 |
| [DA-CLIP, Controlling Vision-Language Models for Universal Image Restoration, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/467e109270678b3e2b3c2e7fa6b3d575-Paper-Conference.pdf)，Section3.2；[作者代码](https://github.com/Algolzw/daclip-uir) | 用预训练VLM的内容/退化表征条件化图像恢复，包括low-light任务。 | 不是误差矩阵QP，也不是视频V-JEPA；它说明预训练语义/退化特征作为恢复条件本身不是新原则。换成V-JEPA输入需要证明特定机制与可迁移收益。 |
| [Xu et al., SNR-Aware Low-Light Image Enhancement, CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html) | 利用可观测SNR先验引导特征融合，区分低信噪区域的处理方式。 | 未发现与CsNet相同的完整误差矩阵算子；但“低光可靠性引导融合”已经建立，不能用换一个uncertainty标签代替具体机制。 |
| [Lin et al., ANCHOR: Revisiting the Current Frame, arXiv2026](https://arxiv.org/abs/2608.09342) | 视频恢复输出修正：从物理迹象估计空间trust field，在恢复提案和当前观测间取舍。 | 本审计仅确认其primary摘要与方法结构，没有核验其复现成绩；不是相同的三候选二阶矩QP。它进一步限制“anchor-preserving可靠性修正”这一宽泛创新表述。 |

以上对应关系是对原文公式/机制的比较，不是将不同任务的已发表PSNR拼成排行榜。

## 数学上需要说清楚的内容

设候选为 `c_i`，真实目标为 `y`，误差 `e_i=c_i−y`，所有期望均条件于推理
时可观测信息 `z`。则

以下先用单像素/通道的候选误差向量记号；RGB/patch版本把乘积换成统一归一化
内积，bias项相应为均值误差的Gram，所有风险与候选距离必须使用相同归一化。

`M(z)=E[ee^T|z]=Cov(e|z)+b(z)b(z)^T`, `b(z)=E[e|z]`。

因此使用中心化 covariance 会漏掉条件bias项。**这一正确区分已经被CsNet的
定义及脚注明确包含；不能把重新命名为“bias-aware second-moment risk”当贡献。**
“经典无偏GLS”不是最准确称呼，因为这里允许偏差和非负约束；更准确的已有
对象是 simplex-constrained minimum-MSE combination。3候选的active-set枚举、
Cholesky/PSD参数化、对角稳定项也不会自动产生新原理。

还有一个重要的可检验恒等式：

`2 M_ij(z) = M_ii(z)+M_jj(z) − E[||c_i−c_j||²|z]`。

这是CsNet Eq4的条件版本。若 `z`包含当前候选patch，最后的候选距离就是可直接
计算的量。故三个对角风险加上候选几何差异已经约束所有交叉项；自由预测六个
对称元素可能破坏这一一致性。若 `z`只含粗特征，最后一项仍是条件期望，不能
未经说明直接用单次距离替代它。完整矩阵更有用不等于它是新算子。

`zero gate => exact HVI anchor` 是可测试的退化/初始化性质，不是部署后绝不掉分
的保证。`M`的PSD也只保证二次优化良定，不保证矩阵在新域校准正确。只有真实
条件矩阵给出相应的条件MSE最优性；源域预测矩阵不自动带来目标域保证。

## 最有信息量的后续对照建议

以下是建议，**尚未写入或改变任何冻结实验协议**。

1. **直接条件CsNet基线**：同一组固定对齐候选、同一可观测输入、同一source
   FIT预算，预测3个对角MSE，再用候选距离关系构造交叉项/PSD修正/同一个
   simplex solver。不加booster。必须比较它，才能区分新机制和已知公式的局部应用。
2. **直接系数预测基线**：相同小backbone预测3个softmax系数，只混合相同候选，
   不生成RGB；这应称为匹配的kernel/weight-head控制，不能冒充原版KPN复现。
   它检验显式风险中间量是否优于普通学习融合。
3. **完整矩阵与对角矩阵**：除分别训练的matched控制外，加同一已训练矩阵头
   的推理干预 `M -> diag(M)`，保持输入、预测头、初始化与训练历史不变；同时
   报真实风险校准与最终重建。只比较6输出与3输出的不同网络会混入容量因素。
4. **先验条件是否必要**：完整矩阵下比较像素/flow/mask条件、匹配容量的V-JEPA
   条件、同clip打乱先验条件。若打乱先验或像素条件匹配效果，不能将收益归因于
   frozen video prior。source FIT监督、target不训练是正确协议，不是模块创新。
5. **无需真实图像即可证伪的机制测试**：复制同一个donor不能凭空增加独立证据；
   full模型不应像独立对角模型那样把重复观测误当降方差样本。另造相同中心化
   covariance、不同候选bias向量的合成误差，检验是否真使用未中心化风险。
   这些测试证明实现/行为，仍不证明新颖性。

若要成为主方法模块，还缺一个不能被上述局部CsNet与直接系数头轻易替代的
具体operator或因果机制，例如明确刻画对齐/遮挡诱发的共享误差结构，并在
匹配控制下证明这种结构估计能改善校准和恢复、且依赖真正的时序先验。
这只是研究缺口，不是声称该方向尚无文献或承诺一定新颖。

**执行建议：保留second-moment方案作为低成本强baseline有意义；把它直接
包装成主创新没有意义。** 即使full胜diag，也首先支持“建模相关误差有用”这一
已有原则。若不能同时超越条件CsNet、直接权重头并通过先验必要性对照，应
停止将其作为主贡献扩跑。当前冻结主实验继续按原协议完成。
