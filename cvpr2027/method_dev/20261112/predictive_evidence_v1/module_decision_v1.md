# 下一模块决策：把联合预测贡献变成观测运输预算

2026-11-12。仅方法规格与可证伪判断；本文件没有实现模块，没有读取新 GT / internal / target，没有启动推理、训练或数据导出。

**结论：下面是一个今天可以实现的小算子，但目前仍不足以认定为核心原创模块。** 它比“把 surprise 拼进 CNN 再预测 gate”更明确，能隔离一个结构假设：两个 donor 的联合可预测性收益，应限制它们总共能改动多少真实观测。但 Shapley 归因、容量约束、水位分配和保留 anchor 都不是新的；若它不能胜过使用同样输入的普通 gate，或联合贡献不比独立贡献有效，应停止这个候选，而不是再命名。

## 为什么只验证这一个假设

纠正颜色后，HVI 在固定 source FIT102 / 51 scenes 的 mean clip RGB PSNR 仍为 **15.322981**。预先固定的时域平均 lambda .25 / .5 / 1 分别提高 **.233501 / .372559 / .276572 dB**，三个条件都使 51 个 scene 的 mean MSE 下降；没有选择 lambda。这证明该 anchor / 旧 flow 配方有源域时域余量，不证明 JEPA 的贡献或新模块的必要性。[实际结果](../color_contract_v1/RESULTS.md)

source 的绝对误差很大，不应让一个自由 RGB 头通过修正源域曝光风格制造“时序机制有效”。本候选只运输真实邻帧细节，固定每个 patch 的 anchor RGB 均值；该限制是混杂控制，不是创新点。固定平均依然是必须保留的强基线。旧 joint matching 已失败，故不再同时引入新的对应搜索、噪声模型和 RGB decoder。

## 唯一候选的输入与可执行定义

输入只有 canonical RGB LQ `x`、冻结 HVI 输出 `b`、现有 target-to-source flow / validity，以及**真正匹配的**冻结 V-JEPA2 encoder / predictor。第一个版本的 donor 是不重叠的前后 tubelet；2-frame tubelet 必须作为完整干预单元。旧 2.1 distilled encoder1024 与 predictor1664 不可组合使用。

### 1. 四个上下文，固定一个可观测预测目标

对目标 tubelet 的 patch `p`，`H` 是被预测的目标位置，`C` 是不含目标及两个 donor 的固定上下文。对 `S = empty, {1}, {2}, {1,2}`，在 **patch embedding 后、第一层 transformer 前**按官方 mask 语义保留 `C union S`，位置编号保持不变：

```
L_p(S) = || normalize(P(E_mask(x, C union S), H)_p)
            - normalize(z_H,p) ||²
v_p(S) = L_p(empty) - L_p(S),   v_p(empty) = 0.
```

`z_H` 来自同一匹配目标 encoder 对目标 tubelet 自身的编码，其他 tubelet 在编码前移除；所有四个 `S` 共用完全相同的 `z_H`。这避免 donor 先污染 full-context teacher target，但 **target-local 编码本身偏离官方 full-context teacher 分布**，是必须先检验的风险，不可宣称已解决。使用 canonical LQ 和官方预处理；不使用 GT、GT 亮度匹配或按 coalition 分别做曝光归一化。

不能删除已有 full-context block17 / 64维缓存后称为真实 donor removal；该缓存已经混入被删除 donor 的信息。新 critic 需要实际 masked encoder 调用。`v` 只是该模型对这一个可观测目标的预测帮助，不是干净 RGB 风险、相机噪声、条件互信息或因果贡献。

### 2. 联合贡献限制总运输量，而不是只进入一个 softmax

省略 patch 下标。两 donor 的精确 Shapley 值和交互为：

```
I = v12 - v1 - v2
phi1 = (v1 + v12 - v2) / 2
phi2 = (v2 + v12 - v1) / 2
```

这里明确使用已有 Shapley 公式，不能把它作为新理论。令 `tau > 0`，定义个体容量及联合总预算：

```
c_d = min(1, max(phi_d, 0) / tau)
B12 = min(1, max(v12, 0) / tau)
```

若 `B12=0`，所有运输为零。否则主算子和无 caps 对照都用同一个 `r = masked_softmax(phi/tau, valid_d)` 作为未约束分配比例；主算子再做两变量有上限的水位分配：

```
w_d = min(c_d, k * r_d),   choose k so w1+w2 = B12.
```

两个变量可枚举“均未饱和 / donor1 饱和 / donor2 饱和”三种情况，完全不需要大网络或迭代优化。由于 `phi1+phi2=v12`，正部容量之和足够容纳 `B12`；仍应对有限精度写断言。无效 donor 容量为零，只有一个有效 donor 时退化为其单独贡献，两个都无效则保留 base。不另加 photometric 权重或 uncertainty 模型；主算子与无 caps 对照必须共享上述 `r`，否则会把分配偏好变化混入“去容量约束”的比较。

**关键代数退化，不能包装成额外复杂算子：** 当 `phi1,phi2 >= 0` 且 `0 <= v12/tau <= 1`，每个 `phi_d/tau <= 1`，所以 `B12 = c1+c2`。结合 `w_d <= c_d` 和 `w1+w2=B12`，唯一解就是 `w_d=phi_d/tau`，完全不依赖 `r` 或水位分配。若一个 `phi` 为负且 `0 < v12/tau < 1`，负信用 donor 容量为零，唯一正信用 donor 取得全部 `B12`；`v12<=0` 则全部关闭。因此常见的小信用、`tau` 初值1区间，主算子实质是截断/缩放的 Shapley 混合，而不是新的复杂运输求解器。在非负信用且总预算进入上限后，只有尚有分配自由度的区域才可能真正使用 `r`。

任何后续机制试验必须先记录全部有效 patch 中的零预算比例、`v12/tau` 分布、总预算达到上限比例、个体容量达到上限比例、实际 `w_d=c_d` 约束活跃比例，以及上述唯一解退化区间的占比；要区分“容量数值被1截断”和“分配碰到容量”两个概念。若多数 patch 落在退化区，不能凭 caps / water-filling 名称声称额外结构贡献。main 对同质量无 caps 对照的差异在这些区域只是 Shapley 线性分配与 softmax 分配的差别；可检验机制主要剩下真实联合预测信用是否比加性信用和普通可观测可靠性更有用。

**结构假设：** 固定 `v1,v2`，改变真实联合贡献 `v12` 会改变可行的总运输预算。两个独立 sigmoid 或只对单 donor score 做归一化不能自动满足这一约束；但一个足够灵活、能看到 `v12` 的普通 gate 完全可能学会同样映射，所以不能宣称超越了 gate 的函数表达能力。这是可检验的约束归纳偏置，不是一个新注意力类别，也不是全局 optimal transport。

**必须加入同总质量、无个体容量的对照：** 保留主算子实际计算的同一个 `B12`，只在有效 donor 上分配：

```
w_d_mass_control = B12 * masked_softmax(phi_d / tau, valid_d).
```

无有效 donor 时权重全零；其余情况下两个权重之和与主算子严格相同，均为 `B12`。该对照不增加参数，使用相同 `gamma / tau`、DC 投影、值、mask 和 source 计划，只删除 `w_d <= c_d` 个体容量约束。主算子胜过它才可能支持容量分配约束有用，不能仅以相对普通三路 softmax 的收益作判断。相同总权重并不保证实际 RGB 残差范数相同，因为两个 donor 的值可能相消或幅度不同；应同时记录输出残差 RMS，但不事后重标定或以残差范数选配置。

### 3. 真实值运输、曝光约束和精确零初始化

只用同一份固定 flow 得到真实邻帧 HVI 值 `a_d = warp(b_d)`；不使用 predicted latent 生成 RGB，也不取期望坐标后重新搜索。令 `M_d` 为固定有效支持，对每个不重叠 patch16 定义：

```
D_d = M_d * ((a_d-b) - mean_valid(a_d-b, M_d))
u = b + gamma * (w1*D1 + w2*D2)
y = projection_[0,1],same_patch_mean_as_b(u), only on union(M1,M2).
```

投影是逐通道带盒约束的固定和欧氏投影：`clip(u+c,0,1)`，选择标量 `c` 使有效区域的和等于 `b`；有效支持外严格保持 `b`。这是已有数值操作，作用是避免夹断后破坏均值约束。不能用 `I-boxblur9` 冒充精确零均值投影；也不能声称保存了全部低频，只保存声明的 patch DC。硬 patch 边界和抑制真实低频修复都是代价。

仅两个可学习标量：`gamma=tanh(g)`、`g` 每步投影到 `[0,4]`，初始化 `g=0`；`tau=exp(q)`，初始化 `q=0`、有界正数。所有对照同 state / plan / optimizer，零初始化严格 `y=b`。无 decoder、无 learned correspondence、无 new backbone。GT 若以后使用，仅用于 source FIT 的 RGB 监督/机制诊断；预测接口没有 GT，目标域不优化。

建议接口：

```
prepare_masked_credit(x_rgb, encoder, predictor, target_tubelet_ids)
    -> losses[B,U,P,4], mask_ids, checkpoint/source/normalization fingerprints
transport(base_rgb, warped_donors[B,2,3,U,H,W], valid, losses)
    -> prediction, {v1,v2,v12,I,phi,capacity,total_budget,weights}
```

CPU transport 很小；真实成本来自新 masked encoding。四 coalition + 一次 target-local teacher，单个目标 tubelet 约 5 次编码；全 8 个目标 tubelet 不能谎称仍是原来一次 encoder。先做固定中央 tubelet 的 source 机制试验，不能未经计时直接承诺全136高吞吐导出。既有 source LQ / RGB HVI / flow 可复用，已有静态 prior token 不能替代这些调用。

## 重复、冗余与互补的明确行为

- **同一物理 donor 重复登记：** 先按 `(clip, source_tubelet, sampling_grid)` 去重，再算 coalition；重复不能增加上下文 token、预算或值质量。这是输入一致性，不是学出的独立性。
- **两个不同位置却完全冗余的 donor：** 若 critic 给出 `v1=v2=v12=v>0`，则各得 `v/2` 的未饱和容量，总运输 `min(1,v/tau)`，而不是加倍。若两个 warped 值完全相同，输出与等信用单 donor 一致。实际 predictor 不一定满足这个假设；时间位置和上下文长度可能让重复内容虚增信用，必须测试，不能宣称天然复制不变。
- **独立或互补：** `v12>v1+v2` 提高联合预算并分配正交互；例如单独都无收益但合起来有收益，两者都可得到质量。负交互则收紧总预算。它并不自动说明联合 RGB 恢复更好——两个共享错误也可能更容易预测。
- **不可改善保证：** 非负预算、合法范围及 anchor identity 只保证代数行为；不保证 PSNR 不下降、时间更稳、风险校准正确或跨域成立。

## 最近工作会如何否定“强创新”表述

完整来源和限制见 [nearest_work.md](nearest_work.md)。本轮另外重新打开了下列三个 primary 页面确认关键边界。

- [WMReward](https://arxiv.org/abs/2601.10553) 已用 V-JEPA2 奖励引导视频生成候选轨迹。因此“预训练世界模型指导视觉恢复”不够；这里最多是实际 donor 的上下文干预与具体运输容量的连接仍待验证。
- [ANCHOR](https://arxiv.org/html/2608.09342v1) 已有可观测证据、trust field、保留当前观测并修正输出。换成 HVI anchor、零 gate、有限幅度都不是新的核心。
- [CsNet](https://arxiv.org/pdf/1711.06712) 已有包括 bias 的误差二阶矩及 simplex 二次风险组合。本规格不预测风险矩阵、不声称求 MSE 最优权，但如果最终优势可由同输入的条件 CsNet 或直接权重头解释，仍缺独立机制贡献。[已有逐式核对](../hvi_temporal_fit_feasibility_v1/checks/second_moment_neighbors.md)
- [Fovea-JEPA](https://arxiv.org/html/2610.34033v1) 的误差干预/贡献与融合先例进一步压缩空间；其 error 是下游分类误差，不应混称本规格 native prediction error。Shapley、coalition credit、受限分配也各自已有。本轮没有证明不存在同类 restoration 算子。

**最严厉的评价：这仍可能只是“已有 Shapley attribution + 有限容量 gate + 观测核融合”的组合。** 目前没有足够证据称为用户要的强模块创新；不能为了赶进度把公式具体化误当原创性已确立。

## 最小会推翻候选的 source-only 试验

先冻结 matched encoder/predictor strict-load、mask/teacher 合同及实际 runtime。然后只取现有 FIT manifest 中预声明的 12 个 scene、各固定一个 pass、中央一个完整目标 tubelet；不读取 internal17 或 target。先不训练两个标量，不做全量导出。所有操作用 canonical RGB 和同一真实值/支持：

1. 检查“复制物理 donor不变、替换为错时 donor、交换 donor次序、共同通道合同、mask无泄漏、零门控恒等”。固定背景仍可预测但无真实细节帮助的负例必须包含。失败就是实现/critic 合同失败，不能继续扩跑。
2. 用固定幅度 `.5` 的同一运输动作，在这些 FIT GT 上仅做诊断，计算加入 donor 的条件 RGB MSE 改善，并去掉 patch DC 的自由度。比较 `v1,v2,v12` 的条件效应能否区分有益/有害 donor，是否超过相同 token 的静态相似度、像素/flow一致性，以及随机 predictor / 时间打乱的同算子。若只能预测亮度或场景难度、不能预测**增量**改善，立即否决。
3. 分开检验交互和容量。交互消融在完整相同算子中设 `v12 := v1+v2`（消除交互，其他输入、参数、候选、支持全相同）。容量的主对照使用上面的 `B12 * masked_softmax(phi/tau)`，其总运输质量与主算子相同，只去掉个体 caps；因此不能把 main 对它的差异解释为总运输预算更小。另保留普通三路 gate：`softmax([0,phi1/tau,phi2/tau])` 的两个 donor 分量，仍乘相同 `gamma`、使用相同 DC 约束/真实值。三路 gate 与 main 同两个标量及初始输出，但 **donor 总质量不同**；main 只胜过它，不能单独归因于 credit 或容量机制更好，可能只是更少运输。若真实交互不胜加性消融，或 main 不胜同质量无 caps 对照，对应组件贡献即不成立；若普通 gate 同样好，也没有证据把整个受限算子推广为更好的核心机制。随后若获准训练，所有模式同两标量、400步和固定source计划，不能调整筛选阈值来救结果。

不能因为这12个开发 FIT scene 正向就称泛化，也不能把已经多次开发使用的 internal17 称新盲测。只有这一最小机制试验值得先投入；若它不通过，**不建议今天直接再启动一个大规模“新模块”screen**。固定时域平均的正向事实应保留，但它不能替代所缺的创新证据。
