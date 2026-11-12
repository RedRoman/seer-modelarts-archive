# 官方基线可执行性核查（2026-11-12）

仅只读查看作者页面、官方代码与论文；TempRetinex 权重仅作匿名 HTTP HEAD，未下载权重正文、未安装依赖、未执行作者代码或使用加速器。完整路径、配置与证据见 `feasibility.json` 和 `github_readonly_evidence.json`。

| 方法 | 固定版本与公开权重 | 本轮协议可执行性 |
|---|---|---|
| TempRetinex | [官方仓库](https://github.com/liyinibristol/TempRetinex/tree/03ca79f57f2ad7b4c6120364de3a8c59afa42eed)，commit `03ca79f57f2ad7b4c6120364de3a8c59afa42eed`。BVI-RLV.pth、DID.pth 各 22,244,986 bytes；RAFT 21,109,305 bytes；固定 commit 的 raw URL 均 HEAD 200。 | 可进一步实现冻结推理适配，但现有权重未证明 SDSD-only 来源。BVI 权重不能作 SDSD→BVI source-only 证据；DID 权重只能另列 DID→BVI。 |
| DWTA | [官方仓库](https://github.com/lrr-rachel/DWTA-Net/tree/85b713940afe24e66ebe3d2497a2a8d41f35457d)，commit `85b713940afe24e66ebe3d2497a2a8d41f35457d`。树内唯一 `.pth` 为 UniMatch/GMFlow 光流权重 29,500,177 bytes。 | README、固定树及空 releases 中未找到完整增强模型 checkpoint。另有 Mamba 与自定义 DCN CUDA 依赖，尚不能直接部署冻结 NPU 对比。 |
| VLLVE++ | [论文 v1](https://arxiv.org/html/2602.08699v1) 写明代码和模型拟在发表时释放；本次未确认独立官方 release。 | 作者页面指向 [LLVE_STCD](https://github.com/xiaogang00/LLVE_STCD)，固定 main `1ee87d16a207a14df7d8b9db8445aed7678681dd`，不能将已经评估的 STCD 标成 VLLVE++。未确认不等于证明不存在。 |

TempRetinex 的 [test.py](https://github.com/liyinibristol/TempRetinex/blob/03ca79f57f2ad7b4c6120364de3a8c59afa42eed/test.py) 与 `predict.py` 采用 eval/no_grad，没有目标域梯度优化。但 test 默认额外报告用 **GT histogram matching** 的指标，公平适配必须直接使用预测 H3 和统一评分，排除 GT 后处理。模型内部基于历史预测的直方图匹配则不使用 GT。还需要处理硬编码 cuda:0、默认 `.pt` 与真实 `.pth` 不符、loader 固定 resize 1920×1080、权重过滤加载可能遗漏参数、逐 clip recurrent reset，以及事先固定正向/反向推理规则。

TempRetinex 的 [train.py](https://github.com/liyinibristol/TempRetinex/blob/03ca79f57f2ad7b4c6120364de3a8c59afa42eed/train.py) 默认 RLV、batch 1、5 epochs、seed 2、lr 3e-4、flow scale 1；[论文 v2](https://arxiv.org/html/2511.09609v2) 的 lr 5e-5、训练 flow 下采样 3 与默认值不同。外部 train/test 列表和 checkpoint 实际训练配置未随仓库公开，不能从文件名反推精确训练来源。严格 SDSD-only 需要另做固定源域配置训练与来源记录，不能靠已发布 BVI 权重补足。

DWTA 的官方关系由 [Bristol 机构论文页](https://research-information.bris.ac.uk/en/publications/dynamic-weight-based-temporal-aggregation-for-low-light-video-enh/) 确认。[配置](https://github.com/lrr-rachel/DWTA-Net/blob/85b713940afe24e66ebe3d2497a2a8d41f35457d/configs/DWTANet.yml) 是 DID、300k steps、batch 1、patch 512、5 frames、lr 1e-4；测试只处理完整五帧邻域，同协议 16 帧评分还需显式边界规则。公开 flow 文件不能替代完整 DWTA 增强模型。

这些结果只说明基线资源与适配工作量，不证明当前轨迹图机制创新或达到 SOTA。时间特征、光流引导聚合、对应关系修正均已有先例。
