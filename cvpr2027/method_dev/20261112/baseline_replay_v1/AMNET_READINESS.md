# AMNet RGB-only integration audit

Read-only upstream inspection on 2026-11-12; no AMNet score or checkpoint is
claimed. The [official RGB repository](https://github.com/lhfgghc/LLVE-AMNet)
was pinned at `9dcbe4db5e2a0710cc020936a80b32df8ca712a6` and its linked
[official Ascend repository](https://github.com/lhfgghc/LLVE-AMNet-NPU) at
`4c535f6462f5be9669268e35bb7fd2d6010be3ee`. Retrieved source hashes and the
SDSD RGB configurations are preserved in `amnet_audit/`.

The actual GT-free RGB interface is `AMNet.inference_video(x, event=None,
ir=None)`, input/output `[T,3,H,W]`. It uses a ConvLSTM and initializes hidden
state inside each whole-video forward. Both pinned repositories have identical
`models/amnet.py` bytes. The configurations include the RGB backbone, implicit
event/IR translators, SNR-guided fusion and ConvLSTM; RGB-only inference does
not require real auxiliary sensors. This is a feasible native video reference.

The NPU demo uses decoder blocks `[2,2,1,2]`, defaults to the general pretrained
weight, and loads with `strict=False`. A benchmark adapter must select an
explicit dataset-finetuned weight in advance, match its configuration, strictly
load every tensor, reset state for each clip and adopt the common uint8-rounding
scorer. The demo's default pretraining checkpoint and permissive loading are
not accepted as an independently reproduced finetuned baseline.

The official README links SDSD-indoor, SDSD-outdoor and DID RGB finetuned
weights only through Baidu shares; neither repository contains checkpoint
files. One anonymous request to the SDSD-indoor link returned HTTP200 at the
share initialization/extraction page, not a checkpoint (`baidu_access_check.json`).
No login or extraction/download API was attempted. The browser tool's request
for the alternative Google Drive **general pretraining** page returned HTTP401;
that file would not replace the specified finetuned checkpoint anyway. These
observations do not establish that an account is always necessary.

Current status: code and interface confirmed, official finetuned checkpoint
not obtained, strict-load/NPU inference/VAL16 evaluation therefore unrun.
Keep AMNet as a required pending modern video comparison and report the access
dependency explicitly. Do not substitute random weights, a different training
run, or the paper's published full-resolution PSNR for a common-protocol score.
