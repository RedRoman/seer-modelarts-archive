# Source input and color-contract audit

**A source color-contract error is confirmed.** The cached SDSD static `.npy`
values retain the archive's native BGR order, whereas the project labels them
RGB. Their scale, sampling/cropping/flipping and HVI frame batching reproduce
the original producer exactly. This is not evidence of an HVI-exporter scaling
bug. The source/BVI ranking reversal must be reconsidered after canonical RGB
preparation; its full magnitude is not attributed to color by this audit.

The audit was fixed in advance to `split.fit[0]`,
`SDSD_outdoor/MVI_0975`, clips108/109 (`MVI_0975` and `MVI_0975_2`). It inspected
32 distinct original LQ frames only. No GT storage was read, no model was
constructed or run, and no quality score, internal17-scene or target pixel was
read. The paired loaders received64 requests for GT-shaped zero placeholders;
these were never reads of actual GT pixels.

| Check, both clips | Result |
|---|---|
| Frozen SourceClips producer and saved augmentation plan versus cache LQ | Bitwise equal |
| Original ListedSDSD class body replayed with the same RNG versus cache LQ | Bitwise equal |
| HVI restricted LQ-storage reader/export input versus source cache | Bitwise equal; existing HVI manifest LQ hash matches |
| Per-frame HVI unbind versus time/batch flatten | Bitwise equal |
| Original frame scale | uint8 HWC3, normalized once by255; no second normalization |
| Official RGB sequence loader versus cached LQ reversed `[2,1,0]` | Bitwise equal |
| Official RGB loader versus cached LQ without reversal | Different |
| Selected original ZIP members versus extracted files | All32 exact SHA256 and byte-count matches |

Channel reversal changes max/mean absolute LQ values by
0.10588235/0.00886174 for clip108 and0.11764707/0.01009569 for clip109.
The cached maxima are0.40000001 and0.16862746, with finite float32
`[1,3,16,256,256]` tensors. These are input parity statistics, not new
restoration metrics. Frame indices, exact seeds, every selected raw-file/member
hash, loader hashes and all bindings are in [source_input_audit.json](source_input_audit.json).

## Archive and semantic provenance

The original download provenance identifies the static archives from
Retinexformer's public dataset release at commit
`1e9a0efce4b306b6701b824768370ff26066c32a`. Its
[NumPy reader](https://raw.githubusercontent.com/caiyuanhao1998/Retinexformer/1e9a0efce4b306b6701b824768370ff26066c32a/basicsr/data/util.py)
documents BGR values and changes channel order before returning RGB tensors.
The [SDSD author's sequence reader](https://raw.githubusercontent.com/JIA-Lab-research/SDSD/main/data/util.py)
performs the same conversion. Both primary source snapshots are retained.

The archived `outdoor_static_np.z01/.z02/.z03/.zip` download receipt contains
file sizes, locally computed SHA256 values and a completed archive CRC check.
The audit streamed **only the32 selected input members** from those split ZIP
parts with the existing7zz executable and compared their bytes to the disk
files. This rules out an intervening RGB conversion for the inspected arrays;
the conclusion is not based on a directory name. Publisher digests were not
available, entire27GB archives were not rehashed now, and no original PNG
counterpart was used. It is a verified official-loader/data contract, with
numerical pixel inspection limited to this fixed FIT scene.

The machine gate [source_color_receipt.json](source_color_receipt.json) has
`status=closed`, `native_source_color=BGR`, `verified=true`, the fixed source
provenance SHA, all32 member/disk hashes, two clip checks and five hashed proof
files. Receipt SHA256:
`fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70`.
Remote copy:
`/cache/zihaohe/seer_source_input_audit_20261112_v2/source_color_receipt_closed.json`.
The complete audit JSON SHA256 is
`2bcd5521be473cf8a379b370b389957d12e79a2f5efc5ada60657fddbe60f86d`.

## Canonical RGB adapter boundary

Let `P` reverse channels0/2. For the legacy Seer source observations,
`LQ_rgb=P*LQ_native`, `GT_rgb=P*GT_native`, and
`SeerBase_rgb=P*SeerBase_native`; this permutation preserves their full RGB MSE.
Source Y metrics must use canonical RGB weights after the conversion. A new
manifest must bind color explicitly rather than rewrite old caches.

For an unchanged legacy Seer checkpoint, an external canonical-RGB interface
can be `P*Seer_native(P*x_rgb, VJEPA(P*x_rgb))`. This reproduces the native
branch/encoder inputs and expresses the result in RGB; it does not repair the
fact that the pretrained V-JEPA historically consumed BGR under RGB channel
normalization. Replacing only the encoder input by RGB would change the
conditioning distribution of the already-trained pixel branch. This requires
a separate declared treatment, not silent checkpoint reinterpretation.

The old HVI source output is `HVI(LQ_native_BGR)`, from a model with an RGB
contract. **Reversing that output is not a substitute for rerunning
`HVI(P*LQ_native)`**: HVI is not guaranteed channel-permutation equivariant.
Similarly, previous V-JEPA token addresses cannot be corrected by swapping
feature dimensions; canonical-RGB prior/critic features need a new forward.
Pixel addresses and effective-error calibration must be rebuilt against the
new anchor. Saved Seer/RAFT flows can only be reused as explicitly labeled
legacy proposals, since RAFT also received the old color representation.

No adapter or cache change is implemented here. Root owns the new protocol.
The old HVI FIT15.41-versus-Seer28.56 result and the BVI Y ranking cannot now be
used alone to diagnose a shared exposure bias: the source HVI input contract
was wrong, and the two headline numbers also involve different populations
and RGB/Y metrics.

## Loss implication after canonical inputs are established

For a frozen canonical anchor `B`, label `G` and actual-observation correction
`d`, incremental squared risk is
`R(B+d)-R(B)=2<d,B-G>+||d||²`. Subtracting the anchor loss removes its constant
error term, but a low-frequency exposure error still contaminates the cross
term whenever `d` changes low frequencies. Relative risk alone is not exposure
invariant. This quadratic identity is already represented by the project's
`correction_risk.py`; relabeling it is not a module innovation.

One FIT-only critic target worth testing after color repair is incremental
gradient/high-pass utility, for example
`u(d)=[2<Dd,D(G-B)>-||Dd||²]/[||Dd||²+sigma_fit²]`, where `D` is a fixed spatial
derivative and `sigma_fit` is frozen from FIT data. It measures the detail
correction's usefulness relative to the same anchor, and kills a spatially
constant additive bias. It does not remove multiplicative exposure errors,
changing illumination, clipping, misregistration or chromatic bias. A box
high-pass is not an orthogonal projection. Common valid supports and appropriate
filter-support erosion remain necessary.

Such targets must use source FIT labels only, never a GT-derived gain as an
inference input. Preserve an exact zero-correction anchor path and still check
canonical full-RGB fidelity: a detail-only loss can reward noisy sharpening,
halos or texture while ignoring appearance errors. Donors share scene/model
biases, so agreement is not calibrated reliability. In particular, applying
the [Noise2Noise](https://arxiv.org/abs/1803.04189) argument would require its
noise/target-mean conditions rather than merely having neighboring frames.
Frequency losses, residual ranking and identity fallbacks are established
ingredients; the separate predictor-based evidence mechanism needs its own
matched-module evidence to establish innovation.
