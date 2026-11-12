# Frozen native-color replay, version 1

This repair checks the unchanged SDSD-only E17 small seed0 final20k bases with
pretrained V-JEPA (P) and no tokens (N). It tests their declared color interface;
it is neither a new method nor an isolated causal estimate of pretraining. The
two original pixel branches did not have fully matched initial states.

The official archive/loader receipt closes static source arrays as native BGR.
Canonical RGB inference is `P Seer_native(P RGB)`, where `P=[2,1,0]` surrounds
the **whole** native pipeline: both the pixel branch and its historical encoder
receive native BGR. The encoder retains its historical ImageNet normalization.
No encoder-only color fix, retraining, gain fitting, parameter search, GT-mean
alignment, selector, candidate search or optical-flow computation is performed.

P checkpoint: `5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740`.
N checkpoint: `fa3c0c9e720a28b56f01069f7a2cd407bc5b4d752a2a1d4482f1bd63557b5516`.
Both are the original shared `inputs/e17_small_*_s0_final_cpu.pt` files. P uses
the unchanged V-JEPA checkpoint `7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`.
The color helper is frozen at `7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60`;
closed source receipt is `fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70`.

Before any target pixels, execute exactly source FIT indices 108 and 109 in
scene `SDSD_outdoor/MVI_0975` for each arm. The restricted reader opens only LQ
and old-base tensor storage; GT storage remains unopened. Inputs are original
float32 values, with no uint8 reencoding. Both arms must have identical LQ.
The wrapper captures its actual native input and output: input must equal the
native cache bitwise, and its RGB output must equal the captured output's exact
channel permutation bitwise. Pixel weights load with zero missing/unexpected
keys, and state/reconstruction must match original provenance before and after.
The P encoder's used video keys must match its loaded checkpoint exactly; any
unused image-only keys are listed explicitly in the receipt.

**New repair compatibility thresholds, declared before real forward:** each
replayed native base must have max absolute difference <=`1e-4` AND mean
absolute difference <=`1e-5` against its original source base, in unit-range
float RGB/BGR. These correspond to 0.0255/0.00255 uint8 levels. These are new
gates, not purported historical base tolerances. Wrapper permutation identity
is exact. A failed gate is retained and investigated without widening it.

Runtime is frozen NPU BF16 autocast for the original encoder and pixel forward,
float32 clamped output, batch1, four CPU threads. Choose only physical NPU6/7
after a fresh `npu-smi` confirms no process on the chosen device. Never stop
other jobs. Hash all release Python files, upstream V-JEPA sources, metric
source, weights, split, metadata and closed color proof chain. Save runtime
versions and unchanged states; no predicted RGB tensors are saved.

The target entry point is prepared but requires a separate parent authorization
receipt binding this protocol and both successful source receipts. Its only
evaluation is frozen BVI VAL16, center16 frames, center256 crop, original paired
filenames, RGB PNG inputs. All two-arm outputs are retained without selecting
an arm. Use the existing `score_prediction`: rounded uint8 RGB/Y PSNR, LPIPS,
unwarped tLP and |tLP-GT tLP|; no GT exposure alignment or GT-derived flow.
GT is used only after prediction. This phase's initial authorization covers
preparation and source parity, **not target execution**. No target pixels are
read by source mode, tests, freezing or package verification.
