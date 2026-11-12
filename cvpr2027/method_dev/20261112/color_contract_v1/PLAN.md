# Explicit color contract repair, version 1

Date: 2026-11-12. This is an implementation and experiment record, not a manuscript contribution. Fixing an input contract is not module innovation or evidence of SOTA.

## Trigger and evidence boundary

The historical generic NPY reader preserves stored channel order, while PNG readers yield RGB. The official SDSD/Retinexformer NPY reader exchanges channels `[2,1,0]` before returning RGB. A separate fixed audit covers only the two previously selected FIT clips from `SDSD_outdoor/MVI_0975`: original archive-member bytes, current NPY bytes, historical loader/crop/flip replay, source-cache LQ, and HVI export input. The machine-readable source-color receipt must bind these observations and the original source provenance before corrected inference or scores. No color order is inferred from a file extension, visual preference or target score.

Source provenance SHA256: `0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0`. All source input identities and the existing FIT/internal split stay fixed.

## Immutable historical boundary

Do not edit `code_full/data/common.py`, historical releases, checkpoints, source caches, flow fields, scores or their frozen protocols. New code, manifests, caches and reports live under this version. Preserve the original experiments as legacy stored-order evidence and identify the interpretation limits in the impact ledger. No guessed numerical correction of prior scores is allowed.

## Implementation

`color_contract.py` supplies two explicit boundaries:

1. Declared BGR NPY arrays become canonical RGB; ordinary image files are read with PIL's RGB contract. Unknown NPY orders fail before reading pixels.
2. A frozen BGR-native model is exposed as an RGB function by converting RGB input to BGR, calling the **entire** original pipeline once, and converting its BGR prediction to RGB. Its pixel and encoder branches retain their historical input convention. This does not repair an RGB-pretrained encoder that was conditioned on BGR during training. Changing only encoder color is a separate intervention that requires new source conditioning/training evidence.

The boundary helper and its test file were byte-frozen before remote Torch testing. Eight synthetic tests cover known color patches, range/layout conversion, rejection of unknown order/nonfinite inputs, an intentionally non-equivariant native pipeline, one-call behavior, gradient propagation, input preservation and output-shape validation. Actual Torch 2.7.1 CPU tests pass; their record also preserves the initial wrong-interpreter-path launch failure (no tests executed in that attempt).

## Corrected source sequence

1. **LQ-only HVI export:** use the frozen selective archive reader to retrieve only source LQ storage. Convert declared native BGR to RGB before the unchanged, strict-loaded HVI model. Produce all 136 source predictions in a new cache, with native/RGB input tensor hashes, source identities, weight/state hashes, unchanged-model check, and the source-color receipt. This step reads no GT storage and computes no quality scores.
2. **FIT-only diagnostic:** after export completeness and exact manifest identities are frozen, evaluate the same 102 FIT clips / 51 scenes. Convert legacy GT BGR to RGB. Keep the previously fixed seven rows: HVI, temporal mean with lambda 0.25/0.5/1, and same-frame identity controls at the same lambdas. No parameter selection, internal-17 evaluation, or target data access. Use the existing stored flow/validity proposals unchanged and explicitly label them as legacy proposals, not canonical-RGB flow re-estimation. RGB MSE/PSNR only.
3. **Independent verification and backup:** recompute reported aggregates and deltas from saved sufficient statistics; verify identity controls and provenance. Preserve raw failures. Copy completed evidence to a new exclusive SFS directory and verify bytes before committing compact text evidence.

The export and FIT protocols record their own exact file dependencies and runtime receipts. Any score-bearing execution must bind the final color receipt and new HVI manifest, rather than a mutable directory name alone.

## Interpretation limits and next method gate

This repair does not establish a cross-domain benefit. Historical SDSD-to-BVI deployment combined scene transfer with an input/output convention change for SDSD-only checkpoints; a new native-adapter replay would need a separate fixed protocol and source parity check before new target inference. No such target replay is authorized by this protocol. BVI has already been observed and must never be relabeled globally blind.

The current joint matching module failed to beat matched controls. Fixed HVI temporal averaging is an established operation. Predictor-based critics also have close primary precedents, and the available predictor output (1664) does not match the packaged encoder target (1024). Do not relabel these as novel or silently project incompatible spaces. A future core module needs a concrete distinct operator and source-only evidence beyond a matched conventional gate, descriptor matching, shuffled prior and same-frame control. Corrected data contracts are a prerequisite for that decision, not a substitute for it.
