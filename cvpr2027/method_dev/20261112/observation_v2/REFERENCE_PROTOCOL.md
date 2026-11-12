# Additional frozen TempRetinex reference

Specified before this reference's BVI evaluation. This separate reference does
not alter the seven v2 variants, 42 training heads or 31 methods per base arm.

Use the [author-linked TempRetinex repository](https://github.com/liyinibristol/TempRetinex/tree/03ca79f57f2ad7b4c6120364de3a8c59afa42eed)
at commit03ca79f57f2ad7b4c6120364de3a8c59afa42eed and its author-named DID.pth
release. Its actual training split/configuration has not been independently
verified. Do not use the BVI-named release. This is an external reference with
different training data, rather than a matched SDSD-only comparison.

Follow the public predict.py forward sequential inference: flow scale1,
official full RAFT with20 iterations, built-in four-view spatial ensemble,
prediction-history histogram matching, original warp coordinate/padding
conventions, and output H3 (result[1]). No reverse inference. The official
test.py default flow scale3 differs; we fix the predict.py scale before seeing
target scores. Preserve these choices without target-based tuning.

Evaluate the same complete BVI VAL16 list, center16/crop256, and reset recurrent
state at the beginning of each clip. Use direct crop256 tensors, without the
official loader's forced1920x1080 resize. Thus this is the project's common
clip protocol, not the author's original full-video benchmark.

GT is used only by the common rounded-uint8 RGB/Y PSNR, LPIPS and per-sequence
absolute tLP-to-GT-gap scoring. Exclude the GT histogram-matched metrics in the
official test script. Internal histogram matching to a previous prediction is
observable and is retained. No target optimizer, adaptation or selection.

Require strict loading of the44 enhancement-state keys and179 RAFT-state keys,
input/checkpoint/source hashes, unchanged states after inference, complete
16-sequence coverage and finite metrics. Synthetic CPU smoke/reset checks are
implementation evidence only. Accelerator dispatch requires a fresh free-card
check and the same isolated scoring environment as the main experiment.

Report all results and the training/context differences. This reference alone
does not establish a fair source-matched SOTA ranking.
