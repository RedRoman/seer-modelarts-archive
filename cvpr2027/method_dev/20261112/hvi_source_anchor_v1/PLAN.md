# HVI source anchor cache v1 — preparation only

This independent preparation does not change the frozen joint-search screen.
Export a spatial anchor for the original SDSD TRAIN136/68 scenes only, using
the exact HVI-CIDNet generalization checkpoint already replayed. No module
training, GT-pixel inspection, source score, held-out score, target score,
checkpoint choice, brightness fitting or new claim is authorized by this cache.

Use the original source cache's `tensors.lq` float32 `[1,3,16,256,256]` values
directly, with no uint8 reconstruction or rounding. The source `.pt` archives
also contain unrelated tensors. A restricted metadata parser represents tensor
references without constructing their values, then reads **only the LQ storage
entry**. It must reject storage sharing with another tensor. Whole-file SHA256
reads raw archive bytes solely for identity; it never interprets GT pixels.
Tests must prove correct strided LQ reconstruction, no-GT fixtures, refusal of
GT-storage access, shared-storage rejection and an allowlist of pickle globals.

Freeze HVI's191 strict state keys, default CIDNet configuration and unmodified
four upstream source files from baseline replay. Checkpoint SHA256 is
`6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27`;
state SHA256 is
`afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b`.
Inference is independent per frame, batch1, float32, no augmentation/ensemble,
clamp to [0,1] and **no rounding**. No original base, GT, flow or other donor
tensor enters HVI. Weight/source/config/adapter hashes are frozen before export.

Require strict-load CPU smoke, GT-free fixture tests and a synthetic CPU/NPU
float-output comparison before real export. Freeze the comparison tolerance as
maximum absolute error <=.01 and mean absolute error <=.001, reflecting the
existing FP32 NPU stack; report actual differences. This synthetic comparison
contains no real image or label. Require fresh NPU7 free-process evidence
before dispatch and never touch another user's card/process.

Output one payload per source clip with **only** `name`, `scene`, float32
`base`, and observable `lineage`. The lineage records full source-file SHA,
exact canonical float-LQ tensor SHA, source-provenance identity, selected archive
entry, fixed model/config/release identities and output tensor SHA. Never copy
LQ/GT images, original base/flow, target inputs or scores. Validate output shape,
dtype, finite values, [0,1] range and hashes only; log no image statistics or
quality metric. Require exactly136 outputs,68 scenes and unchanged model state.

Use a new exclusive runtime `/cache/zihaohe/seer_hvi_source_anchor_20261112`
and new SFS backup `method_dev_20261112/hvi_source_anchor_v1_final_v1`.
Copy only the new compact float32 HVI outputs and reproducibility records;
preserve old caches/results. This prepares a future source experiment and
does not establish that HVI-plus-module works, that the current module should
be replaced, or that any new method is SOTA.
