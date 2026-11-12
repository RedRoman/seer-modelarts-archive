# Critic runtime contract

This is an experiment implementation record, not a novelty or quality claim.
The critic predicts one fixed noisy-LQ target-local EMA tubelet. It does not
observe GT or HVI; the static flow descriptor baseline reads a separately frozen
GT-free flow/LQ/HVI observable sidecar. Full-context descriptors may include donor
information and are used only as a strong descriptive baseline.

The unchanged critic code SHA256 is
`72c8040d21a4928c8a1f2bfc703acc65bf2e660cddad58bb0a08159545665e11`.
The declared 256 crop is preserved for transport alignment; ImageNet RGB mean/std
is reused, but the official hub resize292/center256 operation is not performed.
The context and target both use the matching frozen EMA encoder, following the
official inference interface rather than claiming online/EMA training semantics.

## Device decision and preserved failures

- The default NPU four-frame synthetic comparison failed the fixed normalized-loss
  absolute max `.01` / mean `.001` tolerance: max `.44734508`, mean `.13140345`.
  Exact masked-context and masked-prediction pixel-intervention invariance passed.
- The next precision script imported `torch_npu` before checkpoint loading;
  its serialization wrapper rejects a Path plus mmap. It failed before a model
  forward and is retained separately.
- Registering NPU after CPU checkpoint loading, with must_keep_origin_dtype and
  both convolution/matmul HF32 disabled, still failed the unchanged gate.
  Trained loss max/mean difference was `.02572155` / `.00517298`; random loss was
  `.00536168` / `.00142577`. Encoder context/target and normalized prediction
  differences are recorded in the raw synthetic report.
- Source critic export therefore uses the original official CPU SDPA implementation
  and float32 weights. There is no NPU source inference and no attention substitution.

## Compute schedule and provenance

A single CPU worker was interrupted during the 16-frame synthetic preflight to
use four fixed disjoint CPU workers on host31051. Its complete synthetic receipt
and source output directory did not exist, which together with the frozen driver
ordering establishes that no source LQ reader had been reached. No synthetic pass
is claimed for that interrupted attempt.

The first sharded driver had a duplicate exclusive checks-directory creation.
All four workers failed after model loading and before any synthetic/data forward.
Their four exceptions and zero-source-output evidence are retained. The new v2
driver removes only that extra mkdir and uses new exclusive directories. An
independent review also exercised eight synthetic merge fixtures, including
missing completion, mixed states, wrong identities/lineage/patch order and NaN.

The final CPU schedule is exactly indices `[0,1,2]`, `[3,4,5]`, `[6,7,8]`,
`[9,10,11]` from the fixed12 source-FIT pass1 list. Each worker uses four threads,
independently strict-loads the same checkpoint and seeded random predictor, and
executes the full 16-frame synthetic mask/loss schema plus exact masked-pixel
invariance gate before any source LQ is read. Outputs are exclusive worker
directories; the merger requires all four complete receipts, identical unchanged
states, exact source/RGB input hashes and unique coverage12 before writing the
joint manifest. It performs no quality-based worker or shard selection.

The frozen GT-free observable sidecar SHA256 is
`580df11ab460f2fb74043f1ae6292de6f0d74a674d86b1c5fb5620ee7f200f23`.
The source selective reader only opens metadata, byte order and its separate LQ
storage entry. Whole source archive SHA256 validation reads complete opaque bytes;
it does not interpret or open GT tensor storage entries. No excluded17, target
input or target score is used. No raw video or checkpoint binary is saved here.

Final data and closure status are given by the merged manifest and four completion
receipts; the presence of this runtime note alone does not assert run completion.


## Completed technical export

All 12 declared FIT pass1 clips completed once on the four fixed CPU workers.
The full 16-frame synthetic preflights took261.2–265.6 seconds each. Actual per-clip
export took220.5–237.8 seconds (mean 228.4 seconds). **These are complete probe costs**,
including all four coalition encodings, trained/random/wrong-time controls,
full/local static descriptor controls and metadata checks. Synthetic timing also
includes the additional exact masked-pixel invariance checks. They are not the
latency of a standalone transport module, nor a deployment throughput claim.
Each CPU with 4 threads worker took979.6–1004.6 seconds including model loading,
preflight, three clips and final state/hash checks.

All four preflight result arrays are exactly identical under canonical JSON
hash `db13a015512f2dc5205c93017f3b082ef143be33bffc3e66713106559d1938fe`.
Encoder/trained-predictor/random-predictor state digests match before/after and
across all four workers. The strict merger verified 36,864 finite loss values;
a separate verifier checked 108 input-lineage and pre-attention shape conditions.
The merged 12-clip manifest SHA256 is
`beb8748f72c7512da68b4f6cbcfe9a910b3bd137a16f6a42757488bc6844070b`.

The exclusive SFS backup contains 103 payloads / 5,402,264 bytes, each verified by
read-back SHA256. Its manifest hash is
`ce4284ed9c051506f3779f7a3e096940e90975985d72bca3ed59d9e8e58d88d2`.
The 1,366,780-byte compact archive has hash
`acd19d8a205ee5b955acc6f81ed592beb531e114de5c24f681a9541255b135c6`.
Local seven-part transfer and all 103 payload bytes/hashes were verified; only 97
UTF8 text files were mirrored into Git evidence. Raw source/video/model binaries
remain outside Git. Final owned-process and NPU 6/7 snapshots are separate receipts.
The statistical utility decision is recorded by the independent analysis phase;
technical export completion is not evidence that predictive credit is useful.
