# HVI source anchor v1 — ready, no quality evaluation

The original SDSD TRAIN136/68-scene float HVI anchor cache is complete and
verified. This is independent preparation for a future experiment; the frozen
joint-search experiment, its cache, scripts, controls and scores were not changed.
No GT pixel, source/held-out/target quality score or module training was used.

Ready runtime cache:
`/cache/zihaohe/seer_hvi_source_anchor_20261112/cache_v1`.
Its [manifest](evidence/cache_manifest.json) SHA256 is
`f70b1d9cb6b334202e4901c6635336c9a2fc694a1f3de7816757834fe8fdb3fb`.
The136 output files total1,711,673,288 bytes. Export completed at
2026-11-12T05:33:07Z after170.66 seconds of actual export work.

Each `hvi_000.pt` through `hvi_135.pt` contains only `name`, `scene`, float32
`base` `[1,3,16,256,256]`, and observable `lineage`. The model consumes original
source-cache float LQ values directly; neither input nor output is uint8
reencoded, and outputs are clamped without rounding. There is no copied LQ,
GT, original-base, flow or target tensor. Lineage binds each output to its
source-file SHA, exact canonical float-LQ tensor SHA, source provenance,
fixed model/config/release identity and output tensor SHA.

The source reader does not `torch.load` the mixed original archives. Its
allowlisted metadata parser constructs tensor-storage descriptions, then opens
only the unique LQ storage entry. Shared storage is rejected before values are
read. The production compatibility probe opened exactly `data.pkl`, `byteorder`
and LQ `data/0`. Whole-file SHA hashing reads raw archive bytes for identity;
it does not interpret the other tensors' pixels. All136 production payloads
record one LQ storage entry and no other tensor-storage read.

Six actual CPU tests passed with zero skips: strided LQ reconstruction,
non-LQ-storage read rejection, absence of a GT key, shared-storage rejection,
pickle-global allowlisting, and exact float/GT-free output-schema preservation.
The fixed191-key HVI weight loaded strictly. Its synthetic CPU/NPU comparison
had maximum absolute difference0.0003399551 and mean0.0000453938; both passed
the frozen tolerances. Both model states matched
`afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b`
and remained unchanged. No real image was involved in this comparison.

Protocol/config/adapter/tests were frozen before source export, with release
manifest SHA256
`2df36567d068c0ead173c89ebe50611e8c340aa7c6e3e731f502fce1f58ea11b`.
Fresh physical NPU7 checks preceded synthetic execution and actual export;
other users' processes were untouched. The final check found zero producer
inference processes and all10 frozen release payloads unchanged.

[Independent output verification](checks/outputs_verification.json) reopened
all136 new HVI payloads and checked exact schema, coverage, shape, float32,
finite [0,1] range, file/tensor hashes and source lineage. It accessed no
original source tensor storage and computed no quality metric. Its source
bindings use the already frozen source metadata only.

The new exclusive SFS backup is
`/temp/zihaohe/seer_cvpr/method_dev_20261112/hvi_source_anchor_v1_final_v1`;
its `cache_v1` subdirectory is the durable equivalent of the runtime cache.
All164 files/1,719,880,997 bytes, including compact float outputs, fixed weight,
sources, manifests, tests and logs, were copied and SHA-verified. Backup manifest
SHA256 is
`b08eaf4606e5aa05cffa887185d3bc4599241c4a514f668929369de7270dc134`.
Original RGB/GT source images were not duplicated.

The66,427-byte text archive was downloaded and SHA-verified as
`8f72043312db6b787266fd79172e8cf6c1bb8f49bee26f3d479fa5abcee52d5a`.
All27 downloaded text/source payload hashes match the shared manifest and
the local adapter/config/protocol match the frozen runtime. The136 float
outputs and one weight remain in the verified remote backup, outside Git.

A future consumer must bind the final manifest, exact source mapping and a
separately frozen protocol. This cache alone does not show any new module
improvement or justify replacing the currently running screen.
