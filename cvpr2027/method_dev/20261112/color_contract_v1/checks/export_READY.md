# Canonical-RGB HVI source export: complete

Completed 2026-11-12 06:29:40 UTC; inference 174.678 seconds on physical NPU 7 after a fresh no-running-process check. This is source preparation only: no GT tensor deserialization or pixel-value access/computation, no quality scores, no target inputs, no training.

**Read-statistic meaning:** source payload SHA256 is computed over each entire original `.pt` as opaque bytes; those reads include the archive's GT storage bytes. The selective ZIP reader opens zero GT tensor data entries and never deserializes or interprets GT tensor values. Recorded `gt_storage_entries_read=0` refers to that selective-reader operation, **not** zero whole-program GT-byte reads. Opaque full-file identity hashing is allowed and was performed both before inference and at final identity verification. The old export uses the same distinction; no frozen implementation or protocol was changed.

## Frozen execution

- Entry: `bash /cache/zihaohe/seer_color_contract_20261112/export_release_v1/export_run_v1.sh` (already executed; refuses an existing output directory).
- Release: `/cache/zihaohe/seer_color_contract_20261112/export_release_v1`; 22 frozen files, manifest SHA `3c17f59800b263bb1ef8c064493884dae2200dcf387d0aca54dc3950ae7ac390`.
- Export script SHA `bb9ebb3d3808e4ee263d567bd56a1407c8896a8ce749b3ebcc0ac934ac261aae`; protocol SHA `417ea20f4ae15e52664731aed3b5567f68ab4988b026d98fac6a9966254cc8d3`.
- Strict HVI state: 191 keys, weight SHA `6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27`, state SHA `afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b`; unchanged after inference.
- Reused exact frozen legacy selective reader, strict model loader and float predictor, adapter SHA `26c0c36656ad7b7d3b0ce04275578a45286c27931a476c9dcfb1ce1981e69db2`. Shared boundary helper reference SHA `7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60` is frozen in dependencies; its source was not changed. Explicit permutation is in the new exporter.
- CPU fixture tests: 8/8 pass on remote Torch 2.7.1; actual HVI strict-load plus synthetic RGB inference also passes. Local system Python 3.14 lacks Torch, so it was used only for syntax compilation and archive verification; no packages were installed.

## Color evidence and inputs

Native source cache is declared BGR using the closed original-archive audit, not inferred from scores. The final receipt SHA is `fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70`; all 5 primary proof-file hashes were checked before inference. The earlier draft receipt was not used.

Original source provenance SHA `0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0`; source summary SHA `63df74b887dc66543137569d58b6fd57c6fbdb931c6f4c6077c53b77585031ae`. Exactly the original SDSD TRAIN136 clips / 68 scenes were exported. The immutable selective archive reader opens metadata, byteorder and the LQ storage only; whole-file SHA reads establish byte identity without deserializing GT.

Boundary: source `BGR` -> channel index `[2,1,0]` -> HVI `RGB` -> output `RGB` without another swap. Float inputs are not reencoded to uint8. HVI predictions are float32, clamped to `[0,1]`, without rounding.

## Consumer interface

Cache: `/cache/zihaohe/seer_color_contract_20261112/cache_rgb_hvi_v1`; `manifest.json` SHA `5f25fdfcca5bc7d581c1484245fa760ef321da642f5e741077c128376616a94a`, `complete=true`.

136 payloads `hvi_rgb_000.pt` … `hvi_rgb_135.pt`, total 1,711,809,608 bytes. Each contains only `name`, `scene`, `base`, `lineage`. `base` is canonical RGB float32 `[1,3,16,256,256]`; no LQ, GT or original base tensor is copied. `lineage` binds native-LQ and RGB-LQ exact tensor hashes separately, source file hash, source provenance/split, channel boundary, checkpoint/model state, export script/config/release and BGR evidence.

`lq_tensor_sha256` retains the legacy reader's meaning (native BGR) and equals `native_lq_tensor_sha256`; `rgb_lq_tensor_sha256` identifies the actual HVI input. Consumers must not treat the legacy alias as the RGB hash. The manifest's `per_sequence` rows bind file/base tensor hashes, source hash and both input hashes.

Independent new-output reopening verified all 136 payloads, coverage, shape/dtype, finite/unit range, file and tensor hashes and color/checkpoint lineage. Final verification rehashed all 22 new release files, all 10 old release files and all 136 original source payloads unchanged; no export process remained active. See `export_outputs_verification.json` and `export_final_identity_verification.json`. The architecture owner has been notified that this complete manifest is ready for separately authorized FIT-only replay.

## Backup and local evidence

Exclusive SFS directory: `/temp/zihaohe/seer_cvpr/method_dev_20261112/color_contract_rgb_hvi_source_v1_final_v1`. All 176 payload files / 1,720,160,220 bytes were SHA-verified, including 136 HVI bases and one checkpoint. No original LQ/GT videos were copied. Backup manifest SHA `8ceceeb472ee164405d21682e353b004fed3c674c4bc8aeb8505dcefb18488df`.

Compact text archive: `/Users/hezihao/Documents/New project/seer_color_contract_20261112/export_compact_text_v1.tar.gz`, 93,571 bytes, SHA `66911af194831d0a75b10460efe171a9130b079a967718234f5f5951dfe4ec2b`. All 40 text files were extracted safely; 39 payload hashes plus the backup manifest were verified locally. No `.pt` file was downloaded. Complete extracted evidence is beside the archive; selected logs/manifests/receipts are under this repository's `checks/export_*` paths.

This export establishes a color contract and source artifact. It does not establish an enhancement gain or module novelty.
