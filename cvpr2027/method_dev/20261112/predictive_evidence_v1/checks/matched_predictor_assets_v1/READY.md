# Matched V-JEPA2-L predictor asset: available

2026-11-12. Availability verification only. No dataset pixel, GT pixel, target example, quality score, training or NPU use. This is a **new independent critic asset**, not a replacement for the historical V-JEPA2.1 distilled-L restorer conditioning and not a module novelty claim.

## Authenticated asset

- Official original URL: `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt`, linked by the [official model README](https://github.com/facebookresearch/vjepa2#models).
- Official HTTPS checksum source: [facebook/vjepa2-vitl-fpc64-256 original/model.pth](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/original/model.pth), independently read as its [raw LFS pointer](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/raw/main/original/model.pth). The model repository head displayed was `b3c1679b7c34d3255ef3547f27c7b226aefab26f`; a pinned raw-pointer web request was unavailable, as explicitly recorded in `checksum_primary_receipt.json`.
- Exact size: **5,127,726,842 bytes**. SHA256: **`5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012`**.
- New binary: `/cache/zihaohe/seer_matched_predictor_assets_20261112_v1/weights/vitl.pt`.

The FBA HTTPS certificate failed as expired on both local and 31051. HF HTTPS was network-unreachable from 31051. Official FBA HTTP HEAD/ranges worked; no TLS verification was bypassed. The entire HTTP asset was authenticated against the public **HTTPS** HF SHA and size before any `torch.load`, then rehashed again immediately before loading. Transport failures and bounded retries are retained: single stream stopped after 865,075,200 bytes due slow throughput; an eight-range attempt hit its 180-second limit with one 56,777,952-byte suffix missing; a final bounded suffix request plus joining completed in 21.07 seconds and produced the exact public SHA. No incomplete/mismatched asset was loaded.

The full V-JEPA2.1 ViT-G teacher HEAD was HTTP200 / **30,238,058,912 bytes**, and was not downloaded. The inspected deployed weights directories contained the historical distilled-L checkpoint, with 1024-dimensional encoders and 1664-dimensional predictor output, but no matched original ViT-L or full teacher asset. The new original ViT-L asset resolves that dimensional/interface availability issue independently.

## Frozen local implementation

The deployed source tree `/temp/zihaohe/seer_cvpr/modelzoo/vjepa2-main` was copied to the new root's `upstream/`; all **36 source files** are pinned in `evidence/upstream_manifest.json`. The selected architecture follows the [official V-JEPA2 constructor](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py): ViT-L, patch16, tubelet2, encoder width1024, predictor width384/depth12/heads12, ten mask tokens, predictor output1024, RoPE.

The frozen hub file contains a testing `http://localhost:8300` base URL. It was inspected and hashed, **never executed**. The script imports the frozen local `src.models.vision_transformer` and `src.models.predictor` directly and asserts their actual module paths. There is no `torch.hub`, remote-main execution, source patch, learned bridge, random predictor head or upstream modification.

Checkpoint top-level keys are `encoder`, `predictor`, `opt`, `scaler`, `target_encoder`, `epoch`, `loss`, `batch_size`, `world_size`, `lr`. Only the requested model branches are used; no saved loss/optimizer values are reported or used. Complete tensor key/shape/dtype inventories for `target_encoder`, `predictor` and the online `encoder` are in `evidence/checkpoint_tensor_manifest.json`.

## Exact loading and synthetic API

| Component | Actual checkpoint key / result |
|---|---|
| Context encoder | `target_encoder` EMA; strict **292 keys**, 303,885,312 parameters |
| Target observation encoder | The same frozen EMA module, matching the official hub's inference convention |
| Predictor | `predictor`; strict **160 keys**, 22,086,016 parameters |
| Omitted / missing / unexpected keys | **Zero**; no absolute-position tensor needed omission |
| Encoder `patch_embed.proj.weight` | float32 `[1024,3,2,16,16]` |
| Encoder `norm.weight` | float32 `[1024]` |
| Predictor `predictor_embed.weight` | float32 `[384,1024]` |
| Predictor `predictor_proj.weight` | float32 `[1024,384]` |
| Predictor `mask_tokens.0` | float32 `[1,1,384]`; ten mask tokens |

Executed with CPU Torch2.7.1, backend autoload disabled, four CPU threads, no NPU. Constructor positions use the native 256px spatial grid and nominal64 frames; the bounded synthetic test uses four synthetic frames/two whole tubelets, supported by the encoder's dynamic frame-count API. It is an interface test, not a full64-frame runtime/performance claim.

```python
context = encoder(video, masks=[context_ids])
prediction = predictor(context, masks_x=[context_ids], masks_y=[target_ids], mask_index=0)
target_full = encoder(video)  # same fixed synthetic observable target
```

Synthetic input `[1,3,4,256,256]`, context indices `[1,32]`, target indices `[1,16]` yield context `[1,32,1024]`, prediction `[1,16,1024]`, observed target `[1,16,1024]`; all are finite and can form a same-shaped residual. A first-block hook records only `[1,32,1024]` tokens entering attention. Altering only the fully withheld tubelet leaves context and prediction **bitexact unchanged**, confirming that this API masks before contextual attention rather than deleting cached contextual features afterward.

Encoder state SHA before/after: `c22efaa60d915bcd3567eca1e3611ce8125b4fb3a24614a9bb155636867e7a1d`. Predictor state SHA before/after: `2e22d798996cf7c8bc52cb877ae30d443324f00f37aa9b77c74218f52356f9a4`. Checkpoint and frozen upstream hashes are unchanged. Synthetic execution completed at 06:48:37 UTC, 51.69 seconds; details are in `evidence/synthetic_interface.json`.

**Interpretation boundary:** the verified context branch uses EMA, as the official hub does. The online `encoder` was inventoried but not separately loaded/tested; online-context/EMA-target training pairing and target-normalization conventions must be specified separately for a future method. Same width, successful loading and finite residuals establish availability, not calibrated restoration reliability or useful donor credit.

## Backup and evidence

SFS: `/temp/zihaohe/seer_cvpr/method_dev_20261112/matched_predictor_assets_v1_final_v1`; 62 payload files / 5,128,128,567 bytes, all SHA-verified, including the one authenticated complete checkpoint. Incomplete transfer scratch was excluded. Backup manifest SHA `e5b0d78266a245e26a986e2a6f09d2f2f1e749aabe899889edc446f9a00eecc2`.

Local compact archive: `/Users/hezihao/Documents/New project/seer_matched_predictor_assets_20261112_v1/assets_compact_v1.tar.gz`, 77,938 bytes, SHA `50856d4c0e8c814bc2c0074da08d2855bee6a49a42f5c7bcaae13e44736a77cb`. Its 62 text files were safely extracted and verified (61 payload hashes plus the backup manifest). No binary was downloaded locally or placed in Git. Full source/text evidence remains beside that archive; this repository holds compact manifests, logs, receipts and the workflow scripts only.
