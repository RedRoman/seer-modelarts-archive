# Historical Seer checkpoint color contract

Read-only audit, 2026-11-12. No model, loader, checkpoint, cached tensor or frozen evaluator was modified. No target or internal-holdout image was read or scored. This note distinguishes verified code behavior from the separate audit of the original SDSD NPY channel convention.

## Verified historical code path

The following files were inspected directly on host 31051 under `/temp/zihaohe/seer_cvpr/code/vjepa_llve`. These are historical runtime files; the current repository's reconstruction fixes do not add color conversion.

| File | SHA256 | Relevant behavior |
|---|---|---|
| `data/common.py` | `6041b43e832ab67def75b9e7285927dddd77d45d0bc25945bb600167b04e4ded` | `read_image_01`, lines 19–47: NPY layout/range conversion only; no channel permutation. PIL image formats are converted to RGB. |
| `tools/seer_train_cvpr.py` | `633ec8b1c772980b47c69502a9431935dc7514c5f5715dbb1c93354d7ec8a61d` | `ListedSDSD` loads both input and GT through that reader; lines 371 and 401 feed the same `lq` to `backbone(lq)` and `pb(lq, zp)`. Pixel losses use the correspondingly loaded GT. |
| `tools/seer_train_e4.py` | `8e535ca835b8bcd5d00433145cc51dabe2e01bf1c7a401cff056ab403a56e994` | Imports `ListedSDSD`; BVI/LOL use the PIL reader. Lines 299 and 302 likewise send the same `lq` to both branches. |
| `models/vjepa_backbone.py` | `8d8e4ca44cecad797aec05af5f627b50927b9d72860187b76d466794f2d3e4bf` | Resize/resample, then `(x-mean)/std` using `(0.485,0.456,0.406)` and `(0.229,0.224,0.225)` in existing channel order. No BGR-to-RGB conversion. |
| `tools/eval_e4.py` | `cf5cbdad66779121a4cd49a99da8729f3922c8f95bc7414e9ca5b2be46c8b3b2` | `run_clip` only converts THWC uint8 to BCTHW float; lines 260 and 263 feed identical `x` to encoder and pixel branch. Output conversion also preserves channel order. |

The actual shared metric/data reader `/temp/zihaohe/seer_cvpr/code/baselines/common/eval_lib.py`, SHA256 `a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d`, documents RGB but returns NPY arrays without swapping channels (lines 82–88). Non-NPY images use `imageio.imread` (line 89), hence ordinary BVI PNGs use RGB. The repository preserves this file at `method_dev/20261111/source_transfer_v1/references/runtime_sources/common_eval_lib.py`.

Current repository references: `code_full/data/common.py:19`, `code_full/tools/seer_train_cvpr.py:58`, `code_full/tools/seer_train_cvpr.py:346`, `code_full/tools/eval_e4.py:263`, and `code_full/models/vjepa_backbone.py:37`. No hidden swap was found between loader, `clip_to_tensor`, encoder, pixel branch and output conversion.

**What this establishes:** training and evaluation preserve the NPY's stored channel order, whereas the BVI PNG reader yields RGB. **What code alone does not establish:** that the original SDSD NPY bytes are BGR. That conclusion requires the official dataset preprocessing/loader contract and the separate FIT-only raw/cache parity audit; a filename or a misleading RGB docstring is not evidence. Once that audit confirms BGR, the consequences below follow without additional model evaluation.

## Native adapter for an existing SDSD-only checkpoint

Let `S` exchange channels 0 and 2, let `r` be canonical RGB input, let `E` include the historical encoder preprocessing, and let `P` be the trained pixel branch. For an SDSD-only checkpoint trained on BGR NPY arrays, native deployment in an RGB pipeline is:

```text
native = S(r)
tokens = E(native)                     # retain the historical encoder input convention
output_rgb = S(P(native, tokens))      # convert the trained output convention back to RGB
```

For the none arm, use its saved zero-token construction instead of `E`. An exposure/refiner branch, if present in another checkpoint, must also retain its trained native-input convention. For canonicalized SDSD input `r=S(npy)`, `native=npy` exactly, so this adapter preserves the historical model computation while exposing an RGB interface. For BVI input, it restores the checkpoint's native channel convention before inference. This is a fixed dataset/model contract, not a target-score-selected choice.

Keep three cases distinct:

| Case | Pixel input | Encoder input | Output interpretation |
|---|---|---|---|
| Historical BVI replay | RGB | RGB | Existing evaluator's unchanged array convention; retain as historical evidence. |
| Native SDSD-only checkpoint adapter | BGR | BGR | BGR prediction swapped to RGB before metrics or further RGB processing. |
| Encoder-color intervention | BGR | RGB | BGR prediction swapped to RGB; this changes conditioning relative to training and is a separate controlled experiment. |

Switching only the encoder to RGB may improve compatibility with its pretraining, but the pixel branch was trained against `E(BGR)`. It therefore cannot silently replace the native adapter. RGB-native training of a new pixel branch would be a separate training condition. Do not infer that every checkpoint is BGR-native: BVI-only checkpoints are RGB-native, and mixed-domain checkpoints trained through these loaders saw domain-dependent conventions.

## Consequences for evidence and cache reuse

- An unadapted SDSD-to-BVI result mixes scene/domain transfer with a channel-convention change if SDSD NPY is BGR. Neither the magnitude nor direction of a correction can be inferred from code; preserve old numbers and measure an explicitly labeled adapter condition before revising claims.
- Jointly swapping prediction and GT leaves RGB MSE/PSNR invariant. RGB-weighted luma PSNR, LPIPS, RGB-pretrained RAFT and their temporal metrics are generally not invariant. Historical SDSD Y/LPIPS/temporal numbers cannot be relabeled canonical RGB metrics by a scalar correction. Training-time `rgb_to_y` also used the stored channel order.
- Historical pretrained-versus-none source comparisons used the same loader, but the pretrained encoder received the stored channel order through RGB ImageNet normalization. This limits claims about an RGB-native foundation prior; it does not by itself erase the numerical within-protocol comparisons.
- Cached source block17/prior addresses were produced from the historical stored-order LQ and are not interchangeable with newly canonical RGB predictor features. Existing Seer correction caches and selectors also have this lineage; new RGB consumers require explicit conversion/reconstruction and provenance.
- HVI is a separately pretrained RGB consumer. If its source export read stored BGR LQ without conversion, simply swapping its saved output does not reproduce `HVI(S(lq))`: generally `S(HVI(lq)) != HVI(S(lq))`. A corrected HVI source cache must be recomputed from canonical RGB input. Do not overwrite the old cache or interpret its results as a correctly adapted HVI reference.
- For downstream fusion in a canonical RGB pipeline, convert the native Seer output back to RGB before new RGB flow/candidate/metric stages, and ensure LQ/GT use that same canonical contract. A legacy downstream selector trained on stored-order features needs its own native-contract adapter or retraining; fixing only the base is insufficient.

## Predictor asset finding retained from the interrupted interface audit

The existing `vjepa2_1_vitl_dist_vitG_384.pt` is 5,151,198,524 bytes and contains `encoder`, `predictor` and `ema_encoder`. The predictor has 162 tensor keys / 22,973,056 parameters: 12 blocks, width 384, eight mask tokens and an output width of 1664. Both packaged encoder branches have width 1024. No separate 1664-dimensional teacher checkpoint was present in the inspected weights directory.

The official [V-JEPA 2.1 loader](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py) selects `ema_encoder`, enables modality embeddings with `img_temporal_dim_size=1`, and loads the predictor with `return_all_tokens=True`, `n_output_distillation=1`, and `teacher_embed_dim=1664`. A CPU synthetic interface test strictly loaded all predictor keys, produced finite target/context predictions of shapes `[1,16,1664]` / `[1,32,1664]`, and used no dataset images. This verifies interface availability, not restoration utility.

The attempted real FIT-LQ smoke stopped with `KeyError: 'lq'` before encoder forward because the cache stores tensors under a nested dictionary. It was not rerun after color-contract triage took priority; no GT tensor values, target frames or holdout scores were read in that attempt. No NPU was allocated.

Prediction-change features remain implementable without a new world model, but a 1664-dimensional predictor output cannot be subtracted directly from the available 1024-dimensional encoder target. Removing a donor only from already computed full-context block17 tokens is also not a valid leave-donor-out intervention: attention has already propagated donor information. A future experiment must mask before encoder attention and explicitly distinguish prediction sensitivity from calibrated restoration reliability. The legacy `predictor_lora.py` bypasses the masked API and replaces the output head; it is not an implementation of this proposed frozen masked-predictor measurement.
