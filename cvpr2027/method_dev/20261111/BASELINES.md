# Executable baseline card — 2026-11-11

Purpose: select and reproduce strong RGB-only low-light **video** baselines. These are external authors' reported numbers, not reproduced Seer results or one common leaderboard. No dependencies installed, accelerator jobs run, or model weights downloaded for this card.

## Primary sources and target numbers

Numbers are PSNR / SSIM / LPIPS where available; retain each source's protocol.

| Baseline | Primary paper / official code | Reported reference | Execution status / comparison boundary |
|---|---|---|---|
| STCD / VLLVE | [IJCAI 2025 paper](https://www.ijcai.org/proceedings/2025/0238.pdf), [official repo](https://github.com/xiaogang00/LLVE_STCD) | SDSD indoor 28.93 / .88; outdoor 26.32 / .82 | Public implementation and indoor checkpoint. Five input frames. Strong direct video baseline; reproduce its split and metric rather than compare to Seer39 PSNR-Y. |
| VLLVE++ | [2026 extended paper, Table V and implementation](https://arxiv.org/html/2602.08699v1) | SDSD indoor 29.78 / .90; outdoor 27.47 / .85 | Paper explicitly says code/models will be released upon publication. The STCD repo is not verified to implement ++. Treat as a published numerical reference, not an executable released checkpoint. |
| AMNet RGB-only | [2026 paper](https://arxiv.org/abs/2606.11186), [official CUDA repo and weights](https://github.com/lhfgghc/LLVE-AMNet), [official NPU repo](https://github.com/lhfgghc/LLVE-AMNet-NPU) | SDSD indoor 29.03 / .92; outdoor 26.37 / .84 | Released weights and NPU code. Uses extra synthetic multimodal pretraining. Its released SDSD split differs substantially from STCD; numbers cannot establish which is better. NPU assessment below. |
| CRISP | [2026-10 paper, Table 2](https://arxiv.org/html/2610.24299v1) | BVI in-domain 32.78 / .923 / .098; same paper's re-evaluated STCD 33.43 / .919 / .104; BVI-Mamba 31.22 / .912 / .071 | Authors report Full-HD, separate in-domain training and common re-evaluation, except WaveLUT. No release repo verified from the official paper. Strong foundation-prior comparator: frozen DINOv3, AdaLN, recurrent warping, curriculum. |
| BVI-Mamba | [2026 preprint](https://arxiv.org/abs/2604.23655), [official repo](https://github.com/edhuang1/BVI-Mamba) | Use the CRISP BVI-only re-evaluation above when discussing in-domain numbers. | Its own paper also has **cross-test-domain averages**: do not relabel those as BVI-only. PCD/TSA alignment plus VSS reconstruction. Audit the original LPIPS evaluator before re-use. |

Additional design controls, not interchangeable video benchmark scores: [DWTA-Net](https://arxiv.org/abs/2510.09450) already uses flow-guided dynamic recurrent aggregation; [MirrorDistill](https://arxiv.org/abs/2609.25331) already uses clean-reference feature teachers with no teacher at inference; [ICML 2026 encoding-mismatch paper and official code](https://proceedings.mlr.press/v306/tian26k.html) already studies per-image low rank versus rotating dataset subspaces. Retain ordinary feature KD, SNR/flow gates, and static versus adaptive basis as controls if a new method uses these ingredients.

## Protocols to freeze before a benchmark run

Two result tracks must remain distinct:

1. **Method development / attribution:** current frozen Seer train136 / val16 / test39, center16 temporal frames and center256 spatial crop, rounded uint8 RGB PSNR **and** PSNR-Y, normalized RGB LPIPS, per-sequence output. Current 400-step pilots are validation-only and are not SOTA results. Use the same data plan, initial common weights, optimization budget, and seed list for matched comparisons.
2. **Public benchmark reproduction:** exact named external split, spatial resolution, boundary handling, context length, recurrence reset, quantization, metric implementation, and averaging. Retrain all compared methods on the same track. Record extra pretraining separately. Report indoor/outdoor separately on SDSD. Do not copy track 1 scores into track 2 tables.

### SDSD

- [STCD official indoor test YAML](https://raw.githubusercontent.com/xiaogang00/LLVE_STCD/main/options/test/SDSD_indoor_test.yml): `pair13,pair15,pair21,pair23,pair31,pair33,pair50,pair52,pair58,pair60,pair68,pair70`, five frames. [Test implementation](https://raw.githubusercontent.com/xiaogang00/LLVE_STCD/main/test.py) quantizes predictions/GT to uint8 and computes three-channel PSNR/SSIM, with both sequence-average log output and a separate frame-weighted printout. Specify which average is reproduced.
- [AMNet official indoor split](https://raw.githubusercontent.com/lhfgghc/LLVE-AMNet/main/configs/splits/sdsd_indoor_split.json): test is `pair11,pair1,pair21,pair19,pair4,pair9`. Only `pair21` overlaps STCD's test set. Obtain the outdoor list from its corresponding [official JSON](https://raw.githubusercontent.com/lhfgghc/LLVE-AMNet/main/configs/splits/sdsd_outdoor_split.json); do not infer it from indoor or strip `_2` suffixes in our archive.
- [AMNet RGB training config](https://raw.githubusercontent.com/lhfgghc/LLVE-AMNet/main/configs/experiments/amnet/sdsd/sdsd_in_rgb.yaml) uses 8-frame clips; `clip_stride=2` is the **window-start stride**, while frames within a clip are consecutive. Its loader resizes the short side to 256 then crops 128, including at test time. This is not native-resolution or Seer center256 evaluation.
- Pair frames by exact basename; assert shape, dtype/range, frame count, sequence count, and zero split overlap. Avoid the upstream loaders' silent `min(len(LQ),len(GT))` truncation. Capture the exact source release/layout: the current data are `indoor_static_np` and `outdoor_static_np`, not proof of identity to every paper's source images.

### BVI-RLV

- [Official author project](https://lrr-rachel.github.io/BVI-RLV/) and [dataset paper](https://arxiv.org/html/2407.03535v3): 40 scenes, 31,800 paired Full-HD frames, 25 fps, two illumination levels. `low_light_10` / `low_light_20` mean **10% / 20% normal light intensity**, not 10 / 20 fps. Legacy `fps` variable names in our code do not change the data grouping.
- Official benchmark uses 32 train / 8 test scenes; same-scene illumination variants stay together. Our 24 train / 8 validation / 8 test scene development track has less training data and is not automatically identical to the public training protocol. Use exact official lists, not a newly invented random or lexicographic 80/20 partition.
- The dataset paper's cross-dataset tables average four target test domains equally; columns can denote **training dataset**. Do not interpret those as BVI-only evaluations.
- [BVI-Mamba full-resolution evaluator](https://raw.githubusercontent.com/edhuang1/BVI-Mamba/main/test_fullresolution.py) uses five-frame windows and pads 1080 to 1088 before unpadding; its [loader](https://raw.githubusercontent.com/edhuang1/BVI-Mamba/main/datasets/data_loader.py) excludes the first/last two frames. The original evaluator passes OpenCV BGR values in `[0,1]` directly to default LPIPS. A common evaluator must explicitly use RGB and the chosen LPIPS normalization; label corrected re-evaluation rather than claiming bit-identical original LPIPS.
- Keep single-frame fidelity, perceptual quality, and temporal quality separate. Raw adjacent-frame tLPIPS can favor static/blurred predictions; include GT-relative temporal distance and motion/occlusion-aware diagnostics. Do not use GT brightness fitting or post-hoc GT exposure correction unless separately labeled oracle processing.

## AMNet-NPU source inspection and minimum integration

External read-only checkout: `/Users/hezihao/Documents/New project/seer_method_dev_20261111/reference_repos/LLVE-AMNet-NPU`

Pinned source: [`4c535f6462f5be9669268e35bb7fd2d6010be3ee`](https://github.com/lhfgghc/LLVE-AMNet-NPU/tree/4c535f6462f5be9669268e35bb7fd2d6010be3ee). Shallow clone, clean worktree, about 69 MB including example frames. AST parsing passed for all 23 Python files without importing them. **This is static interface validation, not an NPU forward/backward smoke test or weight validation.** Keep this reference checkout unchanged; put adaptations in a separate working copy/wrapper.

### Dependencies and device

Imports require `torch`, `torchvision`, `numpy`, `Pillow`, `PyYAML`, `tqdm`, `einops`, `pytorch-msssim`, and `timm` exposing `ModelEmaV3`. NPU execution requires the host's mutually compatible Ascend driver/firmware, CANN, PyTorch and `torch_npu`; upstream does not pin that matrix. Do not replace an existing working Ascend environment with the CUDA README's pip command. Training selects NPU automatically if available; inference also accepts `--device npu:0`. The inspected code has no custom CUDA extension, but `torch.fft.rfft2/irfft2`, complex tensors, non-reentrant checkpointing, EMA and optional AMP still need an actual host smoke test. Shipped RGB configs disable AMP; preserve that for first validation.

### Weights — official download locations only; not downloaded

Place weights in the adaptation's `checkpoints/` directory; retain filename, source URL, SHA256, source config, and load-key diagnostics. Links come from the [official parent README](https://github.com/lhfgghc/LLVE-AMNet#1-download-pretrained-weights).

| Artifact | Official URL | Expected local filename |
|---|---|---|
| Shared pretrained initialization | [Google Drive](https://drive.google.com/file/d/1Jyo9xI5ymPHFdB56Gq83hbCSjswfi3PK/view?usp=drive_link), [Baidu, code `vuux`](https://pan.baidu.com/s/1VzNG1yPM9A5INUnGpmfI1g) | `checkpoints/amnet_pretrained.pth` |
| Indoor RGB fine-tuned | [Baidu, code `3a4v`](https://pan.baidu.com/s/1ZsNOP4tskXxMKRFgru9vMQ) | `checkpoints/sdsd_in_rgb.pth` |
| Outdoor RGB fine-tuned | [Baidu, code `1ng4`](https://pan.baidu.com/s/1nhtWTKTvNeqcOC2h1Q41xg) | `checkpoints/sdsd_out_rgb.pth` |

No BVI-specific released weight was listed in that README. The shared pretrained weight uses extra data; label a fine-tuned-pretrained baseline accordingly. Training from scratch with unavailable initialization is a different experiment.

**Download lab note, 2026-11-11:** the official parent README still links the Google Drive ID above and names the expected file `amnet_pretrained.pth`. An anonymous metadata-only check from this workspace returned HTTP 401 for the Drive view page; HEAD requests to the official download endpoints redirected to Google sign-in HTML, without a checkpoint `Content-Disposition` or byte length. The official Baidu alternative could not be opened by the web reader. Thus anonymous downloadability and actual filename/size remain **unconfirmed**, not proven unavailable. No weights were downloaded, no account login or permission request was made, and no dependencies were installed. Recheck accessibility when staging the baseline and retain the eventual file SHA; do not infer file integrity from the README filename.

The [official paper, Appendix C](https://arxiv.org/html/2606.11186v1#A3) identifies extra pretraining on Vimeo90K, REDS, YouTube-VOS, Inter4K, UVO and MOSEv2 (1,693,034 frames), with synthetic low light, v2e events and ThermalGen infrared. Preserve this provenance for any RGB-only BVI adaptation. Also retain the recipe distinction: paper Table 6 gives batch32 / LR `2e-4`, whereas the pinned NPU SDSD configs give batch4 and indoor LR `1e-4` or outdoor LR `1e-6` with a frozen backbone. There is no official BVI recipe in the inspected checkout; record the selected source recipe and every adaptation rather than silently combining settings.

### Existing data to loader contract

Confirmed archive roots are under `/temp/zihaohe/seer_cvpr/datasets/`; resolve symlinks and verify availability on the chosen host before use. The frozen split is `/temp/zihaohe/seer_cvpr/datasets/splits_cvpr.json`.

| Interface | Minimum connection |
|---|---|
| SDSD training | `SDSDVideoDatasetRGB(root, clip_len, clip_stride, base_size, crop_size, split_json, split)` already reads `<root>/{input,GT}/<sequence>/*.npy` as uint8 HWC RGB. Set root to the exact existing `indoor_static_np` or `outdoor_static_np` root. For our split, export its exact train/val lists to upstream's JSON structure; its key named `test` must contain **development validation** during selection. |
| BVI training | No BVI dataset class/config exists upstream. Add a small adapter that reads paired PNGs `<scene>/low_light_{10,20}` and `<scene>/normal_light_{10,20}`, preserves scene-level splits, and emits the same six-item tuple as SDSD. No PNG-to-JPEG conversion or bulk duplicate frame export is needed. |
| Training tuple | `(x, y, event_zeros, ir_zeros, 0, 0)`: per item `x/y` float RGB `[T,3,H,W]` in `[0,1]`; zero placeholders `[T,10,H,W]` and `[T,1,H,W]`. Trainer converts modality flags to absence, so RGB-only needs no event/IR data. |
| Model training | `AMNet(**model_cfg)`; `loss, log = model(x, y=y, event=None, ir=None, loss_cfg=...)` for `[B,T,3,H,W]`. Keep implicit event/IR translator branches enabled for the published RGB-only architecture; RGB-only does not mean deleting these modules. |
| Model evaluation | `model.inference_video(x, event_use_real=False, ir_use_real=False)` returns `[T,3,H,W]` for unbatched RGB input. Reset recurrence at the agreed sequence/clip boundary, preserve every target frame, pad/unpad spatially as needed for the /8 encoder. Choose and record raw versus EMA weights. |

Original entry point, **after creating an adapted config and fixing selection/evaluation below**, is `python scripts/train.py --cfg <adapted-config.yaml>`. No runnable benchmark command is claimed before these changes and the NPU smoke test pass. The demo `scripts/inference_full.py` only accepts per-video `*.jpg`, has no GT or metrics, and is not the SDSD/BVI benchmark evaluator.

### Required changes before a valid run

1. **Split isolation:** original `scripts/train.py` loads `split="test"` every epoch, then the trainer ranks checkpoints by that result. Route our validation list here; reserve test for a single frozen final evaluation. Freeze the public reproduction selection rule separately.
2. **Full validation coverage:** original test DataLoader has `batch_size=4`, but `Trainer.evaluate` uses only `x[0]` / `y[0]`. Set evaluation batch size to 1 or iterate every batch member. Also replace overlapping-window duplicate frame counting with the explicit target-frame policy and record counts.
3. **Resolution/metric adapter:** the upstream dataset applies short-side resize256 and center128 crop to test too. Use a dedicated common evaluator for center256 development and native-resolution benchmark tracks. Emit per-sequence names, RGB PSNR, PSNR-Y where relevant, correctly normalized RGB LPIPS, and the exact temporal definitions. Upstream evaluation logs only floating RGB PSNR / SSIM.
4. **Weight integrity:** `_load_pretrain` warns then continues when a path is missing; `strict=False` only prints counts of missing/unexpected keys. Fail closed on absent required initialization or unintended key mismatches. The demo loads `checkpoint['model']`, not `model_ema`; reconcile this with training's EMA evaluation and save the chosen key in result metadata.
5. **Training recipe:** indoor config trains the backbone at `1e-4`; outdoor config freezes it and uses `1e-6`. Preserve these as explicit published-recipe variants. Do not accidentally freeze random weights when training from scratch; that would not be a meaningful baseline. Use fresh output directories to prevent the automatic `latest.pth` resume from silently changing initialization.
6. **Small preflight, before sustained training:** assert paired filenames and splits; verify RGB/range/shape and 1-batch forward/backward on the intended NPU; ensure gradients reach intended trainable branches and required pretrained keys loaded; compare complete evaluator counts with dataset manifests. Then run the agreed budget and seed list. No preflight has been executed by this card's author.

Source locations for the above audit: [training entry](https://github.com/lhfgghc/LLVE-AMNet-NPU/blob/4c535f6462f5be9669268e35bb7fd2d6010be3ee/scripts/train.py), [SDSD dataset](https://github.com/lhfgghc/LLVE-AMNet-NPU/blob/4c535f6462f5be9669268e35bb7fd2d6010be3ee/datasets/sdsd_video_dataset_rgb.py), [trainer](https://github.com/lhfgghc/LLVE-AMNet-NPU/blob/4c535f6462f5be9669268e35bb7fd2d6010be3ee/engine/trainer.py), [model API](https://github.com/lhfgghc/LLVE-AMNet-NPU/blob/4c535f6462f5be9669268e35bb7fd2d6010be3ee/models/amnet.py), [FFT translator](https://github.com/lhfgghc/LLVE-AMNet-NPU/blob/4c535f6462f5be9669268e35bb7fd2d6010be3ee/models/adapter.py), [demo evaluator](https://github.com/lhfgghc/LLVE-AMNet-NPU/blob/4c535f6462f5be9669268e35bb7fd2d6010be3ee/scripts/inference_full.py).

## Lab note: correction-risk fusion has direct prior art

This note records a novelty boundary, not a paper claim or a change to the frozen three-arm selector experiment.

- **Closest: CsNet, Optimal Combination of Image Denoisers (TIP 2019).** [Original paper, §§II-A, III-B](https://arxiv.org/html/1711.06712), [official implementation](https://github.com/jhchoi0908/consensusNet). It supervises a network with GT patch MSE (L1 regression), averages nonoverlapping 64×64 patch predictions into each candidate's image MSE, reconstructs error second moments from those MSEs and observable pairwise image differences, then optimizes nonnegative sum-to-one fusion. The released [network.py, `network1.run`](https://raw.githubusercontent.com/jhchoi0908/consensusNet/master/network.py) averages MSE predictions; [utils.py, `combine`](https://raw.githubusercontent.com/jhchoi0908/consensusNet/master/utils.py) returns **one global weight vector per image**. Patch-level prediction is already present; spatially varying fusion is not the reported final weighting. A local version remains a straightforward extension, not an established novelty gap.
- **KPN, Burst Denoising with Kernel Prediction Networks (CVPR 2018).** [Original paper, §4](https://arxiv.org/html/1712.02327). Predicts per-pixel spatial/temporal kernels for joint burst alignment and averaging, with signal-dependent noise input and per-frame auxiliary reconstruction losses annealed away. Kernels can be negative: it is not a positive-softmax-only fusion baseline. Our fixed restored-RGB donors and learned two-moment head differ from its trainable filters, but learning local temporal weights or adding a noise map is already covered.
- **Oracle parameter supervision is also established.** [HyperDn, 2026, §§2.1–2.3](https://arxiv.org/html/2605.20479v2) predicts model-based denoiser hyperparameters using GT-searched oracle labels, studying transfer across configurations. It is not the same mixing algorithm, but replacing costly GT-selected parameters with an input-only predictor is not itself a new principle. Our current experiment regresses moments, not oracle weights; direct oracle-weight/output supervision is an appropriate future control, not a description of the current loss.

**Exact algebraic relation to CsNet (our derivation).** Let the three candidate images be `z0=b`, `z1=b+d0`, `z2=b+d1`. With the same patch-mean reduction, define `mi=mean((GT−zi)^2)`, `e=GT−b`, `ai=mean(e*di)` and `Gij=mean(di*dj)`. Then

```text
ai = (m0 + Gii − mi) / 2
Σij = (mi + mj − mean((zi−zj)^2)) / 2
γ = (1−w0−w1, w0, w1)
γᵀΣγ = m0 + wᵀGw − 2aᵀw,   γ>=0, sum(γ)=1.
```

Thus our simplex risk is **exactly the same MSE fusion objective in residual coordinates**, not a new risk formulation. This identity applies to true/common moments or the reconstructed covariance before CsNet's optional PSD repair; different learned statistics and that repair need not produce identical practical weights. Predicting two relative moments instead of three absolute MSEs removes an irrelevant common offset. Observable `G` gives a PSD quadratic term without estimating/projecting a full error matrix; this and RMS target scaling are numerical/learning choices. They do not by themselves establish a method contribution. Current raw-MSE and fit-scalar-normalized-MSE controls test learnability and optimization conditioning on identical candidates; a win would not prove a new fusion principle.

**Follow-up experiments (not yet run):**

1. Test **conditional message utility under nonlinear fusion**: for fixed background messages `S`, measure `U(j|S)=L(R(S),GT)−L(R(S∪{j}),GT)` on training data. Ask whether an input-only, context-conditioned selector predicts utility changes when another donor is present and retains actual gains on held-out scenes. Compare independent quality gates, matched direct-MSE training, the full-Gram fixed-candidate control, and oracle-action supervision. Linear-candidate complementarity alone is already captured by the quadratic program. Because the frozen base sees the clip, these are added-message interventions, not removal of all information from a frame.
2. Test **candidate information complementarity**, before adding network components: under fixed compute, compare the present correlated all-clip restored donors with proposals that have explicitly restricted donor evidence, and measure source error correlation, conditional oracle headroom and learned recovered gain by motion/occlusion/noise strata. Any source-isolation claim requires a dataflow audit; different frame indices alone do not ensure independent evidence. A useful result must improve the learned outcome and cost frontier against the same direct and risk selectors, not merely enlarge a GT oracle bound. This is an experimental direction; its novelty remains to be checked once the concrete mechanism exists.
