# Source-only observation controls and trajectory reconstruction, v2

Frozen before this round's source training and BVI evaluation. This is an
experimental protocol, not a paper draft. The objective is a useful method with
positive, reproducible quality and temporal results. Neither novelty nor SOTA is
established by this pilot.

## Motivation and hypotheses

The preceding SDSD-to-BVI experiment improved PSNR and LPIPS using raw-detail
correction. Learned weights did not beat equal thirds in PSNR-Y, and temporal
gap increased. The next experiment asks whether information from correctly
aligned neighboring observations matters, and whether jointly reconstructed
adjacent states can improve the quality/temporal tradeoff beyond a matched
node-only predictor. Graph regularization, frequency separation, gain transfer,
and convex fusion are established ideas; using them alone is not a novelty claim.

## Fixed source and target scope

- Two historical E17 small final20k base checkpoints: pretrained seed0 and none
  seed0. These are one base-training seed per arm; the original pixel-model
  initializations were not exactly matched between arms. Selector seeds below
  do not constitute independent base-training replication.
- Source: the existing 136 SDSD indoor/outdoor training sequences, 68 scenes,
  one fixed 16-frame/crop256 cache per sequence, data seed20261113. Keep the
  previous scene split (51 fit scenes, 17 internal scenes; seed20261114).
  Internal scenes were seen during historical base training. Only fit scenes
  train the new heads or determine feature/loss normalization.
- Target: all 16 BVI VAL sequences, eight scenes with two illumination
  conditions, fixed center16/crop256. Target GT supplies scores only, after all
  predictions. No target-dependent training, tuning, stopping or model selection
  within this round. BVI VAL informed earlier development, so the entire
  project's history is not target blind. TEST remains unused.
- Existing source/target pairing, uint8 rounding, PSNR-RGB, PSNR-Y, LPIPS and
  per-sequence absolute tLP gap follow the preceding common evaluation library.

## Observable candidates

Let L be a reflect-padded 9x9 box filter, b the frozen base and x the low-light
input. Define q=clamp(L(b)+clamp(L(b)/(L(x)+1e-4),0.25,16)*(x-L(x)),0,1).
The raw candidate uses high-frequency content from q in the two +/-2 frame
donors warped to the target. Preserve the existing mask and LP support erosion.

Three candidate families use the same original direction-specific validity:

1. raw_detail: correctly aligned donor values, byte-compatible with v1.
2. same_frame_detail: use the current frame's q values. This removes explicit
   donor-value access; the video base itself can still contain neighbor information.
3. mismatched_detail: swap donor-frame values while keeping each original
   direction's flow/mask; clamp requested source indices at clip endpoints.
   Record requested/actual indices and fallback, without cyclic wrapping or
   relaxing temporal validity.

GT never enters candidates, flow, graph, feature construction or model inference.

## Reconstruction mechanism and seven variants

A 19->32->32->2 node CNN supplies equal-thirds-initialized convex corrections
from base and the two candidates (6,626 parameters). True +/-1 RAFT flows
provide adjacent constraints; the prior +/-2 graph alone has disconnected odd
and even chains. Each bilinear corner defines a separate undirected pair with
its nonnegative interpolation weight. Reciprocal directions remain parallel
undirected edges. The learned edge precision uses own-observation discrepancy
and the other donor's observable discrepancy/correlation (193 parameters).

With a=node_prediction-L(b), minimize the formal energy
E(z)=0.5*sum_i||z_i-a_i||^2+0.5*sum_edges w_uv||z_u-z_v||^2.
Use exactly two Jacobi steps initialized at a and clamp L(b)+z. Both endpoints
receive every edge's contribution; I+L_graph is symmetric positive definite.
This pairwise energy is not ||z_t-W z_s||^2, and two iterations are not a
converged optimizer. All graph/candidate arithmetic uses CPU float32.

| Variant | Candidate | Reconstruction |
|---|---|---|
| raw_node | raw_detail | node only |
| same_frame_node | same_frame_detail | node only |
| mismatched_node | mismatched_detail | node only |
| trajectory | raw_detail | conditioned edge precision |
| trajectory_fixed_edge | raw_detail | precision fixed at 0.5 |
| trajectory_no_cross | raw_detail | other-donor features zeroed |
| trajectory_scrambled_cross | raw_detail | other-donor features shifted by T/2 |

All variants allocate 6,819 parameters and share per-seed initialization and
sampling. Node-only and fixed-edge variants deactivate the edge head, so active
capacity and FLOPs differ. Conditioned and scrambled-cross retain equal active
capacity. Graph modes initially use precision0.5; node-only has a different
initial function by design. Coupling1, patch8, LP9, two iterations, hidden32
and all other mechanism settings are fixed for this round.

## Training and full evaluation coverage

- Seeds0/1/2, seven variants per base arm: 42 new final heads in total.
- Each head: Adam lr1e-3, default betas/eps, no weight decay, 1,000 updates.
  CPU4 threads, one fit-source clip per update sampled with replacement; an
  aligned random128 crop from the cached256, all16 frames as context, eight
  unique sampled frames in the source RGB reconstruction loss. Tighten adjacent
  masks for cropped-out donor coordinates. Divide MSE by fit-only base MSE.
- Per-seed clip/frame/crop plans are identical across variants and base arms.
  No intermediate holdout evaluation, target selection, or early stopping.
  Final1000 is the only checkpoint evaluated as a new result.
- Target per arm: base +21 new heads +three frozen v1 raw/direct heads +six
  fixed controls (equal thirds and donor average for each candidate family)
  =31 methods, all evaluated. Old v1 heads used eight independent crop256
  frames/update; they are frozen references with a different training budget.
- Every target method shares one base prediction and the same RAFT +/-2 and
  extra +/-1 flows per clip. Candidate construction and all heads use CPU
  float32. Extra graph/flow work is additional cost; no equal-latency claim.
- Immutable source manifests, input/checkpoint SHA256, actual sample plans and
  initial states, complete training curves, and unchanged frozen base/RAFT/head
  state checks accompany results. Original v1 sources and snapshots stay intact.

## Analysis fixed before reading new target results

Report every seed and variant, all four metrics, and both base arms. Average
the three selector seeds for each sequence, then average the two illumination
conditions within each scene. Use paired scene bootstrap95% intervals over all
eight scenes with the preceding fixed bootstrap settings. Seed variation is
reported separately, not treated as extra independent target scenes.

Primary comparisons are raw_node versus same_frame_node and mismatched_node,
and trajectory versus raw_node. Also report raw_node versus base, equal thirds
and v1 raw/direct; trajectory versus fixed_edge, no_cross, scrambled_cross,
base, equal thirds and v1 raw/direct. For PSNR, larger is better; for LPIPS and
tLP gap, smaller is better. Report competing metric directions explicitly.
Show paired differences in correction gains between pretrained and none, with
the historical initialization/training confound stated. No favorable subset
replaces the full table. A successful pilot still requires broader datasets,
matched strong baselines, and replication before any cross-domain SOTA claim.

## Execution boundaries

Use process-free NPU cards only after a fresh process inventory. Keep tensor
outputs under host-local /cache, because shared /temp SFS lacks atomic rename.
Transport base-only source exports and adjacent flow outputs through exclusive
shared scratch, verify hashes, then remove only our verified transport copies.
Do not alter other users' jobs, system environments, frozen v1 files or runtime
snapshots. Runtime dependency fixes use isolated versioned copies with evidence.
