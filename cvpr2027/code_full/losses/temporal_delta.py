"""A loss on how the prediction's latents evolve in time.

`tools/token_probe.py` showed the encoder does represent temporal structure on
this data -- reordering the input frames changes the block-23 tokens by 58% --
while the converged model's output moves by 0.05 dB under the same reordering.
The information is present and unused, and the reason is visible in the
objective: Charbonnier, Retinex and TV are all per-frame, so nothing rewards
getting the temporal evolution right.

This adds the missing term. Tokens are taken for the prediction and for the
ground truth, differenced along the tubelet axis, and the two difference fields
are matched. Differencing is what makes the term specific: the static semantic
component cancels, so the loss cannot be satisfied by improving per-frame
appearance, which the pixel terms already drive.

How this differs from the two earlier attempts, both of which failed for
identifiable reasons:

  M1/M2  compared one latent to another latent, so no gradient reached the
         pixels, and the pretrained predictor head had been replaced by a random
         projection, discarding the predictions it was meant to use
  LPM    matched a single scalar "unpredictability" value and failed a
         specificity probe -- spatial blur moved it 1.8x more than flicker

Here the gradient path runs prediction -> frozen encoder -> loss, and the
quantity matched is a field of temporal differences rather than a scalar, so a
spatially-uniform change cannot satisfy it.

The decisive check is not the metric but the mechanism: if the term works, the
trained model's output should become sensitive to the order of its input
frames, which `tools/temporal_ablate.py` measures directly and which currently
reads 0.05 dB.
"""
from __future__ import annotations

from typing import Tuple

import torch


def tubelet_deltas(tokens: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
    """(B, N, D) -> (B, T_v - 1, H_v*W_v, D) first differences along time."""
    b, n, d = tokens.shape
    t_v, h_v, w_v = grid
    if n != t_v * h_v * w_v:
        raise ValueError(f"N={n} does not match grid {grid}")
    x = tokens.reshape(b, t_v, h_v * w_v, d)
    return x[:, 1:] - x[:, :-1]


def temporal_delta_loss(tok_pred: torch.Tensor, tok_gt: torch.Tensor,
                        grid: Tuple[int, int, int], eps: float = 1e-6
                        ) -> torch.Tensor:
    """Relative L1 between the prediction's and the target's latent dynamics.

    Normalising by the target's own delta magnitude keeps the term comparable
    across clips whose motion differs by an order of magnitude, so a nearly
    static scene cannot dominate the batch simply by having small deltas.
    """
    dp = tubelet_deltas(tok_pred, grid)
    dg = tubelet_deltas(tok_gt, grid)
    num = (dp - dg).abs().mean()
    den = dg.abs().mean().clamp_min(eps)
    return num / den


def temporal_delta_cosine_loss(tok_pred: torch.Tensor, tok_gt: torch.Tensor,
                               grid: Tuple[int, int, int], eps: float = 1e-6
                               ) -> torch.Tensor:
    """Direction-first version of the term above (E24, Amendment 14).

    `temporal_delta_loss` normalises by the *target*'s delta magnitude, which
    leaves a trivial optimum: a prediction with no temporal variation at all
    scores exactly 1.0, and reaching it only requires smoothing the output,
    whereas going below 1.0 requires reconstructing the motion. E22 took the
    cheap route -- `tools/td_probe.py` on its step 5000 reads magnitude 0.202
    against the converged baseline's 1.063, with alignment *falling* from
    +0.034 to +0.014.

    Here the direction term is scale-invariant in `dp`, so shrinking the
    prediction's dynamics buys nothing, and the magnitude is restored by a
    separate penalty on the log ratio -- symmetric, because overshooting the
    target's motion is as wrong as undershooting it and a one-sided penalty
    would not say so.

    The cosine is taken per token over the channel axis, the same quantity
    `td_probe.py` reports, so the training signal and the diagnostic that
    judges it cannot drift apart.
    """
    dp = tubelet_deltas(tok_pred, grid)
    dg = tubelet_deltas(tok_gt, grid)
    cos = torch.nn.functional.cosine_similarity(
        dp.flatten(0, 2), dg.flatten(0, 2), dim=-1, eps=eps).mean()
    ratio = dp.abs().mean().clamp_min(eps) / dg.abs().mean().clamp_min(eps)
    return (1.0 - cos) + ratio.log().abs()
