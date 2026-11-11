

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
