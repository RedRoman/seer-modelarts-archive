"""Two-source correction-risk minimization; no model or training integration.

For a patch error e = GT - base and correction directions d_0, d_1,
    a_i = mean(e * d_i), G_ij = mean(d_i * d_j).
The change in squared reconstruction error is
    q(w) = w.T @ G @ w - 2 * a.T @ w.
We minimize q on w >= 0, sum(w) <= 1. The zero correction is always feasible.
This is classical quadratic minimization, not a new method claim.

The solver accepts float32/float64 PyTorch tensors, including NPU float32.
It uses elementwise operations only: no inverse, solve, or decomposition.
Torch is imported lazily so the arithmetic kernel can also be checked against
an independent NumPy dense grid without installing torch on a development host.
"""
from __future__ import annotations


def _solve_components(a0, a1, g00, g01, g11, *, where):
    """Normalized scalar-array kernel; ``where`` is torch.where or np.where.

    All inputs have the same batch shape and have already been jointly scaled.
    The small private kernel is intentionally backend-neutral for numerical QA.
    It returns (w0, w1, q) and preserves the first solution on exact ties.
    """
    zero, one = a0 * 0, a0 * 0 + 1

    def edge_parameter(b, curvature):
        # Minimize curvature*t^2 - 2*b*t on [0,1]. Divide ONLY after
        # 0 < b < curvature, so the quotient is bounded and cannot overflow.
        interior = (curvature > 0) & (b > 0) & (b < curvature)
        numerator = where(interior, b, zero)
        denominator = where(interior, curvature, one)
        return where(b >= curvature, one,
                     where(interior, numerator / denominator, zero))

    # The three edges include their endpoints. Explicit vertices also handle
    # tiny negative curvatures from roundoff in otherwise PSD input matrices.
    t0 = edge_parameter(a0, g00)
    t1 = edge_parameter(a1, g11)
    t2 = edge_parameter(a0 - a1 + g11 - g01, g00 + g11 - 2 * g01)
    candidates = [(one, zero), (zero, one), (t0, zero),
                  (zero, t1), (t2, one - t2)]

    determinant = g00 * g11 - g01 * g01
    numerator0 = a0 * g11 - a1 * g01
    numerator1 = a1 * g00 - a0 * g01
    inside = ((g00 > 0) & (g11 > 0) & (determinant > 0)
              & (numerator0 >= 0) & (numerator1 >= 0)
              & (numerator0 + numerator1 <= determinant))
    # Feasibility is tested in multiplied form BEFORE dividing by a potentially
    # small determinant. A valid numerator is in [0, determinant]. Singular
    # optima have an equally good boundary solution, already enumerated above.
    denominator = where(inside, determinant, one)
    wi0 = where(inside, numerator0, zero) / denominator
    wi1 = where(inside, numerator1, zero) / denominator
    wi1 = where(wi1 > one - wi0, one - wi0, wi1)  # roundoff at sum(w)=1
    candidates.append((wi0, wi1))

    best0, best1, best_risk = zero, zero, zero
    for w0, w1 in candidates:
        risk = (g00 * w0 * w0 + 2 * g01 * w0 * w1 + g11 * w1 * w1
                - 2 * (a0 * w0 + a1 * w1))
        better = risk < best_risk
        best0 = where(better, w0, best0)
        best1 = where(better, w1, best1)
        best_risk = where(better, risk, best_risk)
    return best0, best1, best_risk


def _check_tensors(first, second, first_name, second_name):
    import torch
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"{first_name} and {second_name} must be torch tensors")
    if first.dtype not in (torch.float32, torch.float64) or second.dtype != first.dtype:
        raise TypeError(f"{first_name} and {second_name} must share float32 or float64 dtype")
    if first.device != second.device:
        raise ValueError(f"{first_name} and {second_name} must be on the same device")
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise ValueError(f"{first_name} and {second_name} must be finite")
    return torch


def solve_simplex2(a, gram):
    """Return optimal ``(weights, risk_change)`` for each independent patch.

    a: (..., 2); gram: (..., 2, 2), symmetric positive semidefinite.
    Output shapes: (..., 2), (...,). No batch broadcasting is performed.
    Risk is the squared-error CHANGE, not absolute MSE; negative is favorable.
    Its reduction convention is whatever was used when constructing a/gram.

    Shape, device, dtype and finite checks are strict. Gram symmetry/PSD checks
    allow 32 machine eps of relative roundoff and symmetrize its off-diagonal.
    Exact tied minimizers may have different weights; the returned risk is the
    invariant quantity. Piecewise autograd is retained away from active-set ties.
    Input validation synchronizes the device; this is an experimental oracle,
    not an unchecked fused inference operator.
    """
    torch = _check_tensors(a, gram, "a", "gram")
    if a.ndim < 1 or a.shape[-1] != 2 or gram.shape != a.shape[:-1] + (2, 2):
        raise ValueError("Expected a (...,2) and gram (...,2,2) with identical batch shapes")
    tiny = torch.finfo(a.dtype).tiny
    tolerance = 32 * torch.finfo(a.dtype).eps
    # Validate relative to Gram's own scale, not a's possibly much larger one.
    gram_scale = gram.abs().amax(dim=(-2, -1)).clamp_min(tiny)
    normalized_gram = gram / gram_scale[..., None, None]
    g00, g11 = normalized_gram[..., 0, 0], normalized_gram[..., 1, 1]
    forward, backward = normalized_gram[..., 0, 1], normalized_gram[..., 1, 0]
    if bool(((forward - backward).abs() > tolerance).any()):
        raise ValueError("gram must be symmetric")
    off_diagonal = forward * .5 + backward * .5
    determinant = g00 * g11 - off_diagonal.square()
    if bool(((g00 < -tolerance) | (g11 < -tolerance) | (determinant < -tolerance)).any()):
        raise ValueError("gram must be positive semidefinite")

    scale = torch.maximum(a.abs().amax(dim=-1), gram_scale)
    aa, gg = a / scale[..., None], gram / scale[..., None, None]
    w0, w1, risk = _solve_components(
        aa[..., 0], aa[..., 1], gg[..., 0, 0],
        gg[..., 0, 1] * .5 + gg[..., 1, 0] * .5, gg[..., 1, 1], where=torch.where)
    weights = torch.stack((w0, w1), dim=-1)
    risk = risk * scale
    if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(risk).all()):
        raise OverflowError("Optimal risk is not representable in the input dtype; rescale inputs")
    return weights, risk


def train_only_patch_statistics(gt_minus_base, residuals):
    """TRAIN-ONLY GT helper: detached mean-error statistics for flattened patches.

    gt_minus_base: (..., D); residuals: (..., 2, D), D > 0.
    Flatten patch pixels AND color channels into D before calling this function.
    Returns a (...,2), gram (...,2,2), using MEANS over D consistently.
    These GT-derived quantities must never be fed to a test-time selector.
    The production solver accepts *predicted* a and observable residual Gram.
    """
    torch = _check_tensors(gt_minus_base, residuals, "gt_minus_base", "residuals")
    if (gt_minus_base.ndim < 1 or gt_minus_base.shape[-1] < 1
            or residuals.shape != gt_minus_base.shape[:-1] + (2, gt_minus_base.shape[-1])):
        raise ValueError("Expected gt_minus_base (...,D), residuals (...,2,D), D > 0")
    with torch.no_grad():
        error, directions = gt_minus_base.detach(), residuals.detach()
        a = (directions * error.unsqueeze(-2)).mean(dim=-1)
        gram = (directions.unsqueeze(-2) * directions.unsqueeze(-3)).mean(dim=-1)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(gram).all()):
        raise OverflowError("Patch statistics overflowed; rescale errors and residuals")
    return a, gram


def solve_two_candidate_risk(a, gram):
    """Named oracle entry point; same contract as :func:`solve_simplex2`."""
    return solve_simplex2(a, gram)


def patch_moments(error, corrections, patch=8):
    """TRAIN-ONLY GT patch moments for channel-first video tensors.

    error = GT - base: [B,3,T,H,W].
    corrections: [B,2,3,T,H,W]. H and W must be divisible by patch.
    Returns a [B,T,H/patch,W/patch,2], Gram [...,2,2], detached.
    Each moment is a mean over all RGB channels and patch*patch pixels.
    This helper does not read data or run any model, matching or warping.
    """
    _check_tensors(error, corrections, "error", "corrections")
    if not isinstance(patch, int) or isinstance(patch, bool) or patch < 1:
        raise ValueError("patch must be a positive integer")
    if error.ndim != 5 or error.shape[1] != 3:
        raise ValueError("error must have shape [B,3,T,H,W]")
    batch, channels, time, height, width = error.shape
    if corrections.shape != (batch, 2, channels, time, height, width):
        raise ValueError("corrections must have shape [B,2,3,T,H,W] matching error")
    if min(time, height, width) < 1 or height % patch or width % patch:
        raise ValueError("T/H/W must be positive and H/W divisible by patch")
    rows, columns, size = height // patch, width // patch, channels * patch * patch
    errors = error.reshape(batch, channels, time, rows, patch, columns, patch)
    errors = errors.permute(0, 2, 3, 5, 1, 4, 6).reshape(batch, time, rows, columns, size)
    directions = corrections.reshape(batch, 2, channels, time, rows, patch, columns, patch)
    directions = directions.permute(0, 3, 4, 6, 1, 2, 5, 7).reshape(
        batch, time, rows, columns, 2, size)
    return train_only_patch_statistics(errors, directions)
