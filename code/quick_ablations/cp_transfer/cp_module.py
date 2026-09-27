"""Bayesian Chroma Preservation (CP), reimplemented from scratch per
sections/03_method.tex Eq.5-6 (Seer paper). No original CP code survives on
this host; this is a from-scratch, careful reimplementation so the baseline
transfer experiment is on equal footing with how Seer's own alpha was fit.

Definitions (matching the paper exactly):
  Y_bullet(img) = 0.299*R + 0.587*G + 0.114*B      (standard Retinex/BT.601 luma,
                                                     matches models/local_illumination_cues.py
                                                     rgb_to_luminance used elsewhere in this repo)
  Yhat_cp = X * Y(Yhat) / (Y(X) + eps),  eps = 1e-3      (Eq. cp_surrogate_fusion, line 1)
  Yhat_star_c(alpha_c) = alpha_c * Yhat_cp_c + (1-alpha_c) * Yhat_c   (per-channel convex combo)

  alpha_c^* = (m22 - m12) / (m11 - 2*m12 + m22)
  where  eps1 = Yhat_cp_c - Y_c , eps2 = Yhat_c - Y_c  (Y = GT here, uncentred 2nd moments)
        m_ij = E[eps_i * eps_j]   (empirical mean over all pixels/frames/pairs in the fit set)

All tensors assumed float32 in [0,1], shape (..., 3, H, W) with channel dim
explicit (works for both (3,H,W) image and (3,T,H,W) video by broadcasting
over leading dims via flattening).
"""
from __future__ import annotations
import numpy as np
import torch


EPS_DIV = 1e-3


def luminance(img: torch.Tensor) -> torch.Tensor:
    """img: (..., 3, H, W) -> (..., 1, H, W). BT.601 luma, matches
    models/local_illumination_cues.py:rgb_to_luminance in this repo."""
    r = img[..., 0:1, :, :]
    g = img[..., 1:2, :, :]
    b = img[..., 2:3, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b


def cp_surrogate(pred: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """Yhat_cp = X * Y(Yhat) / (Y(X) + eps). pred, lq: (...,3,H,W) in [0,1]."""
    y_pred = luminance(pred)
    y_lq = luminance(lq)
    return lq * y_pred / (y_lq + EPS_DIV)


def fuse_per_channel(pred: torch.Tensor, cp: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """alpha: (3,) per-channel convex weight. Returns alpha*cp + (1-alpha)*pred, clamped [0,1]."""
    alpha = alpha.view(*([1] * (pred.dim() - 3)), 3, 1, 1).to(pred.dtype).to(pred.device)
    out = alpha * cp + (1.0 - alpha) * pred
    return out.clamp(0.0, 1.0)


def fit_alpha_star(pred_list, lq_list, gt_list) -> np.ndarray:
    """Fit per-channel alpha* on a validation pool using the closed-form LMMSE
    solution (Eq. alpha_star_general). Uncentred residuals against GT.

    pred_list, lq_list, gt_list: lists of numpy arrays, each (3,H,W) or
    (3,T,H,W) float32 in [0,1], one entry per validation pair. All pixels
    across all pairs (and all frames if video) are pooled into one empirical
    expectation per the paper's "N ~= 15 SDSD validation pairs" protocol
    (pairs are the unit of N; every pixel within each pair contributes to the
    empirical moment average, exactly as an M-estimator over the pooled
    pixel population represented by those N pairs).

    Returns alpha: shape (3,) float64.
    """
    # accumulate per-channel sums for m11, m12, m22 using running sums to
    # avoid holding all pixels of all pairs in memory simultaneously.
    sum_e1e1 = np.zeros(3, dtype=np.float64)
    sum_e1e2 = np.zeros(3, dtype=np.float64)
    sum_e2e2 = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)

    for pred, lq, gt in zip(pred_list, lq_list, gt_list):
        pred_t = torch.from_numpy(pred).float()
        lq_t = torch.from_numpy(lq).float()
        gt_t = torch.from_numpy(gt).float()
        cp_t = cp_surrogate(pred_t, lq_t)

        # move channel dim to last for easy flatten-per-channel
        def chan_flat(x):
            # x: (3, ...) -> (3, N)
            c = x.shape[0]
            return x.reshape(c, -1).numpy().astype(np.float64)

        u1 = chan_flat(cp_t)     # Yhat_cp
        u2 = chan_flat(pred_t)   # Yhat
        y = chan_flat(gt_t)      # GT

        e1 = u1 - y
        e2 = u2 - y

        sum_e1e1 += (e1 * e1).sum(axis=1)
        sum_e1e2 += (e1 * e2).sum(axis=1)
        sum_e2e2 += (e2 * e2).sum(axis=1)
        count += e1.shape[1]

    m11 = sum_e1e1 / count
    m12 = sum_e1e2 / count
    m22 = sum_e2e2 / count

    denom = (m11 - 2 * m12 + m22)
    alpha = np.where(np.abs(denom) > 1e-12, (m22 - m12) / denom, 0.5)
    # alpha is a convex-combination weight in principle in [0,1] but the
    # closed form can exceed that range if residuals are correlated in an
    # unusual way; we do NOT clip here so we can report the raw fitted value
    # faithfully (matches the paper's reported (0.50,0.50,0.68) which are
    # all within [0,1] naturally for Seer; report whatever Retinexformer's
    # data actually gives, clipped only at apply-time via the convex fusion
    # itself which is unconstrained in alpha unless we choose to clip).
    return alpha, dict(m11=m11.tolist(), m12=m12.tolist(), m22=m22.tolist())
