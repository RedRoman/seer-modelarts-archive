"""The token projection used by the training script, shared with evaluation."""
from __future__ import annotations

import torch
import torch.nn as nn


def project_tokens(z, mode, k, basis, li):
    """Keep the archived training arithmetic and dtype conversions unchanged."""
    b, n, d = z.shape
    if mode == "global_rank":
        mu, V = basis["mean"][li], basis["comp"][li][:k]
        x = z.reshape(-1, d) - mu
        return ((x @ V.T) @ V + mu).reshape(b, n, d)
    if mode != "clip_rank":
        raise ValueError(f"unknown token projection: {mode!r}")
    x = z.reshape(-1, d)
    mu = x.mean(0, keepdim=True)
    xc = (x - mu).float()
    V = torch.linalg.svd(xc, full_matrices=False)[2][:k]
    return (((xc @ V.T) @ V) + mu).reshape(b, n, d).to(z.dtype)


class ProjectedBackbone(nn.Module):
    """Preserve the backbone result contract while applying saved conditioning.

    Training computes the encoder under autocast, then projects float32 tokens
    outside autocast. Evaluation must follow that order as well.
    """
    def __init__(self, backbone, mode, rank, basis=None):
        super().__init__()
        self.backbone, self.mode, self.rank = backbone, mode, rank
        self.register_buffer("basis_mean", None if basis is None else basis["mean"])
        self.register_buffer("basis_comp", None if basis is None else basis["comp"])

    def forward(self, video):
        result = dict(self.backbone(video))
        basis = {"mean": self.basis_mean, "comp": self.basis_comp}
        with torch.autocast(device_type=video.device.type, enabled=False):
            result["feats"] = [project_tokens(z.float(), self.mode, self.rank, basis, li)
                               for li, z in enumerate(result["feats"])]
        return result
