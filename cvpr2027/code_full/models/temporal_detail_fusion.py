"""Local cross-frame restoration-feature transport with optional semantic guidance.

The value path uses neighbouring restoration features, never teacher tokens.
Semantic guidance is frozen, clip-centered cosine similarity. A zero output
projection preserves the original restoration network at initialization.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalDetailFusion(nn.Module):
    def __init__(self, dim, *, key_dim=32, radius=2, offset=2, guidance=1.0,
                 token_grid=(8, 16, 16), logit_scale=8.0):
        super().__init__()
        if dim < 1 or key_dim < 1 or radius < 0 or offset < 1:
            raise ValueError("positive channels/offset and nonnegative radius required")
        if guidance not in (0.0, 1.0) or not 0 < logit_scale <= 32:
            raise ValueError("guidance must be 0/1; logit_scale must be in (0, 32]")
        self.dim, self.radius, self.offset = int(dim), int(radius), int(offset)
        self.guidance = float(guidance)  # configuration, deliberately not in state_dict
        self.token_grid = tuple(int(v) for v in token_grid)
        if len(self.token_grid) != 3 or min(self.token_grid) < 1:
            raise ValueError("token_grid must contain three positive dimensions")
        # Shared projection makes initial matching a similarity rather than
        # the dot product of unrelated randomly initialized Q/K embeddings.
        self.match_proj = nn.Conv2d(dim, key_dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))
        self.confidence = nn.Conv2d(2, 1, 1)
        with torch.no_grad():
            self.confidence.weight[0, :, 0, 0] = torch.tensor([2.0, -2.0])
            self.confidence.bias.zero_()
        self.output_proj = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def _indices(self, time, delta, device):
        index = torch.arange(time, device=device) + delta
        return index.clamp(0, time - 1), (index >= 0) & (index < time)

    def _validity(self, batch, time, height, width, device):
        side = 2 * self.radius + 1
        spatial = F.unfold(torch.ones(1, 1, height, width, device=device),
                           kernel_size=side, padding=self.radius).bool()
        spatial = spatial.reshape(1, 1, side * side, height, width)
        masks = []
        for delta in (-self.offset, self.offset):
            _, valid_time = self._indices(time, delta, device)
            masks.append(spatial & valid_time.reshape(1, time, 1, 1, 1))
        return torch.cat(masks, dim=2).expand(batch, -1, -1, -1, -1).reshape(
            batch * time, 2 * side * side, height, width)

    def _feature_scores(self, features, batch, time):
        bt, channels, height, width = features.shape
        side = 2 * self.radius + 1
        query = features.reshape(batch, time, channels, height, width)
        scores = []
        for delta in (-self.offset, self.offset):
            index, _ = self._indices(time, delta, features.device)
            neighbour = query.index_select(1, index).reshape(bt, channels, height, width)
            keys = F.unfold(neighbour, kernel_size=side, padding=self.radius).reshape(
                bt, channels, side * side, height, width)
            scores.append((features.unsqueeze(2) * keys).sum(1))
        return torch.cat(scores, dim=1)

    @torch.no_grad()
    def _semantic_scores(self, tokens, batch, time, height, width):
        tv, hv, wv = self.token_grid
        if tokens.ndim != 3 or tokens.shape[0] != batch or tokens.shape[1] != tv * hv * wv:
            raise ValueError("semantic tokens do not match batch/token_grid")
        if time % tv:
            raise ValueError("restoration frames must be a multiple of token time")
        # Removing the full clip mean prevents the shared DC vector from
        # making all spatial token cosine similarities nearly one.
        field = tokens.detach().float()
        field = F.normalize(field - field.mean(1, keepdim=True), dim=-1, eps=1e-6)
        channels = field.shape[-1]
        field = field.reshape(batch, tv, hv, wv, channels).permute(0, 1, 4, 2, 3)
        field = F.interpolate(field.reshape(batch * tv, channels, hv, wv),
                              size=(height, width), mode="bilinear", align_corners=False)
        field = F.normalize(field.float(), dim=1, eps=1e-6)
        field = field.reshape(batch, tv, channels, height, width).repeat_interleave(time // tv, dim=1)
        query = field.reshape(batch * time, channels, height, width)
        # Do not unfold 1024-channel semantic fields: compute one displacement
        # at a time, retaining only scalar similarity maps.
        scores = []
        r = self.radius
        for delta in (-self.offset, self.offset):
            index, _ = self._indices(time, delta, field.device)
            neighbour = field.index_select(1, index).reshape(batch * time, channels, height, width)
            neighbour = F.pad(neighbour, (r, r, r, r))
            for y in range(2 * r + 1):
                for x in range(2 * r + 1):
                    scores.append((query * neighbour[:, :, y:y + height, x:x + width]).sum(1))
        return torch.stack(scores, dim=1)

    @staticmethod
    def _probabilities(logits, valid):
        probabilities = torch.softmax(logits.float().masked_fill(~valid, -10000.0), dim=1)
        probabilities = probabilities * valid
        return probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-12)

    def forward(self, features, tokens=None, teacher_tokens=None, *, bypass=False, return_aux=False):
        """Return (B,C,T,H,W) features and scalar diagnostics/optional KL loss."""
        if features.ndim != 5 or features.shape[1] != self.dim:
            raise ValueError("features must have shape (B, configured_C, T, H, W)")
        if bypass:
            return features, {"bypassed": features.new_ones(())}
        batch, channels, time, height, width = features.shape
        flat = features.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        descriptor = F.normalize(self.match_proj(flat).float(), dim=1, eps=1e-6)
        similarity = self._feature_scores(descriptor, batch, time)
        valid = self._validity(batch, time, height, width, features.device)
        scale = self.logit_scale.clamp(math.log(0.1), math.log(32.0)).exp()
        semantic = None
        if self.guidance:
            if tokens is None:
                raise ValueError("prior guidance requires low-light semantic tokens")
            semantic = self._semantic_scores(tokens, batch, time, height, width)
        logits = similarity * scale
        if semantic is not None:
            logits = logits + self.guidance * semantic * scale
        probabilities = self._probabilities(logits, valid)
        count = valid.sum(1, keepdim=True)
        has_neighbours = count > 0
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1, keepdim=True)
        entropy = entropy / count.clamp_min(2).float().log()
        peak = similarity.masked_fill(~valid, -1.0).amax(1, keepdim=True)
        confidence_logits = self.confidence(torch.cat([peak, entropy], dim=1))
        confidence = torch.sigmoid(confidence_logits)
        confidence = confidence * has_neighbours

        side = 2 * self.radius + 1
        size = side * side
        by_time = flat.reshape(batch, time, channels, height, width)
        transported = torch.zeros_like(flat)
        for i, delta in enumerate((-self.offset, self.offset)):
            index, _ = self._indices(time, delta, flat.device)
            neighbour = by_time.index_select(1, index).reshape_as(flat)
            values = F.unfold(neighbour, kernel_size=side, padding=self.radius).reshape(
                batch * time, channels, size, height, width)
            weights = probabilities[:, i * size:(i + 1) * size].to(values.dtype)
            transported = transported + (values * weights.unsqueeze(1)).sum(2)
        difference = confidence.to(flat.dtype) * (transported - flat)
        output = flat + self.output_proj(difference)
        output = output.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        aux = {
            "bypassed": flat.new_zeros(()),
            "confidence_mean": confidence.detach().float().mean(),
            "entropy_mean": entropy.detach().mean(),
            "valid_query_fraction": has_neighbours.detach().float().mean(),
            "logit_scale": scale.detach(),
            "semantic_score_std": (semantic[valid].std(unbiased=False).detach()
                                   if semantic is not None and valid.any()
                                   else flat.new_zeros((), dtype=torch.float32)),
        }
        if return_aux:
            # Optional differentiable maps for later counterfactual evidence
            # selection. Default logging retains detached scalars only.
            def as_volume(value):
                return value.reshape(batch, time, 1, height, width).permute(0, 2, 1, 3, 4)
            aux["confidence_logits"] = as_volume(confidence_logits)
            aux["confidence_map"] = as_volume(confidence)
            aux["valid_query_mask"] = as_volume(has_neighbours).detach()
        if teacher_tokens is not None:
            teacher_scores = self._semantic_scores(teacher_tokens, batch, time, height, width)
            with torch.no_grad():
                teacher_p = self._probabilities(teacher_scores * scale.detach(), valid)
                teacher_entropy = -(teacher_p * teacher_p.clamp_min(1e-12).log()).sum(1, keepdim=True)
                teacher_entropy = teacher_entropy / count.clamp_min(2).float().log()
                teacher_weight = (1 - teacher_entropy).clamp(0, 1) * has_neighbours
            kl = (teacher_p * (teacher_p.clamp_min(1e-12).log()
                              - probabilities.clamp_min(1e-12).log())).sum(1, keepdim=True)
            aux["correspondence_loss"] = (kl * teacher_weight).sum() / teacher_weight.sum().clamp_min(1e-6)
            aux["teacher_confidence_mean"] = teacher_weight.detach().mean()
        return output, aux
