"""Experimental video-prior addresses with observed, position-preserving values.

This is a module hypothesis, not a validated novelty/performance claim. Frozen
block-17 V-JEPA tokens supply addresses; calibrated low-light RGB detail supplies
values. Excluding the target frame's values does NOT make the encoder or base
prediction independent of that frame. There is no GT or flow input to this file.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


ADDRESS_MODES = ("prior", "pixel", "shuffled_prior")
RETRIEVAL_MODES = ("cross_frame", "same_frame", "local")
PRIOR_PROJECTION_SEED = 1729
PIXEL_PROJECTION_SEED = 1730
NORMALIZE_EPS = 1e-6


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be an integer in [0,2**63)")


def memory_feature_config(patch=16, descriptor_dim=64,
                          projection_seed=PRIOR_PROJECTION_SEED):
    """JSON recipe; all preprocessing is CPU float32 and may be cached once."""
    _positive_int(patch, "patch")
    _positive_int(descriptor_dim, "descriptor_dim")
    _seed(projection_seed, "projection_seed")
    if patch < 4 or patch % 4:
        raise ValueError("patch must be a positive multiple of 4")
    return {
        "schema_version": 1, "patch": patch, "descriptor_dim": descriptor_dim,
        "backend": "CPU torch.float32", "prior_block": 17, "prior_input_dim": 1024,
        "prior_projection_seed": projection_seed,
        "pixel_projection_seed": PIXEL_PROJECTION_SEED,
        "projection": "CPU torch.randn(input_dim,descriptor_dim), private Generator(seed), /sqrt(descriptor_dim)",
        "prior_centering": "subtract full-clip mean over all tubelet-spatial tokens BEFORE projection",
        "pixel_descriptor": "low RGB then base RGB; each patch adaptive-average-pooled to 4x4; channel,y,x",
        "pixel_centering": "subtract full-clip mean over all frame-spatial patches BEFORE projection",
        "normalization": "F.normalize projected descriptors, dim=-1, eps=1e-6",
        "temporal_expansion": "repeat_interleave each tubelet descriptor to its consecutive frames",
        "crop_policy": "prepare full clip once; crop row-major patch grids; NEVER recenter cropped descriptors",
        "values": "q=clamp(LP9(base)+clamp(LP9(base)/(LP9(low)+1e-4),.25,16)*(low-LP9(low)),0,1); q-LP9(q)",
        "value_layout": "[B,T,P,3*patch*patch]; spatial patches row-major; channels then patch y,x",
        "lowpass": "spatial box9, reflect padding; no time filtering",
        "gt_input": False, "target_fit": False,
        "alignment": "none; patch-relative positions retained; no subpixel-motion compensation",
    }


@lru_cache(maxsize=8)
def _projection(input_dim, output_dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(input_dim, output_dim, generator=generator,
                       dtype=torch.float32, device="cpu") / math.sqrt(output_dim)


def _finite_cpu_float(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} preprocessing must run on CPU")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@torch.no_grad()
def project_prior_tokens(tokens, *, descriptor_dim=64,
                         projection_seed=PRIOR_PROJECTION_SEED):
    """CPU [B,N,1024] -> [B,N,D], full-clip centered then projected/normalized.

    No global RNG is consumed. Pass the full block-17 clip before any crop.
    This helper cannot authenticate a tensor's encoder/layer; the exporter must
    verify the backbone, checkpoint, layer identity and source-only inventory.
    """
    _positive_int(descriptor_dim, "descriptor_dim")
    _seed(projection_seed, "projection_seed")
    _finite_cpu_float(tokens, "tokens")
    if tokens.ndim != 3 or min(tokens.shape[:2]) < 1 or tokens.shape[-1] != 1024:
        raise ValueError("tokens must have shape [B,N,1024]")
    centered = tokens - tokens.mean(dim=1, keepdim=True)
    projected = centered @ _projection(1024, descriptor_dim, projection_seed)
    return F.normalize(projected, dim=-1, eps=NORMALIZE_EPS).contiguous()


def _frames(video):
    b, c, t, h, w = video.shape
    return video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _video(frames, b, t):
    _, c, h, w = frames.shape
    return frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _lowpass(video):
    b, _, t, _, _ = video.shape
    filtered = F.avg_pool2d(F.pad(_frames(video), (4, 4, 4, 4), mode="reflect"), 9, 1)
    return _video(filtered, b, t)


def _patchify(video, patch):
    b, c, t, h, w = video.shape
    return F.unfold(_frames(video), patch, stride=patch).transpose(1, 2).reshape(
        b, t, (h // patch) * (w // patch), c * patch * patch).contiguous()


def _unpatchify(values, height, width, patch):
    b, t, p, d = values.shape
    frames = F.fold(values.reshape(b * t, p, d).transpose(1, 2),
                    (height, width), patch, stride=patch)
    return _video(frames, b, t)


@torch.no_grad()
def prepare_observation_memory_inputs(low, base, prior_tokens, *,
                                      token_grid=(8, 16, 16), patch=16,
                                      descriptor_dim=64, prior_is_projected=False,
                                      projection_seed=PRIOR_PROJECTION_SEED):
    """Prepare full-clip observable fields once, before patch-aligned cropping.

    low/base: CPU float32 [B,3,T,H,W], RGB [0,1]. prior_tokens: [B,N,1024]
    or already projected [B,N,D] when prior_is_projected=True. N=prod(token_grid),
    the spatial grid must be H/patch by W/patch, and T must divide into complete
    tubelets. Projected tokens are used verbatim: no second centering/projection.

    Returns detail_values [B,T,P,3*patch²], both addresses [B,T,P,D], spatial_shape
    [H,W], and metadata. The tensors are detached; GT is never accepted.
    """
    config = memory_feature_config(patch, descriptor_dim, projection_seed)
    for value, name in ((low, "low"), (base, "base"), (prior_tokens, "prior_tokens")):
        _finite_cpu_float(value, name)
    if low.ndim != 5 or low.shape[1] != 3 or min(low.shape) < 1 or base.shape != low.shape:
        raise ValueError("low and base must have matching shape [B,3,T,H,W]")
    b, _, t, h, w = low.shape
    if min(h, w) <= 4 or h % patch or w % patch:
        raise ValueError("Need reflect-compatible dimensions and complete patches")
    if any(bool(((value < 0) | (value > 1)).any()) for value in (low, base)):
        raise ValueError("low and base RGB must be in [0,1]")
    if not isinstance(prior_is_projected, bool):
        raise TypeError("prior_is_projected must be bool")
    if not isinstance(token_grid, (tuple, list)) or len(token_grid) != 3:
        raise ValueError("token_grid must contain tubelet,height,width")
    for size in token_grid:
        _positive_int(size, "token_grid dimension")
    tv, gh, gw = token_grid
    if (gh, gw) != (h // patch, w // patch) or t % tv:
        raise ValueError("Token grid must match spatial patches and complete temporal tubelets")
    expected_dim = descriptor_dim if prior_is_projected else 1024
    if prior_tokens.shape != (b, tv * gh * gw, expected_dim):
        raise ValueError("prior_tokens shape does not match the declared token grid/dimension")
    if prior_is_projected:
        if bool((torch.linalg.vector_norm(prior_tokens, dim=-1) > 1 + 2e-5).any()):
            raise ValueError("Projected descriptors must have norm <=1 (zero descriptors allowed)")
        projected = prior_tokens.detach()
    else:
        projected = project_prior_tokens(prior_tokens, descriptor_dim=descriptor_dim,
                                         projection_seed=projection_seed)
    prior_address = projected.reshape(b, tv, gh * gw, descriptor_dim).repeat_interleave(
        t // tv, dim=1).contiguous()

    low_l, base_l = _lowpass(low), _lowpass(base)
    gain = (base_l / (low_l + 1e-4)).clamp(.25, 16.)
    calibrated = (base_l + gain * (low - low_l)).clamp(0, 1)
    detail_values = _patchify(calibrated - _lowpass(calibrated), patch)

    # Spatial sub-bins in the address descriptor; full within-patch RGB is kept
    # in detail_values, not averaged into these descriptors.
    pixel_patches = _patchify(torch.cat((low, base), dim=1), patch)
    pixel_raw = F.adaptive_avg_pool2d(pixel_patches.reshape(-1, 6, patch, patch), (4, 4))
    pixel_raw = pixel_raw.reshape(b, t * gh * gw, 96)
    pixel_raw = pixel_raw - pixel_raw.mean(dim=1, keepdim=True)
    pixel_projected = pixel_raw @ _projection(96, descriptor_dim, PIXEL_PROJECTION_SEED)
    pixel_address = F.normalize(pixel_projected, dim=-1, eps=NORMALIZE_EPS).reshape(
        b, t, gh * gw, descriptor_dim).contiguous()
    metadata = dict(config, token_grid=list(token_grid), frames=t,
                    projected_input=prior_is_projected, spatial_shape=[h, w])
    return {"detail_values": detail_values.detach(), "prior_address": prior_address.detach(),
            "pixel_address": pixel_address.detach(), "spatial_shape": [h, w],
            "metadata": metadata}


class ObservationMemory(nn.Module):
    """Same parameter shapes and initialization in every address/read mode.

    A[t,p,k] is a normalized learned assignment to shared slots. For cross-frame
    reads, slot sums/masses are formed separately from all s!=t, then read using
    A[t]. Values retain RGB and every within-patch coordinate. This is finite
    soft-assignment memory, not optical flow, a graph solver, or independent noise.

    `local` reads only the same patch position at t-1/t+1, weighted by similarity
    between their learned slot assignments. It has the same trainable state but
    different operations/FLOPs. `same_frame` writes and reads only the target
    frame. `shuffled_prior` permutes address locations only, not values.
    """
    def __init__(self, address_mode="prior", retrieval_mode="cross_frame", *,
                 descriptor_dim=64, slots=32, address_dim=16, patch=16,
                 hidden_dim=16, shuffle_seed=2718):
        super().__init__()
        if address_mode not in ADDRESS_MODES or retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError("Unknown address_mode or retrieval_mode")
        for name, value in (("descriptor_dim", descriptor_dim), ("slots", slots),
                            ("address_dim", address_dim), ("hidden_dim", hidden_dim)):
            _positive_int(value, name)
        memory_feature_config(patch, descriptor_dim)
        _seed(shuffle_seed, "shuffle_seed")
        self.address_mode, self.retrieval_mode = address_mode, retrieval_mode
        self.descriptor_dim, self.slots = descriptor_dim, slots
        self.address_dim, self.patch = address_dim, patch
        self.hidden_dim, self.shuffle_seed = hidden_dim, shuffle_seed
        self.address_projection = nn.Linear(descriptor_dim, address_dim, bias=False)
        self.slot_keys = nn.Parameter(torch.randn(slots, address_dim) / math.sqrt(address_dim))
        self.logit_scale = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.decoder = nn.Sequential(nn.Conv2d(6, hidden_dim, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden_dim, 3, 3, padding=1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def get_config(self):
        """Constructor kwargs; modes are not tensors, so save this with state_dict."""
        return {"address_mode": self.address_mode, "retrieval_mode": self.retrieval_mode,
                "descriptor_dim": self.descriptor_dim, "slots": self.slots,
                "address_dim": self.address_dim, "patch": self.patch,
                "hidden_dim": self.hidden_dim, "shuffle_seed": self.shuffle_seed}

    def _validate(self, base, detail_values, prior_address, pixel_address):
        values = (base, detail_values, prior_address, pixel_address)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("All inputs must be tensors")
        if base.ndim != 5 or base.shape[1] != 3 or min(base.shape) < 1:
            raise ValueError("base must have shape [B,3,T,H,W]")
        b, _, t, h, w = base.shape
        if h % self.patch or w % self.patch or t < 2:
            raise ValueError("Need complete patches and at least two frames")
        p = (h // self.patch) * (w // self.patch)
        if detail_values.shape != (b, t, p, 3 * self.patch * self.patch):
            raise ValueError("detail_values shape disagrees with base and patch size")
        if any(value.shape != (b, t, p, self.descriptor_dim)
               for value in (prior_address, pixel_address)):
            raise ValueError("Address shape disagrees with base and descriptor_dim")
        if any(value.device != base.device or value.dtype != torch.float32 for value in values):
            raise TypeError("Inputs must share a device and be float32")
        if self.slot_keys.device != base.device or self.slot_keys.dtype != torch.float32:
            raise TypeError("Model parameters must share the input device and be float32")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("Inputs must be finite")
        if bool(((base < 0) | (base > 1)).any()):
            raise ValueError("base RGB must be in [0,1]")

    def _assignments(self, prior_address, pixel_address):
        address = pixel_address if self.address_mode == "pixel" else prior_address
        b, t, p, d = address.shape
        if self.address_mode == "shuffled_prior":
            generator = torch.Generator(device="cpu").manual_seed(self.shuffle_seed)
            permutation = torch.randperm(t * p, generator=generator, device="cpu").to(address.device)
            address = address.reshape(b, t * p, d).index_select(1, permutation).reshape(b, t, p, d)
        queries = F.normalize(self.address_projection(address), dim=-1, eps=NORMALIZE_EPS)
        keys = F.normalize(self.slot_keys, dim=-1, eps=NORMALIZE_EPS)
        logits = (queries @ keys.T) * self.logit_scale.clamp(.1, 16.)
        return torch.softmax(logits, dim=-1)

    def _read(self, assignments, values):
        b, t, p, k = assignments.shape
        if self.retrieval_mode == "local":
            times = torch.arange(t, device=values.device)
            source = torch.stack((times - 1, times + 1), dim=-1)
            valid = (source >= 0) & (source < t)
            source = source.clamp(0, t - 1)
            neighbors = assignments[:, source]  # [B,T,2,P,K]
            queries = F.normalize(assignments, dim=-1, eps=NORMALIZE_EPS)
            keys = F.normalize(neighbors, dim=-1, eps=NORMALIZE_EPS)
            similarity = (queries[:, :, None] * keys).sum(-1) * self.logit_scale.clamp(.1, 16.)
            weights = torch.softmax(similarity.masked_fill(~valid[None, :, :, None], -torch.inf), dim=2)
            read = (weights[..., None] * values[:, source]).sum(dim=2)
            return read, {"local_weights": weights, "local_source_indices": source,
                          "local_source_valid": valid}

        sums = torch.einsum("btpk,btpv->btkv", assignments, values)
        masses = assignments.sum(dim=2)
        if self.retrieval_mode == "cross_frame":
            # Explicit zero diagonal gives exact zero derivatives with respect
            # to target-frame VALUES, without cancellation in total-own sums.
            off_diagonal = 1 - torch.eye(t, device=values.device, dtype=values.dtype)
            sums = torch.einsum("ts,bskv->btkv", off_diagonal, sums)
            masses = torch.einsum("ts,bsk->btk", off_diagonal, masses)
        memory = sums / masses.clamp_min(1e-12)[..., None]
        read = torch.einsum("btpk,btkv->btpv", assignments, memory)
        return read, {"slot_mass": masses}

    def forward(self, base, detail_values, prior_address, pixel_address, *, return_aux=False):
        self._validate(base, detail_values, prior_address, pixel_address)
        if not isinstance(return_aux, bool):
            raise TypeError("return_aux must be bool")
        assignments = self._assignments(prior_address, pixel_address)
        retrieved, diagnostics = self._read(assignments, detail_values)
        b, _, t, h, w = base.shape
        memory_detail = _unpatchify(retrieved, h, w, self.patch)
        residual = _video(self.decoder(torch.cat((_frames(base), _frames(memory_detail)), dim=1)), b, t)
        prediction = (base + residual).clamp(0, 1)
        aux = {"assignment_entropy": float((-(assignments.clamp_min(1e-12).log() * assignments)
                                               .sum(-1).mean()).detach()),
               "memory_rms": float(memory_detail.square().mean().sqrt().detach())}
        if return_aux:
            aux.update(assignments=assignments, retrieved_values=retrieved, **diagnostics)
        return {"pred_pixels": prediction, "memory_detail": memory_detail, "aux": aux}
