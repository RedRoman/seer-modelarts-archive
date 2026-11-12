"""Bounded measurement-coordinate updates with a frozen RGB observation selector.

The only learned output is a target-to-source flow increment. There is no
trainable RGB decoder. This tests a prior-conditioned alignment hypothesis;
flow refinement/deformable sampling are established operations, not new claims.
GT is not accepted. The original source-fitted selector and normalization stay
frozen, including during backpropagation through their observable inputs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CANDIDATES = _load("_geometry_frozen_candidates", "observation_candidates.py")
_FEATURES = _load("_geometry_frozen_features", "correction_selector.py")
ADDRESS_MODES = ("prior", "pixel", "shuffled_prior")


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _check(isinstance(value, int) and not isinstance(value, bool) and value > 0,
           f"{name} must be a positive integer")


def _frames(video):
    b, c, t, h, w = video.shape
    return video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _geometric_mask(flows, offset):
    b, _, _, t, h, w = flows.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=flows.device),
                            torch.arange(w, device=flows.device), indexing="ij")
    times = torch.arange(t, device=flows.device)
    outputs = []
    for direction, delta in enumerate((-offset, offset)):
        x = xx[None, None] + flows[:, direction, 0]
        y = yy[None, None] + flows[:, direction, 1]
        source = times + delta
        temporal = ((source >= 0) & (source < t))[None, :, None, None]
        outputs.append((temporal & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1))[:, None])
    return torch.stack(outputs, 1)


def differentiable_raw_candidates(low, base, flows, masks, *, offset=2):
    """Exact frozen raw-detail recipe with its outer no_grad wrapper removed.

    The existing implementation has only that outer wrapper; its lowpass,
    bilinear sampling, gain and clamp operations retain their original math.
    Caller must supply conservative valid masks for the actual supplied flow.
    This helper does not relax or silently repair invalid input masks.
    """
    function = getattr(_CANDIDATES.build_observation_candidates, "__wrapped__", None)
    if not callable(function):
        raise RuntimeError("Pinned candidate helper must expose its no_grad-wrapped function")
    return function(low, base, flows, masks, mode="raw_detail", offset=offset)


def differentiable_patch_features(low, base, corrections, valid_mask, patch=8):
    """The frozen selector's 19 features, retaining derivatives through values.

    This is the minimal math copy of observable_patch_features, whose internal
    no_grad block cannot be removed by unwrapping. Exact parity is tested.
    The returned Gram uses observations only, never GT error correlations.
    """
    _integer(patch, "patch")
    _check(isinstance(low, torch.Tensor) and low.ndim == 5 and low.shape[1] == 3,
           "low must have shape [B,3,T,H,W]")
    b, c, t, h, w = low.shape
    _check(min(b, t, h, w) > 0 and h % patch == 0 and w % patch == 0,
           "Complete positive patches required")
    _check(base.shape == low.shape and corrections.shape == (b, 2, c, t, h, w)
           and valid_mask.shape == (b, 2, 1, t, h, w), "Observable feature shapes differ")
    for value in (low, base, corrections, valid_mask):
        _check(value.device == low.device and bool(torch.isfinite(value).all()),
               "Finite observable inputs on a shared device required")
    _check(all(value.dtype == torch.float32 for value in (low, base, corrections)),
           "Observable RGB tensors must be float32")
    _check(bool(((valid_mask >= 0) & (valid_mask <= 1)).all()), "Validity must lie in [0,1]")
    _check(not bool((corrections * (1-valid_mask.float())).abs().amax() > 1e-7),
           "Corrections must be zero outside validity")
    pool = _FEATURES._patch_pool
    low_mean, base_mean = pool(low, patch), pool(base, patch)
    d0, d1 = corrections[:, 0], corrections[:, 1]
    d0_mean, d1_mean = pool(d0, patch), pool(d1, patch)
    low_var = (pool(low.square(), patch)-low_mean.square()).clamp_min(0).mean(2, keepdim=True)
    base_var = (pool(base.square(), patch)-base_mean.square()).clamp_min(0).mean(2, keepdim=True)
    g00, g01, g11 = (pool(value, patch).mean(2) for value in
                    (d0.square(), d0*d1, d1.square()))
    gram = torch.stack((g00, g01, g01, g11), -1).reshape(b, t, h//patch, w//patch, 2, 2)
    valid = [pool(valid_mask[:, j].float(), patch) for j in range(2)]
    features = torch.cat((low_mean, base_mean, d0_mean, d1_mean, low_var, base_var,
                          g00[:, :, None], g01[:, :, None], g11[:, :, None], *valid), 2)
    return features, gram


class ObservationGeometry(nn.Module):
    """Shared direction network; prior/pixel/shuffled have identical state shapes.

    Inputs are full-clip cached RGB and original [-offset,+offset] flows. Only
    the caller's LOSS may be cropped; source sampling never crops the video.
    Addresses are cached full-clip descriptors, in spatial row-major order.
    The original flow samples source addresses at predictor-patch centers;
    average original flow within each predictor patch locates that sample.
    Own/source/difference descriptors plus original flow/validity predict a
    bilinearly upsampled increment. The coordinate update is bounded by two
    pixels per axis, and is zero where the original mask is invalid.

    The RGB result is precisely the frozen selector S(refined_flow). At zero
    increment it equals S(original_flow); no free RGB compensation is added.
    """
    def __init__(self, normalization, frozen_node_state, address_mode="prior", *,
                 descriptor_dim=64, address_patch=16, predictor_patch=16,
                 hidden_dim=32, address_dim=16, offset=2, max_shift=2.0,
                 shuffle_seed=2718):
        super().__init__()
        _check(address_mode in ADDRESS_MODES, "Unknown address mode")
        for name, value in (("descriptor_dim", descriptor_dim), ("address_patch", address_patch),
                            ("predictor_patch", predictor_patch), ("hidden_dim", hidden_dim),
                            ("address_dim", address_dim), ("offset", offset)):
            _integer(value, name)
        _check(address_patch == 16 and predictor_patch in (8, 16) and offset == 2,
               "Declared geometry uses address patch16, predictor patch8/16, and offset2")
        _check(isinstance(max_shift, (float, int)) and not isinstance(max_shift, bool)
               and max_shift == 2, "Declared per-axis displacement bound is 2 pixels")
        _check(isinstance(shuffle_seed, int) and not isinstance(shuffle_seed, bool)
               and 0 <= shuffle_seed < 2**63, "Invalid private shuffle seed")
        mean = torch.tensor(normalization["feature_mean"], dtype=torch.float32)
        std = torch.tensor(normalization["feature_std"], dtype=torch.float32)
        _check(mean.shape == std.shape == (19,) and bool(torch.isfinite(mean).all())
               and bool(torch.isfinite(std).all()) and bool((std > 0).all()),
               "Frozen source normalization must contain 19 finite means/positive stds")
        self.register_buffer("feature_mean", mean.reshape(1, 1, 19, 1, 1))
        self.register_buffer("feature_std", std.reshape(1, 1, 19, 1, 1))
        self.address_mode, self.descriptor_dim = address_mode, descriptor_dim
        self.address_patch, self.predictor_patch = address_patch, predictor_patch
        self.hidden_dim, self.address_dim = hidden_dim, address_dim
        self.offset, self.max_shift, self.shuffle_seed = offset, float(max_shift), shuffle_seed

        self.node = nn.Sequential(nn.Conv2d(19, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 1), nn.ReLU(), nn.Conv2d(32, 2, 1))
        expected = set(self.node.state_dict())
        state = dict(frozen_node_state)
        if any(key.startswith("node.") for key in state):
            allowed = {"node."+key for key in expected} | {
                "edge.0.weight", "edge.0.bias", "edge.2.weight", "edge.2.bias"}
            _check(set(state) <= allowed and {"node."+key for key in expected} <= set(state),
                   "Unexpected or missing frozen trajectory state keys")
            state = {key.removeprefix("node."): value for key, value in state.items()
                     if key.startswith("node.")}
        _check(set(state) == expected, "Frozen node state must match all six tensors exactly")
        _check(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                   and bool(torch.isfinite(v).all()) for v in state.values()), "Invalid frozen node state")
        self.node.load_state_dict(state, strict=True)
        self.node.requires_grad_(False)
        self.node.eval()

        self.address_projection = nn.Conv2d(descriptor_dim, address_dim, 1, bias=False)
        self.offset_net = nn.Sequential(nn.Conv2d(3*address_dim+3, hidden_dim, 3, padding=1),
                                        nn.ReLU(), nn.Conv2d(hidden_dim, 2, 1))
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.node.eval()
        return self

    def get_config(self):
        """Constructor kwargs excluding the separately supplied frozen state."""
        return {"normalization": {"feature_mean": self.feature_mean.flatten().tolist(),
                                   "feature_std": self.feature_std.flatten().tolist()},
                "address_mode": self.address_mode, "descriptor_dim": self.descriptor_dim,
                "address_patch": self.address_patch, "predictor_patch": self.predictor_patch,
                "hidden_dim": self.hidden_dim, "address_dim": self.address_dim,
                "offset": self.offset, "max_shift": self.max_shift, "shuffle_seed": self.shuffle_seed}

    def _validate(self, low, base, flows, masks, prior_address, pixel_address):
        _CANDIDATES._validate(low, base, flows, masks, self.offset)
        _check(low.device.type == "cpu" and self.feature_mean.device.type == "cpu",
               "Geometry and frozen selector run on CPU float32")
        b, _, t, h, w = low.shape
        _check(h % self.address_patch == 0 and w % self.address_patch == 0,
               "Full source geometry must contain complete address patches")
        p = (h//self.address_patch)*(w//self.address_patch)
        for address in (prior_address, pixel_address):
            _check(isinstance(address, torch.Tensor) and address.shape == (b, t, p, self.descriptor_dim)
                   and address.device.type == "cpu" and address.dtype == torch.float32
                   and bool(torch.isfinite(address).all()), "Invalid full-clip cached address tensor")
        _check(not bool((masks.bool() & ~_geometric_mask(flows, self.offset)).any()),
               "Original mask marks out-of-frame or out-of-image coordinates valid")

    def _address_fields(self, prior_address, pixel_address, flows, masks):
        b, _, _, t, h, w = flows.shape
        ah, aw = h//self.address_patch, w//self.address_patch
        ph, pw = h//self.predictor_patch, w//self.predictor_patch
        address = pixel_address if self.address_mode == "pixel" else prior_address
        if self.address_mode == "shuffled_prior":
            rng = torch.Generator(device="cpu").manual_seed(self.shuffle_seed)
            permutation = torch.randperm(t*ah*aw, generator=rng)
            address = address.reshape(b, t*ah*aw, self.descriptor_dim).index_select(1, permutation)
        address = address.reshape(b, t, ah, aw, self.descriptor_dim).permute(0, 1, 4, 2, 3)
        address = self.address_projection(address.reshape(b*t, self.descriptor_dim, ah, aw))
        address = address.reshape(b, t, self.address_dim, ah, aw)
        own = F.interpolate(address.reshape(b*t, self.address_dim, ah, aw), size=(ph,pw),
                             mode="bilinear", align_corners=False)
        yy, xx = torch.meshgrid(torch.arange(ph, dtype=torch.float32),
                                torch.arange(pw, dtype=torch.float32), indexing="ij")
        center_x = (xx+.5)*self.predictor_patch-.5
        center_y = (yy+.5)*self.predictor_patch-.5
        times = torch.arange(t)
        results = []
        for direction, delta in enumerate((-self.offset, self.offset)):
            average_flow = _FEATURES._patch_pool(flows[:, direction], self.predictor_patch)
            average_flow = average_flow.reshape(b*t, 2, ph, pw)
            x, y = center_x[None]+average_flow[:, 0], center_y[None]+average_flow[:, 1]
            grid = torch.stack(((x+.5)*(2/w)-1, (y+.5)*(2/h)-1), -1)
            source = address.index_select(1, (times+delta).clamp(0,t-1))
            source = F.grid_sample(source.reshape(b*t, self.address_dim, ah, aw), grid,
                                   mode="bilinear", padding_mode="border", align_corners=False)
            valid = _FEATURES._patch_pool(masks[:, direction].float(), self.predictor_patch)
            valid = valid.reshape(b*t, 1, ph, pw)
            source = source * valid
            scaled_flow = average_flow / float(max(h,w))
            results.append(torch.cat((own, source, source-own, scaled_flow, valid), 1))
        return torch.stack(results, 1).reshape(b*t*2, 3*self.address_dim+3, ph, pw)

    def _selector(self, low, base, corrections, valid):
        b, _, t, h, w = base.shape
        features, _ = differentiable_patch_features(low, base, corrections, valid, patch=8)
        standardized = (features-self.feature_mean)/self.feature_std
        logits = self.node(standardized.reshape(b*t, 19, h//8, w//8))
        weights = torch.softmax(torch.cat((torch.zeros_like(logits[:, :1]), logits), 1), 1)[:, 1:]
        weights = F.interpolate(weights, size=(h,w), mode="nearest").reshape(b,t,2,h,w).permute(0,2,1,3,4)
        prediction = (base+(weights[:, :, None]*corrections).sum(1)).clamp(0,1)
        return prediction, features, weights

    def forward(self, low, base, flows, masks, prior_address, pixel_address, *, return_aux=False):
        self._validate(low, base, flows, masks, prior_address, pixel_address)
        _check(isinstance(return_aux, bool), "return_aux must be bool")
        # Cache observables are fixed; only the new geometry module is trained.
        low, base, flows, masks = (x.detach() for x in (low, base, flows, masks))
        prior_address, pixel_address = prior_address.detach(), pixel_address.detach()
        b, _, _, t, h, w = flows.shape
        inputs = self._address_fields(prior_address, pixel_address, flows, masks)
        coarse = self.max_shift*torch.tanh(self.offset_net(inputs))
        dense = F.interpolate(coarse, size=(h,w), mode="bilinear", align_corners=False)
        offsets = dense.reshape(b,t,2,2,h,w).permute(0,2,3,1,4,5).contiguous()
        offsets = offsets*masks.float()
        refined_flow = flows+offsets
        valid_input = masks.bool() & _geometric_mask(refined_flow, self.offset)
        corrections, valid, _ = differentiable_raw_candidates(low, base, refined_flow, valid_input,
                                                               offset=self.offset)
        prediction, features, weights = self._selector(low, base, corrections, valid)
        aux = {"mean_abs_offset": float(offsets.detach().abs().mean()),
               "max_abs_offset": float(offsets.detach().abs().amax()),
               "original_valid_fraction": float(masks.float().mean()),
               "refined_valid_fraction": float(valid_input.float().mean()),
               "detail_valid_fraction": float(valid.float().mean())}
        if return_aux:
            aux.update(refined_flows=refined_flow, candidate_masks=valid,
                       conservative_masks=valid_input, corrections=corrections,
                       features=features, weights=weights)
        return {"pred_pixels": prediction, "offsets": offsets, "aux": aux}
