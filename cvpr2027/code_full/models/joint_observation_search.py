"""Explicit paired correspondence search over fixed observable costs.

The learned quantities are four scalars, not an RGB decoder or offset CNN.
Effective observation-error calibration is supplied by a source-only cache;
it is not a claim of independent, physically calibrated camera noise.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


_spec = importlib.util.spec_from_file_location(
    "_joint_search_frozen_geometry", Path(__file__).with_name("observation_geometry.py"))
_G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_G)
_check = _G._check
SHIFTS_XY = tuple((dx, dy) for dy in (-2, 0, 2) for dx in (-2, 0, 2))
AMPLITUDE_RULE = "tanh_projected_nonnegative_zero_init"
CONFIDENCE_RULE = "one_minus_marginal_entropy_over_log_valid_support"


def _masked_distribution(cost, valid, temperature):
    """A proper zero distribution when no candidate is valid; never NaN."""
    logits = (-cost / temperature).masked_fill(~valid, -1e30)
    probability = torch.softmax(logits, -1) * valid.float()
    return probability / probability.sum(-1, keepdim=True).clamp_min(1e-30)


def _confidence(probability, support):
    count = support.sum(-1)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
    concentration = (1 - entropy / count.clamp_min(2).float().log()).clamp(0, 1)
    return torch.where(count == 1, torch.ones_like(concentration), concentration) * (count > 0)


class JointObservationSearch(nn.Module):
    """K=9, row-major (dy then dx), pair axes=(past candidate,future candidate).

    Costs: unary[B,2,T,N,9], pair[B,T,N,9,9], N=(H/16)*(W/16).
    `prepared` has unary_observation, pair_observation, unary_constant,
    pair_constant, descriptor_costs (prior/pixel/shuffled_prior), valid_unary,
    valid_pair, and optionally metadata. All costs are fixed CPU float32;
    validity is bool. Descriptor/calibration provenance belongs to the cache.
    Both modes use identical pair feasibility; independent is unary-only
    evidence under that feasibility, and factorizes only on Cartesian support.
    Both modes use single-donor inference where the other donor
    has no evidence. If both have evidence but no pair is valid, both abstain.

    Offset = tanh(clamp(amplitude_raw,0,4)) * confidence * posterior expected shift,
    then bilinear upsampling and original pixel validity. Negative amplitude
    is disallowed. Call project_parameters_() after each optimizer update.
    Only actual observed pixels are sampled, once, by the frozen RGB selector.
    """
    def __init__(self, normalization, frozen_node_state, search_mode="joint",
                 address_mode="prior", uncertainty_mode="calibrated", *,
                 search_patch=16, offset=2, max_shift=2.0,
                 amplitude_rule=AMPLITUDE_RULE, confidence_rule=CONFIDENCE_RULE):
        super().__init__()
        _check(search_mode in ("joint", "independent"), "Unknown search mode")
        _check(address_mode in _G.ADDRESS_MODES, "Unknown address mode")
        _check(uncertainty_mode in ("calibrated", "constant"), "Unknown uncertainty mode")
        _check(search_patch == 16 and offset == 2 and max_shift == 2,
               "Fixed search patch16, temporal offset2, and spatial bound2 required")
        _check(amplitude_rule == AMPLITUDE_RULE and confidence_rule == CONFIDENCE_RULE,
               "Search rules must match the declared frozen recipe")
        mean = torch.tensor(normalization["feature_mean"], dtype=torch.float32)
        std = torch.tensor(normalization["feature_std"], dtype=torch.float32)
        _check(mean.shape == std.shape == (19,) and bool(torch.isfinite(mean).all())
               and bool(torch.isfinite(std).all()) and bool((std > 0).all()),
               "Frozen normalization requires 19 finite means/positive stds")
        self.register_buffer("feature_mean", mean.reshape(1, 1, 19, 1, 1))
        self.register_buffer("feature_std", std.reshape(1, 1, 19, 1, 1))
        self.register_buffer("shifts_xy", torch.tensor(SHIFTS_XY, dtype=torch.float32))
        self.search_mode, self.address_mode = search_mode, address_mode
        self.uncertainty_mode = uncertainty_mode
        self.search_patch, self.offset, self.max_shift = search_patch, offset, float(max_shift)
        self.node = nn.Sequential(nn.Conv2d(19, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 1), nn.ReLU(), nn.Conv2d(32, 2, 1))
        expected, state = set(self.node.state_dict()), dict(frozen_node_state)
        if any(key.startswith("node.") for key in state):
            required = {"node." + key for key in expected}
            allowed = required | {"edge.0.weight", "edge.0.bias", "edge.2.weight", "edge.2.bias"}
            _check(required <= set(state) <= allowed, "Unexpected/missing frozen state keys")
            state = {k.removeprefix("node."): v for k, v in state.items() if k.startswith("node.")}
        _check(set(state) == expected, "Frozen node requires its exact six tensors")
        _check(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                   and bool(torch.isfinite(v).all()) for v in state.values()), "Invalid frozen node state")
        self.node.load_state_dict(state, strict=True)
        self.node.requires_grad_(False)
        self.node.eval()
        self.amplitude_raw = nn.Parameter(torch.tensor(0.))
        self.descriptor_log_weight = nn.Parameter(torch.tensor(0.))
        self.regularization_log_weight = nn.Parameter(torch.tensor(math.log(.05)))
        self.log_temperature = nn.Parameter(torch.tensor(0.))

    def train(self, mode=True):
        super().train(mode)
        self.node.eval()
        return self

    @torch.no_grad()
    def project_parameters_(self):
        self.amplitude_raw.clamp_(0, 4)
        self.descriptor_log_weight.clamp_(-4, 4)
        self.regularization_log_weight.clamp_(-6, 2)
        self.log_temperature.clamp_(-4, 4)
        return self

    def get_config(self):
        """Constructor kwargs; frozen node weights are deliberately separate."""
        return {"normalization": {"feature_mean": self.feature_mean.flatten().tolist(),
                                   "feature_std": self.feature_std.flatten().tolist()},
                "search_mode": self.search_mode, "address_mode": self.address_mode,
                "uncertainty_mode": self.uncertainty_mode, "search_patch": self.search_patch,
                "offset": self.offset, "max_shift": self.max_shift,
                "amplitude_rule": AMPLITUDE_RULE, "confidence_rule": CONFIDENCE_RULE}

    def _validate_prepared(self, prepared, shape):
        b, t, n = shape
        unary_shape, pair_shape = (b, 2, t, n, 9), (b, t, n, 9, 9)
        required = {"unary_observation", "pair_observation", "unary_constant", "pair_constant",
                    "descriptor_costs", "valid_unary", "valid_pair"}
        _check(isinstance(prepared, dict) and required <= set(prepared) <= required | {"metadata"},
               "Exact observable prepared fields required; GT/labels are not inputs")
        descriptors = prepared["descriptor_costs"]
        _check(isinstance(descriptors, dict) and set(descriptors) == set(_G.ADDRESS_MODES),
               "All three declared descriptor costs must be present")
        costs = [(prepared[k], unary_shape) for k in ("unary_observation", "unary_constant")]
        costs += [(prepared[k], pair_shape) for k in ("pair_observation", "pair_constant")]
        costs += [(v, unary_shape) for v in descriptors.values()]
        for value, expected in costs:
            _check(isinstance(value, torch.Tensor) and value.shape == expected
                   and value.dtype == torch.float32 and value.device.type == "cpu"
                   and bool(torch.isfinite(value).all()) and bool((value >= 0).all()),
                   "Fixed finite nonnegative CPU float32 costs have wrong shape/type")
        for key, expected in (("valid_unary", unary_shape), ("valid_pair", pair_shape)):
            value = prepared[key]
            _check(isinstance(value, torch.Tensor) and value.shape == expected
                   and value.dtype == torch.bool and value.device.type == "cpu", "Validity must be CPU bool")
        vu, vp = prepared["valid_unary"], prepared["valid_pair"]
        _check(not bool((vp & ~(vu[:, 0, ..., :, None] & vu[:, 1, ..., None, :])).any()),
               "Pair validity cannot exceed its unary supports")

    def correspondence_posterior(self, prepared):
        """Validated fixed costs -> differentiable marginals and confidence."""
        shape = prepared["unary_observation"].shape
        _check(len(shape) == 5, "Unary costs must have five dimensions")
        self._validate_prepared(prepared, (shape[0], shape[2], shape[3]))
        suffix = "observation" if self.uncertainty_mode == "calibrated" else "constant"
        unary = prepared["unary_" + suffix].detach()
        pair = prepared["pair_" + suffix].detach()
        descriptor = prepared["descriptor_costs"][self.address_mode].detach()
        valid = prepared["valid_unary"].detach()
        pair_valid = prepared["valid_pair"].detach()
        wdesc = self.descriptor_log_weight.clamp(-4, 4).exp()
        wreg = self.regularization_log_weight.clamp(-6, 2).exp()
        temperature = self.log_temperature.clamp(-4, 4).exp()
        cost = unary + wdesc * descriptor + wreg * self.shifts_xy.square().sum(-1)
        independent = _masked_distribution(cost, valid, temperature)
        energy = cost[:, 0, ..., :, None] + cost[:, 1, ..., None, :]
        if self.search_mode == "joint":
            energy = energy + pair
        joint = _masked_distribution(energy.flatten(-2), pair_valid.flatten(-2), temperature)
        joint = joint.reshape_as(energy)
        marginals = torch.stack((joint.sum(-1), joint.sum(-2)), 1)
        support = torch.stack((pair_valid.any(-1), pair_valid.any(-2)), 1)
        available = valid.any(-1)
        only_one = available & ~available.flip(1)
        marginals = torch.where(only_one[..., None], independent, marginals)
        support = torch.where(only_one[..., None], valid, support)
        confidence = _confidence(marginals, support)
        expected = torch.einsum("bdtnk,kc->bdtnc", marginals, self.shifts_xy)
        return {"marginals": marginals, "confidence": confidence, "expected_shifts": expected,
                "descriptor_weight": wdesc, "regularization_weight": wreg, "temperature": temperature}

    def forward(self, low, base, flows, masks, prepared, *, return_aux=False):
        _G._CANDIDATES._validate(low, base, flows, masks, self.offset)
        _check(low.device.type == self.feature_mean.device.type == "cpu", "Search runs on CPU float32")
        _check(isinstance(return_aux, bool), "return_aux must be bool")
        b, _, t, h, w = base.shape
        _check(h % 16 == 0 and w % 16 == 0, "Full frame must contain complete patch16 cells")
        _check(not bool((masks.bool() & ~_G._geometric_mask(flows, self.offset)).any()),
               "Original validity exceeds geometric support")
        self._validate_prepared(prepared, (b, t, (h//16)*(w//16)))
        low, base, flows, masks = (v.detach() for v in (low, base, flows, masks))
        posterior = self.correspondence_posterior(prepared)
        amplitude = self.amplitude_raw.clamp(0, 4).tanh()
        coarse = amplitude * posterior["confidence"][..., None] * posterior["expected_shifts"]
        coarse = coarse.reshape(b, 2, t, h//16, w//16, 2).permute(0, 1, 2, 5, 3, 4)
        dense = F.interpolate(coarse.reshape(b*2*t, 2, h//16, w//16), size=(h,w),
                              mode="bilinear", align_corners=False)
        offsets = dense.reshape(b, 2, t, 2, h, w).permute(0, 1, 3, 2, 4, 5) * masks.float()
        refined = flows + offsets
        conservative = masks.bool() & _G._geometric_mask(refined, self.offset)
        corrections, valid, _ = _G.differentiable_raw_candidates(low, base, refined, conservative)
        prediction, features, weights = _G.ObservationGeometry._selector(self, low, base, corrections, valid)
        aux = {"amplitude": float(amplitude.detach()),
               "mean_abs_offset": float(offsets.detach().abs().mean()),
               "max_abs_offset": float(offsets.detach().abs().amax()),
               "mean_confidence": float(posterior["confidence"].detach().mean()),
               "descriptor_weight": float(posterior["descriptor_weight"].detach()),
               "regularization_weight": float(posterior["regularization_weight"].detach()),
               "temperature": float(posterior["temperature"].detach()),
               "refined_valid_fraction": float(conservative.float().mean())}
        if return_aux:
            aux.update(posterior, refined_flows=refined, conservative_masks=conservative,
                       candidate_masks=valid, corrections=corrections, features=features, weights=weights)
        return {"pred_pixels": prediction, "offsets": offsets, "aux": aux}
