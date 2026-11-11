"""Two finite Jacobi steps on an observation-conditioned, undirected pixel graph.

The energy is 1/2 sum_i ||z_i-a_i||^2 + 1/2 sum_e w_e ||z_u-z_v||^2.
Every bilinear corner is a separate pair; this is NOT ||z_t-W z_s||^2.
Both ends of every pair receive the same weight. Positive unit anchor precision
makes I+L SPD. We execute two iterations, not a converged optimizer. All inputs
are observable; source GT belongs only in the caller's reconstruction loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("conditioned", "node_only", "fixed_edge", "no_cross", "scrambled_cross")


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _flat(video):
    return video.permute(0, 2, 3, 4, 1).reshape(-1, video.shape[1])


def _video(flat, shape):
    b, t, h, w = shape
    return flat.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()


def lowpass(video):
    b, c, t, h, w = video.shape
    _check(min(h, w) > 4, "LP9 reflect padding needs H,W > 4")
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    result = F.avg_pool2d(F.pad(frames, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
    return result.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def validate_geometry_only(flows, masks, shape):
    """Cheap node-only validation without allocating the eight-corner graph."""
    b, t, h, w = shape
    _check(flows.shape == (b, 2, 2, t, h, w) and masks.shape == (b, 2, 1, t, h, w), "Invalid adjacent geometry shape")
    _check(t >= 2 and flows.device.type == "cpu" and flows.dtype == torch.float32
           and masks.device == flows.device, "CPU float32 adjacent flows required")
    _check(bool(torch.isfinite(flows).all()) and bool(torch.isfinite(masks).all())
           and bool(((masks == 0) | (masks == 1)).all()), "Invalid geometry values/masks")
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for direction, delta in enumerate((-1, 1)):
        source_time = torch.arange(t) + delta
        x, y = xx[None, None] + flows[:, direction, 0], yy[None, None] + flows[:, direction, 1]
        geometric = ((source_time >= 0) & (source_time < t))[None, :, None, None]
        geometric = geometric & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        _check(not bool((masks[:, direction, 0].bool() & ~geometric).any()), "Mask marks invalid time/image coordinates as valid")


def build_bilinear_graph(flows, masks):
    """CPU graph for adjacent [-1,+1] target-to-source pixel flows.

    flows [B,2,2,T,H,W], masks [B,2,1,T,H,W]. Returned source indices
    [2,B*T*H*W,4] address flattened (batch,time,y,x) nodes. Corners are
    (floor y/floor x, floor y/ceil x, ceil y/floor x, ceil y/ceil x).
    Reciprocal directed observations remain separate parallel undirected edges.
    """
    _check(flows.ndim == 6 and flows.shape[1:3] == (2, 2), "Invalid adjacent flow shape")
    b, _, _, t, h, w = flows.shape
    _check(min(b, t, h, w) > 0 and t >= 2, "Need positive dimensions and at least two frames")
    _check(flows.device.type == "cpu" and flows.dtype == torch.float32, "Graph construction requires CPU float32")
    _check(masks.shape == (b, 2, 1, t, h, w) and masks.device == flows.device, "Invalid graph mask shape/device")
    _check(bool(torch.isfinite(flows).all()) and bool(torch.isfinite(masks).all()), "Nonfinite graph geometry")
    _check(bool(((masks == 0) | (masks == 1)).all()), "Graph masks must be binary")
    # Geometry is frozen. Only learned precision weights carry gradients.
    flows, masks = flows.detach(), masks.detach().bool()
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    times = torch.arange(t)
    batches = torch.arange(b)[:, None, None, None]
    all_indices, all_weights = [], []
    for direction, delta in enumerate((-1, 1)):
        x = xx[None, None] + flows[:, direction, 0]
        y = yy[None, None] + flows[:, direction, 1]
        source_time = times + delta
        valid_time = ((source_time >= 0) & (source_time < t))[None, :, None, None]
        geometric = valid_time & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        valid = masks[:, direction, 0]
        _check(not bool((valid & ~geometric).any()), "Mask marks invalid time/image coordinates as valid")
        x0, y0 = x.floor(), y.floor()
        dx, dy = x - x0, y - y0
        x0, y0 = x0.long().clamp(0, w - 1), y0.long().clamp(0, h - 1)
        x1, y1 = (x.floor().long() + 1).clamp(0, w - 1), (y.floor().long() + 1).clamp(0, h - 1)
        temporal = (batches * t + source_time.clamp(0, t - 1)[None, :, None, None]) * h * w
        indices = torch.stack((temporal + y0 * w + x0, temporal + y0 * w + x1,
                               temporal + y1 * w + x0, temporal + y1 * w + x1), -1)
        weights = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), -1)
        weights = weights.clamp(0, 1) * valid[..., None]
        all_indices.append(indices.reshape(-1, 4))
        all_weights.append(weights.reshape(-1, 4))
    return {"shape": (b, t, h, w), "source_index": torch.stack(all_indices),
            "bilinear_weight": torch.stack(all_weights), "mask": masks}


def directed_gather(video, graph):
    """Masked bilinear W, returning [B,2,C,T,H,W]."""
    flat = _flat(video)
    outputs = []
    for indices, weights in zip(graph["source_index"], graph["bilinear_weight"]):
        selected = flat[indices.reshape(-1)].reshape(len(flat), 4, -1)
        outputs.append(_video((selected * weights[..., None]).sum(1), graph["shape"]))
    return torch.stack(outputs, 1)


def directed_adjoint(values, graph):
    """Exact transpose W^T, summed over the two directed observation fields."""
    b, t, h, w = graph["shape"]
    _check(values.shape[:2] == (b, 2) and values.shape[3:] == (t, h, w), "Invalid adjoint shape")
    result = values.new_zeros((b * t * h * w, values.shape[2]))
    for direction, (indices, weights) in enumerate(zip(graph["source_index"], graph["bilinear_weight"])):
        messages = _flat(values[:, direction])[:, None, :] * weights[..., None]
        result = result.index_add(0, indices.reshape(-1), messages.reshape(-1, values.shape[2]))
    return _video(result, graph["shape"])


def _weighted_edges(graph, precision):
    b, t, h, w = graph["shape"]
    _check(precision.shape == (b, 2, 1, t, h, w), "Expected precision [B,2,1,T,H,W]")
    _check(precision.device.type == "cpu" and precision.dtype == torch.float32, "CPU float32 precision required")
    _check(bool(torch.isfinite(precision).all()) and bool(((precision >= 0) & (precision <= 1)).all()),
           "Edge precision must be finite and in [0,1]")
    scale = precision[:, :, 0].permute(1, 0, 2, 3, 4).reshape(2, -1, 1)
    return graph["bilinear_weight"] * scale


def _degree(graph, weights):
    result = weights.sum((0, 2))
    for indices, values in zip(graph["source_index"], weights):
        result = result.index_add(0, indices.reshape(-1), values.reshape(-1))
    return result


def _adjacency(flat, graph, weights):
    result = torch.zeros_like(flat)
    for indices, values in zip(graph["source_index"], weights):
        donor = flat[indices.reshape(-1)].reshape(len(flat), 4, -1)
        result = result + (values[..., None] * donor).sum(1)
        incoming = values[..., None] * flat[:, None, :]
        result = result.index_add(0, indices.reshape(-1), incoming.reshape(-1, flat.shape[1]))
    return result


def pair_laplacian(video, graph, precision):
    """Symmetric graph Laplacian D-A, including outgoing AND incoming terms."""
    weights = _weighted_edges(graph, precision)
    flat = _flat(video)
    return _video(_degree(graph, weights)[:, None] * flat - _adjacency(flat, graph, weights), graph["shape"])


def reconstruct_two_steps(anchor, graph, precision):
    """Exactly two Jacobi iterations for (I+L)z=a, starting at z0=a."""
    _check(anchor.device.type == "cpu" and anchor.dtype == torch.float32, "CPU float32 anchor required")
    _check((anchor.shape[0], *anchor.shape[2:]) == graph["shape"], "Graph/anchor dimensions differ")
    weights = _weighted_edges(graph, precision)
    degree = _degree(graph, weights)
    a = _flat(anchor)
    state = a
    for _ in range(2):
        state = (a + _adjacency(state, graph, weights)) / (1 + degree[:, None])
    return _video(state, graph["shape"]), _video(degree[:, None], graph["shape"])


def _pool(video, patch):
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return F.avg_pool2d(frames, patch, patch).reshape(b, t, c, h // patch, w // patch)


def observable_edge_features(detail_observations, graph, patch=8):
    """Own5 / other5 channels; no GT, oracle moments or learned output input.

    Own: RGB mean absolute observation discrepancy, discrepancy energy, validity.
    Other: RGB mean discrepancy between donors, normalized residual correlation,
    other donor validity. Correlation uses only the joint valid support.
    """
    aligned = directed_gather(detail_observations, graph)
    valid = graph["mask"].float()
    residual = (aligned - detail_observations[:, None]) * valid
    own, other = [], []
    joint = valid[:, 0] * valid[:, 1]
    for j in range(2):
        k = 1 - j
        own.append(torch.cat((_pool(residual[:, j].abs(), patch),
                              _pool(residual[:, j].square().mean(1, keepdim=True), patch),
                              _pool(valid[:, j], patch)), dim=2))
        x, y = residual[:, j] * joint, residual[:, k] * joint
        dot = _pool((x * y).mean(1, keepdim=True), patch)
        ex, ey = (_pool(v.square().mean(1, keepdim=True), patch) for v in (x, y))
        correlation = (dot / (ex * ey + 1e-12).sqrt()).clamp(-1, 1)
        other.append(torch.cat((_pool((aligned[:, j] - aligned[:, k]).abs() * joint, patch),
                                correlation, _pool(valid[:, k], patch)), dim=2))
    return torch.stack(own, 1), torch.stack(other, 1)


class TrajectoryReconstruction(nn.Module):
    """Common node network plus explicitly conditioned graph precision.

    Graph modes share allocated parameters. node_only/fixed_edge intentionally
    deactivate the edge head and are ablations, not equal-active-capacity claims.
    conditioned and scrambled_cross keep identical active parameter counts.
    """
    def __init__(self, feature_channels=19, hidden=32, coupling=1.0, mode="conditioned", patch=8):
        super().__init__()
        _check(mode in MODES and coupling == 1.0 and patch == 8, "Use declared mode/coupling1/patch8")
        self.feature_channels, self.hidden, self.mode, self.patch = feature_channels, hidden, mode, patch
        self.node = nn.Sequential(nn.Conv2d(feature_channels, hidden, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(self.node[-1].weight)
        nn.init.zeros_(self.node[-1].bias)
        self.edge = nn.Sequential(nn.Conv2d(10, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)

    def get_config(self):
        return {"schema_version": 1, "feature_channels": self.feature_channels, "hidden": self.hidden,
                "mode": self.mode, "patch": self.patch, "coupling": 1.0, "anchor_precision": 1.0,
                "steps": 2, "offsets": [-1, 1], "state": "RGB node_prediction minus LP9(base)",
                "energy": "unit anchored undirected bilinear-corner pair Laplacian; not squared gather residual",
                "edge_precision": "sigmoid [0,1]; fixed_edge=0.5; node_only=0",
                "cross_scramble": "other5 features cyclic shift T//2; own,node,geometry unchanged",
                "backend": "CPU float32", "node_initialization": "equal thirds",
                "edge_initialization": "constant 0.5; graph/node-only initial outputs can differ"}

    def forward(self, base, corrections, node_features, detail_observations,
                adjacent_flows, adjacent_masks, return_aux=False):
        _check(base.ndim == 5 and base.shape[1] == 3, "Expected RGB base [B,3,T,H,W]")
        b, _, t, h, w = base.shape
        _check(min(h, w) >= self.patch and h % self.patch == 0 and w % self.patch == 0, "Complete patch8 grid required")
        _check(corrections.shape == (b, 2, 3, t, h, w) and detail_observations.shape == base.shape,
               "Correction/observation shape differs")
        _check(node_features.shape == (b, t, self.feature_channels, h // self.patch, w // self.patch), "Node feature shape differs")
        for value in (base, corrections, node_features, detail_observations):
            _check(value.device.type == "cpu" and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                   "Finite CPU float32 observable tensors required")
        _check(bool(((base >= 0) & (base <= 1)).all()), "Base must lie in [0,1]")
        logits = self.node(node_features.reshape(b * t, self.feature_channels, h // self.patch, w // self.patch))
        mixture = torch.softmax(torch.cat((torch.zeros_like(logits[:, :1]), logits), 1), 1)[:, 1:]
        mixture = F.interpolate(mixture, size=(h, w), mode="nearest").reshape(b, t, 2, h, w).permute(0, 2, 1, 3, 4)
        node_prediction = (base + (mixture[:, :, None] * corrections).sum(1)).clamp(0, 1)
        if self.mode == "node_only":
            validate_geometry_only(adjacent_flows, adjacent_masks, (b, t, h, w))
            aux = {"mean_edge_precision": 0., "mean_degree": 0., "mean_abs_state_update": 0., "steps": 0}
            if return_aux:
                anchor = node_prediction - lowpass(base)
                aux.update(node_prediction=node_prediction, anchor=anchor, state=anchor,
                           edge_precision=torch.zeros_like(adjacent_masks, dtype=torch.float32),
                           degree=torch.zeros_like(base[:, :1]), node_weights=mixture)
            return {"pred_pixels": node_prediction, "trajectory_aux": aux}
        graph = build_bilinear_graph(adjacent_flows, adjacent_masks)
        _check(graph["shape"] == (b, t, h, w), "Base/adjacent-flow dimensions differ")
        own, other = observable_edge_features(detail_observations, graph, self.patch)
        if self.mode == "no_cross":
            other = torch.zeros_like(other)
        elif self.mode == "scrambled_cross":
            other = torch.roll(other, shifts=t // 2, dims=2)
        edge_inputs = torch.cat((own, other), 3).reshape(b * 2 * t, 10, h // self.patch, w // self.patch)
        precision = torch.sigmoid(self.edge(edge_inputs))
        precision = F.interpolate(precision, size=(h, w), mode="nearest").reshape(b, 2, t, 1, h, w).permute(0, 1, 3, 2, 4, 5)
        if self.mode == "node_only":
            precision = torch.zeros_like(precision)
        elif self.mode == "fixed_edge":
            precision = torch.full_like(precision, 0.5)
        base_low = lowpass(base)
        anchor = node_prediction - base_low
        state, degree = reconstruct_two_steps(anchor, graph, precision)
        prediction = node_prediction if self.mode == "node_only" else (base_low + state).clamp(0, 1)
        aux = {"mean_edge_precision": float(precision.detach().mean()),
               "mean_degree": float(degree.detach().mean()),
               "mean_abs_state_update": float((state - anchor).detach().abs().mean()), "steps": 2}
        if return_aux:
            aux.update(node_prediction=node_prediction, anchor=anchor, state=state,
                       edge_precision=precision, degree=degree, node_weights=mixture)
        return {"pred_pixels": prediction, "trajectory_aux": aux}
