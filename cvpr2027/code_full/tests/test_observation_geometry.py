"""CPU parity/gradient/firewall tests for bounded observation geometry."""
import importlib.util
import inspect
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1] / "models"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is not installed")
class ObservationGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load("_test_observation_geometry", "observation_geometry.py")
        cls.old = load("_test_geometry_trajectory_reference", "trajectory_reconstruction.py")

    def setUp(self):
        torch.manual_seed(7)
        self.normalization = {"feature_mean": [.03]*19, "feature_std": [.7]*19}
        self.reference = self.old.TrajectoryReconstruction(mode="node_only").eval()
        with torch.no_grad():
            self.reference.node[-1].weight.normal_(0, .05)
            self.reference.node[-1].bias.copy_(torch.tensor([.1, -.1]))
        self.frozen = {k: v.clone() for k, v in self.reference.state_dict().items()}

    def fixture(self, batch=1, frames=6, height=32):
        low = torch.rand(batch, 3, frames, height, height)*.35+.03
        base = torch.rand_like(low)*.4+.3
        flows = torch.full((batch, 2, 2, frames, height, height), .2)
        masks = torch.zeros(batch, 2, 1, frames, height, height, dtype=torch.bool)
        masks[:, 0, :, 2:, 1:-1, 1:-1] = True
        masks[:, 1, :, :-2, 1:-1, 1:-1] = True
        prior = torch.randn(batch, frames, (height//16)**2, 8)
        pixel = torch.randn_like(prior)
        return low, base, flows, masks, prior, pixel

    def model(self, mode="prior", predictor_patch=16):
        return self.m.ObservationGeometry(self.normalization, self.frozen,
            address_mode=mode, descriptor_dim=8, hidden_dim=4, address_dim=4,
            predictor_patch=predictor_patch)

    def old_prediction(self, inputs):
        low, base, flows, masks, _, _ = inputs
        corrections, validity, _ = self.m._CANDIDATES.build_observation_candidates(
            low, base, flows, masks, mode="raw_detail", offset=2)
        features, _ = self.m._FEATURES.observable_patch_features(low, base, corrections, validity)
        mean = torch.tensor(self.normalization["feature_mean"])[None, None, :, None, None]
        std = torch.tensor(self.normalization["feature_std"])[None, None, :, None, None]
        # Node-only graph arguments are validated but do not enter its output.
        adjacent = torch.zeros_like(flows)
        adjacent_masks = torch.zeros_like(masks)
        adjacent_masks[:, 0, :, 1:] = True
        adjacent_masks[:, 1, :, :-1] = True
        return self.reference(base, corrections, (features-mean)/std,
                              torch.zeros_like(base), adjacent, adjacent_masks)["pred_pixels"]

    def test_candidate_exact_parity_and_flow_derivative(self):
        low, base, flow, masks, _, _ = self.fixture()
        expected = self.m._CANDIDATES.build_observation_candidates(low, base, flow, masks,
                                                                   mode="raw_detail", offset=2)
        variable = flow.clone().requires_grad_()
        actual = self.m.differentiable_raw_candidates(low, base, variable, masks)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(actual[2], expected[2])
        derivative, = torch.autograd.grad(actual[0].square().mean(), variable)
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        self.assertGreater(float(derivative.abs().sum()), 0)

    def test_features_exact_parity_and_correction_derivative(self):
        low, base, flow, masks, _, _ = self.fixture()
        corrections, valid, _ = self.m.differentiable_raw_candidates(low, base, flow, masks)
        expected = self.m._FEATURES.observable_patch_features(low, base, corrections, valid)
        variable = corrections.clone().requires_grad_()
        actual = self.m.differentiable_patch_features(low, base, variable, valid)
        for result, reference in zip(actual, expected):
            self.assertTrue(torch.equal(result, reference))
        derivative, = torch.autograd.grad(actual[0].square().sum()+actual[1].square().sum(), variable)
        self.assertTrue(bool(torch.isfinite(derivative).all()))
        self.assertGreater(float(derivative.abs().sum()), 0)

    def test_zero_offset_exact_frozen_anchor_for_all_modes_and_predictor_patches(self):
        inputs = self.fixture(batch=2)
        expected = self.old_prediction(inputs)
        for mode in self.m.ADDRESS_MODES:
            for patch in (8, 16):
                model = self.model(mode, patch)
                actual = model(*inputs)
                self.assertTrue(torch.equal(actual["offsets"], torch.zeros_like(actual["offsets"])))
                self.assertTrue(torch.equal(actual["pred_pixels"], expected))
                self.assertFalse(torch.equal(actual["pred_pixels"], inputs[1]))

    def test_same_mode_initial_state_and_config_reload(self):
        inputs = self.fixture()
        state = None
        for mode in self.m.ADDRESS_MODES:
            torch.manual_seed(13)
            model = self.model(mode)
            if state is None:
                state = {k:v.clone() for k,v in model.state_dict().items()}
            self.assertEqual(set(state), set(model.state_dict()))
            self.assertTrue(all(torch.equal(state[k],v) for k,v in model.state_dict().items()))
            config = model.get_config()
            self.assertNotIn("frozen_node_state", config)
            restored = self.m.ObservationGeometry(frozen_node_state=self.frozen, **config)
            restored.load_state_dict(model.state_dict(), strict=True)
            self.assertTrue(torch.equal(restored(*inputs)["pred_pixels"], model(*inputs)["pred_pixels"]))

    def test_geometry_bounds_conservative_validity_and_no_invalid_output(self):
        inputs = self.fixture()
        model = self.model()
        with torch.no_grad():
            model.offset_net[-1].bias.copy_(torch.tensor([100., -100.]))
        result = model(*inputs, return_aux=True)
        self.assertLessEqual(float(result["offsets"].abs().max()), 2.)
        self.assertTrue(torch.equal(result["offsets"]*(~inputs[3]).float(), torch.zeros_like(result["offsets"])))
        valid = result["aux"]["conservative_masks"]
        self.assertFalse(bool((valid & ~inputs[3]).any()))
        self.assertLess(int(valid.sum()), int(inputs[3].sum()))
        geometric = self.m._geometric_mask(result["aux"]["refined_flows"], 2)
        self.assertFalse(bool((valid & ~geometric).any()))
        corrections = result["aux"]["corrections"]
        self.assertTrue(torch.equal(corrections*(~result["aux"]["candidate_masks"]).float(),
                                    torch.zeros_like(corrections)))

    def test_direction_frame_and_component_layout(self):
        inputs = list(self.fixture())
        inputs[2].zero_()
        inputs[3].zero_()
        inputs[3][:, 0, :, 2:] = True
        inputs[3][:, 1, :, :-2] = True
        inputs[4].zero_()
        for t in range(6):
            inputs[4][:, t, :, 0] = t
        model = self.model()
        with torch.no_grad():
            model.address_projection.weight.zero_()
            model.address_projection.weight[0, 0, 0, 0] = 1
            model.offset_net[0].weight.zero_()
            model.offset_net[0].bias.zero_()
            model.offset_net[0].bias[0] = 10
            model.offset_net[0].weight[0, 2*model.address_dim, 1, 1] = 1
            model.offset_net[-1].weight.zero_()
            model.offset_net[-1].weight[0, 0, 0, 0] = .01
        offsets = model(*inputs)["offsets"]
        # At t=2: past descriptor=0, own=2, future=4. No axis/time exchange.
        torch.testing.assert_close(offsets[0, 0, 0, 2, 16, 16], 2*torch.tanh(torch.tensor(.08)))
        torch.testing.assert_close(offsets[0, 1, 0, 2, 16, 16], 2*torch.tanh(torch.tensor(.12)))
        self.assertTrue(torch.equal(offsets[:, :, 1], torch.zeros_like(offsets[:, :, 1])))

    def test_optimizer_updates_geometry_only_and_frozen_node_is_unchanged(self):
        inputs = self.fixture()
        target = inputs[1]*.8+.08
        for mode in self.m.ADDRESS_MODES:
            model = self.model(mode).train()
            self.assertFalse(model.node.training)
            before = {k:v.clone() for k,v in model.node.state_dict().items()}
            initial = model(*inputs)["pred_pixels"].detach()
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = model(*inputs)
                loss = (result["pred_pixels"]-target).square().mean()
                loss.backward()
                for p in model.parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(bool(torch.isfinite(p.grad).all()))
                    else:
                        self.assertIsNone(p.grad)
                optimizer.step()
            self.assertGreater(float(model.address_projection.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.offset_net[-1].weight.grad.abs().sum()), 0)
            self.assertFalse(torch.equal(model(*inputs)["pred_pixels"], initial))
            self.assertTrue(all(torch.equal(before[k], v) for k,v in model.node.state_dict().items()))

    def test_all_invalid_evidence_is_exact_base_even_with_nonzero_offsets(self):
        inputs = list(self.fixture())
        inputs[3].zero_()
        model = self.model()
        with torch.no_grad():
            model.offset_net[-1].bias.fill_(1.)
        result = model(*inputs)
        self.assertTrue(torch.equal(result["pred_pixels"], inputs[1]))
        self.assertTrue(torch.equal(result["offsets"], torch.zeros_like(result["offsets"])))

    def test_shuffle_rng_isolation_and_address_intervention(self):
        inputs = self.fixture()
        plain, shuffled = self.model(), self.model("shuffled_prior")
        shuffled.load_state_dict(plain.state_dict())
        before = torch.random.get_rng_state().clone()
        args = (inputs[4], inputs[5], inputs[2], inputs[3])
        first = shuffled._address_fields(*args)
        second = shuffled._address_fields(*args)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, plain._address_fields(*args)))

    def test_gt_firewall_and_invalid_inputs(self):
        names = set(inspect.signature(self.m.ObservationGeometry.forward).parameters)
        self.assertTrue(names.isdisjoint({"gt", "target", "oracle", "a", "gram", "error"}))
        inputs = list(self.fixture())
        bad = list(inputs)
        bad[3] = inputs[3].clone()
        bad[3][:, 0, :, 0, 4, 4] = True
        with self.assertRaises(ValueError):
            self.model()(*bad)
        with self.assertRaises(ValueError):
            self.m.ObservationGeometry(self.normalization, self.frozen, max_shift=3)
        with self.assertRaises(ValueError):
            self.m.ObservationGeometry({"feature_mean":[0]*19,"feature_std":[0]*19}, self.frozen)
        with self.assertRaises(ValueError):
            self.m.ObservationGeometry(self.normalization, dict(self.frozen, unknown=torch.tensor(0.)))
        with self.assertRaises(ValueError):
            self.model()(*inputs[:4], inputs[4][:, :, :-1], inputs[5])
        node_only = {k.removeprefix("node."):v for k,v in self.frozen.items() if k.startswith("node.")}
        model = self.m.ObservationGeometry(self.normalization, node_only)
        self.assertTrue(all(not p.requires_grad for p in model.node.parameters()))


if __name__ == "__main__":
    unittest.main()
