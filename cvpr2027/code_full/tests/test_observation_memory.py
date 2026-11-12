"""Small CPU contracts for the experimental observable-value memory module."""
import importlib.util
import inspect
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "observation_memory.py"


@unittest.skipIf(torch is None, "torch is not installed")
class ObservationMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_test_observation_memory", MODULE_PATH)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def setUp(self):
        torch.manual_seed(123)

    def make_inputs(self, frames=4, height=8, width=8):
        low = torch.rand(1, 3, frames, height, width) * .3
        base = torch.rand_like(low) * .5 + .2
        tokens = torch.randn(1, (frames // 2) * (height // 4) * (width // 4), 1024)
        fields = self.m.prepare_observation_memory_inputs(
            low, base, tokens, token_grid=(frames // 2, height // 4, width // 4),
            patch=4, descriptor_dim=8)
        return low, base, tokens, fields

    def model(self, address="prior", retrieval="cross_frame"):
        return self.m.ObservationMemory(address, retrieval, descriptor_dim=8,
                                        slots=3, address_dim=4, patch=4, hidden_dim=4)

    @staticmethod
    def forward(model, base, fields, **kwargs):
        return model(base, fields["detail_values"], fields["prior_address"],
                     fields["pixel_address"], **kwargs)

    def test_projection_exact_recipe_rng_and_constant_offset(self):
        tokens = torch.randn(2, 6, 1024)
        before = torch.random.get_rng_state().clone()
        actual = self.m.project_prior_tokens(tokens, descriptor_dim=8)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        generator = torch.Generator(device="cpu").manual_seed(1729)
        matrix = torch.randn(1024, 8, generator=generator) / (8 ** .5)
        expected = torch.nn.functional.normalize(
            (tokens - tokens.mean(1, keepdim=True)) @ matrix, dim=-1, eps=1e-6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        shifted = self.m.project_prior_tokens(tokens + 2, descriptor_dim=8)
        torch.testing.assert_close(actual, shifted, rtol=2e-5, atol=2e-6)
        zero = self.m.project_prior_tokens(torch.ones(1, 4, 1024), descriptor_dim=8)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))

    def test_projected_cache_parity_and_tubelet_repeat(self):
        low, base, tokens, fields = self.make_inputs()
        projected = self.m.project_prior_tokens(tokens, descriptor_dim=8)
        cached = self.m.prepare_observation_memory_inputs(
            low, base, projected, token_grid=(2, 2, 2), patch=4,
            descriptor_dim=8, prior_is_projected=True)
        for key in ("detail_values", "prior_address", "pixel_address"):
            self.assertTrue(torch.equal(fields[key], cached[key]))
        address = fields["prior_address"]
        self.assertTrue(torch.equal(address[:, 0], address[:, 1]))
        self.assertTrue(torch.equal(address[:, 2], address[:, 3]))
        self.assertFalse(torch.equal(address[:, 0], address[:, 2]))

    def test_calibration_values_and_patch_phase_are_preserved(self):
        low, base, _, fields = self.make_inputs()
        # Independent explicit frame-wise reference for the observable recipe.
        def lp(x):
            out = []
            for frame in range(x.shape[2]):
                padded = torch.nn.functional.pad(x[:, :, frame], (4, 4, 4, 4), mode="reflect")
                out.append(torch.nn.functional.avg_pool2d(padded, 9, 1))
            return torch.stack(out, dim=2)
        low_l, base_l = lp(low), lp(base)
        q = (base_l + (base_l / (low_l + 1e-4)).clamp(.25, 16) * (low - low_l)).clamp(0, 1)
        expected = q - lp(q)
        reconstructed = self.m._unpatchify(fields["detail_values"], 8, 8, 4)
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
        for row in range(2):
            for column in range(2):
                expected_patch = expected[:, :, :, row*4:(row+1)*4, column*4:(column+1)*4]
                expected_patch = expected_patch.permute(0, 2, 1, 3, 4).reshape(1, 4, 48)
                torch.testing.assert_close(fields["detail_values"][:, :, row*2+column],
                                           expected_patch, rtol=0, atol=0)

    def test_crop_fields_without_recentring(self):
        _, base, _, fields = self.make_inputs(height=16, width=16)
        # Slice bottom-left 8x8 at the patch grid, keeping full-clip descriptors.
        cropped = {}
        for name in ("detail_values", "prior_address", "pixel_address"):
            value = fields[name]
            cropped[name] = value.reshape(1, 4, 4, 4, -1)[:, :, 2:4, :2].reshape(1, 4, 4, -1)
        output = self.forward(self.model(), base[:, :, :, 8:16, :8], cropped)
        self.assertTrue(torch.equal(output["pred_pixels"], base[:, :, :, 8:16, :8]))
        restored = self.m._unpatchify(cropped["detail_values"], 8, 8, 4)
        full = self.m._unpatchify(fields["detail_values"], 16, 16, 4)
        self.assertTrue(torch.equal(restored, full[:, :, :, 8:16, :8]))

    def test_all_modes_identical_initial_state_identity_and_reload(self):
        _, base, _, fields = self.make_inputs()
        reference = None
        for address, retrieval in (("prior", "cross_frame"), ("pixel", "cross_frame"),
                                    ("shuffled_prior", "cross_frame"), ("prior", "same_frame"),
                                    ("pixel", "local")):
            torch.manual_seed(8)
            model = self.model(address, retrieval)
            state = model.state_dict()
            if reference is None:
                reference = {key: value.clone() for key, value in state.items()}
            self.assertEqual(set(reference), set(state))
            self.assertTrue(all(torch.equal(value, reference[key]) for key, value in state.items()))
            output = self.forward(model, base, fields)
            self.assertTrue(torch.equal(output["pred_pixels"], base))
            self.assertTrue(all(isinstance(value, float) for value in output["aux"].values()))
            restored = self.m.ObservationMemory(**model.get_config())
            restored.load_state_dict(state, strict=True)
            torch.testing.assert_close(self.forward(restored, base, fields)["memory_detail"],
                                       output["memory_detail"], rtol=0, atol=0)

    def test_shared_memory_matches_explicit_leave_one_out_sum(self):
        model = self.model()
        assignments = torch.softmax(torch.randn(1, 3, 4, 3), -1)
        values = torch.randn(1, 3, 4, 48)
        actual, _ = model._read(assignments, values)
        expected = torch.zeros_like(actual)
        for target in range(3):
            for patch in range(4):
                for slot in range(3):
                    numerator = torch.zeros(48)
                    denominator = torch.tensor(0.)
                    for source in range(3):
                        if source == target:
                            continue
                        for source_patch in range(4):
                            mass = assignments[0, source, source_patch, slot]
                            numerator += mass * values[0, source, source_patch]
                            denominator += mass
                    expected[0, target, patch] += assignments[0, target, patch, slot] * numerator / denominator
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_target_values_have_exactly_zero_memory_derivative(self):
        _, base, _, fields = self.make_inputs()
        values = fields["detail_values"].clone().requires_grad_()
        model = self.model()
        output = model(base, values, fields["prior_address"], fields["pixel_address"], return_aux=True)
        derivative, = torch.autograd.grad(output["aux"]["retrieved_values"][:, 1].sum(), values)
        self.assertTrue(torch.equal(derivative[:, 1], torch.zeros_like(derivative[:, 1])))
        self.assertGreater(float(derivative[:, [0, 2, 3]].abs().sum()), 0)
        changed = values.detach().clone()
        changed[:, 1] += 100
        changed_output = model(base, changed, fields["prior_address"], fields["pixel_address"], return_aux=True)
        self.assertTrue(torch.equal(output["aux"]["retrieved_values"][:, 1],
                                    changed_output["aux"]["retrieved_values"][:, 1]))

    def test_same_frame_memory_excludes_other_frame_values(self):
        assignments = torch.softmax(torch.randn(1, 4, 4, 3), -1)
        values = torch.randn(1, 4, 4, 48, requires_grad=True)
        read, _ = self.model(retrieval="same_frame")._read(assignments, values)
        derivative, = torch.autograd.grad(read[:, 1].sum(), values)
        self.assertTrue(torch.equal(derivative[:, [0, 2, 3]], torch.zeros_like(derivative[:, [0, 2, 3]])))
        self.assertGreater(float(derivative[:, 1].abs().sum()), 0)

    def test_local_read_time_edges_no_wrap_and_no_far_values(self):
        assignments = torch.full((1, 4, 4, 3), 1/3)
        values = torch.arange(4, dtype=torch.float32)[None, :, None, None].expand(1, 4, 4, 48).clone()
        values.requires_grad_()
        read, aux = self.model("pixel", "local")._read(assignments, values)
        expected = torch.tensor([1., 1., 2., 2.])[None, :, None, None].expand_as(read)
        torch.testing.assert_close(read, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(aux["local_weights"][:, 0, 0], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(aux["local_weights"][:, -1, 1], torch.zeros(1, 4)))
        derivative, = torch.autograd.grad(read[:, 1].sum(), values)
        self.assertTrue(torch.equal(derivative[:, [1, 3]], torch.zeros_like(derivative[:, [1, 3]])))

    def test_uniform_patch_pattern_is_not_spatially_averaged(self):
        pattern = torch.zeros(48)
        pattern[7] = 1
        values = pattern[None, None, None].expand(1, 4, 4, 48).clone()
        assignments = torch.softmax(torch.randn(1, 4, 4, 3), -1)
        for mode in ("cross_frame", "same_frame", "local"):
            read, _ = self.model(retrieval=mode)._read(assignments, values)
            torch.testing.assert_close(read, values, rtol=1e-6, atol=1e-6)

    def test_shuffle_is_deterministic_address_only_and_rng_isolated(self):
        _, base, _, fields = self.make_inputs()
        plain, shuffled = self.model(), self.model("shuffled_prior")
        shuffled.load_state_dict(plain.state_dict())
        before = torch.random.get_rng_state().clone()
        first = self.forward(shuffled, base, fields, return_aux=True)
        second = self.forward(shuffled, base, fields, return_aux=True)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first["aux"]["assignments"], second["aux"]["assignments"]))
        plain_output = self.forward(plain, base, fields, return_aux=True)
        self.assertFalse(torch.allclose(first["aux"]["assignments"], plain_output["aux"]["assignments"]))
        # A location shuffle preserves the multiset of unmodified assignments.
        first_sorted = first["aux"]["assignments"].reshape(-1, 3).sort(dim=0).values
        plain_sorted = plain_output["aux"]["assignments"].reshape(-1, 3).sort(dim=0).values
        torch.testing.assert_close(first_sorted, plain_sorted, rtol=0, atol=0)

    def test_real_optimizer_step_unblocks_address_and_slot_gradients(self):
        _, base, _, fields = self.make_inputs()
        target = (base * .8 + .1).detach()
        for address, retrieval in (("prior", "cross_frame"), ("pixel", "cross_frame"),
                                    ("shuffled_prior", "cross_frame"), ("prior", "same_frame"),
                                    ("pixel", "local")):
            model = self.model(address, retrieval)
            initial = model.decoder[-1].weight.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.forward(model, base, fields)["pred_pixels"]
                loss = (prediction - target).square().mean()
                loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                optimizer.step()
            self.assertFalse(torch.equal(initial, model.decoder[-1].weight))
            self.assertGreater(float(model.address_projection.weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.slot_keys.grad.abs().sum()), 0)

    def test_observable_preprocessing_detached_and_signatures_gt_free(self):
        low, base, tokens, _ = self.make_inputs()
        fields = self.m.prepare_observation_memory_inputs(low.requires_grad_(), base.requires_grad_(),
                    tokens.requires_grad_(), token_grid=(2, 2, 2), patch=4, descriptor_dim=8)
        for key in ("detail_values", "prior_address", "pixel_address"):
            self.assertFalse(fields[key].requires_grad)
        for function in (self.m.prepare_observation_memory_inputs, self.m.ObservationMemory.forward):
            names = set(inspect.signature(function).parameters)
            self.assertTrue(names.isdisjoint({"gt", "target", "oracle", "errors", "a", "gram"}))
        self.assertFalse(fields["metadata"]["gt_input"])
        self.assertFalse(fields["metadata"]["target_fit"])

    def test_invalid_shapes_values_and_configs_fail_closed(self):
        low, base, tokens, fields = self.make_inputs()
        with self.assertRaises(ValueError):
            self.m.prepare_observation_memory_inputs(low, base, tokens, token_grid=(1, 2, 2), patch=4)
        with self.assertRaises(ValueError):
            self.m.prepare_observation_memory_inputs(low - 2, base, tokens, token_grid=(2, 2, 2), patch=4)
        with self.assertRaises(TypeError):
            self.m.project_prior_tokens(tokens.double())
        with self.assertRaises(ValueError):
            self.m.project_prior_tokens(torch.full_like(tokens, float("nan")))
        with self.assertRaises(ValueError):
            self.m.ObservationMemory(address_mode="oracle")
        with self.assertRaises(ValueError):
            self.m.ObservationMemory(patch=3)
        with self.assertRaises(ValueError):
            self.model()(base, fields["detail_values"][:, :, :-1], fields["prior_address"], fields["pixel_address"])
        with self.assertRaises(ValueError):
            self.m.prepare_observation_memory_inputs(low, base, torch.full((1, 8, 8), 3.),
                token_grid=(2, 2, 2), patch=4, descriptor_dim=8, prior_is_projected=True)


if __name__ == "__main__":
    unittest.main()
