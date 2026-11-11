"""Graph algebra, boundary semantics and observable-only differentiable contract."""
import importlib.util
import inspect
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


def module():
    path = Path(__file__).resolve().parents[1] / "models/trajectory_reconstruction.py"
    spec = importlib.util.spec_from_file_location("trajectory_test_module", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@unittest.skipIf(torch is None, "Torch required")
class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = module()
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def geometry(self, t=3, h=8, w=8, dx=0., dy=0., batch=1):
        flow = torch.zeros(batch, 2, 2, t, h, w)
        flow[:, :, 0] = dx
        flow[:, :, 1] = dy
        mask = torch.ones(batch, 2, 1, t, h, w)
        mask[:, 0, :, 0] = 0
        mask[:, 1, :, -1] = 0
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        mask *= (((xx + dx >= 0) & (xx + dx <= w - 1)
                  & (yy + dy >= 0) & (yy + dy <= h - 1))[None, None, None, None])
        return flow, mask

    def inputs(self, t=4, batch=1):
        b, h, w = batch, 8, 8
        base = torch.rand(b, 3, t, h, w)
        corrections = torch.randn(b, 2, 3, t, h, w) * .05
        features = torch.randn(b, t, 19, 1, 1)
        details = torch.randn(b, 3, t, h, w) * .1
        flow, mask = self.geometry(t=t, batch=b)
        return base, corrections, features, details, flow, mask

    def dense_laplacian(self, graph, precision):
        b, t, h, w = graph["shape"]
        n = b * t * h * w
        matrix = torch.zeros(n, n)
        weights = self.m._weighted_edges(graph, precision).detach()
        for direction in range(2):
            for target in range(n):
                for corner in range(4):
                    source = int(graph["source_index"][direction, target, corner])
                    value = weights[direction, target, corner]
                    matrix[target, target] += value
                    matrix[source, source] += value
                    matrix[target, source] -= value
                    matrix[source, target] -= value
        return matrix

    def test_gather_translation_and_time_boundaries(self):
        flow, mask = self.geometry(t=3, h=3, w=4, dx=1.)
        graph = self.m.build_bilinear_graph(flow, mask)
        video = torch.arange(36.).reshape(1, 1, 3, 3, 4)
        output = self.m.directed_gather(video, graph)
        torch.testing.assert_close(output[0, 0, 0, 1, :, :-1], video[0, 0, 0, :, 1:])
        self.assertEqual(float(output[:, 0, :, 0].abs().sum()), 0.)
        self.assertEqual(float(output[:, 1, :, -1].abs().sum()), 0.)
        self.assertEqual(float(output[..., -1].abs().sum()), 0.)

    def test_gather_adjoint_inner_product(self):
        flow, mask = self.geometry(t=3, h=3, w=4, dx=.3, dy=.2, batch=2)
        graph = self.m.build_bilinear_graph(flow, mask)
        x = torch.randn(2, 3, 3, 3, 4)
        y = torch.randn(2, 2, 3, 3, 3, 4)
        left = (self.m.directed_gather(x, graph) * y).sum()
        right = (x * self.m.directed_adjoint(y, graph)).sum()
        torch.testing.assert_close(left, right, atol=4e-6, rtol=2e-6)

    def test_pair_laplacian_is_symmetric_psd_and_unit_anchor_spd(self):
        flow, mask = self.geometry(t=3, h=2, w=3, dx=.25)
        graph = self.m.build_bilinear_graph(flow, mask)
        precision = torch.rand(1, 2, 1, 3, 2, 3)
        matrix = self.dense_laplacian(graph, precision)
        torch.testing.assert_close(matrix, matrix.T, atol=1e-6, rtol=0)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix).min()), -1e-6)
        self.assertGreater(float(torch.linalg.eigvalsh(torch.eye(len(matrix)) + matrix).min()), .99999)
        x = torch.randn(1, 3, 3, 2, 3)
        expected = matrix @ self.m._flat(x)
        actual = self.m._flat(self.m.pair_laplacian(x, graph, precision))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_constant_state_preserved(self):
        flow, mask = self.geometry(dx=.25, dy=.25)
        graph = self.m.build_bilinear_graph(flow, mask)
        precision = torch.rand(1, 2, 1, 3, 8, 8)
        anchor = torch.full((1, 3, 3, 8, 8), .37)
        state, _ = self.m.reconstruct_two_steps(anchor, graph, precision)
        torch.testing.assert_close(state, anchor, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(self.m.pair_laplacian(anchor, graph, precision),
                                   torch.zeros_like(anchor), atol=4e-7, rtol=0)

    def test_zero_edges_are_exact_anchor(self):
        flow, mask = self.geometry()
        graph = self.m.build_bilinear_graph(flow, mask)
        anchor = torch.randn(1, 3, 3, 8, 8)
        state, degree = self.m.reconstruct_two_steps(anchor, graph, torch.zeros_like(mask))
        self.assertTrue(torch.equal(state, anchor))
        self.assertEqual(float(degree.sum()), 0.)

    def test_two_steps_match_dense_jacobi(self):
        flow, mask = self.geometry(t=3, h=2, w=3, dx=.2, dy=.3)
        graph = self.m.build_bilinear_graph(flow, mask)
        precision = torch.rand(1, 2, 1, 3, 2, 3)
        laplacian = self.dense_laplacian(graph, precision)
        diagonal = torch.diag(laplacian)
        adjacency = torch.diag(diagonal) - laplacian
        anchor = torch.randn(1, 2, 3, 2, 3)
        a = self.m._flat(anchor)
        expected = a
        for _ in range(2):
            expected = (a + adjacency @ expected) / (1 + diagonal[:, None])
        result, _ = self.m.reconstruct_two_steps(anchor, graph, precision)
        torch.testing.assert_close(self.m._flat(result), expected, atol=2e-7, rtol=2e-6)

    def test_single_pair_updates_both_endpoints_not_unrelated_nodes(self):
        flow, mask = self.geometry(t=3, h=2, w=2)
        mask.zero_()
        mask[0, 0, 0, 1, 0, 0] = 1
        graph = self.m.build_bilinear_graph(flow, mask)
        anchor = torch.zeros(1, 1, 3, 2, 2)
        anchor[0, 0, 1, 0, 0] = 1
        state, _ = self.m.reconstruct_two_steps(anchor, graph, torch.ones_like(mask))
        self.assertAlmostEqual(float(state[0, 0, 0, 0, 0]), .25)
        self.assertAlmostEqual(float(state[0, 0, 1, 0, 0]), .75)
        self.assertEqual(int(torch.count_nonzero(state)), 2)

    def test_node_only_matches_equal_thirds_and_modes_share_state_shapes(self):
        inputs = self.inputs(batch=2)
        original = self.m.TrajectoryReconstruction(mode="node_only")
        output = original(*inputs, return_aux=True)
        weights = output["trajectory_aux"]["node_weights"]
        expected = (inputs[0] + (weights[:, :, None] * inputs[1]).sum(1)).clamp(0, 1)
        self.assertTrue(torch.equal(expected, output["pred_pixels"]))
        self.assertEqual(sum(p.numel() for p in original.parameters()), 6819)
        for mode in self.m.MODES:
            model = self.m.TrajectoryReconstruction(mode=mode)
            model.load_state_dict(original.state_dict(), strict=True)
            result = model(*inputs)
            self.assertEqual(result["pred_pixels"].shape, inputs[0].shape)
            self.assertEqual(model.get_config()["steps"], 2)
            self.assertTrue(bool(torch.isfinite(result["pred_pixels"]).all()))

    def test_source_loss_reaches_node_and_edge_parameters_without_gt_input(self):
        self.assertNotIn("gt", inspect.signature(self.m.TrajectoryReconstruction.forward).parameters)
        inputs = self.inputs()
        model = self.m.TrajectoryReconstruction()
        predicted = model(*inputs)["pred_pixels"]
        target = torch.rand_like(predicted)  # Source-only caller label, never passed to the module.
        (predicted - target).square().mean().backward()
        self.assertGreater(float(model.node[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.edge[-1].weight.grad.abs().sum()), 0.)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_cross_scramble_changes_only_edge_conditioning(self):
        inputs = self.inputs()
        model = self.m.TrajectoryReconstruction()
        with torch.no_grad():
            model.edge[0].weight.fill_(.1)
            model.edge[0].bias.fill_(.1)
            model.edge[-1].weight.fill_(.1)
        conditioned = model(*inputs, return_aux=True)["trajectory_aux"]
        model.mode = "scrambled_cross"
        scrambled = model(*inputs, return_aux=True)["trajectory_aux"]
        self.assertTrue(torch.equal(conditioned["node_prediction"], scrambled["node_prediction"]))
        self.assertTrue(torch.equal(conditioned["anchor"], scrambled["anchor"]))
        self.assertGreater(float((conditioned["edge_precision"] - scrambled["edge_precision"]).abs().max()), 1e-6)

    def test_precision_gradient_matches_finite_difference(self):
        flow, mask = self.geometry(t=3, h=2, w=2)
        graph = self.m.build_bilinear_graph(flow, mask)
        anchor = torch.randn(1, 1, 3, 2, 2)
        precision = torch.full_like(mask, .4, requires_grad=True)
        output, _ = self.m.reconstruct_two_steps(anchor, graph, precision)
        output.square().sum().backward()
        index = (0, 0, 0, 1, 0, 0)
        values = []
        for delta in (.001, -.001):
            changed = precision.detach().clone()
            changed[index] += delta
            result, _ = self.m.reconstruct_two_steps(anchor, graph, changed)
            values.append(float(result.square().sum()))
        self.assertAlmostEqual(float(precision.grad[index]), (values[0] - values[1]) / .002, delta=.0015)

    def test_invalid_mask_and_precision_rejected(self):
        flow, mask = self.geometry()
        bad = mask.clone()
        bad[0, 0, 0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "invalid time"):
            self.m.build_bilinear_graph(flow, bad)
        graph = self.m.build_bilinear_graph(flow, mask)
        with self.assertRaisesRegex(ValueError, "precision"):
            self.m.reconstruct_two_steps(torch.ones(1, 3, 3, 8, 8), graph, torch.full_like(mask, -1))


if __name__ == "__main__":
    unittest.main()
