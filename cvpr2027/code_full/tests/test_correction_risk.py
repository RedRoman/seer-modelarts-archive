"""Numerical QA for the isolated correction-risk solver; no model/data access.

NumPy tests exercise the SAME arithmetic kernel and compare against an
independent dense simplex grid even when torch is unavailable. Torch-specific
contract, GT-helper, objective identity and gradient tests run on remote CPU.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "models/correction_risk.py"
spec = importlib.util.spec_from_file_location("correction_risk_under_test", MODULE_PATH)
risk_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk_module)
try:
    import torch
except ImportError:
    torch = None


def numpy_solve(a, gram):
    a, gram = np.asarray(a), np.asarray(gram)
    scale = np.maximum(np.abs(a).max(axis=-1), np.abs(gram).max(axis=(-2, -1)))
    scale = np.maximum(scale, np.finfo(a.dtype).tiny)
    aa, gg = a / scale[..., None], gram / scale[..., None, None]
    w0, w1, q = risk_module._solve_components(
        aa[..., 0], aa[..., 1], gg[..., 0, 0], gg[..., 0, 1], gg[..., 1, 1], where=np.where)
    return np.stack((w0, w1), axis=-1), q * scale


def dense_grid_minimum(a, gram, divisions=400):
    # Independent exhaustive objective evaluation; no analytical stationarity.
    first, second = np.meshgrid(np.arange(divisions + 1), np.arange(divisions + 1), indexing="ij")
    keep = first + second <= divisions
    weights = np.stack((first[keep], second[keep]), axis=-1) / divisions
    return np.min(np.einsum("ni,ij,nj->n", weights, gram, weights) - 2 * weights @ a)


class NumpyKernelTests(unittest.TestCase):
    def check_case(self, a, gram, expected_risk=None, expected_weights=None):
        a, gram = np.array(a, dtype=np.float64), np.array(gram, dtype=np.float64)
        weights, q = numpy_solve(a, gram)
        self.assertTrue(np.isfinite(weights).all() and np.isfinite(q).all())
        self.assertTrue((weights >= 0).all())
        self.assertLessEqual(float(weights.sum()), 1 + 1e-14)
        self.assertLessEqual(float(q), 0)
        direct = weights @ gram @ weights - 2 * a @ weights
        self.assertAlmostEqual(float(q), float(direct), places=12)
        grid = dense_grid_minimum(a, gram)
        self.assertLessEqual(float(q), grid + 1e-11)
        self.assertLess(grid - float(q), 1e-4 * max(1, np.abs(a).max(), np.abs(gram).max()))
        if expected_risk is not None:
            self.assertAlmostEqual(float(q), expected_risk, places=11)
        if expected_weights is not None:
            np.testing.assert_allclose(weights, expected_weights, atol=1e-12)

    def test_zero_negative_gain_and_one_candidate(self):
        self.check_case([0, 0], [[0, 0], [0, 0]], 0, [0, 0])
        self.check_case([-.4, -.2], [[1, -.8], [-.8, 1]], 0, [0, 0])
        self.check_case([.4, 0], [[2, 0], [0, 0]], -.08, [.2, 0])

    def test_interior_simplex_edge_and_complementary_directions(self):
        self.check_case([.2, .3], np.eye(2), -.13, [.2, .3])
        self.check_case([1, 1], np.eye(2), -1.5, [.5, .5])
        self.check_case([.1, .1], [[1, -.8], [-.8, 1]], -.1, [.5, .5])

    def test_singular_correlated_opposite_and_linear_cases(self):
        self.check_case([.4, .4], [[1, 1], [1, 1]], -.16)
        self.check_case([.3, -.3], [[1, -1], [-1, 1]], -.09)
        self.check_case([.1, .7], [[0, 0], [0, 0]], -1.4, [0, 1])
        self.check_case([.25, .25], [[1, 1 - 1e-12], [1 - 1e-12, 1]])

    def test_random_psd_matrices_match_independent_dense_grid(self):
        rng = np.random.default_rng(20261111)
        for _ in range(48):
            directions = rng.normal(size=(2, 5))
            gram = directions @ directions.T / 5
            a = rng.normal(size=2)
            with self.subTest(a=a.tolist()):
                self.check_case(a, gram)

    def test_float32_extreme_scales_and_singular_ratios_do_not_overflow(self):
        base_a = np.array([[.2, .3], [1, 1], [.4, .4], [1, -1]], dtype=np.float32)
        base_g = np.array([np.eye(2), np.eye(2), [[1, 1], [1, 1]], [[0, 0], [0, 0]]], dtype=np.float32)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            reference, ref_q = numpy_solve(base_a, base_g)
            for scale in (1e-30, 1e-10, 1, 1e10, 1e30):
                weights, q = numpy_solve(base_a * np.float32(scale), base_g * np.float32(scale))
                np.testing.assert_allclose(weights, reference, rtol=3e-6, atol=3e-6)
                np.testing.assert_allclose(q / scale, ref_q, rtol=3e-6, atol=3e-6)
            weights, q = numpy_solve(np.array([1e30, -1e30], np.float32),
                                     np.array([[1e-30, 0], [0, 1e-30]], np.float32))
            np.testing.assert_array_equal(weights, [1, 0])
            self.assertTrue(np.isfinite(q))

    def test_batched_kernel_and_noncontiguous_inputs(self):
        rng = np.random.default_rng(4)
        d = rng.normal(size=(3, 4, 2, 6))
        gram = np.einsum("...id,...jd->...ij", d, d) / 6
        a = rng.normal(size=(3, 4, 2))
        weights, q = numpy_solve(a[:, ::2], gram[:, ::2])
        for row in range(3):
            for column in range(2):
                expected_w, expected_q = numpy_solve(a[row, column * 2], gram[row, column * 2])
                np.testing.assert_array_equal(weights[row, column], expected_w)
                np.testing.assert_array_equal(q[row, column], expected_q)


@unittest.skipIf(torch is None, "torch required; NumPy kernel QA still runs locally")
class TorchSolverTests(unittest.TestCase):
    def test_solver_matches_numpy_and_grid_for_float32_and_float64(self):
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(64, 2, 7))
        grams = np.einsum("nid,njd->nij", directions, directions) / 7
        aa = rng.normal(size=(64, 2))
        for dtype, tol in ((torch.float32, 3e-6), (torch.float64, 1e-11)):
            a, gram = torch.tensor(aa, dtype=dtype), torch.tensor(grams, dtype=dtype)
            weights, q = risk_module.solve_simplex2(a, gram)
            expected_w, expected_q = numpy_solve(a.numpy(), gram.numpy())
            np.testing.assert_allclose(weights.numpy(), expected_w, rtol=tol, atol=tol)
            np.testing.assert_allclose(q.numpy(), expected_q, rtol=tol, atol=tol)
            self.assertEqual(weights.dtype, dtype)
            self.assertEqual(tuple(q.shape), (64,))
            for index in range(8):
                self.assertLessEqual(q[index].item(), dense_grid_minimum(aa[index], grams[index]) + tol)

    def test_torch_degenerate_cases_and_no_linalg_dependency(self):
        from unittest import mock
        a = torch.tensor([[0., 0.], [.4, 0], [.4, .4], [-.2, -.3], [.1, .1]])
        gram = torch.tensor([[[0., 0], [0, 0]], [[2, 0], [0, 0]], [[1, 1], [1, 1]],
                             [[1, 0], [0, 1]], [[1, -.8], [-.8, 1]]])
        with mock.patch.object(torch.linalg, "solve", side_effect=AssertionError("No solve")), \
             mock.patch.object(torch.linalg, "inv", side_effect=AssertionError("No inverse")), \
             mock.patch.object(torch.linalg, "eigh", side_effect=AssertionError("No decomposition")):
            w, q = risk_module.solve_simplex2(a, gram)
        torch.testing.assert_close(q, torch.tensor([0., -.08, -.16, 0., -.1]))
        self.assertTrue(torch.equal(w[0], torch.zeros(2)))
        self.assertTrue(torch.equal(w[3], torch.zeros(2)))

    def test_gt_helper_is_detached_and_risk_equals_actual_mse_change(self):
        torch.manual_seed(17)
        error = torch.randn(2, 3, 24, requires_grad=True)
        residuals = torch.randn(2, 3, 2, 24, requires_grad=True)
        a, gram = risk_module.train_only_patch_statistics(error, residuals)
        self.assertFalse(a.requires_grad or gram.requires_grad)
        self.assertEqual(tuple(a.shape), (2, 3, 2))
        self.assertEqual(tuple(gram.shape), (2, 3, 2, 2))
        w, q = risk_module.solve_simplex2(a, gram)
        corrected_error = error.detach() - (w.unsqueeze(-1) * residuals.detach()).sum(-2)
        direct = corrected_error.square().mean(-1) - error.detach().square().mean(-1)
        torch.testing.assert_close(q, direct, atol=4e-7, rtol=4e-6)
        self.assertTrue((direct <= 4e-7).all())

    def test_autograd_is_finite_away_from_active_set_ties(self):
        a = torch.tensor([.2, .3], requires_grad=True)
        gram = torch.tensor([[1., .1], [.1, 1.]], requires_grad=True)
        w, q = risk_module.solve_simplex2(a, gram)
        (w.square().sum() + q).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(gram.grad).all())

    def test_shape_dtype_nonfinite_and_non_psd_rejection(self):
        a, gram = torch.zeros(2), torch.eye(2)
        bad_shapes = ((torch.zeros(3), gram), (a, torch.zeros(3, 3)),
                      (torch.zeros(1, 2), gram), (torch.tensor(0.), gram))
        for aa, gg in bad_shapes:
            with self.assertRaises(ValueError):
                risk_module.solve_simplex2(aa, gg)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                risk_module.solve_simplex2(torch.tensor([invalid, 0.]), gram)
            with self.assertRaises(ValueError):
                risk_module.solve_simplex2(a, torch.tensor([[invalid, 0.], [0., 1.]]))
        for invalid in (torch.tensor([[1., 2.], [0., 1.]]),
                        torch.tensor([[1., 2.], [2., 1.]]),
                        torch.tensor([[-1., 0.], [0., 1.]])):
            with self.assertRaises(ValueError):
                risk_module.solve_simplex2(a, invalid)
        for aa, gg in ((a.half(), gram.half()), (a, gram.double()), ([0., 0.], gram)):
            with self.assertRaises(TypeError):
                risk_module.solve_simplex2(aa, gg)

    def test_helper_rejects_invalid_contract_and_nonfinite(self):
        for error, residuals in ((torch.zeros(0), torch.zeros(2, 0)),
                                 (torch.zeros(8), torch.zeros(8, 2)),
                                 (torch.zeros(3, 8), torch.zeros(2, 8)),
                                 (torch.full((8,), float("nan")), torch.zeros(2, 8))):
            with self.assertRaises(ValueError):
                risk_module.train_only_patch_statistics(error, residuals)

    def test_video_patch_moments_match_explicit_rgb_spatial_slices(self):
        error = torch.arange(2 * 3 * 3 * 8 * 12, dtype=torch.float32).reshape(2, 3, 3, 8, 12) / 100
        directions = torch.stack((error.sin(), error.cos()), dim=1)
        error.requires_grad_(True)
        directions.requires_grad_(True)
        a, gram = risk_module.patch_moments(error, directions, patch=4)
        self.assertEqual(tuple(a.shape), (2, 3, 2, 3, 2))
        self.assertEqual(tuple(gram.shape), (2, 3, 2, 3, 2, 2))
        self.assertFalse(a.requires_grad or gram.requires_grad)
        for b in range(2):
            for t in range(3):
                for row in range(2):
                    for col in range(3):
                        e = error[b, :, t, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4].detach().flatten()
                        d = directions[b, :, :, t, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4].detach().reshape(2, -1)
                        torch.testing.assert_close(a[b, t, row, col], (d * e).mean(-1))
                        torch.testing.assert_close(gram[b, t, row, col], d @ d.T / e.numel())
        w, q = risk_module.solve_two_candidate_risk(a, gram)
        self.assertEqual(tuple(w.shape), tuple(a.shape))
        self.assertEqual(tuple(q.shape), (2, 3, 2, 3))

    def test_video_patch_moments_reject_shape_or_partial_patches(self):
        error, d = torch.zeros(1, 3, 2, 8, 8), torch.zeros(1, 2, 3, 2, 8, 8)
        for patch in (0, -1, 3, 2.5, True):
            with self.assertRaises(ValueError):
                risk_module.patch_moments(error, d, patch=patch)
        for e, corrections in ((error[:, :2], d), (error, d[:, :1]),
                                (error.squeeze(0), d), (error[..., :7], d[..., :7])):
            with self.assertRaises(ValueError):
                risk_module.patch_moments(e, corrections)

    def test_empty_batch_and_noncontiguous_inputs(self):
        w, q = risk_module.solve_simplex2(torch.zeros(0, 2), torch.zeros(0, 2, 2))
        self.assertEqual(tuple(w.shape), (0, 2))
        self.assertEqual(tuple(q.shape), (0,))
        a = torch.tensor([[.2, .3], [.4, .1], [.1, -.2], [0., 0.]])[::2]
        gram = torch.eye(2).expand(4, 2, 2)[::2]
        w, q = risk_module.solve_simplex2(a, gram)
        wc, qc = risk_module.solve_simplex2(a.contiguous(), gram.contiguous())
        torch.testing.assert_close(w, wc)
        torch.testing.assert_close(q, qc)


if __name__ == "__main__":
    unittest.main()
