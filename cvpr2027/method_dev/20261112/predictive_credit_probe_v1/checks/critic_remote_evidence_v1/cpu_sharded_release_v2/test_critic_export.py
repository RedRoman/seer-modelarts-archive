"""CPU contract checks for masks, normalization and random initialization."""
import unittest
import torch
from critic_export import aligned_cosine, mask_plan, normalized_loss, normalize_rgb, wrong_time_content


class CriticContractTests(unittest.TestCase):
    def test_original_token_coordinates_and_disjoint_coalitions(self):
        plan = mask_plan()
        self.assertEqual(plan['target_ids'], list(range(1024, 1280)))
        self.assertEqual(plan['donor_ids'], [list(range(768, 1024)), list(range(1280, 1536))])
        self.assertEqual(plan['common_tubelets'], [0, 1, 2, 6, 7])
        self.assertEqual([len(value) for value in plan['context_ids'].values()], [1280, 1536, 1536, 1792])
        for ids in plan['context_ids'].values():
            self.assertEqual(ids, sorted(set(ids)))
            self.assertFalse(set(ids) & set(plan['target_ids']))

    def test_donor_order_swap_only_swaps_singleton_context(self):
        first, swapped = mask_plan(), mask_plan(donors=(5, 3))
        self.assertEqual(first['context_ids']['empty'], swapped['context_ids']['empty'])
        self.assertEqual(first['context_ids']['both'], swapped['context_ids']['both'])
        self.assertEqual(first['context_ids']['donor1'], swapped['context_ids']['donor2'])

    def test_physical_duplicate_donor_is_rejected(self):
        with self.assertRaises(ValueError):
            mask_plan(donors=(3, 3))
        with self.assertRaises(ValueError):
            mask_plan(donors=(3, 4))

    def test_normalized_loss_has_unit_vector_geometry(self):
        target = torch.tensor([[[1., 0.], [0., 2.]]])
        self.assertTrue(torch.equal(normalized_loss(target*4, target), torch.zeros(1, 2)))
        self.assertTrue(torch.equal(normalized_loss(-target, target), torch.full((1, 2), 4.)))

    def test_normalization_preserves_pixels_and_geometry(self):
        value = torch.zeros(1, 3, 16, 16, 16)
        before = value.clone()
        result = normalize_rgb(value)
        self.assertEqual(result.shape, value.shape)
        self.assertTrue(torch.equal(before, value))
        expected = -torch.tensor([.485, .456, .406])/torch.tensor([.229, .224, .225])
        self.assertTrue(torch.equal(result[0, :, 0, 0, 0], expected))

    def test_out_of_range_and_mismatched_latent_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_rgb(torch.full((1, 3, 2, 16, 16), 1.01))
        with self.assertRaises(ValueError):
            normalized_loss(torch.ones(1, 2, 4), torch.ones(1, 3, 4))

    def test_wrong_time_is_exact_non_target_bijection(self):
        video = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1, 1).repeat(1, 3, 1, 1, 1)
        result = wrong_time_content(video)
        self.assertTrue(torch.equal(video[:, :, 8:10], result[:, :, 8:10]))
        self.assertEqual(sorted(result[0, 0, :, 0, 0].tolist()), list(range(16)))
        self.assertEqual(result[0, 0, :2, 0, 0].tolist(), [6., 7.])
        self.assertTrue(torch.equal(video[0, 0, :, 0, 0], torch.arange(16, dtype=torch.float32)))

    def test_flow_aligned_descriptor_identity_and_shift(self):
        generator = torch.Generator().manual_seed(32)
        donor = torch.randn(1, 256, 7, generator=generator)
        xy = [[16*(index%16)+7.5, 16*(index//16)+7.5] for index in range(256)]
        result = aligned_cosine(donor, donor, xy)
        self.assertTrue(torch.allclose(result, torch.ones_like(result), atol=2e-6, rtol=0))
        shifted = [[min(255., x+16), y] for x, y in xy]
        self.assertLess(float(aligned_cosine(donor, donor, shifted).mean()), .3)


if __name__ == '__main__':
    unittest.main()
