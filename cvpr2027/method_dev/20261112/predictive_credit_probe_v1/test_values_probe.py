"""Synthetic invariants for a GT-free RGB value action and storage firewall."""
import inspect
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

import values_probe as v


class ValuesTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(9917)
        self.base = torch.rand(1, 3, 16, 32, 32) * .8 + .1
        self.low = self.base * .1
        self.flow = torch.zeros(1, 2, 2, 16, 32, 32)
        self.mask = torch.ones(1, 2, 1, 16, 32, 32)

    def prepare(self):
        return v.prepare_values(self.base, self.low, self.flow, self.mask)

    def test_gt_cannot_change_predictions(self):
        self.assertFalse({"gt", "labels", "fit_labels"} & set(inspect.signature(v.prepare_values).parameters))
        pred, _, _ = self.prepare()
        before = {k: p.clone() for k, p in pred.items()}
        a = v.score_utilities(pred, torch.zeros_like(pred["base"]))
        b = v.score_utilities(pred, torch.ones_like(pred["base"]))
        self.assertTrue(all(torch.equal(pred[k], before[k]) for k in pred))
        self.assertFalse(torch.equal(a["mse_base"], b["mse_base"]))

    def test_dc_range_support_and_exact_zero(self):
        self.mask[..., :8, :] = 0
        predictions, _, constraints = self.prepare()
        self.assertLess(constraints["max_abs_patch_frame_channel_dc_sum_error"], 2e-5)
        self.assertTrue(constraints["support_outside_exact_anchor"])
        self.assertTrue(torch.equal(predictions["base"], predictions["sameframe_zero"]))
        for p in predictions.values():
            self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))
            self.assertTrue(torch.equal(p[..., :8, :], predictions["base"][..., :8, :]))

    def test_common_channel_permutation(self):
        predictions, features, _ = self.prepare()
        permuted, pfeatures, _ = v.prepare_values(self.base[:, [2,1,0]], self.low[:, [2,1,0]], self.flow, self.mask)
        for key in predictions:
            self.assertTrue(torch.equal(predictions[key][[2,1,0]], permuted[key]), key)
        for key in features:
            self.assertTrue(torch.allclose(features[key], pfeatures[key], atol=1e-12, rtol=0), key)
        gt = torch.rand_like(predictions["base"])
        a, b = v.score_utilities(predictions,gt), v.score_utilities(permuted,gt[[2,1,0]])
        self.assertTrue(all(torch.allclose(a[k],b[k],atol=1e-12,rtol=0) for k in a))

    def test_duplicate_donors_pair_is_not_double(self):
        self.base[:,:,10:12] = self.base[:,:,6:8]
        predictions, _, _ = self.prepare()
        self.assertTrue(torch.equal(predictions["single1"], predictions["single2"]))
        self.assertTrue(torch.equal(predictions["single1"], predictions["pair"]))

    def test_empty_donors_exact_base(self):
        self.mask.zero_()
        pred, features, _ = self.prepare()
        self.assertTrue(all(torch.equal(p, pred["base"]) for p in pred.values()))
        self.assertTrue(all(torch.equal(p,torch.zeros_like(p)) for p in features.values()))

    def test_projection_is_euclidean_box_fixed_sum(self):
        base = torch.full((3,2,16,16), .3)
        support = torch.ones(1,2,16,16,dtype=torch.bool)
        candidate = torch.linspace(-2,2,256).reshape(1,1,16,16).expand_as(base).clone()
        result = v.project_box_fixed_sum(candidate,base,support)
        self.assertTrue(bool(((result>=0)&(result<=1)).all()))
        self.assertLess(float((result.double().sum((-1,-2))-base.double().sum((-1,-2))).abs().max()),2e-5)
        free=(result>0)&(result<1)
        shifts=(result-candidate)[free]
        self.assertLess(float(shifts.max()-shifts.min()),1e-6)
        # KKT at clipped endpoints relative to the common free-coordinate shift.
        shift=shifts.mean()
        self.assertTrue(bool((candidate[result==0]+shift<=1e-6).all()))
        self.assertTrue(bool((candidate[result==1]+shift>=1-1e-6).all()))

    def test_geometry_coordinate_units_and_fallback(self):
        self.flow[:,0,0] = 1.25
        self.flow[:,0,1] = -.5
        _, valid = v.warp_two(self.base,self.flow,self.mask)
        coords, fractions = v.observable_geometry(self.flow,valid,32,32)
        self.assertAlmostEqual(coords[0][2][0],8.75,places=5)
        self.assertAlmostEqual(coords[0][2][1],23.,places=5)
        self.assertEqual(fractions[1],[1.,1.,1.,1.])
        valid[0].zero_()
        coords, fractions = v.observable_geometry(self.flow,valid,32,32)
        self.assertEqual(coords[0][0],[7.5,7.5])
        self.assertEqual(fractions[0],[0.,0.,0.,0.])

    def test_reject_excluded_before_any_open(self):
        row={"name":"excluded", "scene":"excluded", "source_path":"/does/not/exist"}
        with self.assertRaisesRegex(ValueError,"Refusing excluded"):
            v.label_source(row,["fit"],["fit"],None)
        with self.assertRaisesRegex(ValueError,"Refusing excluded"):
            v.require_selected({"name":"fit_2","scene":"fit"},["fit_2"],["fit"])

    def test_utility_exact_algebra(self):
        predictions, _, _ = self.prepare()
        scored=v.score_utilities(predictions,torch.rand_like(predictions["base"]))
        self.assertTrue(torch.allclose(scored["u1_given2"],scored["u12"]-scored["u2"],atol=1e-15,rtol=0))
        self.assertTrue(torch.allclose(scored["u2_given1"],scored["u12"]-scored["u1"],atol=1e-15,rtol=0))
        self.assertTrue(torch.equal(scored["mse_base"],scored["mse_sameframe_zero"]))

    def test_selective_reader_never_opens_gt_or_base(self):
        reader_path=Path(os.environ["VALUES_LEGACY_READER"])
        reader=v.module_at("_values_test_metadata",reader_path,"26c0c36656ad7b7d3b0ce04275578a45286c27931a476c9dcfb1ce1981e69db2")
        import io
        shapes={"lq":tuple(self.low.shape),"flow":tuple(self.flow.shape),"valid_mask":tuple(self.mask.shape)}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"source.pt"
            torch.save({"name":"fit","scene":"fit","tensors":{"lq":self.low,"flow":self.flow,
                "valid_mask":self.mask,"gt":torch.rand_like(self.base),"base":self.base}},path)
            with zipfile.ZipFile(path) as archive:
                name=next(n for n in archive.namelist() if n.endswith("/data.pkl"))
                meta=reader.MetadataOnlyUnpickler(io.BytesIO(archive.read(name))).load()
                forbidden={name[:-len("data.pkl")]+"data/"+meta["tensors"][k].storage.key for k in ("gt","base")}
            class Firewall(zipfile.ZipFile):
                def read(self,name,*args,**kwargs):
                    if name in forbidden:
                        raise AssertionError("GT/base storage was opened")
                    return super().read(name,*args,**kwargs)
            name,scene,tensors,lineage=v.read_observables(path,reader,archive_class=Firewall,expected_shapes=shapes)
            self.assertEqual((name,scene),("fit","fit"))
            self.assertEqual(set(tensors),set(shapes))
            self.assertEqual(lineage["gt_storage_entries_read"],0)
            self.assertTrue(all(torch.equal(tensors[k],p) for k,p in (("lq",self.low),("flow",self.flow),("valid_mask",self.mask))))
            torch.save({"name":"fit","scene":"fit","tensors":{"lq":self.low,"flow":self.flow,
                "valid_mask":self.mask,"gt":self.low}},path)
            with self.assertRaisesRegex(ValueError,"aliases excluded"):
                v.read_observables(path,reader,expected_shapes=shapes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
