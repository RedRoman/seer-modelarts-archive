"""Synthetic checks only: no real source, HVI, holdout or target data."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fit_feasibility as run


def fixture():
    generator=torch.Generator().manual_seed(521)
    base=.1+.7*torch.rand(1,3,5,8,8,generator=generator)
    flow=torch.zeros(1,2,2,5,8,8)
    mask=torch.ones(1,2,1,5,8,8,dtype=torch.bool)
    return base,flow,mask


class HviFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_exact_seven_rows_and_sameframe_arithmetic_identity(self):
        base,flow,mask=fixture()
        predictions,coverage=run.fixed_predictions(base,flow,mask)
        self.assertEqual(set(predictions),set(run.METHODS))
        self.assertEqual(len(predictions),7)
        for value in run.LAMBDAS:
            self.assertTrue(torch.equal(predictions[f"sameframe_identity_lambda_{value:g}"],base))
        self.assertAlmostEqual(sum(coverage.values()),1.,6)

    def test_two_and_single_donor_fixed_arithmetic(self):
        base,flow,mask=fixture()
        for i,value in enumerate((.1,.2,.3,.4,.7)):base[:,:,i]=value
        correction,count=run.aligned_hvi_mean(base,flow,mask)
        self.assertTrue(torch.equal(count[:,:,2],torch.full_like(count[:,:,2],2)))
        self.assertTrue(torch.allclose(correction[:,:,2],torch.full_like(correction[:,:,2],.1),atol=1e-7))
        self.assertTrue(torch.equal(count[:,:,0],torch.ones_like(count[:,:,0])))
        self.assertTrue(torch.allclose(correction[:,:,0],torch.full_like(correction[:,:,0],.2),atol=1e-7))
        predictions,_=run.fixed_predictions(base,flow,mask)
        self.assertTrue(torch.allclose(predictions["temporal_mean_lambda_0.25"][:,:,2],torch.full_like(base[:,:,2],.325),atol=1e-7))

    def test_no_valid_donor_returns_exact_anchor(self):
        base,flow,mask=fixture();mask.zero_()
        predictions,coverage=run.fixed_predictions(base,flow,mask)
        self.assertEqual(coverage["no_donor_fraction"],1.)
        for value in predictions.values():self.assertTrue(torch.equal(value,base))

    def test_out_of_image_and_time_geometry_ignores_border_padding(self):
        base,flow,mask=fixture();flow[:]=10000
        correction,count=run.aligned_hvi_mean(base,flow,mask)
        self.assertFalse(bool(count.any()));self.assertFalse(bool(correction.any()))
        # Masks deliberately mark all times valid; helper must enforce availability.
        flow.zero_();_,count=run.aligned_hvi_mean(base,flow,mask)
        self.assertTrue(torch.equal(count[:,:,:2],torch.ones_like(count[:,:,:2])))
        self.assertTrue(torch.equal(count[:,:,3:],torch.ones_like(count[:,:,3:])))

    def test_target_to_source_direction_matches_known_translation(self):
        base,flow,mask=fixture()
        target=torch.arange(8,dtype=torch.float32).reshape(1,1,1,8).expand(1,3,8,8)/10
        base[:,:,2]=target
        base[:,:,0]=torch.roll(target,1,-1);base[:,:,4]=torch.roll(target,1,-1)
        flow[:,0,0,2]=1;flow[:,1,0,2]=1
        correction,count=run.aligned_hvi_mean(base,flow,mask)
        self.assertLess(float(correction[:,:,2,:,:7].abs().max()),1e-7)
        self.assertFalse(bool(count[:,:,2,:,7].any()))

    def test_original_mask_controls_actual_valid_neighbor_mean(self):
        base,flow,mask=fixture()
        base[:,:,0]=.1;base[:,:,2]=.3;base[:,:,4]=.7
        mask[:,0,:,:,0:4]=False
        correction,count=run.aligned_hvi_mean(base,flow,mask)
        self.assertTrue(torch.allclose(correction[:,:,2,:4],torch.full_like(correction[:,:,2,:4],.4),atol=1e-7))
        self.assertTrue(torch.equal(count[:,:,2,:4],torch.ones_like(count[:,:,2,:4])))
        self.assertTrue(torch.allclose(correction[:,:,2,4:],torch.full_like(correction[:,:,2,4:],.1),atol=1e-7))

    def test_prediction_helpers_have_no_label_parameter(self):
        for function in (run.aligned_hvi_mean,run.fixed_predictions):
            self.assertFalse({"gt","labels","fit_labels"}&set(inspect.signature(function).parameters))
        with self.assertRaises(TypeError):run.fixed_predictions(*fixture(),gt=torch.zeros(1))

    def test_production_loader_refuses_holdout_before_tensor_load(self):
        with patch("torch.load",side_effect=AssertionError("No tensor may be opened")) as loader:
            with self.assertRaisesRegex(ValueError,"excluded"):
                run.load_fit_sample({"name":"b","scene":"holdout"},["fit"])
            loader.assert_not_called()

    def test_evaluation_filters_rows_before_invoking_label_loader(self):
        inputs=fixture();seen=[]
        rows=[{"name":"a","scene":"fit"},{"name":"b","scene":"holdout"}]
        split={"fit":["fit"],"internal_holdout":["holdout"]}
        def loader(row,fit_scenes):
            self.assertIn(row["scene"],fit_scenes)
            seen.append(row["name"])
            return inputs,torch.full_like(inputs[0],.9)
        scores=run.evaluate_fit_rows(rows,split,loader)
        self.assertEqual(seen,["a"]);self.assertEqual([r["name"] for r in scores],["a"])

    def test_metric_uses_mean_clip_psnr_not_psnr_of_mean_mse(self):
        rows=[]
        for i,mse in enumerate((.01,.04)):
            values={m:{"rgb_mse_float":mse,"psnr_rgb_from_clip_mse":-10*np.log10(mse)} for m in run.METHODS}
            rows.append({"name":str(i),"scene":"fit","methods":values})
        result=run.aggregate(rows)
        self.assertAlmostEqual(result["mean"]["hvi"]["mean_clip_psnr_rgb"],16.989700043360187,12)
        self.assertAlmostEqual(result["mean"]["hvi"]["mean_rgb_mse"],.025,12)
        self.assertGreater(abs(result["mean"]["hvi"]["mean_clip_psnr_rgb"]+10*np.log10(.025)),.9)
        self.assertEqual(result["per_scene"][0]["clips"],2)
        self.assertFalse(result["lambda_selection_performed"])

    def test_invalid_masks_rejected(self):
        base,flow,mask=fixture()
        with self.assertRaisesRegex(ValueError,"binary"):
            run.aligned_hvi_mean(base,flow,mask.float()*.5)


if __name__=="__main__":unittest.main()
