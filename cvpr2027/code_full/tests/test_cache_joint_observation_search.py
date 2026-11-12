"""Numerical and data-firewall tests for joint donor cost preparation."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/"tools"))
import cache_joint_observation_search as cache


def sample():
    generator=torch.Generator().manual_seed(75)
    low=.15+.05*torch.rand((1,3,5,32,32),generator=generator)
    base=(low*2).clamp(0,1)
    flows=torch.zeros(1,2,2,5,32,32)
    masks=torch.ones(1,2,1,5,32,32,dtype=torch.bool)
    masks[:,0,:,:2]=False;masks[:,1,:,3:]=False
    prior=F.normalize(torch.randn(1,5,4,64,generator=generator),dim=-1)
    pixel=F.normalize(torch.randn(1,5,4,64,generator=generator),dim=-1)
    return low,base,flows,masks,prior,pixel


class JointCostsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.inputs=sample()
        cls.costs=cache.build_joint_costs(*cls.inputs,[.001,.0001],global_variance=.002)

    def test_layout_finiteness_and_candidate_masks(self):
        c=self.costs
        for key in ("unary_observation","unary_constant"):
            self.assertEqual(tuple(c[key].shape),(1,2,5,4,9))
            self.assertTrue(bool(torch.isfinite(c[key]).all() & (c[key]>=0).all()))
        for key in ("pair_observation","pair_constant"):
            self.assertEqual(tuple(c[key].shape),(1,5,4,9,9))
            self.assertTrue(bool(torch.isfinite(c[key]).all() & (c[key]>=0).all()))
        self.assertEqual(c["valid_unary"].dtype,torch.bool)
        self.assertEqual(c["valid_pair"].dtype,torch.bool)
        self.assertFalse(bool(c["valid_unary"][:,0,:2].any()))
        self.assertFalse(bool(c["valid_unary"][:,1,3:].any()))
        self.assertFalse(bool(c["valid_pair"][:,:2].any()))
        self.assertFalse(bool(c["valid_pair"][:,3:].any()))
        self.assertTrue(bool(c["valid_pair"][:,2,:,:,4].any()))

    def test_inference_signature_has_no_labels(self):
        names=set(inspect.signature(cache.build_joint_costs).parameters)
        self.assertFalse(names & {"gt","target","labels","fit_labels"})
        with self.assertRaises(TypeError):
            cache.build_joint_costs(*self.inputs,[.001,.0001],global_variance=.002,gt=self.inputs[1])

    def test_nnls_recovers_known_coefficients(self):
        x=torch.tensor([[1.,1.],[2.,1.],[3.,1.]],dtype=torch.float64)
        truth=torch.tensor([.2,.3],dtype=torch.float64);y=x@truth
        fit=cache.fit_nonnegative_variance({"xx":x.T@x,"xy":x.T@y,"yy":y@y,"count":3,"target_sum":y.sum()})
        self.assertTrue(torch.allclose(torch.tensor(fit["coefficients"],dtype=torch.float64),truth,atol=1e-12))
        self.assertAlmostEqual(fit["global_effective_variance"],float(y.mean()),12)

    def test_nnls_selects_boundary(self):
        x=torch.tensor([[1.,1.],[2.,1.],[3.,1.]],dtype=torch.float64)
        y=torch.tensor([3.,2.,1.],dtype=torch.float64)
        fit=cache.fit_nonnegative_variance({"xx":x.T@x,"xy":x.T@y,"yy":y@y,"count":3})
        self.assertEqual(fit["coefficients"][0],0.)
        self.assertAlmostEqual(fit["coefficients"][1],2.,12)

    def test_fit_calibration_never_loads_holdout(self):
        low,base,*_=self.inputs
        seen=[]
        rows=[{"name":"a","scene":"fit"},{"name":"b","scene":"holdout"}]
        def loader(row):
            if row["scene"]!="fit":
                raise AssertionError("Holdout must never be loaded for calibration")
            seen.append(row["name"])
            return {"lq":low,"base":base,"gt":base*.9}
        result=cache.fit_calibration(rows,["fit"],loader)
        self.assertEqual(seen,["a"])
        self.assertEqual(result["fit_names"],["a"])
        self.assertFalse(result["internal_holdout_labels_read"])

    def test_common_support_normalization_not_padding(self):
        a=torch.zeros(1,1,12);b=torch.tensor([1.,3.,100.,100.]*3).reshape(1,1,12)
        mask=torch.tensor([[[True,True,False,False]]])
        va=torch.ones_like(a);vb=torch.ones_like(a)
        score,constant,valid,fraction=cache._supported_ssd(a,b,va,vb,mask,2,.5,2.)
        self.assertAlmostEqual(float(score),2.5,6)
        self.assertAlmostEqual(float(constant),1.25,6)
        self.assertTrue(bool(valid));self.assertEqual(float(fraction),.5)
        rejected=cache._supported_ssd(a,b,va,vb,mask,2,.75,2.)
        self.assertFalse(bool(rejected[2]));self.assertEqual(float(rejected[0]),0.)

    def test_invalid_costs_cannot_benefit_from_padding(self):
        c=self.costs
        for key in ("unary_observation","unary_constant"):
            self.assertFalse(bool(c[key][~c["valid_unary"]].any()))
        for key in ("pair_observation","pair_constant"):
            self.assertFalse(bool(c[key][~c["valid_pair"]].any()))
        self.assertTrue(bool((c["metadata"]["unary_support_fraction"]>=.5)[c["valid_unary"]].all()))

    def test_effective_variance_scale_changes_only_cost_scale(self):
        doubled=cache.build_joint_costs(*self.inputs,[.002,.0002],global_variance=.004)
        for key in ("unary_observation","pair_observation","unary_constant","pair_constant"):
            self.assertTrue(torch.allclose(doubled[key]*2,self.costs[key],atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.equal(doubled["valid_pair"],self.costs["valid_pair"]))

    def test_private_shuffle_preserves_rng_and_values(self):
        prior=self.inputs[4];before=torch.random.get_rng_state().clone()
        a=cache.shuffled_addresses(prior);b=cache.shuffled_addresses(prior)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,prior))
        self.assertTrue(torch.equal(a.reshape(-1,64).sort(0).values,prior.reshape(-1,64).sort(0).values))

    def test_descriptor_identity_and_mode_distinction(self):
        low,base,flows,masks,prior,pixel=self.inputs
        fixed=prior[:,0:1].expand_as(prior).clone()
        c=cache.build_joint_costs(low,base,flows,masks,fixed,pixel,[.001,.0001],global_variance=.002)
        self.assertLess(float(c["descriptor_costs"]["prior"][...,4].max()),1e-6)
        valid=c["valid_unary"]
        self.assertGreater(float((c["descriptor_costs"]["prior"]-c["descriptor_costs"]["pixel"])[valid].abs().max()),.01)

    def test_zero_motion_unary_matches_independent_pixel_oracle(self):
        low,base,*_=self.inputs
        m=cache.observable_module();q,gain,low_l=cache.calibrated_observation(low,base)
        hp=q-m._lowpass(q)
        var=.001*gain.square()*low_l+.0001*gain.square()
        # t2 <- t0, zero candidate: no warp approximation or patch pooling before SSD.
        expected=((hp[:,:,2]-hp[:,:,0]).square()/(var[:,:,2]+var[:,:,0])).reshape(1,3,2,16,2,16).permute(0,2,4,1,3,5).reshape(1,4,-1).mean(-1)
        actual=self.costs["unary_observation"][:,0,2,:,4]
        self.assertTrue(torch.allclose(actual,expected,atol=2e-6,rtol=2e-6))

    def test_fractional_flow_warps_rgb_before_highpass(self):
        low,base,flows,masks,prior,pixel=self.inputs
        flows=flows.clone();flows[:,:,:,2,4:-4,4:-4]=.35
        # Original masks must remain geometrically valid at unshifted edges.
        c=cache.build_joint_costs(low,base,flows,masks,prior,pixel,[.001,.0001],global_variance=.002)
        m=cache.observable_module();q,gain,low_l=cache.calibrated_observation(low,base)
        yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing="ij")
        f=flows[0,0,:,2];grid=torch.stack((2*(xx+f[0]+.5)/32-1,2*(yy+f[1]+.5)/32-1),-1)[None]
        warped=F.grid_sample(q[:,:,0],grid,align_corners=False,padding_mode="border")
        donor=warped-m._lowpass(warped[:,:,None])[:,:,0]
        target=(q-m._lowpass(q))[:,:,2]
        variance=.001*gain.square()*low_l+.0001*gain.square()
        wvar=F.grid_sample(variance[:,:,0],grid,align_corners=False,padding_mode="border")
        expected=((target-donor).square()/(variance[:,:,2]+wvar)).reshape(1,3,2,16,2,16).permute(0,2,4,1,3,5).reshape(1,4,-1).mean(-1)
        self.assertTrue(torch.allclose(c["unary_observation"][:,0,2,:,4],expected,atol=2e-6,rtol=2e-6))

    def test_spawn_worker_preserves_order_and_cost_values(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        low,base,flow,mask,prior,pixel=self.inputs
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/"costs";out.mkdir()
            torch.save({"name":"synthetic","scene":"fit","tensors":{
                "lq":low,"base":base,"flow":flow,"valid_mask":mask}},root/"source.pt")
            torch.save({"name":"synthetic","scene":"fit","prior_address":prior,
                "pixel_address":pixel},root/"addresses.pt")
            row={"name":"synthetic","scene":"fit","source_path":str(root/"source.pt"),
                "address_path":str(root/"addresses.pt"),"source_sha256":"unit-source",
                "address_sha256":"unit-address"}
            args=[(i,row,str(out),[.001,.0001],.002) for i in range(2)]
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
                entries=list(pool.map(cache._cache_row,args))
            self.assertEqual([e["cache_path"] for e in entries],["joint_000.pt","joint_001.pt"])
            for entry in entries:
                value=torch.load(str(out/entry["cache_path"]),weights_only=True)["costs"]
                for key in ("unary_observation","pair_observation","unary_constant","pair_constant","valid_pair","valid_unary"):
                    self.assertTrue(torch.equal(value[key],self.costs[key]),key)
                self.assertEqual(cache.digest(out/entry["cache_path"]),entry["cache_sha256"])


if __name__=="__main__":
    unittest.main()
