"""Small real-Torch tests of posterior math, frozen-anchor parity, and gradients."""
import copy
import importlib.util
import inspect
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class JointObservationSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/"models"/"joint_observation_search.py"
        spec = importlib.util.spec_from_file_location("_test_joint_search", path)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def setUp(self):
        torch.manual_seed(19)
        node = torch.nn.Sequential(torch.nn.Conv2d(19,32,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32,32,1), torch.nn.ReLU(), torch.nn.Conv2d(32,2,1))
        self.state = {k:v.clone() for k,v in node.state_dict().items()}
        self.norm = {"feature_mean":[.03]*19,"feature_std":[.7]*19}

    def model(self, **kwargs):
        return self.m.JointObservationSearch(self.norm, self.state, **kwargs)

    def fixture(self, batch=1, frames=6, height=32):
        low = torch.rand(batch,3,frames,height,height)*.3+.05
        base = torch.rand_like(low)*.4+.3
        flow = torch.full((batch,2,2,frames,height,height),.2)
        masks = torch.zeros(batch,2,1,frames,height,height,dtype=torch.bool)
        masks[:,0,:,2:,1:-1,1:-1] = True
        masks[:,1,:,:-2,1:-1,1:-1] = True
        shape = (batch,2,frames,(height//16)**2,9)
        vu = torch.ones(shape,dtype=torch.bool)
        vu[:,0,:2] = False
        vu[:,1,-2:] = False
        pair = vu[:,0,...,:,None] & vu[:,1,...,None,:]
        p = {"unary_observation":torch.rand(shape)*3,
             "unary_constant":torch.rand(shape)*2,
             "pair_observation":torch.rand(pair.shape)*2,
             "pair_constant":torch.rand(pair.shape)*4,
             "descriptor_costs":{k:torch.rand(shape)*2 for k in self.m._G.ADDRESS_MODES},
             "valid_unary":vu,"valid_pair":pair}
        return low,base,flow,masks,p

    def test_zero_pair_reduces_exactly_to_independent_posterior(self):
        p = self.fixture()[-1]
        p["pair_observation"].zero_()
        a = self.model().correspondence_posterior(p)
        b = self.model(search_mode="independent").correspondence_posterior(p)
        for key in ("marginals","confidence","expected_shifts"):
            torch.testing.assert_close(a[key],b[key],rtol=2e-5,atol=3e-7)
        # Arbitrary non-Cartesian feasibility is also shared by both modes.
        p["valid_pair"][...,::2,1::2] = False
        a = self.model().correspondence_posterior(p)
        b = self.model(search_mode="independent").correspondence_posterior(p)
        for key in ("marginals","confidence","expected_shifts"):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_pair_intervention_changes_only_joint_and_constant_replaces_only_scale(self):
        p = self.fixture()[-1]
        altered = copy.deepcopy(p)
        altered["pair_observation"][...,0,:] += 10
        independent = self.model(search_mode="independent")
        torch.testing.assert_close(independent.correspondence_posterior(p)["marginals"],
            independent.correspondence_posterior(altered)["marginals"],rtol=0,atol=0)
        joint = self.model()
        self.assertFalse(torch.equal(joint.correspondence_posterior(p)["marginals"],
            joint.correspondence_posterior(altered)["marginals"]))
        constant = self.model(uncertainty_mode="constant")
        substituted = copy.deepcopy(p)
        substituted["unary_observation"] = p["unary_constant"]
        substituted["pair_observation"] = p["pair_constant"]
        torch.testing.assert_close(constant.correspondence_posterior(p)["marginals"],
            joint.correspondence_posterior(substituted)["marginals"],rtol=0,atol=0)

    def test_single_missing_donor_and_no_pair_fallback(self):
        p = self.fixture()[-1]
        p["valid_unary"][:,0] = False
        p["valid_pair"].zero_()
        a = self.model().correspondence_posterior(p)
        b = self.model(search_mode="independent").correspondence_posterior(p)
        self.assertEqual(float(a["marginals"][:,0].sum()),0)
        torch.testing.assert_close(a["marginals"][:,1],b["marginals"][:,1],rtol=0,atol=0)
        p["valid_unary"].fill_(True)
        a = self.model().correspondence_posterior(p)
        self.assertEqual(float(a["marginals"].sum()),0)
        self.assertEqual(float(a["confidence"].sum()),0)

    def test_uniform_uncertain_and_single_certain_hypothesis(self):
        p = self.fixture()[-1]
        p["unary_observation"].zero_()
        for v in p["descriptor_costs"].values():
            v.zero_()
        p["valid_unary"].zero_()
        p["valid_unary"][...,[0,2,6,8]] = True
        p["valid_pair"] = p["valid_unary"][:,0,...,:,None] & p["valid_unary"][:,1,...,None,:]
        out = self.model(search_mode="independent").correspondence_posterior(p)
        torch.testing.assert_close(out["confidence"],torch.zeros_like(out["confidence"]),atol=1e-7,rtol=0)
        p["valid_unary"].zero_()
        p["valid_unary"][...,3] = True
        p["valid_pair"] = p["valid_unary"][:,0,...,:,None] & p["valid_unary"][:,1,...,None,:]
        out = self.model().correspondence_posterior(p)
        self.assertTrue(torch.equal(out["confidence"],torch.ones_like(out["confidence"])))
        self.assertTrue(torch.equal(out["expected_shifts"],torch.tensor([-2.,0.]).expand_as(out["expected_shifts"])))

    def test_five_modes_same_initial_state_and_exact_frozen_anchor(self):
        inputs = self.fixture(batch=2)
        low,base,flows,masks,_ = inputs
        address = torch.zeros(2,6,4,8)
        old = self.m._G.ObservationGeometry(self.norm,self.state,descriptor_dim=8,
                                             hidden_dim=4,address_dim=4)
        expected = old(low,base,flows,masks,address,address)["pred_pixels"]
        configs = [{},{"search_mode":"independent"},{"uncertainty_mode":"constant"},
                   {"address_mode":"pixel"},{"address_mode":"shuffled_prior"}]
        first = None
        for config in configs:
            model = self.model(**config)
            if first is None:
                first = {k:v.clone() for k,v in model.state_dict().items()}
            self.assertTrue(all(torch.equal(first[k],v) for k,v in model.state_dict().items()))
            self.assertEqual(sum(v.numel() for v in model.parameters() if v.requires_grad),4)
            result = model(*inputs)
            self.assertTrue(torch.equal(result["pred_pixels"],expected))
            self.assertEqual(float(result["offsets"].abs().sum()),0)
            clone = self.m.JointObservationSearch(frozen_node_state=self.state,**model.get_config())
            clone.load_state_dict(model.state_dict(),strict=True)
            self.assertTrue(torch.equal(clone(*inputs)["pred_pixels"],expected))

    def test_candidate_axis_and_dense_batch_direction_time_layout(self):
        inputs = self.fixture(batch=2)
        p = inputs[-1]
        p["valid_unary"].zero_()
        for b in range(2):
            for d in range(2):
                for t in range(6):
                    p["valid_unary"][b,d,t,:,(b*3+d*2+t)%9] = True
        p["valid_pair"] = p["valid_unary"][:,0,...,:,None] & p["valid_unary"][:,1,...,None,:]
        model = self.model(search_mode="independent")
        with torch.no_grad():
            model.amplitude_raw.fill_(1)
        result = model(*inputs,return_aux=True)
        for b in range(2):
            for d in range(2):
                for t in range(2,4):
                    xy = torch.tensor(self.m.SHIFTS_XY[(b*3+d*2+t)%9],dtype=torch.float32)
                    torch.testing.assert_close(result["offsets"][b,d,:,t,16,16],xy*torch.tanh(torch.tensor(1.)))
        self.assertLessEqual(float(result["offsets"].abs().max()),2)
        self.assertFalse(bool((result["aux"]["conservative_masks"] & ~inputs[3]).any()))

    def test_two_optimizer_steps_retain_all_gradients_and_freeze_selector(self):
        for mode in ("joint","independent"):
            inputs = self.fixture()
            model = self.model(search_mode=mode).train()
            with torch.no_grad():
                model.amplitude_raw.fill_(.35)
                target = model(*inputs)["pred_pixels"].clone()
                model.amplitude_raw.zero_()
            original = {k:v.clone() for k,v in model.node.state_dict().items()}
            initial = model(*inputs)["pred_pixels"].detach()
            optim = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=.01)
            for _ in range(2):
                optim.zero_grad(set_to_none=True)
                (model(*inputs)["pred_pixels"]-target).square().mean().backward()
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    else:
                        self.assertIsNone(parameter.grad)
                optim.step()
                model.project_parameters_()
            self.assertGreater(float(model.amplitude_raw),0)
            self.assertGreater(float(model.descriptor_log_weight.grad.abs()),0)
            self.assertGreater(float(model.regularization_log_weight.grad.abs()),0)
            self.assertGreater(float(model.log_temperature.grad.abs()),0)
            self.assertFalse(torch.equal(model(*inputs)["pred_pixels"],initial))
            self.assertFalse(model.node.training)
            self.assertTrue(all(torch.equal(original[k],v) for k,v in model.node.state_dict().items()))

    def test_fixed_observable_costs_and_rgb_have_no_gradient(self):
        inputs = list(self.fixture())
        for value in inputs[:3]:
            value.requires_grad_()
        p = inputs[-1]
        for key in ("unary_observation","pair_observation","unary_constant","pair_constant"):
            p[key].requires_grad_()
        for value in p["descriptor_costs"].values():
            value.requires_grad_()
        model = self.model()
        with torch.no_grad():
            model.amplitude_raw.fill_(.2)
        model(*inputs)["pred_pixels"].square().mean().backward()
        self.assertTrue(all(v.grad is None for v in inputs[:3]))
        self.assertTrue(all(v.grad is None for v in p["descriptor_costs"].values()))
        self.assertIsNone(p["unary_observation"].grad)
        self.assertIsNone(p["pair_observation"].grad)

    def test_all_invalid_exact_base_and_parameter_projection(self):
        inputs = list(self.fixture())
        inputs[3].zero_()
        inputs[-1]["valid_unary"].zero_()
        inputs[-1]["valid_pair"].zero_()
        model = self.model()
        with torch.no_grad():
            model.amplitude_raw.fill_(-10)
            model.log_temperature.fill_(100)
            model.descriptor_log_weight.fill_(-100)
            model.regularization_log_weight.fill_(100)
        model.project_parameters_()
        self.assertEqual(float(model.amplitude_raw),0)
        self.assertEqual(float(model.log_temperature),4)
        self.assertEqual(float(model.descriptor_log_weight),-4)
        self.assertEqual(float(model.regularization_log_weight),2)
        result = model(*inputs)
        self.assertTrue(torch.equal(result["pred_pixels"],inputs[1]))
        self.assertEqual(float(result["offsets"].abs().sum()),0)

    def test_fail_closed_costs_validity_config_and_gt_firewall(self):
        inputs = self.fixture()
        model = self.model()
        self.assertNotIn("gt",inspect.signature(model.forward).parameters)
        bad = copy.deepcopy(inputs[-1]); bad["gt"] = inputs[1]
        with self.assertRaises(ValueError): model(*inputs[:-1],bad)
        bad = copy.deepcopy(inputs[-1]); bad["unary_observation"].flatten()[0] = float("nan")
        with self.assertRaises(ValueError): model(*inputs[:-1],bad)
        bad = copy.deepcopy(inputs[-1]); bad["valid_unary"] = bad["valid_unary"].float()
        with self.assertRaises(ValueError): model(*inputs[:-1],bad)
        bad = copy.deepcopy(inputs[-1]); bad["valid_pair"].fill_(True)
        with self.assertRaises(ValueError): model(*inputs[:-1],bad)
        with self.assertRaises(ValueError): self.model(search_patch=8)
        with self.assertRaises(ValueError): self.model(amplitude_rule="unknown")
        with self.assertRaises(ValueError): self.model(uncertainty_mode="target_fitted")


if __name__ == "__main__":
    unittest.main()
