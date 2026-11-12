"""Integration contracts for source-only joint-search loading and optimization.

All tensors/labels in these tests are synthetic. The two-step test executes the
real trainer and real model; only metadata/plan/filesystem setup is shortened.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "real Torch CPU is required")
class TrainJointObservationSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code = Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(code/"tools"))
        spec = importlib.util.spec_from_file_location("_test_joint_search_trainer",
            code/"tools/train_joint_observation_search.py")
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def reference(self):
        torch.manual_seed(23)
        node = torch.nn.Sequential(torch.nn.Conv2d(19,32,3,padding=1),torch.nn.ReLU(),
            torch.nn.Conv2d(32,32,1),torch.nn.ReLU(),torch.nn.Conv2d(32,2,1))
        return {"model":{k:v.clone() for k,v in node.state_dict().items()}}

    def metadata(self,row):
        return {"normalization":{"feature_mean":[.03]*19,"feature_std":[.7]*19},
            "per_sequence":[row],"reference_head_path":"synthetic_reference.pt",
            "split":{"fit":[row["scene"]],"internal_holdout":["SDSD_indoor/heldout"]},
            "calibration_sha256":"a"*64,"reference_head_sha256":"b"*64,
            "fit_only_mse_normalizer":.01}

    def write_fixture(self,folder,full=False):
        h,t = (256,16) if full else (32,6)
        torch.manual_seed(31)
        low=torch.rand(1,3,t,h,h)*.3+.05
        base=torch.rand_like(low)*.4+.3
        flow=torch.full((1,2,2,t,h,h),.2)
        masks=torch.zeros(1,2,1,t,h,h,dtype=torch.bool)
        masks[:,0,:,2:,1:-1,1:-1]=True
        masks[:,1,:,:-2,1:-1,1:-1]=True
        shape=(1,2,t,(h//16)**2,9)
        vu=torch.ones(shape,dtype=torch.bool);vu[:,0,:2]=False;vu[:,1,-2:]=False
        vp=vu[:,0,...,:,None]&vu[:,1,...,None,:]
        costs={"unary_observation":torch.rand(shape)*3,"unary_constant":torch.rand(shape)*2,
            "pair_observation":torch.rand(vp.shape)*2,"pair_constant":torch.rand(vp.shape)*4,
            "valid_unary":vu,"valid_pair":vp,
            "descriptor_costs":{k:torch.rand(shape)*2 for k in ("prior","pixel","shuffled_prior")}}
        identity={"name":"SDSD_indoor/pair1_2","scene":"SDSD_indoor/pair1"}
        row=dict(identity,source_path=str(folder/"source.pt"),cache_path="cost.pt",
                 source_sha256="c"*64,address_sha256="d"*64)
        source=dict(identity,tensors={"lq":low,"base":base,"flow":flow,"valid_mask":masks,
                                     "gt":torch.full_like(low,.5)})
        cached=dict(identity,source_sha256=row["source_sha256"],address_sha256=row["address_sha256"],costs=costs)
        torch.save(source,row["source_path"]);torch.save(cached,folder/row["cache_path"])
        (folder/"manifest.json").write_text('{"synthetic_test": true}\n')
        return row,source,cached

    def test_load_sample_preserves_context_but_gt_changes_only_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row,source,cached=self.write_fixture(root)
            inputs,labels=self.runner.load_sample(root,row)
            self.assertEqual(len(inputs),5)
            for actual,key in zip(inputs[:4],("lq","base","flow","valid_mask")):
                self.assertTrue(torch.equal(actual,source["tensors"][key]))
            self.assertNotIn("gt",inputs[-1])
            self.assertTrue(torch.equal(labels,source["tensors"]["gt"]))
            # load_sample uses mmap; preserve values before this deliberate
            # synthetic overwrite (actual source caches are immutable).
            original_labels=labels.clone()
            source["tensors"]["gt"].zero_();torch.save(source,row["source_path"])
            changed_inputs,changed_labels=self.runner.load_sample(root,row)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs[:4],changed_inputs[:4])))
            self.assertTrue(torch.equal(inputs[-1]["pair_observation"],changed_inputs[-1]["pair_observation"]))
            self.assertFalse(torch.equal(original_labels,changed_labels))

    def test_load_sample_rejects_source_cost_identity_and_payload_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row,source,cached=self.write_fixture(root)
            for which,field in ((0,"name"),(0,"scene"),(1,"name"),(1,"scene"),
                                 (1,"source_sha256"),(1,"address_sha256")):
                with self.subTest(payload=which,field=field):
                    payloads=[dict(source),dict(cached)];payloads[which][field]="wrong"
                    torch.save(payloads[0],row["source_path"]);torch.save(payloads[1],root/row["cache_path"])
                    with self.assertRaises(ValueError):self.runner.load_sample(root,row)

    def test_five_declared_modes_freeze_same_selector_and_have_four_scalars(self):
        expected={"joint_prior":("joint","prior","calibrated"),
            "independent_prior":("independent","prior","calibrated"),
            "constant_prior":("joint","prior","constant"),
            "joint_pixel":("joint","pixel","calibrated"),
            "joint_shuffled":("joint","shuffled_prior","calibrated")}
        self.assertEqual(self.runner.MODES,expected)
        meta=self.metadata({"name":"SDSD_indoor/pair1","scene":"SDSD_indoor/pair1"})
        reference=self.reference();fingerprints=set()
        for mode,triple in expected.items():
            model=self.runner.construct(meta,reference,mode).train()
            conf=model.get_config()
            self.assertEqual(tuple(conf[k] for k in ("search_mode","address_mode","uncertainty_mode")),triple)
            trainable={k:v for k,v in model.named_parameters() if v.requires_grad}
            self.assertEqual(set(trainable),{"amplitude_raw","descriptor_log_weight",
                                             "regularization_log_weight","log_temperature"})
            self.assertEqual(sum(v.numel() for v in trainable.values()),4)
            self.assertTrue(all(not p.requires_grad for p in model.node.parameters()))
            self.assertFalse(model.node.training)
            self.assertTrue(all(torch.equal(reference["model"][k],v) for k,v in model.node.state_dict().items()))
            fingerprints.add(self.runner.state_digest(model.state_dict()))
        self.assertEqual(len(fingerprints),1)

    def test_actual_two_step_trainer_projects_parameters_and_saves_finite_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/"prepared";prepared.mkdir()
            row,source,cached=self.write_fixture(prepared,full=True)
            meta=self.metadata(row);reference=self.reference()
            model=self.runner.construct(meta,reference,"joint_prior")
            with torch.no_grad():
                model.amplitude_raw.fill_(.35)
                inputs,_=self.runner.load_sample(prepared,row)
                source["tensors"]["gt"]=model(*inputs)["pred_pixels"].clone()
                model.amplitude_raw.zero_()
            torch.save(source,row["source_path"])
            initial=self.runner.state_digest(model.state_dict())
            frozen=self.runner.state_digest(model.node.state_dict())
            plan={"seed":0,"steps":2,"batches":[{"clip_index":0,"top":64,"left":64,
                    "frames":list(range(0,16,2))} for _ in range(2)]}
            def new_directory(path):
                path=Path(path);path.mkdir(exist_ok=False);return path
            with mock.patch.object(self.runner,"STEPS",2), \
                 mock.patch.object(self.runner,"checked_metadata",return_value=meta), \
                 mock.patch.object(self.runner,"make_plan",return_value=plan), \
                 mock.patch.object(self.runner,"reference_checkpoint",return_value=reference), \
                 mock.patch.object(self.runner,"new_directory",side_effect=new_directory), \
                 mock.patch.object(self.runner,"source_hashes",return_value={"synthetic":"e"*64}), \
                 mock.patch.object(self.runner,"construct",return_value=model), \
                 mock.patch.object(model,"project_parameters_",wraps=model.project_parameters_) as project:
                self.runner.train(prepared,root/"run","joint_prior",0)
            self.assertEqual(project.call_count,2)
            ck=torch.load(root/"run/final.pt",map_location="cpu",weights_only=True)
            done=json.loads((root/"run/complete.json").read_text())
            curve=[json.loads(x) for x in (root/"run/curve.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in curve],[1,2])
            self.assertTrue(all(0<=r["amplitude"]<1 for r in curve))
            self.assertEqual(done["checkpoint_sha256"],self.runner.digest(root/"run/final.pt"))
            self.assertEqual(done["frozen_node_state_sha256"],frozen)
            self.assertTrue(done["frozen_node_unchanged"])
            self.assertEqual(ck["steps"],2)
            self.assertEqual(ck["provenance"]["initial_state_sha256"],initial)
            self.assertNotEqual(ck["model_state_sha256"],initial)
            self.assertEqual(self.runner.state_digest(model.node.state_dict()),frozen)
            for p in model.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
                else:self.assertIsNone(p.grad)

    def test_budget_mismatch_rejected_before_model_or_data_loading(self):
        meta=self.metadata({"name":"SDSD_indoor/pair1","scene":"SDSD_indoor/pair1"})
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(self.runner,"checked_metadata",return_value=meta), \
             mock.patch.object(self.runner,"new_directory",return_value=Path(directory)), \
             mock.patch.object(self.runner,"make_plan",return_value={"steps":399}), \
             mock.patch.object(self.runner,"load_sample") as loader, \
             mock.patch.object(self.runner,"construct") as model:
            with self.assertRaises(ValueError):self.runner.train("unused",directory,"joint_prior",0)
            loader.assert_not_called();model.assert_not_called()


if __name__=="__main__":unittest.main()
