"""Color-boundary tests using observable synthetic images and native functions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location("_declared_colors",Path(__file__).with_name("color_contract.py"))
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


class FileColorTests(unittest.TestCase):
    def test_official_bgr_array_and_rgb_png_have_identical_canonical_pixels(self):
        rgb=np.zeros((4,5,3),dtype=np.uint8)
        rgb[...,0]=243;rgb[...,1]=91;rgb[...,2]=17
        with tempfile.TemporaryDirectory() as folder:
            npy=Path(folder)/"stored.npy";png=Path(folder)/"stored.png"
            np.save(npy,rgb[...,::-1]);Image.fromarray(rgb).save(png)
            a=C.read_declared_image(npy,array_order="BGR")
            b=C.read_declared_image(png,array_order="BGR")
            self.assertTrue(np.array_equal(a,b))
            self.assertFalse(np.array_equal(C.read_declared_image(npy,array_order="RGB"),b))
            self.assertTrue(a.flags.c_contiguous)

    def test_uint16_chw_preserves_scaling_and_permutation(self):
        bgr=np.stack([np.full((5,7),v,dtype=np.uint16) for v in (100,2000,50000)])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"frame.npy";np.save(path,bgr)
            result=C.read_declared_image(path,array_order="BGR")
            self.assertEqual(result.shape,(5,7,3))
            np.testing.assert_array_equal(result[0,0],np.array((50000,2000,100),dtype=np.float32)/65535.)

    def test_unknown_array_order_is_rejected_before_reading(self):
        for order in (None,"auto","bgr","unknown"):
            with self.subTest(order=order),self.assertRaises(ValueError):
                C.read_declared_image("missing.npy",array_order=order)

    def test_nonfinite_and_malformed_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.npy"
            for array in (np.full((4,5,3),np.nan),np.zeros((4,5,2))):
                np.save(path,array)
                with self.assertRaises(ValueError):C.read_declared_image(path,array_order="BGR")


class NativeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch

    def test_entire_pipeline_receives_native_channels_once_and_returns_rgb(self):
        t=self.torch
        native=t.arange(2*3*4*5*7,dtype=t.float32).reshape(2,3,4,5,7)/1000
        rgb=native[:,[2,1,0]].clone();unchanged=rgb.clone();calls=[]
        def pipeline(x):
            calls.append(x.clone())
            # Non-permutation-equivariant: catches output-only channel swapping.
            return t.stack((x[:,0]+2*x[:,1],x[:,1]+3*x[:,2],x[:,2]+4*x[:,0]),1)
        expected=pipeline(native)[:,[2,1,0]];calls.clear()
        result=C.predict_native_as_rgb(pipeline,rgb,native_order="BGR")
        self.assertEqual(len(calls),1)
        self.assertTrue(t.equal(calls[0],native))
        self.assertTrue(t.equal(result,expected))
        self.assertTrue(t.equal(rgb,unchanged))

    def test_rgb_native_is_identity_contract_and_gradient_is_preserved(self):
        t=self.torch;x=t.ones(1,3,2,4,4,requires_grad=True)
        result=C.predict_native_as_rgb(lambda v:v*2,x,native_order="RGB")
        result.sum().backward();self.assertTrue(t.equal(x.grad,t.full_like(x,2)))
        self.assertIs(C.convert_video(x,source_order="RGB",target_order="RGB"),x)

    def test_bad_native_order_rejected_before_predictor(self):
        t=self.torch;calls=[]
        with self.assertRaises(ValueError):
            C.predict_native_as_rgb(lambda x:calls.append(x),t.ones(1,3,2,4,4),native_order="auto")
        self.assertFalse(calls)

    def test_bad_native_output_shape_rejected(self):
        t=self.torch
        with self.assertRaises(ValueError):
            C.predict_native_as_rgb(lambda v:v[:,:,:1],t.ones(1,3,2,4,4),native_order="BGR")


if __name__=="__main__":unittest.main()
