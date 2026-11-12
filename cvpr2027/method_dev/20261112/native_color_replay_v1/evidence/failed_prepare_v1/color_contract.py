"""Explicit RGB/native color boundaries; no inferred checkpoint or array order.

Historical releases stay unchanged. Dataset provenance must declare array order
before calling this adapter. File extensions alone cannot identify NPY color.
"""
from pathlib import Path

import numpy as np
from PIL import Image

ORDERS=("RGB","BGR")


def _order(value):
    if value not in ORDERS:
        raise ValueError("Declare RGB or BGR explicitly; unknown/automatic order is unsupported")
    return value


def read_declared_image(path,*,array_order):
    """Read image files as RGB; explicitly declare NPY's stored channel order."""
    path=Path(path)
    if path.suffix.lower()!=".npy":
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"),dtype=np.float32)/255.
    _order(array_order)
    value=np.load(path,allow_pickle=False)
    if value.ndim==3 and value.shape[0]==3 and value.shape[-1]!=3:
        value=value.transpose(1,2,0)
    if value.ndim==2:value=np.repeat(value[...,None],3,axis=-1)
    if value.ndim!=3 or value.shape[-1]!=3:
        raise ValueError("Expected HWC/CHW three-channel or grayscale NPY image")
    if value.dtype==np.uint8:scale=255.
    elif value.dtype==np.uint16:scale=65535.
    elif value.dtype.kind=="f":scale=255. if value.max()>1.5 else 1.
    else:raise ValueError("Unsupported NPY image dtype")
    value=value.astype(np.float32)/scale
    if not np.isfinite(value).all() or value.min()<0 or value.max()>1:
        raise ValueError("Expected finite unit-range image")
    if array_order=="BGR":value=value[...,::-1]
    return np.ascontiguousarray(value)


def convert_video(value,*,source_order,target_order):
    """Non-mutating, differentiable permutation of a B,3,T,H,W tensor."""
    import torch
    _order(source_order);_order(target_order)
    if not isinstance(value,torch.Tensor) or value.ndim!=5 or value.shape[1]!=3:
        raise ValueError("Expected B,3,T,H,W video")
    if source_order==target_order:return value
    return value.index_select(1,torch.tensor((2,1,0),device=value.device))


def predict_native_as_rgb(predict_native,low_rgb,*,native_order):
    """Adapt the WHOLE frozen native pipeline, including its internal encoder.

    This restores a declared input/output contract. It does not retrain or repair
    any pretrained encoder that historically received an incorrect color order.
    Prediction accepts no labels, scores, automatic order or quality selection.
    """
    native=convert_video(low_rgb,source_order="RGB",target_order=native_order)
    prediction=predict_native(native)
    if (getattr(prediction,"shape",None)!=low_rgb.shape
            or prediction.dtype!=low_rgb.dtype or prediction.device!=low_rgb.device):
        raise ValueError("Native pipeline must return a matching video tensor")
    return convert_video(prediction,source_order=native_order,target_order="RGB")
