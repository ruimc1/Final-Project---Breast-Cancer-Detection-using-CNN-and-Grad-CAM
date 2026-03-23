# TODO: Basic unit tests. need to add more 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import torch
import numpy as np
from unittest.mock import patch
from PIL import Image
import cv2

from core.data import roi_crop_square, MammogramDataset
from core.models import build_model
from core.transforms import get_transforms


def test_roi_crop_returns_valid_image():
    with patch('cv2.imread') as mock_imread:
        mock_imread.return_value = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        img = roi_crop_square("dummy.jpg", None, target_size=224)
    assert isinstance(img, Image.Image)
    assert img.size == (224, 224) 


def test_build_model_returns_valid_model():
    model = build_model("SwinDAMFN")
    assert isinstance(model, torch.nn.Module)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_transforms_return_tensors():
    tf = get_transforms(mean=0.4782, std=0.2124, train=True)
    sample = {"image": np.random.randint(0, 255, (224, 224), dtype=np.uint8)}
    transformed = tf(**sample)
    assert isinstance(transformed["image"], torch.Tensor)
    # must be 1 channel
    assert transformed["image"].shape == (1, 224, 224)


def test_dataset_loads_without_error():
    df = pd.DataFrame([{
        "full_jpeg_path": "dummy.jpg",
        "roi_jpeg_path": None,
        "label": 1
    }])
    with patch('cv2.imread') as mock_imread:
        mock_imread.return_value = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        ds = MammogramDataset(df, get_transforms(mean=0.4782, std=0.2124, train=False))
        img, label = ds[0]
    
    assert len(ds) == 1
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, int)
