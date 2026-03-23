
""""
https://medium.com/analytics-vidhya/test-time-augmentation-using-pytorch-3da02d0a3188
https://docs.pytorch.org/docs/stable/index.html
"""

import albumentations as A #not a fan of this naming but that is hoe the example imported it
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def get_train_transform(mean: float, std: float, image_size: int = 224) -> A.Compose:
    """
    As in the litriture add augmentations to try to boost performance
    Examples from https://albumentations.ai/docs/examples/example/ and others
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, fill_value=0, p=0.3),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])


def get_eval_transform(mean: float, std: float, image_size: int = 224) -> A.Compose:
    """Clean eval transform (used for validation, test, and Grad-CAM inference)."""
    return A.Compose([
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])


def get_transforms(mean: float, std: float, image_size: int = 224, train: bool = True) -> A.Compose:
    """
    Simple wrapper for convinience to get either train or eval transforms based on the train flag
    """
    if train:
        return get_train_transform(mean, std, image_size)
    return get_eval_transform(mean, std, image_size)