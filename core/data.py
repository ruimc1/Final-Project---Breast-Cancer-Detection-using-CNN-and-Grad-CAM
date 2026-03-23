import os
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Dict, Tuple
from torch.utils.data import Dataset
import torch
import pandas as pd

def extract_uid(path_str: object) -> Optional[str]:
    """
    Extract UID from path string 
    handling NaN and weird formats had once 
    """
    if pd.isna(path_str):
        return None
    path_str = str(path_str).strip().splitlines()[0]
    parts = path_str.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else None


def map_label(pathology: object) -> Optional[int]:
    """MALIGNANT = 1, BENIGN = 0"""

    if pathology == "MALIGNANT":
        return 1
    if pathology in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        return 0
    return None


def build_dicom_lookup(dicom_csv_path: str, jpeg_root: str) -> Dict[Tuple[str, str], str]:
    """
    Build lookup for full mammogram roi mask and the crpped images
    """
    dicom_df = pd.read_csv(dicom_csv_path)
    dicom_df["jpeg_path"] = dicom_df["image_path"].apply(
        lambda x: x.replace("CBIS-DDSM/jpeg", jpeg_root).replace("/", os.sep)
    )
    dicom_df["uid"] = dicom_df["image_path"].apply(extract_uid)
    dicom_df["series_norm"] = dicom_df["SeriesDescription"].fillna("").str.strip().str.lower()

    keep_types = {"full mammogram images", "cropped images", "roi mask images"}
    dicom_df = dicom_df[dicom_df["series_norm"].isin(keep_types)].copy()

    return {
        (row["uid"], row["series_norm"]): row["jpeg_path"]
        for _, row in dicom_df.iterrows()
    }


def prepare_case_df(case_csv_path: str, dicom_lookup: Dict[Tuple[str, str], str]) -> pd.DataFrame:
    """
    Jpeg paths were messed up in the original csv so need to rebuild them using the dicom lookup
    Also extract labels and filter out rows without valid paths or labels
    """
    df = pd.read_csv(case_csv_path)
    df["full_uid"] = df["image file path"].apply(extract_uid)
    df["roi_uid"] = df["ROI mask file path"].apply(extract_uid)

    df["full_jpeg_path"] = df["full_uid"].apply(lambda uid: dicom_lookup.get((uid, "full mammogram images")))
    df["roi_jpeg_path"] = df["roi_uid"].apply(lambda uid: dicom_lookup.get((uid, "roi mask images")))

    df["label"] = df["pathology"].apply(map_label)
    df = df.dropna(subset=["label", "full_jpeg_path"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["full_jpeg_path"].apply(os.path.exists)].copy()
    df["has_roi"] = df["roi_jpeg_path"].apply(lambda x: isinstance(x, str) and os.path.exists(x))

    return df.reset_index(drop=True)


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name} rows: {len(df)}")
    print(df["label"].value_counts().sort_index())
    if "has_roi" in df.columns:
        print("Has ROI:", df["has_roi"].value_counts())


# ============================================================
# Image rocessing 
# ============================================================

def apply_clahe(gray_img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def roi_crop_square(img_path: str, roi_path: Optional[str], target_size: int = 224, pad: int = 20) -> Image.Image:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    if roi_path is None or not isinstance(roi_path, str) or not os.path.exists(roi_path):
        img = apply_clahe(img)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)

    mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        img = apply_clahe(img)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        img = apply_clahe(img)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(img)

    x_min, x_max = xs.min() - pad, xs.max() + pad
    y_min, y_max = ys.min() - pad, ys.max() + pad
    h, w = img.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    side = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    crop = img[y1:y2, x1:x2]
    crop = apply_clahe(crop)
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(crop)


# ============================================================
# Dataset
# ============================================================

class MammogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, image_size: int = 224, use_roi_crop: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.use_roi_crop = use_roi_crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        full_path = row["full_jpeg_path"]
        roi_path = row.get("roi_jpeg_path") if self.use_roi_crop else None
        label = int(row["label"])

        img_pil = roi_crop_square(full_path, roi_path, target_size=self.image_size)

        if self.transform:
            img_np = np.array(img_pil)
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(np.array(img_pil)).unsqueeze(0).float() / 255.0

        return img_tensor, label


# ============================================================
# Preview of images for report
# ============================================================

import matplotlib.pyplot as plt

def preview_single_roi_raw_and_clahe(df: pd.DataFrame, idx: int, target_size: int = 384):
    
    row = df.iloc[idx]
    full_path = row["full_jpeg_path"]
    roi_path = row["roi_jpeg_path"]

    print("Full image path:", full_path)
    print("ROI mask path: ", roi_path)
    print("Label:", row["label"])
    print("-" * 100)

    full_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE) if isinstance(roi_path, str) else None

    # Crop without CLAHE first to see difference  
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0) if mask is not None else ([], [])
    if len(xs) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        pad = 20
        h, w = img.shape
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        side = max(x_max - x_min, y_max - y_min)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        raw_crop = img[y1:y2, x1:x2]
        raw_crop_resized = cv2.resize(raw_crop, (target_size, target_size))
    else:
        raw_crop_resized = np.zeros((target_size, target_size), dtype=np.uint8)

    clahe_crop = roi_crop_square(full_path, roi_path, target_size=target_size)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 4, 1); plt.title("Full Mammogram"); plt.imshow(full_img, cmap="gray"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.title("ROI Mask")
    if mask is not None:
        plt.imshow(mask, cmap="gray")
    else:
        plt.text(0.5, 0.5, "No ROI", ha="center", va="center")
    plt.axis("off")
    plt.subplot(1, 4, 3); plt.title("Raw Square Crop (no CLAHE)"); plt.imshow(raw_crop_resized, cmap="gray"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.title("CLAHE Crop (model input)"); plt.imshow(clahe_crop, cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.show()
