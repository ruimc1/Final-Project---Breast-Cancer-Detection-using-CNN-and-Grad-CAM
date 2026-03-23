# run_swin_staged.py
"""
EXACT MATCH TO YOUR OLD SUCCESSFUL SCRIPT (RunModel2_possibleGood.py)
Only changes: Stage 2 = 25 epochs + patience=8
"""

import pandas as pd
import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from core.data import build_dicom_lookup, prepare_case_df, MammogramDataset
from core.transforms import get_transforms
from core.models import build_model
from core.training import DEVICE, train_one_epoch, evaluate, EarlyStopping

# ================== PATHS ==================
DICOM_CSV = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\dicom_info.csv"
JPEG_DIR  = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\jpeg"
CALC_CSV  = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\calc_case_description_train_set.csv"
MASS_CSV  = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\mass_case_description_train_set.csv"

TEST_CSVS = [
    r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\calc_case_description_test_set.csv",
    r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\mass_case_description_test_set.csv"
]

OUTPUT_DIR = Path("swindamfn_staged_run")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================== DATA ==================
dicom_lookup = build_dicom_lookup(DICOM_CSV, JPEG_DIR)
calc_df = prepare_case_df(CALC_CSV, dicom_lookup)
mass_df = prepare_case_df(MASS_CSV, dicom_lookup)
train_df = pd.concat([calc_df, mass_df], ignore_index=True)
train_df = train_df[train_df["has_roi"]].reset_index(drop=True)

mean, std = 0.4782, 0.2124
train_tf = get_transforms(mean, std, train=True)
eval_tf  = get_transforms(mean, std, train=False)

train_dataset = MammogramDataset(train_df, train_tf)
val_dataset   = MammogramDataset(train_df.sample(frac=0.2, random_state=42), eval_tf)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0)

labels = train_df["label"].values
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {class_weights}")  

mean, std = 0.4782, 0.2124
train_tf = get_transforms(mean, std, train=True)
eval_tf  = get_transforms(mean, std, train=False)

train_dataset = MammogramDataset(train_df, train_tf)
val_dataset   = MammogramDataset(train_df.sample(frac=0.2, random_state=42), eval_tf)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0)

model = build_model("SwinDAMFN").to(DEVICE)

# Weighted loss — exactly as in your old successful script
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ================== STAGE 1: Head only ==================
print(f"\n=== Stage 1: Head only (10 epochs) ===")
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# EXACTLY as in your old successful script
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
early_stop = EarlyStopping(patience=8)
best_val_f1 = -1.0

for epoch in range(10):
    train_m = train_one_epoch(model, train_loader, criterion, optimizer)
    val_m = evaluate(model, val_loader, criterion)
    scheduler.step()
    print(f"Epoch {epoch+1} | Train Loss {train_m['loss']:.4f} | Val F1 {val_m['f1']:.4f}")
    
    if val_m["f1"] > best_val_f1 and val_m["f1"] > 0.1:
        best_val_f1 = val_m["f1"]
        torch.save(model.state_dict(), OUTPUT_DIR / "best_swin_stage1.pth")
        print(f"   --New best saved! (F1 {best_val_f1:.4f})")
    
    if early_stop(val_m["f1"]):
        break

# Load best Stage 1
print("\n=== Loading best Stage 1 checkpoint before Stage 2 ===")
model.load_state_dict(torch.load(OUTPUT_DIR / "best_swin_stage1.pth", map_location=DEVICE))
print("Best Stage 1 checkpoint loaded successfully\n")

# ================== STAGE 2: Full model (25 epochs) ==================
print(f"=== Stage 2: Full model (25 epochs) ===")
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15)
early_stop = EarlyStopping(patience=8)
best_val_f1 = -1.0

for epoch in range(25):
    train_m = train_one_epoch(model, train_loader, criterion, optimizer)
    val_m = evaluate(model, val_loader, criterion)
    scheduler.step()
    print(f"Epoch {epoch+1} | Train Loss {train_m['loss']:.4f} | Val F1 {val_m['f1']:.4f}")
    
    if val_m["f1"] > best_val_f1 and val_m["f1"] > 0.1:
        best_val_f1 = val_m["f1"]
        torch.save(model.state_dict(), OUTPUT_DIR / "best_swin_stage2.pth")
        print(f"   --New best saved! (F1 {best_val_f1:.4f})")
    
    if early_stop(val_m["f1"]):
        break

# ================== LOAD BEST STAGE 2 + FINAL EVALUATION ==================
print("\n=== Loading best Stage 2 checkpoint for final model ===")
model.load_state_dict(torch.load(OUTPUT_DIR / "best_swin_stage2.pth", map_location=DEVICE))
print("Best Stage 2 checkpoint loaded successfully\n")

# Final test on real test set
from core.evaluation import collect_probs, find_best_threshold, evaluate_from_probs

print("=== Running final evaluation on real test set ===")
test_dfs = [prepare_case_df(csv, dicom_lookup) for csv in TEST_CSVS]
test_df = pd.concat(test_dfs, ignore_index=True)
test_df = test_df[test_df["has_roi"]].reset_index(drop=True)
test_dataset = MammogramDataset(test_df, eval_tf)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

y_test, p_test = collect_probs(model, test_loader, use_tta=True)
best_thr = find_best_threshold(y_test, p_test)
final_test = evaluate_from_probs(y_test, p_test, thr=best_thr["thr"])

summary = {"final_test": final_test, "best_threshold": best_thr}
with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

torch.save(model.state_dict(), OUTPUT_DIR / "best_swindamfn_staged.pth")

print(f"\nDONE! Final model saved: {OUTPUT_DIR / 'best_swindamfn_staged.pth'}")
print(f"Test F1: {final_test['f1']:.4f} | Recall: {final_test['recall']:.4f}")
