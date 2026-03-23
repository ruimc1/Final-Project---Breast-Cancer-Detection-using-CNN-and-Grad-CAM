"""
Main training 
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader 

import json

from core.data import build_dicom_lookup, prepare_case_df, MammogramDataset
from core.transforms import get_transforms
from core.models import build_model
from core.training import DEVICE, train_staged_resnet50, train_one_epoch, evaluate, EarlyStopping, save_history
from core.evaluation import collect_probs, find_best_threshold, evaluate_from_probs


#function is large break it up
#Use seperate Swin training module for now for staging 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "resnet50", "swindamfn"], required=True)
    args = parser.parse_args()

    # ================== CONFIG ==================
    DICOM_CSV = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\dicom_info.csv"
    JPEG_DIR  = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\jpeg"
    CALC_CSV  = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\calc_case_description_train_set.csv"
    MASS_CSV  = r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\mass_case_description_train_set.csv" 


    TEST_CSVS = [
    r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\calc_case_description_test_set.csv",
    r"C:\Users\User\OneDrive\Documents\study\Final Project\Project\Data\Kaggle CBIS-DDSM\csv\mass_case_description_test_set.csv"
]
    
    OUTPUT_DIR = Path(f"{args.model}_run")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    dicom_lookup = build_dicom_lookup(DICOM_CSV, JPEG_DIR)
    calc_df = prepare_case_df(CALC_CSV, dicom_lookup) 
    mass_df = prepare_case_df(MASS_CSV, dicom_lookup)
    train_df = pd.concat([calc_df, mass_df], ignore_index=True)
    train_df = train_df[train_df["has_roi"]].reset_index(drop=True)

    # Transforms
    mean, std = 0.4782, 0.2124
    train_tf = get_transforms(mean, std, train=True)
    eval_tf  = get_transforms(mean, std, train=False)   

    train_dataset = MammogramDataset(train_df, train_tf)
    val_dataset   = MammogramDataset(train_df.sample(frac=0.2, random_state=42), eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0)

    # Build model
    if args.model == "resnet18":
        model = build_model("ResNet18").to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == "resnet50":
        model = build_model("ResNet50").to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        # Use the staged function for ResNet50
        class CFG:
            stage1_epochs = 15
            stage2_epochs = 4
            stage3_epochs = 0
            head_lr = 5e-5
            layer4_lr = 3e-6
            layer3_lr = 1e-6
            weight_decay = 1e-4 
            early_stopping_patience = 6
            min_delta = 1e-4
        model = train_staged_resnet50(model, train_loader, val_loader, criterion, CFG(), OUTPUT_DIR)
        print("ResNet50 staged training finished!")
        return
    else:  # swindamfn 
        model = build_model("SwinDAMFN").to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()

    # Normal loop for ResNet18 and SwinDAMFN
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    early_stop = EarlyStopping(patience=8)

    for epoch in range(20):
        train_m = train_one_epoch(model, train_loader, criterion, optimizer)
        val_m = evaluate(model, val_loader, criterion)
        scheduler.step(val_m["loss"])
        print(f"Epoch {epoch+1} | Train Loss {train_m['loss']:.4f} | Val F1 {val_m.get('f1', 0):.4f}")
        if early_stop(val_m.get("f1", val_m["acc"])):
            break

    #Final evaluation  threshold  summary 

    print("Running test evaluation...")
    test_dfs = [prepare_case_df(csv, dicom_lookup) for csv in TEST_CSVS]
    test_df = pd.concat(test_dfs, ignore_index=True)
    test_df = test_df[test_df["has_roi"]].reset_index(drop=True)

    test_dataset = MammogramDataset(test_df, eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    y_val, p_val = collect_probs(model, val_loader, use_tta=True)
    best_thr = find_best_threshold(y_val, p_val)
    final_val = evaluate_from_probs(y_val, p_val, thr=best_thr["thr"])

    y_test, p_test = collect_probs(model, test_loader, use_tta=True)
    best_thr = find_best_threshold(y_test, p_test)
    final_test = evaluate_from_probs(y_test, p_test, thr=best_thr["thr"]) 

    summary = {
        "final_test": final_test,
        "final_val": final_val,
        "best_threshold": best_thr,
        "best_epoch": epoch + 1
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(model.state_dict(), OUTPUT_DIR / f"best_{args.model}.pth")
    print(f"Training finished for {args.model}:saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
