"""
all training related code here including staged training and early stopping and so on
"""

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.amp import autocast, GradScaler  

# Check gpu is good now
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Using AMP to try speed up training


#Training and evaluation functions
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                     optimizer: torch.optim.Optimizer, use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    labels_all, preds_all = [], []

    scaler = GradScaler(enabled=use_amp)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast(DEVICE.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(out, dim=1)
        labels_all.extend(y.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

    return {
        "loss": running_loss / len(loader.dataset),
        "acc": accuracy_score(labels_all, preds_all),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    labels_all, preds_all, probs_all = [], [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(DEVICE.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(out, dim=1)
        probs = torch.softmax(out, dim=1)[:, 1]

        labels_all.extend(y.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())

    return {
        "loss": running_loss / len(loader.dataset),
        "acc": accuracy_score(labels_all, preds_all),
        "f1": f1_score(labels_all, preds_all, zero_division=0),
        "auc": roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else 0.0,
    }


#Stopping and history
#TODO: history is giving grief come back to this

class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def save_history(history: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


#Build a staged training

def configure_stage(model: nn.Module, stage_name: str, cfg) -> torch.optim.Optimizer:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    param_groups = [{"params": model.fc.parameters(), "lr": cfg.head_lr}]

    if stage_name in {"layer4_finetune", "layer3_layer4_finetune"}:
        for param in model.layer4.parameters():
            param.requires_grad = True
        param_groups.append({"params": model.layer4.parameters(), "lr": cfg.layer4_lr})
    if stage_name == "layer3_layer4_finetune":
        for param in model.layer3.parameters():
            param.requires_grad = True
        param_groups.append({"params": model.layer3.parameters(), "lr": cfg.layer3_lr})

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def train_staged_resnet50(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                           criterion: nn.Module, cfg, output_dir: Path) -> nn.Module:
    history = {
        "epoch": [], "stage": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []
    }

    best_path = output_dir / "best_resnet50_staged.pth"
    global_epoch = 0
    best_val_f1 = -1.0
    best_stage = ""

    stage_plan = [
        ("head_only", getattr(cfg, "stage1_epochs", 10)),
        ("layer4_finetune", getattr(cfg, "stage2_epochs", 5)),
        ("layer3_layer4_finetune", getattr(cfg, "stage3_epochs", 3)),
    ]

    for stage_name, stage_epochs in stage_plan:
        optimizer = configure_stage(model, stage_name, cfg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        early_stop = EarlyStopping(patience=getattr(cfg, "early_stopping_patience", 6))

        print(f"\n=== Stage: {stage_name} | epochs={stage_epochs} ===")

        for _ in range(stage_epochs):
            global_epoch += 1
            train_m = train_one_epoch(model, train_loader, criterion, optimizer)
            val_m = evaluate(model, val_loader, criterion)

            scheduler.step(val_m["loss"])

            history["epoch"].append(global_epoch)
            history["stage"].append(stage_name)
            history["train_loss"].append(train_m["loss"])
            history["train_acc"].append(train_m["acc"])
            history["val_loss"].append(val_m["loss"])
            history["val_acc"].append(val_m["acc"])
            history["val_f1"].append(val_m["f1"])
            history["val_auc"].append(val_m["auc"])

            print(f"Epoch {global_epoch} | {stage_name} | Train Loss {train_m['loss']:.4f} | Val F1 {val_m['f1']:.4f}")

            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_stage = stage_name
                torch.save(model.state_dict(), best_path)
                print(f"   -- New best saved! (F1 {best_val_f1:.4f})")

            if early_stop(val_m["f1"]):
                print(f"Early stopping stage {stage_name}")
                break

    print(f"\nTraining finished! Best stage: {best_stage} | Best val F1: {best_val_f1:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    save_history(history, output_dir)
    return model
