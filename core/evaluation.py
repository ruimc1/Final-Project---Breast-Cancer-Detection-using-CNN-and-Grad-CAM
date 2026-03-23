"""
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
https://docs.pytorch.org/docs/stable/index.html
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from typing import Dict, Tuple
from PIL import Image
import cv2
from core.transforms import get_eval_transform
from core.data import MammogramDataset
from core.models import DEVICE

CLASS_NAMES = {0: "Benign", 1: "Malignant"}


@torch.no_grad()
def tta_predict(model, img_tensor):
    """
    TTA
    """
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    logits = model(img_tensor)
    logits += model(torch.flip(img_tensor, dims=[-1]))
    for k in range(1, 3):
        rot = torch.rot90(img_tensor, k, dims=[-2, -1]) 
        logits += model(rot)
        logits += model(torch.flip(rot, dims=[-1]))
    return logits / 7.0


def collect_probs(model, df, use_tta=True):
    """Safe version - forces CPU to avoid device mismatch"""
    model = model.to("cpu")          # Force model to CPU
    model.eval()
    
    loader = torch.utils.data.DataLoader(
        MammogramDataset(df, get_eval_transform(mean=[0.4782], std=[0.2124])),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    y_true = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to("cpu")          # Force input to CPU

            if use_tta:
                logits = tta_predict(model, x)
            else:
                logits = model(x)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs)

    return np.array(y_true), np.array(y_prob)


def find_best_threshold(y_true, probs):
    best = {"thr": 0.5, "f1": -1.0}
    for thr in np.arange(0.2, 0.81, 0.01):
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best["f1"]:
            best = {
                "thr": float(thr),
                "acc": float(accuracy_score(y_true, preds)), 
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1": float(score), 
            }
    return best


def evaluate_from_probs(y_true, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "acc": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)), 
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "auc": float(roc_auc_score(y_true, probs)),
        "cm": cm.tolist(),
    }


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Old working loop version (this is what made it look good before)
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), output.detach()

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def predict_with_gradcam(model, target_layer, image_path: str):
    """Fixed version that matches your old working Grad-CAM (grayscale + correct normalization)"""
    from PIL import Image
    import torchvision.transforms as transforms

    # Load as grayscale (1 channel) - exactly as your ResNet50 expects
    original = Image.open(image_path).convert("L")

    # Use the same normalization as your training pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4782], std=[0.2124])
    ])
    tensor = transform(original).unsqueeze(0).to(DEVICE)

    # Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    cam, raw_output = grad_cam.generate(tensor)
    grad_cam.remove()

    probs = torch.softmax(raw_output, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # Old overlay style that looked good
    original_resized = original.resize((224, 224))
    original_np = np.array(original_resized.convert("RGB"))  # convert only for overlay

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

    result_text = (
        f"Prediction: {CLASS_NAMES[pred_idx]}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Probabilities -> Benign: {probs[0]:.4f}, Malignant: {probs[1]:.4f}"
    )

    return original_resized, Image.fromarray(overlay), result_text

