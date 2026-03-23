from pathlib import Path

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "demo_images"

MODEL_OPTIONS = {
    "ResNet18 Baseline": BASE_DIR / "resnet18_run"/ "best_resnet18.pth",
    "ResNet50": BASE_DIR / "resnet50_run" / "best_resnet50_staged.pth",
    "SwinDAMFN (Best Metrics)": BASE_DIR / "swindamfn_staged_run" / "best_swindamfn_staged.pth",
}