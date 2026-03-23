# app.py
from pathlib import Path
import gradio as gr

from config import MODEL_OPTIONS, IMAGE_DIR
from core.models import load_model, get_target_layer
from core.evaluation import predict_with_gradcam

# Cache loaded models + target layers (fast switching in the demo)
_MODEL_CACHE = {}


def list_demo_images():
    """List all images in the demo_images folder."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts]
    return sorted([str(p) for p in files])


def run_inference(model_label: str, image_path: str):
    """Run prediction + Grad-CAM using the clean core modules."""
    if not image_path:
        return None, None, "Please select an image."

    # Load model + target layer (cached for speed)
    if model_label not in _MODEL_CACHE:
        model = load_model(model_label, MODEL_OPTIONS[model_label])
        target_layer = get_target_layer(model)
        _MODEL_CACHE[model_label] = (model, target_layer)

    model, target_layer = _MODEL_CACHE[model_label]

    # Use the centralised predict_with_gradcam from core/evaluation.py
    original, overlay, result_text = predict_with_gradcam(
        model, target_layer, image_path
    )

    return original, overlay, result_text


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="Breast Cancer Detection Demo") as demo:
    gr.Markdown(
        """
        # Breast Cancer Detection Demo
        **Select a trained model and a mammogram image** to see the prediction and Grad-CAM explanation.
        
        Models available:
        - ResNet18 Baseline
        - ResNet50 + CLAHE (your best staged model)
        - ResNet50 + Performance + CLAHE
        """
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value=list(MODEL_OPTIONS.keys())[0],
            label="Select Model",
        )

        image_dropdown = gr.Dropdown(
            choices=list_demo_images(),
            value=list_demo_images()[0] if list_demo_images() else None,
            label="Select Demo Image",
        )

    run_button = gr.Button("Run Prediction", variant="primary")

    with gr.Row():
        original_output = gr.Image(label="Original Mammogram", type="pil")
        gradcam_output = gr.Image(label="Grad-CAM Explanation", type="pil")

    result_box = gr.Textbox(label="Prediction Result", lines=4)

    run_button.click(
        fn=run_inference,
        inputs=[model_dropdown, image_dropdown],
        outputs=[original_output, gradcam_output, result_box],
    )

    gr.Markdown("**Tip:** Try different models to compare explanations!")

if __name__ == "__main__":
    demo.launch(share=False)   # set share=True if you want a public link