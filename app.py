from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

IMG_SIZE = (224, 224)
THRESHOLD = 0.5
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"

# Display sizes (pixels). Change these to make images bigger/smaller.
PREVIEW_WIDTH = 240      # MRI preview shown after selecting/uploading
GRADCAM_WIDTH = 260      # Original MRI + Grad-CAM overlay

# All models used for the prediction / majority vote
MODEL_CONFIG = {
    "EfficientNetB0": {"path": MODEL_DIR / "efficientnetb0_final.keras"},
    "ConvNeXt-BCE": {"path": MODEL_DIR / "convnext_tiny_bce.keras"},
    "ConvNeXt-Focal": {"path": MODEL_DIR / "convnext_tiny_focal.keras"},
}

# Grad-CAM is generated ONLY for this model
GRADCAM_MODEL_NAME = "EfficientNetB0"

SAMPLE_IMAGES = [
    ASSETS_DIR / "sample_1.jpg",
    ASSETS_DIR / "sample_2.jpg",
    ASSETS_DIR / "sample_3.jpg",
    ASSETS_DIR / "sample_4.jpg",
]


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource(show_spinner="Loading classification models...")
def load_all_models():
    """Load every configured model that exists; report problems without hiding them."""
    loaded = {}
    errors = {}

    for name, config in MODEL_CONFIG.items():
        model_path = config["path"]

        if not model_path.exists():
            errors[name] = f"Model file not found: {model_path.name}"
            continue

        try:
            loaded[name] = tf.keras.models.load_model(
                str(model_path), compile=False
            )
        except Exception as exc:
            errors[name] = str(exc)

    return loaded, errors


# -----------------------------
# Image preprocessing and prediction
# -----------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare RGB input. Training used float32 RGB pixels in [0, 255]."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_one_model(model, batch):
    """Return tumour probability, class label, and predicted-class score."""
    output = model(batch, training=False)
    output = np.asarray(output).reshape(-1)

    if output.size != 1:
        raise ValueError(
            f"Expected one sigmoid output for binary classification; "
            f"received {output.size}."
        )

    tumour_probability = float(np.clip(output[0], 0.0, 1.0))
    is_tumour = tumour_probability >= THRESHOLD

    return {
        "label": "Tumourous" if is_tumour else "Non-Tumourous",
        "tumour_probability": tumour_probability,
        "non_tumour_probability": 1.0 - tumour_probability,
        "confidence": (
            tumour_probability if is_tumour else 1.0 - tumour_probability
        ),
    }


def majority_vote(results: dict):
    """
    Majority vote across the models that produced a prediction.
    Returns (label, tumour_votes, total_votes, used_tiebreak).
    A tie (possible only with an even number of models) is broken by the
    average tumour probability.
    """
    total = len(results)
    tumour_votes = sum(r["label"] == "Tumourous" for r in results.values())

    if tumour_votes * 2 > total:
        return "Tumourous", tumour_votes, total, False
    if tumour_votes * 2 < total:
        return "Non-Tumourous", tumour_votes, total, False

    mean_prob = float(np.mean([r["tumour_probability"] for r in results.values()]))
    label = "Tumourous" if mean_prob >= THRESHOLD else "Non-Tumourous"
    return label, tumour_votes, total, True


# -----------------------------
# Grad-CAM helpers
# -----------------------------
def find_last_conv_layer(model):
    """Find the last Conv2D/DepthwiseConv2D layer and its top-level nested owner."""
    candidates = []

    def walk(layer, owner=None):
        for child in getattr(layer, "layers", []):
            if isinstance(child, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                candidates.append((child, owner))
            if getattr(child, "layers", None):
                next_owner = child if owner is None else owner
                walk(child, next_owner)

    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            candidates.append((layer, None))
        if getattr(layer, "layers", None):
            walk(layer, layer)

    if not candidates:
        return None, None
    return candidates[-1]


def build_gradcam_model(model, target_layer, backbone):
    """
    Return a model producing [target activations, final predictions].
    For nested backbones, build the activation graph within the backbone and
    replay the outer classifier layers after that backbone.
    """
    if backbone is None:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[target_layer.output, backbone.output],
    )

    model_input = model.inputs[0]
    activations, features = backbone_grad_model(model_input, training=False)
    x = features

    try:
        backbone_index = model.layers.index(backbone)
    except ValueError as exc:
        raise ValueError(
            "The selected convolutional layer's backbone is not a direct "
            "child of the outer model. An architecture-specific Grad-CAM "
            "path is needed for this saved model."
        ) from exc

    for layer in model.layers[backbone_index + 1:]:
        x = layer(x)

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[activations, x],
    )


def make_gradcam_heatmap(batch: np.ndarray, model, target_layer, backbone, tumour_class: bool):
    """Create a normalized Grad-CAM heatmap for the selected binary class."""
    grad_model = build_gradcam_model(model, target_layer, backbone)
    image_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_activations, predictions = grad_model(image_tensor, training=False)
        tumour_score = predictions[:, 0]
        class_score = tumour_score if tumour_class else (1.0 - tumour_score)

    gradients = tape.gradient(class_score, conv_activations)
    if gradients is None:
        raise ValueError(
            "Gradients are disconnected from the selected convolutional layer. "
            "Try a different target layer or inspect the saved model architecture."
        )

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_activations, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0.0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )
    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.38):
    """Overlay a Grad-CAM heatmap on the original MRI image."""
    original = np.asarray(image.convert("RGB"))
    resized_heatmap = cv2.resize(
        heatmap, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    heatmap_uint8 = np.uint8(np.clip(resized_heatmap, 0, 1) * 255)
    # OpenCV's JET colormap is BGR; convert to RGB before overlay.
    colored = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    overlay = (1.0 - alpha) * original.astype(np.float32) + alpha * colored.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8), heatmap_uint8


# -----------------------------
# User interface
# -----------------------------
st.title("Brain MRI Classification")
st.write(
    "Upload a brain MRI image to obtain a binary prediction from multiple models, "
    "a majority-vote result, and a Grad-CAM visualization of image regions that "
    "influenced the EfficientNetB0 prediction."
)

st.info(
    "Research demonstration only. This app is not a medical diagnostic tool. "
    "A qualified radiologist must interpret MRI scans and make clinical decisions."
)

with st.sidebar:
    st.header("Model settings")
    st.write("Models: " + ", ".join(MODEL_CONFIG.keys()))
    st.write("Consensus: majority vote")
    st.write(f"Grad-CAM model: {GRADCAM_MODEL_NAME}")
    st.write(f"Decision threshold: {THRESHOLD:.2f}")
    st.caption("Preprocessing assumes RGB, 224 × 224, float32 pixel values in [0, 255].")


# Let the user choose a sample MRI or upload their own image.
st.subheader("Try a sample MRI or upload your own")

available_samples = [path for path in SAMPLE_IMAGES if path.exists()]
sample_options = ["Upload my own MRI"] + [path.name for path in available_samples]

selected_sample = st.selectbox(
    "Choose a sample image or upload your own",
    sample_options,
)

uploaded_file = None
if selected_sample == "Upload my own MRI":
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=["png", "jpg", "jpeg"],
    )

image = None

if selected_sample != "Upload my own MRI":
    selected_path = next(
        path for path in available_samples if path.name == selected_sample
    )
    image = Image.open(selected_path).convert("RGB")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

if image is not None:
    st.image(image, caption="MRI selected for analysis", width=PREVIEW_WIDTH)


# Run all classifiers, show each prediction, the majority vote, and Grad-CAM.
if image is not None and st.button("Analyze MRI", type="primary"):
    batch = preprocess_image(image)

    with st.spinner("Loading models and running predictions..."):
        models, load_errors = load_all_models()

    for model_name, error in load_errors.items():
        st.warning(f"{model_name} could not be loaded and was skipped: {error}")

    if not models:
        st.error("No models loaded. Check the files inside your models folder.")
        st.stop()

    results = {}

    for model_name, keras_model in models.items():
        try:
            results[model_name] = predict_one_model(keras_model, batch)
        except Exception as exc:
            st.warning(f"Prediction failed for {model_name}: {exc}")

    if not results:
        st.error("All model predictions failed.")
        st.stop()

    # ----- Majority-vote consensus -----
    final_label, tumour_votes, total_votes, used_tiebreak = majority_vote(results)
    agreeing = tumour_votes if final_label == "Tumourous" else total_votes - tumour_votes
    mean_tumour_prob = float(np.mean([r["tumour_probability"] for r in results.values()]))

    st.divider()
    st.subheader("Final consensus prediction (majority vote)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Consensus", final_label)
    col2.metric("Models agreeing", f"{agreeing} / {total_votes}")
    col3.metric("Average tumour probability", f"{mean_tumour_prob * 100:.2f}%")

    st.write(
        f"**{tumour_votes} of {total_votes}** models predict *Tumourous*; "
        f"**{total_votes - tumour_votes} of {total_votes}** predict *Non-Tumourous*."
    )

    if used_tiebreak:
        st.caption(
            "The vote was tied, so the average tumour probability across models "
            "was used as the tiebreaker."
        )
    if total_votes < len(MODEL_CONFIG):
        st.caption(
            f"Only {total_votes} of {len(MODEL_CONFIG)} configured models were "
            "available, so the vote uses those models."
        )
    st.caption(
        "Majority voting is not a calibrated ensemble or proof that the "
        "prediction is correct."
    )

    # ----- Individual model predictions -----
    st.subheader("Individual model predictions")

    columns = st.columns(len(results))
    for column, (model_name, result) in zip(columns, results.items()):
        with column:
            st.markdown(f"**{model_name}**")
            st.write(f"Prediction: **{result['label']}**")
            st.metric("Predicted-class score", f"{result['confidence'] * 100:.2f}%")
            st.write(f"Tumourous: {result['tumour_probability'] * 100:.2f}%")
            st.write(f"Non-Tumourous: {result['non_tumour_probability'] * 100:.2f}%")

    # ----- Grad-CAM (EfficientNetB0 only) -----
    st.divider()
    st.subheader("Explainable AI: Grad-CAM")

    if GRADCAM_MODEL_NAME not in models:
        st.warning(f"{GRADCAM_MODEL_NAME} is unavailable, so Grad-CAM cannot be generated.")
    elif GRADCAM_MODEL_NAME not in results:
        st.warning(f"{GRADCAM_MODEL_NAME} prediction failed, so Grad-CAM cannot be generated.")
    else:
        gradcam_model = models[GRADCAM_MODEL_NAME]
        gradcam_result = results[GRADCAM_MODEL_NAME]

        try:
            target_layer, backbone = find_last_conv_layer(gradcam_model)

            if target_layer is None:
                st.warning("No convolutional target layer was found.")
            else:
                st.caption(
                    f"Grad-CAM model: {GRADCAM_MODEL_NAME} · "
                    f"Target layer: {target_layer.name}"
                )

                with st.spinner(f"Generating {GRADCAM_MODEL_NAME} Grad-CAM..."):
                    heatmap = make_gradcam_heatmap(
                        batch,
                        gradcam_model,
                        target_layer,
                        backbone,
                        tumour_class=(gradcam_result["label"] == "Tumourous"),
                    )
                    overlay, _ = overlay_heatmap(image, heatmap)

                original_col, heatmap_col, _spacer = st.columns([1, 1, 2])

                with original_col:
                    st.image(image, caption="Original MRI", width=GRADCAM_WIDTH)

                with heatmap_col:
                    st.image(
                        overlay,
                        caption="Heatmap (Grad-CAM overlay)",
                        width=GRADCAM_WIDTH,
                    )

                st.caption(
                    f"The heatmap explains {GRADCAM_MODEL_NAME}'s own prediction "
                    f"({gradcam_result['label']}), which may differ from the "
                    "majority vote. It is not a tumour segmentation mask."
                )

        except Exception as exc:
            st.warning(f"{GRADCAM_MODEL_NAME} Grad-CAM failed: {exc}")
