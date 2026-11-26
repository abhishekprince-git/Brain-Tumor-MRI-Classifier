import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# =====================================================
# Configuration
# =====================================================
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (224, 224)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_NAME = "EfficientNetB0"
MODEL_FILE = "efficientnetb0_final.keras"

# The model is looked up in these places, in order (repo has it in models/,
# older versions of app.py expected it in the project root).
MODEL_CANDIDATES = [
    MODEL_DIR / MODEL_FILE,
    PROJECT_DIR / MODEL_FILE,
    RESULTS_DIR / MODEL_FILE,
]

SAMPLE_IMAGES = [ASSETS_DIR / f"sample_{i}.jpg" for i in range(1, 5)]

# Optional: put test-set results in assets/metrics.json (or results/metrics.json), e.g.
# {"accuracy": 0.9573, "precision": 0.99, "recall": 0.95, "f1": 0.97}
# If no file is found, the performance section is simply hidden.
METRICS_CANDIDATES = [ASSETS_DIR / "metrics.json", RESULTS_DIR / "metrics.json"]

GREEN = "#2e9e6b"
RED = "#d9534f"
AMBER = "#e0a030"

# =====================================================
# Styling (works in light and dark themes)
# =====================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(88,101,242,0.22), rgba(46,158,107,0.18));
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1.4rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.2rem;}
    .hero p {margin: 0; opacity: .85; font-size: 1.02rem;}

    .pill {
        display: inline-block; padding: .18rem .7rem; margin: .8rem .4rem 0 0;
        border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: rgba(128,128,128,0.18); border: 1px solid rgba(128,128,128,0.25);
    }

    .result-card {
        padding: 1.3rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .result-card.tumour {background: rgba(217,83,79,0.13); border-color: rgba(217,83,79,0.5);}
    .result-card.clear  {background: rgba(46,158,107,0.13); border-color: rgba(46,158,107,0.5);}
    .result-label {font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7;}
    .result-value {font-size: 2rem; font-weight: 700; line-height: 1.2; margin: .15rem 0 .3rem 0;}
    .result-sub {opacity: .8; font-size: .95rem;}

    .bar-wrap {margin: .55rem 0;}
    .bar-head {display: flex; justify-content: space-between; font-weight: 600; margin-bottom: .25rem;}
    .bar-track {width: 100%; height: 14px; border-radius: 999px; background: rgba(128,128,128,0.2); overflow: hidden;}
    .bar-fill {height: 100%; border-radius: 999px;}

    .step-card {
        padding: 1.1rem 1.2rem; border-radius: 14px; height: 100%;
        background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2);
    }
    .step-card h4 {margin: 0 0 .4rem 0;}
    .footer {text-align: center; opacity: .6; font-size: .85rem; margin-top: 2.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Streamlit version-compat helpers
# (use_container_width was replaced by width="stretch" in newer Streamlit)
# =====================================================
def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


def primary_button(label: str) -> bool:
    try:
        return st.button(label, type="primary", width="stretch")
    except TypeError:
        return st.button(label, type="primary", use_container_width=True)


# =====================================================
# Model loading
# =====================================================
def resolve_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    # Last resort: any .keras file in the usual folders
    for folder in (MODEL_DIR, PROJECT_DIR, RESULTS_DIR):
        if folder.exists():
            found = sorted(folder.glob("*.keras"))
            if found:
                return found[0]
    searched = ", ".join(str(p) for p in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model file not found. Looked in: {searched}")


@st.cache_resource(show_spinner="Loading EfficientNetB0 model...")
def load_model():
    path = resolve_model_path()
    return tf.keras.models.load_model(str(path), compile=False)


# =====================================================
# Preprocessing and prediction
# =====================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """RGB, 224x224, float32 pixels in [0, 255] (EfficientNet rescales internally)."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_tumour_probability(model, batch) -> float:
    output = np.asarray(model(batch, training=False)).reshape(-1)
    if output.size != 1:
        raise ValueError(
            f"Expected one sigmoid output for binary classification; got {output.size}."
        )
    return float(np.clip(output[0], 0.0, 1.0))


def build_result(tumour_probability: float, threshold: float) -> dict:
    is_tumour = tumour_probability >= threshold
    return {
        "is_tumour": is_tumour,
        "label": "Tumourous" if is_tumour else "Non-Tumourous",
        "tumour_probability": tumour_probability,
        "non_tumour_probability": 1.0 - tumour_probability,
        "confidence": tumour_probability if is_tumour else 1.0 - tumour_probability,
    }


# =====================================================
# Grad-CAM helpers
# =====================================================
def find_last_conv_layer(model):
    """Find the last Conv2D/DepthwiseConv2D layer and the nested backbone that owns it."""
    candidates = []
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)

    def walk(layer, owner):
        for child in getattr(layer, "layers", []):
            if isinstance(child, conv_types):
                candidates.append((child, owner))
            if getattr(child, "layers", None):
                walk(child, owner)

    for layer in model.layers:
        if isinstance(layer, conv_types):
            candidates.append((layer, None))
        if getattr(layer, "layers", None):
            walk(layer, layer)

    if not candidates:
        return None, None
    return candidates[-1]


def build_gradcam_model(model, target_layer, backbone):
    """Model returning [target activations, final predictions].

    If the conv layer lives inside a nested backbone, every layer of the outer
    model is replayed in order (so any layers before the backbone, such as
    rescaling or augmentation, are kept) and the backbone is swapped for a
    version that also exposes the target activations.
    """
    if backbone is None:
        return tf.keras.Model(
            inputs=model.inputs, outputs=[target_layer.output, model.output]
        )

    inner = tf.keras.Model(
        inputs=backbone.inputs, outputs=[target_layer.output, backbone.output]
    )

    outer_layers = [
        layer
        for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    if not any(layer is backbone for layer in outer_layers):
        raise ValueError(
            "The convolutional layer's backbone is not a direct child of the "
            "outer model; an architecture-specific Grad-CAM path is needed."
        )

    x = model.inputs[0]
    activations = None
    for layer in outer_layers:
        if layer is backbone:
            activations, x = inner(x, training=False)
        else:
            x = layer(x)

    return tf.keras.Model(inputs=model.inputs, outputs=[activations, x])


def make_gradcam_heatmap(batch, grad_model, tumour_class: bool) -> np.ndarray:
    """Normalized Grad-CAM heatmap (0-1) for the tumour or the non-tumour class."""
    image_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_activations, predictions = grad_model(image_tensor, training=False)
        tumour_score = predictions[:, 0]
        class_score = tumour_score if tumour_class else (1.0 - tumour_score)

    gradients = tape.gradient(class_score, conv_activations)
    if gradients is None:
        raise ValueError("Gradients are disconnected from the selected layer.")

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


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Pure-NumPy 'jet' colormap: values in [0, 1] -> uint8 RGB.

    Replaces OpenCV so the app needs no cv2 / libGL on Streamlit Cloud.
    """
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def colorize_heatmap(heatmap: np.ndarray, size) -> np.ndarray:
    """Heatmap -> RGB colour image of the given (width, height)."""
    resized = Image.fromarray(heatmap.astype(np.float32), mode="F").resize(
        size, Image.BILINEAR
    )
    return jet_colormap(np.asarray(resized, dtype=np.float32))


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float):
    original = np.asarray(image.convert("RGB"))
    colored = colorize_heatmap(heatmap, (original.shape[1], original.shape[0]))
    blended = (1.0 - alpha) * original.astype(np.float32) + alpha * colored.astype(np.float32)
    return colored, np.clip(blended, 0, 255).astype(np.uint8)


def to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# =====================================================
# Analysis (stored in session_state so sliders don't reset it)
# =====================================================
def run_analysis(model, image: Image.Image) -> dict:
    start = time.time()
    batch = preprocess_image(image)
    tumour_probability = predict_tumour_probability(model, batch)

    analysis = {
        "tumour_probability": tumour_probability,
        "heat_tumour": None,
        "heat_clear": None,
        "layer_name": None,
        "gradcam_error": None,
    }

    try:
        target_layer, backbone = find_last_conv_layer(model)
        if target_layer is None:
            raise ValueError("No convolutional target layer was found.")
        grad_model = build_gradcam_model(model, target_layer, backbone)
        analysis["heat_tumour"] = make_gradcam_heatmap(batch, grad_model, True)
        analysis["heat_clear"] = make_gradcam_heatmap(batch, grad_model, False)
        analysis["layer_name"] = target_layer.name
    except Exception as exc:  # Grad-CAM must never break the prediction
        analysis["gradcam_error"] = str(exc)

    analysis["seconds"] = time.time() - start
    return analysis


def image_key(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.md5(rgb.tobytes())
    digest.update(str(rgb.size).encode())
    return digest.hexdigest()


# =====================================================
# UI helpers
# =====================================================
def probability_bar(label: str, fraction: float, color: str) -> str:
    pct = fraction * 100
    return f"""
    <div class="bar-wrap">
      <div class="bar-head"><span>{label}</span><span>{pct:.2f}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%; background:{color};"></div></div>
    </div>
    """


def confidence_level(confidence: float):
    if confidence >= 0.90:
        return "High", GREEN
    if confidence >= 0.70:
        return "Moderate", AMBER
    return "Low", RED


def load_metrics():
    for path in METRICS_CANDIDATES:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict) and data:
                    # keep only numeric entries
                    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            except Exception:
                return None
    return None


def open_uploaded_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
    return img.convert("RGB")


# =====================================================
# Hero + sidebar
# =====================================================
st.markdown(
    f"""
    <div class="hero">
      <h1>🧠 Brain MRI Classifier</h1>
      <p>Upload a brain MRI slice to get a Tumourous / Non-Tumourous prediction,
         class probabilities, and a Grad-CAM view of the regions that influenced the model.</p>
      <span class="pill">Deep learning</span>
      <span class="pill">{MODEL_NAME}</span>
      <span class="pill">Transfer learning</span>
      <span class="pill">Explainable AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Images with a tumour probability at or above this value are labelled "
        "Tumourous. Lowering it catches more tumours (higher recall) but gives more "
        "false alarms.",
    )
    overlay_alpha = st.slider(
        "Grad-CAM overlay strength", 0.10, 0.80, 0.40, 0.05,
        help="How strongly the heatmap is blended over the MRI.",
    )
    st.divider()
    st.caption(f"Model: **{MODEL_NAME}**")
    st.caption("Input: RGB, 224 × 224, pixel values 0–255")
    st.caption("Task: binary classification (Tumourous vs Non-Tumourous)")

st.warning(
    "⚠️ **Research demonstration only.** This is not a medical device and must not be "
    "used for diagnosis. MRI scans must be interpreted by a qualified radiologist.",
)

tab_analyze, tab_how, tab_about = st.tabs(["🔍 Analyze", "🧪 How it works", "ℹ️ About & limitations"])

# =====================================================
# Tab 1: Analyze
# =====================================================
with tab_analyze:
    left, right = st.columns([1, 1.15], gap="large")

    image = None
    with left:
        st.subheader("1 · Choose an MRI")
        available_samples = [p for p in SAMPLE_IMAGES if p.exists()]
        options = ["Upload my own image"] + [p.name for p in available_samples]
        source = st.selectbox("Image source", options)

        if source == "Upload my own image":
            uploaded = st.file_uploader(
                "Upload a brain MRI (PNG / JPG)", type=["png", "jpg", "jpeg"]
            )
            if uploaded is not None:
                try:
                    image = open_uploaded_image(uploaded)
                except Exception:
                    st.error("That file could not be read as an image.")
        else:
            chosen = next(p for p in available_samples if p.name == source)
            image = Image.open(chosen).convert("RGB")

        if image is not None:
            show_image(image, caption="Selected MRI")
            analyze_clicked = primary_button("🔬 Analyze MRI")
        else:
            analyze_clicked = False
            st.info("Select a sample or upload an MRI image to begin.")

    current_key = image_key(image) if image is not None else None

    # ---- run analysis when requested ----
    if image is not None and analyze_clicked:
        try:
            model = load_model()
        except Exception as exc:
            st.error(f"Could not load the model: {exc}")
            st.stop()

        with st.spinner("Running prediction and generating Grad-CAM..."):
            try:
                result = run_analysis(model, image)
                result["key"] = current_key
                st.session_state["analysis"] = result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

    analysis = st.session_state.get("analysis")
    has_result = (
        image is not None and analysis is not None and analysis["key"] == current_key
    )

    # ---- results column ----
    with right:
        st.subheader("2 · Result")
        if not has_result:
            st.caption("The prediction will appear here after you click **Analyze MRI**.")
        else:
            res = build_result(analysis["tumour_probability"], threshold)
            card_class = "tumour" if res["is_tumour"] else "clear"
            level, level_color = confidence_level(res["confidence"])

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                  <div class="result-label">Predicted class</div>
                  <div class="result-value">{res['label']}</div>
                  <div class="result-sub">
                    Predicted-class score: <b>{res['confidence'] * 100:.2f}%</b> ·
                    Confidence: <b style="color:{level_color}">{level}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class probabilities, exactly as: Tumourous: 19.08% / Non-Tumourous: 80.92%
            st.markdown("**Class probabilities**")
            st.markdown(
                probability_bar("Tumourous", res["tumour_probability"], RED)
                + probability_bar("Non-Tumourous", res["non_tumour_probability"], GREEN),
                unsafe_allow_html=True,
            )
            st.write(f"Tumourous: {res['tumour_probability'] * 100:.2f}%")
            st.write(f"Non-Tumourous: {res['non_tumour_probability'] * 100:.2f}%")

            if 0.35 <= res["tumour_probability"] <= 0.65:
                st.warning(
                    "The model is close to the decision boundary for this image. "
                    "Treat this prediction with extra caution."
                )

            c1, c2 = st.columns(2)
            c1.metric("Decision threshold", f"{threshold:.2f}")
            c2.metric("Inference time", f"{analysis['seconds']:.1f} s")

            st.caption(
                "Scores are model outputs, not calibrated clinical probabilities, "
                "and do not prove the prediction is correct."
            )

    # ---- Grad-CAM (full width) ----
    if has_result:
        st.divider()
        st.subheader("3 · Explainable AI: Grad-CAM")

        if analysis["gradcam_error"]:
            st.warning(f"Grad-CAM could not be generated: {analysis['gradcam_error']}")
        else:
            res = build_result(analysis["tumour_probability"], threshold)
            heatmap = analysis["heat_tumour"] if res["is_tumour"] else analysis["heat_clear"]
            colored, overlay = overlay_heatmap(image, heatmap, overlay_alpha)

            target_class = "Tumourous" if res["is_tumour"] else "Non-Tumourous"
            st.caption(
                f"Heatmap for the predicted class (**{target_class}**) · "
                f"Model: {MODEL_NAME} · Target layer: `{analysis['layer_name']}`"
            )

            g1, g2, g3 = st.columns(3)
            with g1:
                show_image(image, caption="Original MRI")
            with g2:
                show_image(colored, caption="Grad-CAM heatmap")
            with g3:
                show_image(overlay, caption="Overlay")

            st.caption(
                "Warmer colours (red/yellow) mark regions that pushed the model towards "
                "its prediction. This is **not** a tumour segmentation mask, and the "
                "highlighted area is not guaranteed to be the true lesion."
            )

            st.download_button(
                "⬇️ Download overlay (PNG)",
                data=to_png_bytes(overlay),
                file_name="gradcam_overlay.png",
                mime="image/png",
            )

# =====================================================
# Tab 2: How it works
# =====================================================
with tab_how:
    st.subheader("From MRI slice to prediction")
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1 · Preprocess", "The image is converted to RGB and resized to 224 × 224 pixels."),
        ("2 · Extract features", f"{MODEL_NAME}, pre-trained on ImageNet, extracts visual features from the MRI."),
        ("3 · Classify", "A sigmoid output gives the probability that the scan is Tumourous."),
        ("4 · Explain", "Grad-CAM highlights image regions that influenced the prediction."),
    ]
    for col, (title, text) in zip((s1, s2, s3, s4), steps):
        with col:
            st.markdown(
                f'<div class="step-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(
        """
**Decision threshold.** The model outputs a probability. By default, anything at or
above 0.50 is labelled *Tumourous*. In a screening-style problem, missing a tumour is
usually worse than a false alarm, so a lower threshold (the sidebar slider) trades some
precision for higher recall.

**Grad-CAM.** Gradients of the predicted class are propagated back to the last
convolutional layer. Averaging them gives importance weights for each feature map, and
the weighted sum is shown as a heatmap over the original image.
        """
    )

    metrics = load_metrics()
    if metrics:
        st.subheader("Test-set performance")
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_", " ").title(), f"{value * 100:.2f}%")

# =====================================================
# Tab 3: About
# =====================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
This app classifies brain MRI slices as **Tumourous** or **Non-Tumourous** using
transfer learning ({MODEL_NAME}). It was trained on a public brain-tumour MRI dataset
in which glioma, meningioma and pituitary tumour images form the *Tumourous* class.

**Limitations**
- Binary only: it does not identify the tumour type or location.
- Trained on a single public dataset; performance on scans from other scanners,
  hospitals or imaging protocols is unknown.
- The dataset contains augmented images, so reported scores may be optimistic.
- Grad-CAM shows model attention, not a verified lesion boundary.
- It should not be used for diagnosis, screening or any clinical decision.
        """
    )

st.markdown(
    '<div class="footer">Built with Streamlit & TensorFlow · For research and education only</div>',
    unsafe_allow_html=True,
)
